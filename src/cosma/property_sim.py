"""Sequential Gaussian simulation of a cell property (porosity, VSHALE, ...) per interval and facies.

Each (interval, facies) is its own domain: only cells of that facies in that interval are simulated, conditioned on the cell-scale
hard values of that domain (never on other facies or intervals). The values are normal-score transformed with the domain's
distribution, simulated with simple kriging on a multiple-grid path (the Gaussian kernel of cosma.tgs, hard values exact) and back
transformed. The distribution of every domain is an explicit choice:
  'data'  the empirical distribution of the domain's cell-scale values (needs at least ``min_data`` of them; the back transform
          does not extrapolate beyond the data range), or
  'user'  a normal or log-normal distribution typed by the user (mean and standard deviation of the PROPERTY at cell scale);
          hard values that are very unlikely under it (probability < ``conflict_probability`` in either tail) are a conflict and stop
          the run.
A domain without a source, without enough data for 'data', without a normal-score variogram, or with hard values outside the
bounds stops the run with the reason; nothing is borrowed from another domain. Trends are not part of this step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .progress import report
from scipy.special import ndtri
from scipy.stats import norm

from .sis import SisError, _pack, cell_xyz, make_template
from .tgs import _simulate as _gauss_simulate

KINDS = ("normal", "lognormal")


class PropertySimError(SisError):
    pass


@dataclass(frozen=True)
class Distribution:
    kind: str
    mean: float            # of the property itself
    std: float             # of the property itself

    def __post_init__(self):
        if self.kind not in KINDS:
            raise PropertySimError(f"distribution kind must be one of {', '.join(KINDS)}")
        if not (isinstance(self.mean, (int, float)) and math.isfinite(self.mean)):
            raise PropertySimError("the mean must be a number")
        if not (isinstance(self.std, (int, float)) and math.isfinite(self.std) and self.std > 0):
            raise PropertySimError("the standard deviation must be a number > 0")
        if self.kind == "lognormal" and self.mean <= 0:
            raise PropertySimError("a log-normal distribution needs a mean > 0")

    def _mu_sigma(self):
        s2 = math.log(1.0 + (self.std / self.mean) ** 2)
        return math.log(self.mean) - 0.5 * s2, math.sqrt(s2)

    def cdf(self, v):
        v = np.asarray(v, float)
        if self.kind == "normal":
            return norm.cdf(v, self.mean, self.std)
        mu, sg = self._mu_sigma()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(v > 0, norm.cdf(np.log(np.where(v > 0, v, 1.0)), mu, sg), 0.0)

    def to_z(self, v):
        v = np.asarray(v, float)
        if self.kind == "normal":
            return (v - self.mean) / self.std
        mu, sg = self._mu_sigma()
        return (np.log(v) - mu) / sg

    def from_z(self, z):
        z = np.asarray(z, float)
        if self.kind == "normal":
            return self.mean + self.std * z
        mu, sg = self._mu_sigma()
        return np.exp(mu + sg * z)


@dataclass(frozen=True)
class PropertySetup:
    name: str                       # property name (also the name of its PropertySpec in the data checks, if data are used)
    variogram: str                  # variogram name on each interval; '{facies}' is replaced by the facies name
    sources: tuple                  # (((interval, facies code), 'data' | 'user'), ...)
    user: tuple = ()                # (((interval, facies code), Distribution), ...)
    min_data: int = 1               # smallest number of cell values for 'data'
    conflict_probability: float = 0.001
    lower: float | None = None      # optional bounds of the property; hard values outside them stop the run, simulated ones are clipped
    upper: float | None = None
    max_neighbours: int = 12
    search_factor: float = 1.0
    levels: int = 1
    correlate_with: str | None = None   # name of a property simulated before this one (None = independent of the others)
    correlation: tuple = ()             # (((interval, facies code), rho), ...) correlation of the normal scores, |rho| < 1

    def __post_init__(self):
        if self.correlate_with is None:
            if self.correlation:
                raise PropertySimError("correlations are given but no property to correlate with")
        else:
            if not (isinstance(self.correlate_with, str) and self.correlate_with.strip()) or self.correlate_with == self.name:
                raise PropertySimError("correlate_with must name another property")
            ck = [k for k, _ in self.correlation]
            if len(set(ck)) != len(ck):
                raise PropertySimError("a domain has two correlations")
            for k, r in self.correlation:
                if not (isinstance(r, (int, float)) and not isinstance(r, bool) and math.isfinite(r) and -1 < r < 1):
                    raise PropertySimError(f"domain {k}: the correlation must be a number between -1 and 1 (not -1 or 1)")
        if not (isinstance(self.name, str) and self.name.strip()) or not (isinstance(self.variogram, str) and self.variogram.strip()):
            raise PropertySimError("the property needs a name and a variogram name")
        keys = [k for k, _ in self.sources]
        if len(set(keys)) != len(keys):
            raise PropertySimError("a domain has two sources")
        for k, s in self.sources:
            if s not in ("data", "user"):
                raise PropertySimError(f"domain {k}: source must be 'data' or 'user'")
        ud = dict(self.user)
        for k, d in ud.items():
            if dict(self.sources).get(k) != "user" or not isinstance(d, Distribution):
                raise PropertySimError(f"domain {k}: a distribution is given but the source is not 'user'")
        for k, s in self.sources:
            if s == "user" and k not in ud:
                raise PropertySimError(f"domain {k}: the source is 'user' but no distribution is typed")
        for n, v in (("min_data", self.min_data), ("max_neighbours", self.max_neighbours), ("levels", self.levels)):
            if not (isinstance(v, int) and not isinstance(v, bool) and v >= 1):
                raise PropertySimError(f"{n} must be a whole number >= 1")
        if not (isinstance(self.conflict_probability, (int, float)) and 0 <= self.conflict_probability < 0.5):
            raise PropertySimError("conflict_probability must be a number in [0, 0.5)")
        if not (isinstance(self.search_factor, (int, float)) and math.isfinite(self.search_factor) and self.search_factor > 0):
            raise PropertySimError("search_factor must be a number > 0")
        for n, v in (("lower", self.lower), ("upper", self.upper)):
            if v is not None and not (isinstance(v, (int, float)) and math.isfinite(v)):
                raise PropertySimError(f"{n} must be a number")
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise PropertySimError("lower must be below upper")


@dataclass
class PropertyResult:
    values: np.ndarray                       # (nx, ny, nz) float, NaN where there is no facies (inactive)
    report: str
    per_domain: dict = field(default_factory=dict)


class _Empirical:
    """Normal-score transform of a data set: plotting positions (rank - 0.5) / n; back transform interpolates linearly and holds the
    end values beyond the data range."""

    def __init__(self, v):
        v = np.sort(np.asarray(v, float))
        n = len(v)
        self.v = v
        self.z = ndtri((np.arange(n) + 0.5) / n)
        self.n = n

    def to_z(self, x):
        x = np.asarray(x, float)
        # ties share the mean of their plotting positions
        lo = np.searchsorted(self.v, x, side="left")
        hi = np.searchsorted(self.v, x, side="right")
        if ((hi - lo) == 0).any():
            raise PropertySimError("a hard value is not in the data used for the transform")
        return ndtri(((lo + hi) / 2.0) / self.n)

    def from_z(self, z):
        return np.interp(np.asarray(z, float), self.z, self.v)

    def score(self, x):
        """Continuous normal score of simulated values (the inverse of ``from_z``; values beyond the data range get the end scores)."""
        return np.interp(np.asarray(x, float), self.v, self.z)


class _Geom:
    """Cell centres, thickness and activity of a grid, computed once per call."""

    def __init__(self, project, model):
        from .grid.builder import _cell_thickness
        grid = model.grid
        self.grid = grid
        self.X, self.Y, self.Z = cell_xyz(grid)
        self.thick = _cell_thickness(grid.zcorn, grid.nx, grid.ny, grid.nz)
        self.active = (np.asarray(grid.actnum) > 0) & np.isfinite(self.thick) & (self.thick > 0)
        self.by_name = {iv.name: iv for iv in project.chart.intervals}
        X, Y = self.X, self.Y
        self.dxm = float(np.hypot(np.diff(X[:, :, 0], axis=0).mean() if grid.nx > 1 else 1, np.diff(Y[:, :, 0], axis=0).mean() if grid.nx > 1 else 0))
        self.dym = float(np.hypot(np.diff(X[:, :, 0], axis=1).mean() if grid.ny > 1 else 0, np.diff(Y[:, :, 0], axis=1).mean() if grid.ny > 1 else 1))


def gaussian_field(project, geom: _Geom, name: str, fname: str, vname: str, dom, sl, hard_z: dict, max_neighbours: int,
                   search_factor: float, levels: int, seed: int, err=None):
    """A standard-normal field on the cells ``dom`` of the layer block ``sl`` of interval ``name``, simulated with the normal-score
    variogram ``vname`` attached to that interval and equal to ``hard_z`` {(i, j, local k): score} at those cells.
    -> (field, variogram model, mean cell thickness)."""
    err = err or PropertySimError
    m = project.variogram_for(geom.by_name[name].uid, vname)
    if m is None:
        raise err(f"interval {name!r}, facies {fname!r}: no variogram named {vname!r} is attached to the interval")
    if m.transform != "normal_score":
        raise err(f"interval {name!r}, facies {fname!r}: variogram {vname!r} was made on '{m.transform}' data, not normal scores")
    vg, vaz, vnug, vsill, _ = _pack([m])
    hd = np.full(dom.shape, -1, np.int8)
    zlo = np.full(dom.shape, -np.inf)
    for (i, j, k), z in hard_z.items():
        hd[i, j, k] = 0
        zlo[i, j, k] = float(z)
    rh = search_factor * max(s.range_major for s in m.structures)
    rv = search_factor * max(s.range_vert for s in m.structures)
    dzm = float(np.mean(geom.thick[sl][dom]))
    tdi, tdj, tdk = make_template(rh, rv, max(geom.dxm, 1e-9), max(geom.dym, 1e-9), max(dzm, 1e-9))
    strides = np.array([2 ** q for q in range(levels - 1, -1, -1)], np.int64)
    zval = np.zeros(dom.shape)
    stats = np.zeros(1, np.int64)
    _gauss_simulate(zlo.copy(), zlo.copy(), dom, np.ascontiguousarray(geom.X[sl]), np.ascontiguousarray(geom.Y[sl]),
                    np.ascontiguousarray(geom.Z[sl]), hd, vg, vaz, vnug, vsill, tdi, tdj, tdk, max_neighbours, strides, 0, seed, zval, stats)
    return zval, m, dzm


def simulate_property(project, model, facies: np.ndarray, setup: PropertySetup, data: list, seed: int, progress=None,
                      primary: PropertyResult | None = None) -> PropertyResult:
    """facies: (nx, ny, nz) facies codes of one realization (0 = inactive). data: records with i, j, k, interval (index), facies
    and value (datacheck.cell_values). The same seed gives the same realization. ``primary`` is the result of the property named
    by ``setup.correlate_with`` for the same facies realization (required then)."""
    if not (isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**31):
        raise PropertySimError("the seed must be a whole number between 0 and 2^31 - 1")
    if project.chart is None or project.facies_setup is None:
        raise PropertySimError("define the chart and the facies first")
    scheme = tuple(project.facies_setup.scheme)
    grid = model.grid
    facies = np.asarray(facies)
    if facies.shape != (grid.nx, grid.ny, grid.nz):
        raise PropertySimError("the facies realization belongs to a different grid")
    li = np.asarray(model.report.layer_interval)
    geom = _Geom(project, model)
    thick, active, dxm, dym = geom.thick, geom.active, geom.dxm, geom.dym
    out = np.full(facies.shape, np.nan)
    rho_of = dict(setup.correlation)
    if setup.correlate_with is not None:
        if primary is None or primary.values.shape != facies.shape:
            raise PropertySimError(f"{setup.name} is correlated with {setup.correlate_with!r}, which has to be simulated first on the "
                                   "same facies realization")
    elif primary is not None:
        raise PropertySimError(f"{setup.name} has no correlation set but a primary property was given")
    src, user = dict(setup.sources), dict(setup.user)
    lines, per = [], {}
    done, total = 0, int(active.sum())
    for a, name in enumerate(model.interval_names):
        ks = np.nonzero(li == a)[0]
        sl = (slice(None), slice(None), ks)
        for f in scheme:
            dom = np.ascontiguousarray((facies[sl] == f.code) & active[sl])
            if not dom.any():
                continue
            key = (name, f.code)
            if key not in src:
                raise PropertySimError(f"interval {name!r}, facies {f.name!r}: choose where its distribution comes from "
                                       "('data' or a typed distribution); nothing is borrowed from another domain")
            recs = [r for r in data if r["interval"] == a and r["facies"] == f.code]
            # one value per cell: several wells in the same cell are averaged
            cells: dict = {}
            for r in recs:
                cells.setdefault((r["i"], r["j"], r["k"]), []).append(r["value"])
            vals = {c: float(np.mean(v)) for c, v in cells.items() if dom[c[0], c[1], c[2] - ks[0]]}
            arr = np.array(list(vals.values()), float)
            if src[key] == "data":
                if len(arr) < setup.min_data:
                    raise PropertySimError(f"interval {name!r}, facies {f.name!r}: {len(arr)} cell value(s), {setup.min_data} needed "
                                           "for the data distribution; type a distribution instead")
                tr = _Empirical(arr)
                tgt = (float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"))
            else:
                tr = user[key]
                if tr.kind == "lognormal" and (arr <= 0).any():
                    raise PropertySimError(f"interval {name!r}, facies {f.name!r}: hard values <= 0 cannot follow a log-normal distribution")
                if len(arr):
                    p = tr.cdf(arr)
                    bad = (p < setup.conflict_probability) | (p > 1 - setup.conflict_probability)
                    if bad.any():
                        raise PropertySimError(f"interval {name!r}, facies {f.name!r}: {int(bad.sum())} hard value(s) (e.g. "
                                               f"{float(arr[bad][0]):.4g}) are very unlikely under the typed distribution "
                                               f"(probability < {setup.conflict_probability:g} in a tail); change the distribution")
                tgt = (tr.mean, tr.std)
            if len(arr) and ((setup.lower is not None and (arr < setup.lower).any()) or (setup.upper is not None and (arr > setup.upper).any())):
                raise PropertySimError(f"interval {name!r}, facies {f.name!r}: hard values outside the bounds of the property")
            vname = setup.variogram.replace("{facies}", f.name)
            hard_z = {(i, j, k - ks[0]): float(tr.to_z(v)) for (i, j, k), v in vals.items()}
            rho, z1, corr_note = None, None, ""
            if setup.correlate_with is not None:
                if key not in rho_of:
                    raise PropertySimError(f"interval {name!r}, facies {f.name!r}: enter the correlation of {setup.name} with "
                                           f"{setup.correlate_with}")
                rho = rho_of[key]
                pd = primary.per_domain.get(key)
                if pd is None:
                    raise PropertySimError(f"interval {name!r}, facies {f.name!r}: {setup.correlate_with} was not simulated in this domain")
                z1 = np.zeros(dom.shape)
                with np.errstate(invalid="ignore", divide="ignore"):
                    z1[dom] = pd["score"](primary.values[sl][dom])
                if not np.isfinite(z1[dom]).all():
                    raise PropertySimError(f"interval {name!r}, facies {f.name!r}: {setup.correlate_with} has values without a normal "
                                           "score (for example <= 0 under a log-normal distribution)")
                c = math.sqrt(1.0 - rho * rho)
                if hard_z:
                    cells_h = list(hard_z)
                    z2h = np.array([hard_z[q] for q in cells_h])
                    z1h = np.array([z1[q] for q in cells_h])
                    eh = (z2h - rho * z1h) / c
                    pt = norm.cdf(eh)
                    bad = (pt < setup.conflict_probability) | (pt > 1 - setup.conflict_probability)
                    if bad.any():
                        raise PropertySimError(
                            f"interval {name!r}, facies {f.name!r}: {int(bad.sum())} hard value(s) of {setup.name} contradict the "
                            f"correlation {rho:g} with {setup.correlate_with} in the same cell (probability < "
                            f"{setup.conflict_probability:g} in a tail); change the correlation")
                    if len(cells_h) >= 3 and z1h.std() > 0 and z2h.std() > 0:
                        corr_note = f"; at the {len(cells_h)} hard cells the scores correlate {float(np.corrcoef(z1h, z2h)[0, 1]):.2f}"
                    else:
                        corr_note = f"; too few hard cells ({len(cells_h)}) to compare the correlation with data"
                    hard_z = {q: float(e) for q, e in zip(cells_h, eh)}
                else:
                    corr_note = "; no hard cells to compare the correlation with data"
            fi = f.code
            report(progress, f"{setup.name}, {name} / {f.name}", done, total)
            zval, m, _ = gaussian_field(project, geom, name, f.name, vname, dom, sl, hard_z, setup.max_neighbours, setup.search_factor,
                                        setup.levels, seed * 1009 + a * 31 + fi)
            if rho is not None:
                zval = rho * z1 + math.sqrt(1.0 - rho * rho) * zval
            done += int(dom.sum())
            v = tr.from_z(zval)
            if setup.lower is not None:
                v = np.maximum(v, setup.lower)
            if setup.upper is not None:
                v = np.minimum(v, setup.upper)
            for (i, j, k), x in vals.items():                    # hard values exact (the transform round trip is exact; this
                v[i, j, k - ks[0]] = x                           # also removes bound clipping of data that sat on a bound)
            block = out[sl]
            block[dom] = v[dom]
            out[sl] = block
            sim = v[dom]
            score = tr.score if isinstance(tr, _Empirical) else tr.to_z
            per[key] = dict(cells=int(dom.sum()), data=len(arr), source=src[key], mean=float(sim.mean()), std=float(sim.std()),
                            target_mean=tgt[0], target_std=tgt[1], score=score, variogram=m,
                            cell_volume=float(np.mean(thick[sl][dom]) * dxm * dym))
            lines.append(f"{name} / {f.name}: {int(dom.sum())} cells, {len(arr)} hard value(s), distribution from {src[key]}; "
                         f"simulated mean {sim.mean():.4g}, std {sim.std():.4g} (target mean {tgt[0]:.4g}, std {tgt[1]:.4g})"
                         + (f"; correlation {rho:g} with {setup.correlate_with} (score correlation of this realization "
                            f"{float(np.corrcoef(z1[dom], zval[dom])[0, 1]):.3f}){corr_note}" if rho is not None else ""))
            if rho is not None:
                per[key]["correlation"] = rho
    return PropertyResult(out, "\n".join(lines), per)


def hard_property_data(project, model, setup_name: str, hard=None) -> list:
    """Cell-scale values for the data-check property called ``setup_name`` (curve and averaging come from the data checks)."""
    from .datacheck import DataCheckError, cell_values
    from .facies import FaciesError, classify_all, hard_facies, locate_cells
    ds, fs = project.data_checks, project.facies_setup
    if ds is None or fs is None:
        raise PropertySimError("set up the facies and the data checks first (they define the property curve and its averaging)")
    p = next((q for q in ds.properties if q.name == setup_name), None)
    if p is None:
        raise PropertySimError(f"the data checks have no property called {setup_name!r}")
    try:
        wf, _ = classify_all(project, fs)
        hard = hard if hard is not None else hard_facies(project, fs, model.grid)
        loc = {w: locate_cells(model.grid, d.x, d.y, d.z) for w, d in wf.items()}
        return cell_values(project, p, wf, loc, model.grid, hard, fs, np.asarray(model.report.layer_interval))
    except (DataCheckError, FaciesError) as exc:
        raise PropertySimError(str(exc)) from None

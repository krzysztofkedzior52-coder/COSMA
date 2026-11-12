"""Permeability from simulated porosity, per interval and facies: log10(k) = a + b * phi + scatter, then kV/kH and kY/kX.

Per domain (interval, facies) the relation is chosen explicitly:
  'user'  a, b and the scatter (standard deviation of log10 k about the line) typed by the user;
  'core'  the regression of core porosity-permeability pairs at the level the user names ('facies x interval',
          'facies, all intervals' or 'interval, all facies'); it must have passed the data-check thresholds, otherwise the run
          stops with the reason (nothing falls back silently; DataReport.fallback suggests a level).
A core-to-reservoir shift (log10 units) is added to the intercept of core relations; the user types it (0 = none).
The scatter about the line is either 'independent' from cell to cell, or 'correlated': a standard-normal field simulated per domain
with a normal-score variogram the user names (so high- and low-permeability streaks are continuous at a given porosity). The
correlated field is not conditioned to core permeability and costs one more simulation pass over the grid. Where the simulated porosity lies outside the range of the
core pairs behind a 'core' relation the user's rule applies: 'allow' (extrapolate), 'clamp' (use the nearest end of the range for
the line) or 'stop'; the number of such cells is reported. kV/kH is typed per domain. No hard permeability is honoured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

LEVELS = ("facies x interval", "facies, all intervals", "interval, all facies")
EXTRAPOLATION = ("allow", "clamp", "stop")


class PermSimError(ValueError):
    pass


def _num(v, what, positive=False, nonneg=False):
    if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)):
        raise PermSimError(f"{what} must be a number")
    if positive and v <= 0:
        raise PermSimError(f"{what} must be > 0")
    if nonneg and v < 0:
        raise PermSimError(f"{what} cannot be negative")
    return float(v)


@dataclass(frozen=True)
class Relation:
    source: str                   # 'user' | 'core'
    intercept: float | None = None
    slope: float | None = None
    scatter: float | None = None
    level: str | None = None      # for 'core'

    def __post_init__(self):
        if self.source == "user":
            _num(self.intercept, "intercept")
            _num(self.slope, "slope")
            _num(self.scatter, "scatter", nonneg=True)
            if self.level is not None:
                raise PermSimError("a typed relation has no core level")
        elif self.source == "core":
            if self.level not in LEVELS:
                raise PermSimError(f"core level must be one of {', '.join(LEVELS)}")
            if any(x is not None for x in (self.intercept, self.slope, self.scatter)):
                raise PermSimError("a core relation takes its numbers from the regression")
        else:
            raise PermSimError("relation source must be 'user' or 'core'")


@dataclass(frozen=True)
class PermSetup:
    porosity: str                 # informative: the porosity property simulated before
    relations: tuple              # (((interval, facies code), Relation), ...)
    kv_kh: tuple                  # (((interval, facies code), ratio > 0), ...)
    ky_kx: float                  # PERMY / PERMX, everywhere (1 = isotropic in the horizontal)
    core_shift: float             # log10 units added to core intercepts
    extrapolation: str            # 'allow' | 'clamp' | 'stop'
    lower: float | None = None    # bounds of PERMX in mD (simulated values are clipped)
    upper: float | None = None
    scatter_mode: str = "independent"        # 'independent' | 'correlated'
    scatter_variogram: str | None = None     # correlated: variogram name on each interval ('{facies}' = facies name), normal scores
    scatter_neighbours: int | None = None    # correlated: search settings of the residual field
    scatter_search_factor: float | None = None
    scatter_levels: int | None = None

    def __post_init__(self):
        if not (isinstance(self.porosity, str) and self.porosity.strip()):
            raise PermSimError("name the porosity property")
        for nm, pairs in (("relations", self.relations), ("kv_kh", self.kv_kh)):
            keys = [k for k, _ in pairs]
            if len(set(keys)) != len(keys):
                raise PermSimError(f"a domain appears twice in {nm}")
        for _, r in self.relations:
            if not isinstance(r, Relation):
                raise PermSimError("relations must be Relation objects")
        for _, v in self.kv_kh:
            _num(v, "kv/kh", positive=True)
        _num(self.ky_kx, "ky/kx", positive=True)
        _num(self.core_shift, "core shift")
        if self.extrapolation not in EXTRAPOLATION:
            raise PermSimError(f"extrapolation must be one of {', '.join(EXTRAPOLATION)}")
        for nm, v in (("lower", self.lower), ("upper", self.upper)):
            if v is not None:
                _num(v, nm, positive=True)
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise PermSimError("lower must be below upper")
        extra = (self.scatter_variogram, self.scatter_neighbours, self.scatter_search_factor, self.scatter_levels)
        if self.scatter_mode == "independent":
            if any(x is not None for x in extra):
                raise PermSimError("independent scatter takes no variogram or search settings")
        elif self.scatter_mode == "correlated":
            if not (isinstance(self.scatter_variogram, str) and self.scatter_variogram.strip()):
                raise PermSimError("correlated scatter needs the name of its variogram")
            for nm, v in (("neighbours", self.scatter_neighbours), ("multigrid levels", self.scatter_levels)):
                if not (isinstance(v, int) and not isinstance(v, bool) and v >= 1):
                    raise PermSimError(f"correlated scatter: {nm} must be a whole number >= 1")
            _num(self.scatter_search_factor, "correlated scatter: search factor", positive=True)
        else:
            raise PermSimError("scatter_mode must be 'independent' or 'correlated'")


@dataclass
class PermResult:
    permx: np.ndarray
    permy: np.ndarray
    permz: np.ndarray
    report: str
    per_domain: dict = field(default_factory=dict)


def core_relation_row(regression, interval, facies, level):
    """The data-check regression row for a domain at ``level``; stops if it is missing or did not pass."""
    for r in regression:
        if r["level"] != level:
            continue
        if level == "facies x interval" and (r["interval"], r["facies"]) == (interval, facies):
            return r
        if level == "facies, all intervals" and r["facies"] == facies:
            return r
        if level == "interval, all facies" and r["interval"] == interval:
            return r
    raise PermSimError(f"interval {interval!r}, facies {facies}: the data checks have no regression at level '{level}'")


def simulate_permeability(project, model, facies, phi, setup: PermSetup, regression, seed: int, progress=None) -> PermResult:
    """facies, phi: (nx, ny, nz) of one realization (phi NaN where there is no facies). ``regression``: DataReport.regression rows
    (needed only for 'core' relations)."""
    if not (isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**31):
        raise PermSimError("the seed must be a whole number between 0 and 2^31 - 1")
    if project.facies_setup is None:
        raise PermSimError("define the facies first")
    facies, phi = np.asarray(facies), np.asarray(phi, float)
    if facies.shape != phi.shape or facies.shape != (model.grid.nx, model.grid.ny, model.grid.nz):
        raise PermSimError("facies and porosity must have the shape of the grid")
    li = np.asarray(model.report.layer_interval)
    rel, kvkh = dict(setup.relations), dict(setup.kv_kh)
    permx = np.full(facies.shape, np.nan)
    permy, permz = permx.copy(), permx.copy()
    lines, per = [], {}
    rng = np.random.default_rng(seed)
    geom = None
    if setup.scatter_mode == "correlated":
        from .property_sim import _Geom
        if project.chart is None:
            raise PermSimError("define the chart first")
        geom = _Geom(project, model)
    from .progress import report
    done, total = 0, int((facies > 0).sum())
    for a, name in enumerate(model.interval_names):
        ks = np.nonzero(li == a)[0]
        for f in project.facies_setup.scheme:
            dom = np.zeros(facies.shape, bool)
            dom[:, :, ks] = facies[:, :, ks] == f.code
            if not dom.any():
                continue
            key = (name, f.code)
            if key not in rel:
                raise PermSimError(f"interval {name!r}, facies {f.name!r}: choose the porosity-permeability relation "
                                   "(typed or from core)")
            if key not in kvkh:
                raise PermSimError(f"interval {name!r}, facies {f.name!r}: enter kV/kH")
            p = phi[dom]
            if not np.isfinite(p).all():
                raise PermSimError(f"interval {name!r}, facies {f.name!r}: porosity is missing in some cells of this domain")
            r = rel[key]
            lo = hi = None
            if r.source == "user":
                a0, b0, s0 = r.intercept, r.slope, r.scatter
                how = "typed"
            else:
                row = core_relation_row(regression or [], name, f.code, r.level)
                if not row["ok"]:
                    raise PermSimError(f"interval {name!r}, facies {f.name!r}: the core regression at level '{r.level}' did not pass "
                                       f"the data checks ({row['why']}); choose another level or type the relation")
                a0, b0, s0 = row["intercept"] + setup.core_shift, row["slope"], row["resid_std"]
                lo, hi = row["phi_min"], row["phi_max"]
                how = f"core, {r.level} ({row['n']} pairs, r2 {row['r2']:.2f})"
            pe = p
            outside = 0
            if lo is not None:
                outside = int(((p < lo) | (p > hi)).sum())
                if outside and setup.extrapolation == "stop":
                    raise PermSimError(f"interval {name!r}, facies {f.name!r}: {outside} cell(s) have porosity outside the core range "
                                       f"{lo:.3f}-{hi:.3f}")
                if setup.extrapolation == "clamp":
                    pe = np.clip(p, lo, hi)
            report(progress, f"permeability, {name} / {f.name}", done, total)
            if geom is None or s0 == 0:
                e = rng.standard_normal(p.shape)
            else:
                from .property_sim import gaussian_field
                sl = (slice(None), slice(None), ks)
                block = np.ascontiguousarray(dom[sl] & geom.active[sl])
                if block.sum() != dom.sum():
                    raise PermSimError(f"interval {name!r}, facies {f.name!r}: the facies realization has cells that are not active")
                field_, _, _ = gaussian_field(project, geom, name, f.name, setup.scatter_variogram.replace("{facies}", f.name), block, sl,
                                              {}, setup.scatter_neighbours, setup.scatter_search_factor, setup.scatter_levels,
                                              (seed * 1009 + a * 31 + f.code) % (2**31), err=PermSimError)
                full = np.zeros(facies.shape)
                full[sl] = field_
                e = full[dom]
            done += int(dom.sum())
            logk = a0 + b0 * pe + s0 * e
            k = 10.0 ** logk
            if setup.lower is not None:
                k = np.maximum(k, setup.lower)
            if setup.upper is not None:
                k = np.minimum(k, setup.upper)
            permx[dom] = k
            permy[dom] = k * setup.ky_kx
            permz[dom] = k * kvkh[key]
            per[key] = dict(cells=int(dom.sum()), source=how, outside_core_range=outside, intercept=float(a0), slope=float(b0),
                            scatter=float(s0), mean_log10=float(np.log10(k).mean()), kv_kh=kvkh[key], scatter_mode=setup.scatter_mode)
            lines.append(f"{name} / {f.name}: {int(dom.sum())} cells, log10 k = {a0:.3f} + {b0:.3f} phi, scatter {s0:.3f} ({how})"
                         + (f"; {outside} cell(s) outside the core porosity range ({setup.extrapolation})" if lo is not None else "")
                         + f"; scatter {setup.scatter_mode}; kV/kH {kvkh[key]:g}; mean log10 k {np.log10(k).mean():.3f}")
    return PermResult(permx, permy, permz, "\n".join(lines), per)

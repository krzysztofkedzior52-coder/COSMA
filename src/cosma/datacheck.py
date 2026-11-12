"""Data checks before simulation: is there enough hard data, per interval and per facies, to do what the user wants?

Nothing here decides for the user. Every threshold is the user's (no defaults); every check reports numbers and a
verdict against those thresholds, and says what is missing when it fails. Checks:

  inventory      hard cells, wells and well path per interval x facies (from the upscaled facies)
  vertical       pair counts of the vertical variogram of the log samples of one facies in one interval, per well and
                 pooled, as the variogram tool computes them; the usable lag is the longest lag below which every lag
                 bin holds enough pairs
  horizontal     the same for the horizontal variogram, from cell-scale points, using pairs from different wells in the
                 same layer; with fewer than two wells it says plainly that the variogram cannot come from the data
  distributions  cell-scale values of each property (log samples averaged to cells with the user's averaging rule) per
                 interval x facies, next to the log-scale values, with normal and lognormal fits (KS distance only)
  regression     porosity-permeability from core, per facies x interval and at the pooled levels the user may fall
                 back to, with slope confidence interval and r2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .facies import FaciesError, classify_all, locate_cells, upscale
from .variogram import VariogramError, experimental_variogram

AVERAGING = ("arithmetic", "geometric", "harmonic")


class DataCheckError(ValueError):
    pass


def _pos(x, what, integer=False):
    ok = isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and math.isfinite(x) and x > 0
    if ok and integer:
        ok = float(x) == int(x)
    if not ok:
        raise DataCheckError(f"{what} must be a {'whole number' if integer else 'number'} > 0")
    return int(x) if integer else float(x)


@dataclass(frozen=True)
class Thresholds:
    min_hard_cells: int
    min_wells: int
    lag_v: float              # vertical variogram lag (m)
    nlags_v: int
    min_pairs_v: int
    min_usable_lag_v: float   # the vertical variogram must be supported to at least this lag (m)
    lag_h: float
    nlags_h: int
    min_pairs_h: int
    min_usable_lag_h: float
    same_layer_tol: float     # horizontal pairs must lie within this vertical distance (m)
    min_wells_h: int
    min_core_pairs: int
    min_r2: float | None = None

    def __post_init__(self):
        for n, integer in (("min_hard_cells", 1), ("min_wells", 1), ("nlags_v", 1), ("min_pairs_v", 1), ("nlags_h", 1),
                           ("min_pairs_h", 1), ("min_wells_h", 1), ("min_core_pairs", 1)):
            _pos(getattr(self, n), n, True)
        for n in ("lag_v", "min_usable_lag_v", "lag_h", "min_usable_lag_h", "same_layer_tol"):
            _pos(getattr(self, n), n)
        if self.min_r2 is not None and not (isinstance(self.min_r2, (int, float)) and 0 <= self.min_r2 <= 1):
            raise DataCheckError("min_r2 must be between 0 and 1 (or not set)")


@dataclass(frozen=True)
class PropertySpec:
    name: str
    curve: str           # curve of the same log source as the facies
    averaging: str       # how log samples are averaged to a cell: arithmetic | geometric | harmonic

    def __post_init__(self):
        if not self.name or not self.curve:
            raise DataCheckError("a property needs a name and a curve")
        if self.averaging not in AVERAGING:
            raise DataCheckError(f"averaging must be one of {', '.join(AVERAGING)}")


@dataclass(frozen=True)
class CoreSpec:
    porosity: str
    permeability: str
    depth_tol: float                 # a core sample takes the facies of the nearest classified log sample within this (m)
    source: str | None = None        # core source, needed only if a well has several

    def __post_init__(self):
        if not self.porosity or not self.permeability:
            raise DataCheckError("name the core porosity and permeability columns")
        _pos(self.depth_tol, "depth_tol")


@dataclass(frozen=True)
class DataCheckSetup:
    thresholds: Thresholds
    properties: tuple = ()
    core: CoreSpec | None = None

    def __post_init__(self):
        names = [p.name for p in self.properties]
        if len(set(names)) != len(names):
            raise DataCheckError("property names must be unique")


def to_state(s: DataCheckSetup | None):
    if s is None:
        return None
    t = s.thresholds
    return dict(thresholds={k: getattr(t, k) for k in t.__dataclass_fields__},
                properties=[[p.name, p.curve, p.averaging] for p in s.properties],
                core=None if s.core is None else dict(porosity=s.core.porosity, permeability=s.core.permeability,
                                                      depth_tol=s.core.depth_tol, source=s.core.source))


def from_state(d) -> DataCheckSetup | None:
    if d is None:
        return None
    try:
        c = d.get("core")
        return DataCheckSetup(Thresholds(**d["thresholds"]), tuple(PropertySpec(*p) for p in d.get("properties", [])),
                              None if c is None else CoreSpec(c["porosity"], c["permeability"], c["depth_tol"], c.get("source")))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise DataCheckError(f"bad data-check setup: {exc}") from None


# -- helpers --------------------------------------------------------------------------------------------------------------

def describe(v) -> dict:
    v = np.asarray(v, float)
    if v.size == 0:
        return dict(n=0)
    q = np.percentile(v, [10, 50, 90])
    return dict(n=int(v.size), mean=float(v.mean()), std=float(v.std(ddof=1)) if v.size > 1 else 0.0, min=float(v.min()),
                p10=float(q[0]), p50=float(q[1]), p90=float(q[2]), max=float(v.max()))


def fit_ks(v) -> dict:
    """Normal and lognormal maximum-likelihood fits and their Kolmogorov-Smirnov distances (a distance, not a p-value:
    the samples are spatially correlated, so a p-value would overstate the evidence)."""
    v = np.asarray(v, float)
    out = {}
    if v.size < 3 or np.ptp(v) == 0:
        return out
    mu, sd = float(v.mean()), float(v.std(ddof=0))
    out["normal"] = dict(mean=mu, std=sd, ks=float(stats.kstest(v, lambda x: stats.norm.cdf(x, mu, sd)).statistic))
    if (v > 0).all():
        lv = np.log(v)
        lm, ls = float(lv.mean()), float(lv.std(ddof=0))
        if ls > 0:
            out["lognormal"] = dict(mu=lm, sigma=ls, ks=float(stats.kstest(lv, lambda x: stats.norm.cdf(x, lm, ls)).statistic))
    return out


def average(values, weights, how: str) -> float:
    v, w = np.asarray(values, float), np.asarray(weights, float)
    if how == "arithmetic":
        return float((v * w).sum() / w.sum())
    if (v <= 0).any():
        raise DataCheckError(f"{how} averaging needs values > 0")
    if how == "geometric":
        return float(math.exp((np.log(v) * w).sum() / w.sum()))
    return float(w.sum() / (w / v).sum())


def usable_lag(lags, npairs, min_pairs: int) -> float:
    """Longest lag such that every lag bin up to it holds at least ``min_pairs`` pairs (0 if the first does not)."""
    out = 0.0
    for lag, n in zip(lags, npairs):
        if n < min_pairs:
            break
        out = float(lag)
    return out


def regression(phi, perm) -> dict:
    """log10(k) = a + b*phi by least squares. Pairs with k <= 0 or non-finite are left out and counted."""
    phi, perm = np.asarray(phi, float), np.asarray(perm, float)
    ok = np.isfinite(phi) & np.isfinite(perm) & (perm > 0)
    out = dict(n=int(ok.sum()), dropped=int((~ok).sum()))
    if ok.sum() < 3 or np.ptp(phi[ok]) == 0:
        return out
    y = np.log10(perm[ok])
    r = stats.linregress(phi[ok], y)
    n = int(ok.sum())
    resid = y - (r.intercept + r.slope * phi[ok])
    out.update(slope=float(r.slope), intercept=float(r.intercept), r2=float(r.rvalue ** 2),
               slope_ci=float(stats.t.ppf(0.975, n - 2) * r.stderr), resid_std=float(math.sqrt((resid ** 2).sum() / (n - 2))),
               phi_min=float(phi[ok].min()), phi_max=float(phi[ok].max()))
    return out


# -- the report -----------------------------------------------------------------------------------------------------------

@dataclass
class DataReport:
    intervals: list
    scheme: tuple
    thresholds: Thresholds
    inventory: list = field(default_factory=list)
    vertical: list = field(default_factory=list)
    horizontal: list = field(default_factory=list)
    distributions: list = field(default_factory=list)
    regression: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def row(self, table, interval, facies=None, **kw):
        for r in getattr(self, table):
            if r["interval"] == interval and r.get("facies") == facies and all(r.get(k) == v for k, v in kw.items()):
                return r
        return None

    def fallback(self, interval, facies):
        """The first regression level that passes, in the order facies x interval, facies over all intervals, interval over
        all facies; None if none passes. A suggestion only: the user chooses which level to use."""
        for r in self.regression:
            if r["ok"] and ((r["level"] == "facies x interval" and r["interval"] == interval and r["facies"] == facies)):
                return r
        for r in self.regression:
            if r["ok"] and r["level"] == "facies, all intervals" and r["facies"] == facies:
                return r
        for r in self.regression:
            if r["ok"] and r["level"] == "interval, all facies" and r["interval"] == interval:
                return r
        return None

    def summary(self) -> str:
        t = self.thresholds
        names = {f.code: f.name for f in self.scheme}
        out = []
        out.append("Hard data per interval and facies (thresholds: at least "
                   f"{t.min_hard_cells} cells in {t.min_wells} well(s)):")
        for r in self.inventory:
            out.append(f"  {r['interval']} / {names[r['facies']]}: {r['cells']} cells, {r['wells']} well(s), {r['path']:.1f} m of path "
                       f"-> {'ok' if r['ok'] else 'INSUFFICIENT: ' + r['why']}")
        out.append(f"\nVertical variogram support (lag {t.lag_v:g} m, >= {t.min_pairs_v} pairs per lag, needed to "
                   f"{t.min_usable_lag_v:g} m):")
        for r in self.vertical:
            out.append(f"  {r['interval']} / {names[r['facies']]}: {r['samples']:,} samples, {r['wells']} well(s), supported to "
                       f"{r['usable']:g} m -> {'ok' if r['ok'] else 'INSUFFICIENT: ' + r['why']}")
        out.append(f"\nHorizontal variogram support (cell-scale points, different wells, same layer within {t.same_layer_tol:g} m; "
                   f"lag {t.lag_h:g} m, >= {t.min_pairs_h} pairs, needed to {t.min_usable_lag_h:g} m, >= {t.min_wells_h} wells):")
        for r in self.horizontal:
            out.append(f"  {r['interval']} / {names[r['facies']]}: {r['points']} cells, {r['wells']} well(s), supported to "
                       f"{r['usable']:g} m -> {'ok' if r['ok'] else 'INSUFFICIENT: ' + r['why']}")
        if self.distributions:
            out.append("\nProperty distributions at cell scale (log scale for comparison):")
            for r in self.distributions:
                c, lg = r["cell"], r["log"]
                if c["n"] == 0:
                    out.append(f"  {r['property']}: {r['interval']} / {names[r['facies']]}: no cell-scale values")
                    continue
                best = min(r["fits"].items(), key=lambda kv: kv[1]["ks"])[0] if r["fits"] else "n/a"
                out.append(f"  {r['property']}: {r['interval']} / {names[r['facies']]}: {c['n']} cells ({r['wells']} well(s)), mean {c['mean']:.4g}, "
                           f"std {c['std']:.4g}, P10/P50/P90 {c['p10']:.4g}/{c['p50']:.4g}/{c['p90']:.4g}; log scale std {lg['std']:.4g} "
                           f"(variance ratio cell/log {r['variance_ratio']:.3g}); closest fit: {best}")
        if self.regression:
            out.append(f"\nPorosity-permeability from core (needs >= {t.min_core_pairs} pairs"
                       + (f", r2 >= {t.min_r2:g}" if t.min_r2 is not None else "") + "):")
            for r in self.regression:
                lab = f"{r['level']}: {r['interval'] or 'all intervals'} / {names[r['facies']] if r['facies'] else 'all facies'}"
                if "slope" in r:
                    out.append(f"  {lab}: {r['n']} pairs, log10 k = {r['intercept']:.3f} + {r['slope']:.2f} phi (slope +/- {r['slope_ci']:.2f}), "
                               f"r2 {r['r2']:.2f}, scatter {r['resid_std']:.2f} -> {'ok' if r['ok'] else 'INSUFFICIENT: ' + r['why']}")
                else:
                    out.append(f"  {lab}: {r['n']} pairs -> INSUFFICIENT: {r['why']}")
            out.append("Levels that would work for each interval and facies (you choose; nothing is applied):")
            for iv in self.intervals:
                for f in self.scheme:
                    lv = self.fallback(iv, f.code)
                    out.append(f"  {iv} / {f.name}: " + (lv["level"] if lv else "none passes: enter the relation by hand"))
        out += [f"Note: {n}" for n in self.notes]
        return "\n".join(out)


# -- the checks -----------------------------------------------------------------------------------------------------------

def _cell_centre(grid, i, j, k):
    return grid.cell_corners(i, j, k).mean(axis=0)


def cell_values(project, p, wf, loc, grid, hard, fs, layer_interval, notes=None) -> list:
    """Cell-scale values of property ``p`` (a PropertySpec) at the hard-facies cells: one record per (well, cell) with
    interval index, facies code, the averaged value and the log samples behind it. Cells are used only when the share of the path
    with a valid value reaches the facies setup's minimum coverage."""
    out = []
    for w, d in wf.items():
        rec = project.wells[w]
        if d.src_index is None or p.curve not in rec.logs[d.source].curves:
            if notes is not None:
                notes.append(f"{p.name}: well {w} has no curve {p.curve!r} in its facies log source; not used")
            continue
        val = np.asarray(rec.logs[d.source].curves[p.curve], float)[d.src_index]
        I, J, K, why = loc[w]
        use = (why == 0)
        if not use.any():
            continue
        key = (I * grid.ny + J) * grid.nz + K
        for kk in np.unique(key[use]):
            m = use & (key == kk)
            good = m & np.isfinite(val) & ((val > 0) if p.averaging != "arithmetic" else True)
            path = float(d.length[m].sum())
            if not good.any() or float(d.length[good].sum()) / path + 1e-12 < fs.min_coverage:
                continue
            i, j, k = int(kk // (grid.ny * grid.nz)), int((kk // grid.nz) % grid.ny), int(kk % grid.nz)
            f = int(hard.cells[i, j, k])
            if f == 0:
                continue
            out.append(dict(well=w, i=i, j=j, k=k, interval=int(layer_interval[k]), facies=f,
                            value=average(val[good], d.length[good], p.averaging), log_values=val[good].tolist()))
    return out


def run_checks(project, model, setup=None) -> DataReport:
    """All checks, from the project's facies setup and the data-check setup (``setup``, else the project's), on a built
    model (``ModelBuild``)."""
    fs, ds = project.facies_setup, setup or project.data_checks
    if fs is None:
        raise DataCheckError("set up the facies first (Facies menu)")
    if ds is None:
        raise DataCheckError("set the data-check thresholds first")
    if fs.min_coverage is None or fs.conflict is None:
        raise DataCheckError("choose the minimum classified share and the conflict policy in the facies setup first")
    th = ds.thresholds
    grid = model.grid
    layer_interval = np.asarray(model.report.layer_interval)
    names = list(model.interval_names)
    wf, skipped = classify_all(project, fs)
    if not wf:
        raise DataCheckError("no well could be classified: " + "; ".join(f"{w}: {r}" for w, r in skipped.items()))
    hard = upscale(wf, grid, fs.scheme, min_coverage=fs.min_coverage, conflict=fs.conflict, skipped=skipped)
    rep = DataReport(names, fs.scheme, th)
    for w, r in skipped.items():
        rep.notes.append(f"well {w} skipped: {r}")
    loc = {w: locate_cells(grid, d.x, d.y, d.z) for w, d in wf.items()}
    iv_of = lambda K: np.where(K >= 0, layer_interval[np.clip(K, 0, grid.nz - 1)], -1)             # noqa: E731

    ok_rows = (hard.code > 0) & (hard.status == "ok")
    ok_rows &= np.array([hard.cells[i, j, k] == c for i, j, k, c in zip(hard.i, hard.j, hard.k, hard.code)], bool) \
        if ok_rows.any() else ok_rows
    row_iv = layer_interval[hard.k] if hard.k.size else np.empty(0, int)

    for a, name in enumerate(names):
        ks = np.nonzero(layer_interval == a)[0]
        for f in fs.scheme:
            cells = int((hard.cells[:, :, ks] == f.code).sum()) if ks.size else 0
            sel = ok_rows & (row_iv == a) & (hard.code == f.code)
            wells = sorted(set(hard.well[sel]))
            path = float(hard.path[sel].sum())
            why = []
            if cells < th.min_hard_cells:
                why.append(f"{cells} hard cells, {th.min_hard_cells} needed")
            if len(wells) < th.min_wells:
                why.append(f"{len(wells)} well(s), {th.min_wells} needed")
            rep.inventory.append(dict(interval=name, facies=f.code, cells=cells, wells=len(wells), well_names=wells, path=path,
                                      ok=not why, why="; ".join(why)))

    # vertical and horizontal variogram support
    for a, name in enumerate(names):
        for f in fs.scheme:
            pts, grp = [], []
            for w, d in wf.items():
                I, J, K, why = loc[w]
                m = (d.code == f.code) & (why == 0) & (iv_of(K) == a)
                if m.any():
                    pts.append(np.column_stack([d.x[m], d.y[m], d.z[m]]))
                    grp.append(np.full(int(m.sum()), w, object))
            row = dict(interval=name, facies=f.code, samples=0, wells=0, usable=0.0, ok=False, why="no log samples of this facies here")
            if pts:
                P, G = np.vstack(pts), np.concatenate(grp)
                row.update(samples=int(len(P)), wells=int(len(set(G))))
                try:
                    e = experimental_variogram(P, np.zeros(len(P)), "vertical", lag=th.lag_v, nlags=th.nlags_v, groups=G,
                                               max_points=None)
                    u = usable_lag(e.lag, e.npairs, th.min_pairs_v)
                    row.update(usable=u, npairs=[int(n) for n in e.npairs], ok=u >= th.min_usable_lag_v,
                               why="" if u >= th.min_usable_lag_v else
                               f"supported to {u:g} m, {th.min_usable_lag_v:g} m needed (more samples, a larger lag or fewer lags)")
                except VariogramError as exc:
                    row.update(why=str(exc))
            rep.vertical.append(row)

            cp, cg = [], []
            for r in np.nonzero(ok_rows & (row_iv == a) & (hard.code == f.code))[0]:
                cp.append(_cell_centre(grid, hard.i[r], hard.j[r], hard.k[r]))
                cg.append(hard.well[r])
            hrow = dict(interval=name, facies=f.code, points=len(cp), wells=len(set(cg)), usable=0.0, ok=False, why="")
            if len(set(cg)) < 2:
                hrow["why"] = ("data from fewer than two wells: a horizontal variogram cannot be computed; enter its "
                               "parameters or derive them from a map")
            else:
                try:
                    e = experimental_variogram(np.array(cp), np.zeros(len(cp)), "major", lag=th.lag_h, nlags=th.nlags_h,
                                               tolerance_deg=180.0, groups=np.array(cg, object), vertical_tolerance=th.same_layer_tol,
                                               max_points=None)
                    u = usable_lag(e.lag, e.npairs, th.min_pairs_h)
                    hrow.update(usable=u, npairs=[int(n) for n in e.npairs])
                    why = []
                    if len(set(cg)) < th.min_wells_h:
                        why.append(f"{len(set(cg))} wells, {th.min_wells_h} needed")
                    if u < th.min_usable_lag_h:
                        why.append(f"supported to {u:g} m, {th.min_usable_lag_h:g} m needed")
                    hrow.update(ok=not why, why="; ".join(why))
                except VariogramError as exc:
                    hrow["why"] = str(exc)
            rep.horizontal.append(hrow)

    # property distributions at cell scale
    for p in ds.properties:
        recs = cell_values(project, p, wf, loc, grid, hard, fs, layer_interval, rep.notes)
        per = {}                                         # (a, f) -> (cell values, log values, wells)
        for r in recs:
            slot = per.setdefault((r["interval"], r["facies"]), ([], [], set()))
            slot[0].append(r["value"])
            slot[1].extend(r["log_values"])
            slot[2].add(r["well"])
        for a, name in enumerate(names):
            for f in fs.scheme:
                cv, lv, ws = per.get((a, f.code), ([], [], set()))
                cell, log = describe(cv), describe(lv)
                vr = (cell["std"] ** 2 / log["std"] ** 2) if cell["n"] > 1 and log.get("std", 0) > 0 else float("nan")
                rep.distributions.append(dict(property=p.name, interval=name, facies=f.code, wells=len(ws), cell=cell, log=log,
                                              variance_ratio=vr, fits=fit_ks(cv), values=np.asarray(cv)))

    # core porosity-permeability
    if ds.core is not None:
        rep.regression = _core_levels(project, ds.core, th, grid, wf, layer_interval, names, fs.scheme, rep.notes)
    return rep


def _core_levels(project, cs: CoreSpec, th: Thresholds, grid, wf, layer_interval, names, scheme, notes) -> list:
    pairs = []                    # (interval index, facies code, phi, perm, well)
    for w, rec in project.wells.items():
        if not rec.core:
            continue
        srcs = rec.resolved_core()
        src = cs.source
        if src is None:
            if len(srcs) > 1:
                notes.append(f"core of well {w}: several sources ({', '.join(sorted(srcs))}); name the one to use; skipped")
                continue
            src = next(iter(srcs))
        if src not in srcs:
            notes.append(f"core of well {w}: no source {src!r}; skipped")
            continue
        c = srcs[src]
        if cs.porosity not in c.columns or cs.permeability not in c.columns:
            notes.append(f"core of well {w}: lacks column {cs.porosity!r} or {cs.permeability!r}; skipped")
            continue
        tr = rec.trajectory
        lo, hi = tr.md_range
        md = np.asarray(c.md, float)
        inr = (md >= lo) & (md <= hi)
        if not inr.any():
            continue
        x, y, z = tr.at_md(md[inr])
        I, J, K, why = locate_cells(grid, x, y, z)
        phi = np.asarray(c.columns[cs.porosity], float)[inr]
        perm = np.asarray(c.columns[cs.permeability], float)[inr]
        d = wf.get(w)
        for n in range(md[inr].size):
            if why[n] != 0:
                continue
            code = 0
            if d is not None and d.md.size:
                q = int(np.argmin(np.abs(d.md - md[inr][n])))
                if abs(d.md[q] - md[inr][n]) <= cs.depth_tol:
                    code = int(d.code[q])
            pairs.append((int(layer_interval[K[n]]), code, phi[n], perm[n], w))
    if not pairs:
        notes.append("no core porosity-permeability pairs fall in active cells")
        return []
    A = np.array([p[0] for p in pairs])
    F = np.array([p[1] for p in pairs])
    PHI = np.array([p[2] for p in pairs])
    K_ = np.array([p[3] for p in pairs])
    W = np.array([p[4] for p in pairs], object)

    def level(kind, a, f, mask):
        r = regression(PHI[mask], K_[mask])
        r.update(level=kind, interval=None if a is None else names[a], facies=f, wells=len(set(W[mask])))
        why = []
        if r["n"] < th.min_core_pairs:
            why.append(f"{r['n']} pairs, {th.min_core_pairs} needed")
        if "r2" not in r and r["n"] >= th.min_core_pairs:
            why.append("no usable spread in porosity")
        if th.min_r2 is not None and "r2" in r and r["r2"] < th.min_r2:
            why.append(f"r2 {r['r2']:.2f} below {th.min_r2:g}")
        r.update(ok=not why, why="; ".join(why))
        return r

    out = []
    for a in range(len(names)):
        for f in scheme:
            out.append(level("facies x interval", a, f.code, (A == a) & (F == f.code)))
    for f in scheme:
        out.append(level("facies, all intervals", None, f.code, F == f.code))
    for a in range(len(names)):
        out.append(level("interval, all facies", a, None, A == a))
    return out

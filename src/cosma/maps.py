"""Per-interval property maps: one 2D grid of a property for one stratigraphic interval.

A map comes from one of three places, and says which (`method`):
  imported   read from a surface file (units declared by the user)
  thickness  base surface minus top surface of the interval
  kriged     ordinary kriging of per-well LAYER AVERAGES with the interval's variogram

Layer average of a well = average of the property between the interval's two picks, each
sample weighted by the vertical thickness it represents (gaps in the log are not bridged);
for a facies indicator that average is the facies proportion. Wells covering too little of
the interval are skipped, and reported - never silently dropped, never extrapolated.

Kriging uses the horizontal part of the variogram (nugget plus structures with their major /
minor ranges and azimuth); all wells are used (global neighbourhood; at most MAX_WELLS).
Kriging is exact at a well, and the kriging variance (kept with the map) is zero there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fault_checks import crossing_warnings
from .project import Project, ProjectError
from .strat import BASE_UID
from .surfaces import Surface
from .variogram import VariogramModel
from .variogram_data import collect

MAX_WELLS = 3000
SKIP_BELOW = 0.5        # default minimum fraction of the interval a well must cover
WARN_BELOW = 0.8


class MapError(ValueError):
    pass


@dataclass(frozen=True)
class MapGrid:
    """A regular, possibly rotated grid (same conventions as `Surface`: rotation in degrees
    counter-clockwise from +x, node (i, j) at origin + i*xinc along u + j*yinc along v)."""
    xori: float
    yori: float
    xinc: float
    yinc: float
    ncol: int
    nrow: int
    rotation: float = 0.0

    def __post_init__(self):
        if not (self.xinc > 0 and self.yinc > 0):
            raise MapError("grid increments must be positive")
        if self.ncol < 2 or self.nrow < 2:
            raise MapError("a map grid needs at least 2 x 2 nodes")
        if self.ncol * self.nrow > 25_000_000:
            raise MapError("grid has more than 25 million nodes: use a coarser increment")

    def template(self) -> Surface:
        return Surface(self.xori, self.yori, self.xinc, self.yinc, self.rotation, 1, np.full((self.ncol, self.nrow), np.nan))

    def node_xy(self):
        return self.template().node_xy()


def grid_from_surface(s: Surface) -> MapGrid:
    if s.yflip != 1:
        raise MapError("surface has a reversed y axis")
    return MapGrid(s.xori, s.yori, s.xinc, s.yinc, s.ncol, s.nrow, s.rotation)


def grid_from_wells(xy, inc: float, margin: float) -> MapGrid:
    """Axis-aligned grid covering the wells plus `margin` (m) on every side; the user gives both."""
    xy = np.asarray(xy, float)
    if xy.size == 0:
        raise MapError("no wells to place a grid around")
    if not (inc > 0 and margin >= 0):
        raise MapError("increment must be positive and margin cannot be negative")
    lo, hi = xy.min(axis=0) - margin, xy.max(axis=0) + margin
    n = np.maximum(np.ceil((hi - lo) / inc).astype(int) + 1, 2)
    return MapGrid(float(lo[0]), float(lo[1]), inc, inc, int(n[0]), int(n[1]), 0.0)


@dataclass(frozen=True)
class PropertyMap:
    grid: MapGrid
    values: np.ndarray                  # (ncol, nrow), NaN = undefined
    variable: str
    unit: str                           # "fraction", "mD", "m", ... as declared; never guessed
    method: str                         # imported | thickness | kriged
    variance: np.ndarray | None = None  # kriging variance (kriged maps only)
    info: dict = field(default_factory=dict)      # JSON-able provenance (wells used, variogram, ...)

    def __post_init__(self):
        shape = (self.grid.ncol, self.grid.nrow)
        if self.values.shape != shape or (self.variance is not None and self.variance.shape != shape):
            raise MapError(f"map values must have shape {shape}")
        if self.method not in ("imported", "thickness", "kriged"):
            raise MapError(f"unknown map method {self.method!r}")

    def sample(self, x, y):
        g = self.grid
        return Surface(g.xori, g.yori, g.xinc, g.yinc, g.rotation, 1, self.values).sample(x, y)

    def stats(self) -> dict:
        v = self.values[np.isfinite(self.values)]
        return dict(n=int(v.size), min=float(v.min()), max=float(v.max()), mean=float(v.mean())) if v.size else dict(n=0)


# -- layer averages --------------------------------------------------------------------------------------------------------------------
@dataclass
class LayerData:
    interval: str
    variable: str
    x: np.ndarray
    y: np.ndarray
    value: np.ndarray
    wells: list
    thickness: dict
    coverage: dict
    skipped: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    indicator: int | None = None
    crossings: dict = field(default_factory=dict)       # well -> [Crossing] (faults the well path cuts inside the interval)


def _weights(z: np.ndarray, top: float, base: float) -> np.ndarray:
    """Vertical thickness represented by each sample (z ascending): half the gap to each neighbour, a gap
    larger than twice the median spacing is NOT bridged (each side counts only a median half-spacing)."""
    if z.size == 1:
        return np.array([0.0])
    d = np.diff(z)
    med = float(np.median(d))
    d = np.minimum(d, 2 * med)
    w = np.zeros(z.size)
    w[:-1] += d / 2
    w[1:] += d / 2
    w[0] += min(med / 2, max(z[0] - 0.0, 0.0))             # up to the interval top
    w[-1] += min(med / 2, max((base - top) - z[-1], 0.0))  # down to the interval base
    return w


def layer_averages(project: Project, uid: str, variable: str, *, kind: str = "log", source=None,
                   indicator: int | None = None, min_coverage: float = SKIP_BELOW, wells=None) -> LayerData:
    if not 0 <= min_coverage <= 1:
        raise MapError("min_coverage must be between 0 and 1")
    d = collect(project, uid, variable, kind=kind, source=source, frame="top", wells=wells)
    if indicator is not None:
        d = d.as_indicator(indicator)
    out = LayerData(d.interval, d.variable, np.empty(0), np.empty(0), np.empty(0), [], {}, {}, dict(d.skipped), [], indicator)
    xs, ys, vs = [], [], []
    for w in d.used:
        m = d.groups == w
        z, order = np.sort(d.points[m, 2]), np.argsort(d.points[m, 2], kind="stable")
        px, py, v = d.points[m, 0][order], d.points[m, 1][order], d.values[m][order]
        th = d.thickness[w]
        wt = _weights(z, 0.0, th)
        cov = float(wt.sum() / th)
        if cov < min_coverage:
            out.skipped[w] = f"covers only {cov:.0%} of the interval (minimum {min_coverage:.0%})"
            continue
        if cov < WARN_BELOW:
            out.warnings.append(f"{w} covers {cov:.0%} of the interval")
        s = wt.sum()
        out.wells.append(w)
        out.thickness[w], out.coverage[w] = th, cov
        xs.append(float((px * wt).sum() / s)); ys.append(float((py * wt).sum() / s)); vs.append(float((v * wt).sum() / s))
    if not out.wells:
        raise MapError("no well has enough data in this interval: " + "; ".join(f"{w}: {r}" for w, r in out.skipped.items()))
    out.x, out.y, out.value = np.array(xs), np.array(ys), np.array(vs)
    out.crossings = {w: c for w, c in d.crossings.items() if w in out.wells}
    out.warnings += crossing_warnings(out.crossings)
    return out


# -- kriging ------------------------------------------------------------------------------------------------------------------------------
def krige(model: VariogramModel, x, y, v, nodes_xy, *, names=None, chunk: int = 20000):
    """Ordinary kriging with the horizontal part of `model`. Returns (estimate, variance) for each node.

    Exact at data locations (gamma(0) = 0 even when there is a nugget), weights sum to 1.
    Coincident data points are refused (the system would be singular): resolve them first."""
    model.check()
    x, y, v = (np.asarray(a, float) for a in (x, y, v))
    n = v.size
    if n == 0 or not (x.size == y.size == n):
        raise MapError("kriging needs at least one data point")
    if n > MAX_WELLS:
        raise MapError(f"{n} data points: global kriging is limited to {MAX_WELLS}")
    if not np.all(np.isfinite(v)):
        raise MapError("data values must be finite")
    dx, dy = x[:, None] - x[None], y[:, None] - y[None]
    off = ~np.eye(n, dtype=bool)
    if n > 1 and np.any(np.hypot(dx, dy)[off] < 1e-6):
        who = ""
        if names is not None:
            i, j = np.argwhere((np.hypot(dx, dy) < 1e-6) & off)[0]
            who = f" ({names[i]} and {names[j]})"
        raise MapError(f"two data points are at the same location{who}: kriging would be singular")
    A = np.ones((n + 1, n + 1))
    A[:n, :n] = model.gamma_xyz(dx, dy, np.zeros_like(dx))
    A[n, n] = 0.0
    Ainv = np.linalg.inv(A)
    nodes = np.asarray(nodes_xy, float).reshape(-1, 2)
    est, var = np.empty(len(nodes)), np.empty(len(nodes))
    for a in range(0, len(nodes), chunk):
        nd = nodes[a:a + chunk]
        g0 = model.gamma_xyz(x[None, :] - nd[:, :1], y[None, :] - nd[:, 1:2], np.zeros((len(nd), n)))
        rhs = np.concatenate([g0, np.ones((len(nd), 1))], axis=1)
        sol = rhs @ Ainv                                          # (A is symmetric)
        w = sol[:, :n]
        est[a:a + chunk] = w @ v
        var[a:a + chunk] = np.maximum((sol * rhs).sum(axis=1), 0.0)
    return est, var


# -- maps ----------------------------------------------------------------------------------------------------------------------------------
def _primary_surface(project: Project, uid: str):
    lab = project.primary.get(uid)
    return None if lab is None else project.surfaces[uid][lab]


def interval_bounds(project: Project, uid: str):
    """(top surface, base surface) of an interval from the primary surfaces of its two horizons; None for a missing one."""
    ch = project.chart
    if ch is None:
        raise ProjectError("define the stratigraphic chart first")
    i = ch.interval_index(uid)
    below = ch.intervals[i + 1].uid if i + 1 < len(ch.intervals) else BASE_UID
    return _primary_surface(project, uid), _primary_surface(project, below)


def thickness_map(project: Project, uid: str, grid: MapGrid) -> PropertyMap:
    top, base = interval_bounds(project, uid)
    name = project.chart.interval(uid).name
    if top is None or base is None:
        raise MapError(f"{name}: both its top and base horizons need a surface (missing: "
                       + ", ".join(n for n, s in (("top", top), ("base", base)) if s is None) + ")")
    X, Y = grid.node_xy()
    t = base.sample(X, Y) - top.sample(X, Y)
    crossed = int(np.sum(t < 0))
    t = np.where(t < 0, np.nan, t)
    info = dict(crossed_nodes=crossed)
    return PropertyMap(grid, t, "thickness", "m", "thickness", None, info)


def krige_map(project: Project, uid: str, variable: str, model: VariogramModel, grid: MapGrid, *, kind: str = "log",
              source=None, indicator: int | None = None, min_coverage: float = SKIP_BELOW, wells=None,
              mask: str = "thickness", bounds: tuple | None = None, unit: str = "") -> tuple:
    """Krige per-well layer averages onto `grid`. Returns (PropertyMap, notes).

    mask='thickness': nodes outside the interval (no thickness, or outside either surface) become NaN, if
    both bounding surfaces exist (else a note says it was not possible); mask='none' keeps every node.
    bounds=(lo, hi) clips the estimate to a physical range and counts the nodes it changed."""
    if mask not in ("thickness", "none"):
        raise MapError("mask must be 'thickness' or 'none'")
    if bounds is not None and not bounds[0] < bounds[1]:
        raise MapError("bounds must be (low, high) with low < high")
    ld = layer_averages(project, uid, variable, kind=kind, source=source, indicator=indicator, min_coverage=min_coverage, wells=wells)
    notes = list(ld.warnings) + [f"{w} not used: {r}" for w, r in ld.skipped.items()]
    if len(ld.wells) == 1:
        notes.append("Only one well: the map is that well's value everywhere.")
    elif len(ld.wells) < 4:
        notes.append(f"Only {len(ld.wells)} wells: the map is a smooth guess between them; treat it as a trend.")
    X, Y = grid.node_xy()
    est, var = krige(model, ld.x, ld.y, ld.value, np.c_[X.ravel(), Y.ravel()], names=ld.wells)
    est, var = est.reshape(X.shape), var.reshape(X.shape)
    clipped = 0
    if bounds is not None:
        c = np.clip(est, *bounds)
        clipped = int(np.sum(c != est))
        est = c
        if clipped:
            notes.append(f"{clipped} node(s) were outside {bounds} and were clipped.")
    n_masked = 0
    if mask == "thickness":
        top, base = interval_bounds(project, uid)
        if top is None or base is None:
            notes.append("Not masked to the interval: its top and base both need a surface.")
        else:
            t = base.sample(X, Y) - top.sample(X, Y)
            bad = ~(t > 0)
            n_masked = int(bad.sum())
            est, var = np.where(bad, np.nan, est), np.where(bad, np.nan, var)
    info = dict(wells=list(ld.wells), well_xy={w: [float(a), float(b)] for w, a, b in zip(ld.wells, ld.x, ld.y)}, coverage=dict(ld.coverage), skipped=dict(ld.skipped), clipped=clipped, masked=n_masked,
                variogram=dict(nugget=model.nugget, sill=model.sill, azimuth=model.azimuth,
                               ranges=[[s.range_major, s.range_minor] for s in model.structures]),
                indicator=indicator,
                fault_crossings={w: [dict(fault=c.fault, tvdss=c.tvdss) for c in cs] for w, cs in ld.crossings.items()})
    name = f"{variable}[{indicator}]" if indicator is not None else variable
    return PropertyMap(grid, est, name, unit, "kriged", var, info), notes


def export_map(pm: PropertyMap, path) -> None:
    """Write a map as IRAP ASCII so other software can read it. Undefined nodes are written as undefined."""
    import xtgeo
    g = pm.grid
    s = xtgeo.RegularSurface(ncol=g.ncol, nrow=g.nrow, xinc=g.xinc, yinc=g.yinc, xori=g.xori, yori=g.yori, rotation=g.rotation,
                             values=np.ma.masked_invalid(pm.values))
    s.to_file(str(path), fformat="irap_ascii")

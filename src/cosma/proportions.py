"""Vertical proportion curves (VPCs): the proportion of each facies as a function of position inside an interval.

A curve has bins. Position is either *relative* (0 at the top of the interval, 1 at its base, measured along the column
by cumulative layer thickness, so it follows the interval whatever its thickness) or *depth* (true vertical thickness
from the top of the interval, in metres, so bins have a fixed size and thick columns reach more bins). Each interval has
one curve, from one source chosen by the user:

  wells  computed from the hard-data cells, weighted by cell thickness; optional moving-average smoothing over bins
  user   typed in; the proportions of every bin must sum to 1 (they are checked, never rescaled)

Nothing is assumed: the number of bins, the smoothing window and the minimum amount of data per bin and per interval are
the user's. When an interval lacks the data for a curve from wells, the build says so and stops; it does not borrow
from another interval or fall back to a constant. A curve that gives zero proportion to a facies seen in a well at that
position is reported as a conflict with the hard data.

Combining a curve with a proportion map is a later step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MODES = ("relative", "depth")
SOURCES = ("wells", "user")
SUM_TOL = 1e-6


class ProportionError(ValueError):
    pass


def _posint(x, what):
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1:
        raise ProportionError(f"{what} must be a whole number >= 1")
    return int(x)


@dataclass(frozen=True)
class Settings:
    mode: str
    nbins: int | None = None              # relative: number of bins
    bin_size: float | None = None         # depth: bin thickness (m)
    window: int = 1                       # moving-average window in bins (odd; 1 = no smoothing)
    min_wells: int = 1                    # wells with hard cells in the interval, at least
    min_cells: int = 1                    # hard cells within the smoothing window of every bin, at least

    def __post_init__(self):
        if self.mode not in MODES:
            raise ProportionError(f"position mode must be one of {', '.join(MODES)}")
        if self.mode == "relative":
            _posint(self.nbins, "the number of bins")
            if self.bin_size is not None:
                raise ProportionError("relative mode takes a number of bins, not a bin size")
        else:
            if not (isinstance(self.bin_size, (int, float)) and math.isfinite(self.bin_size) and self.bin_size > 0):
                raise ProportionError("the bin size must be a number > 0 (m)")
            if self.nbins is not None:
                raise ProportionError("depth mode takes a bin size, not a number of bins")
        w = _posint(self.window, "the smoothing window")
        if w % 2 == 0:
            raise ProportionError("the smoothing window must be odd (1 = no smoothing)")
        _posint(self.min_wells, "the minimum number of wells")
        _posint(self.min_cells, "the minimum number of hard cells per bin")


@dataclass(frozen=True)
class Curve:
    interval: str
    mode: str
    bin_size: float                  # relative: 1 / nbins; depth: metres
    codes: tuple                     # facies codes, in the order of the columns of ``values``
    values: tuple                    # per bin, a tuple of proportions (one per code), each row summing to 1
    source: str
    n_cells: tuple | None = None     # wells: hard cells supporting each bin (after smoothing)
    n_wells: int | None = None

    def __post_init__(self):
        if self.mode not in MODES or self.source not in SOURCES:
            raise ProportionError("bad curve mode or source")
        if not (isinstance(self.bin_size, (int, float)) and math.isfinite(self.bin_size) and self.bin_size > 0):
            raise ProportionError("the bin size must be a number > 0")
        if self.mode == "relative" and abs(round(1.0 / self.bin_size) * self.bin_size - 1.0) > 1e-9:
            raise ProportionError("relative bins must divide 0-1 into a whole number of bins")
        if not self.values or not self.codes:
            raise ProportionError("a curve needs at least one bin and one facies")
        if len(set(self.codes)) != len(self.codes):
            raise ProportionError("facies codes of a curve must be unique")
        for b, row in enumerate(self.values):
            if len(row) != len(self.codes):
                raise ProportionError(f"bin {b + 1} has {len(row)} values for {len(self.codes)} facies")
            if any(not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0) for v in row):
                raise ProportionError(f"bin {b + 1}: proportions must be numbers >= 0")
            if abs(sum(row) - 1.0) > SUM_TOL:
                raise ProportionError(f"bin {b + 1}: proportions sum to {sum(row):.6g}, not 1 (they are not rescaled)")

    @property
    def nbins(self):
        return len(self.values)

    def array(self):
        return np.array(self.values, float)

    def bin_index(self, position):
        """Bin of each position (relative 0-1, or metres). Raises if a depth lies beyond the last bin."""
        p = np.asarray(position, float)
        idx = np.floor(p / self.bin_size + 1e-9).astype(int)
        if self.mode == "relative":
            return np.clip(idx, 0, self.nbins - 1)
        if (idx >= self.nbins).any():
            need = float(p.max())
            raise ProportionError(f"interval {self.interval!r}: the curve reaches {self.nbins * self.bin_size:g} m but the "
                                  f"interval is {need:g} m thick somewhere; extend the curve")
        return np.clip(idx, 0, None)

    def proportions(self, position):
        return self.array()[self.bin_index(position)]

    def global_proportions(self, positions, weights):
        """Thickness-weighted mean proportions over cells at ``positions`` with ``weights``."""
        w = np.asarray(weights, float)
        return (self.proportions(positions) * w[:, None]).sum(axis=0) / w.sum()


@dataclass(frozen=True)
class ProportionSetup:
    settings: Settings
    choices: tuple = ()               # ((interval, 'wells' | 'user'), ...)
    user_curves: tuple = ()           # (Curve, ...) for the intervals whose source is 'user'

    def __post_init__(self):
        ch = dict(self.choices)
        if len(ch) != len(self.choices):
            raise ProportionError("an interval has two sources")
        for iv, src in ch.items():
            if src not in SOURCES:
                raise ProportionError(f"interval {iv!r}: source must be one of {', '.join(SOURCES)}")
        uc = {c.interval: c for c in self.user_curves}
        if len(uc) != len(self.user_curves):
            raise ProportionError("two curves for one interval")
        for iv, c in uc.items():
            if ch.get(iv) != "user" or c.source != "user":
                raise ProportionError(f"interval {iv!r}: a typed curve is given but the source is not 'user'")
            if c.mode != self.settings.mode:
                raise ProportionError(f"interval {iv!r}: the typed curve is {c.mode}, the setup is {self.settings.mode}")
        for iv, src in ch.items():
            if src == "user" and iv not in uc:
                raise ProportionError(f"interval {iv!r}: the source is 'user' but no curve is typed")

    def source_of(self, interval):
        return dict(self.choices).get(interval)

    def curve_of(self, interval):
        return {c.interval: c for c in self.user_curves}.get(interval)


def to_state(s):
    if s is None:
        return None
    t = s.settings
    return dict(settings=dict(mode=t.mode, nbins=t.nbins, bin_size=t.bin_size, window=t.window, min_wells=t.min_wells,
                              min_cells=t.min_cells),
                choices=[list(c) for c in s.choices],
                curves=[dict(interval=c.interval, mode=c.mode, bin_size=c.bin_size, codes=list(c.codes),
                             values=[list(r) for r in c.values]) for c in s.user_curves])


def from_state(d):
    if d is None:
        return None
    try:
        curves = tuple(Curve(c["interval"], c["mode"], c["bin_size"], tuple(c["codes"]), tuple(tuple(r) for r in c["values"]),
                             "user") for c in d.get("curves", []))
        return ProportionSetup(Settings(**d["settings"]), tuple((a, b) for a, b in d["choices"]), curves)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        if isinstance(exc, ProportionError):
            raise
        raise ProportionError(f"bad proportion setup: {exc}") from None


# -- geometry -----------------------------------------------------------------------------------------------------------

def interval_geometry(grid, layer_interval, a):
    """For interval index ``a``: layer indices, cell thickness (nx, ny, n), position of each cell centre as cumulative
    thickness from the top of the interval (nx, ny, n), and the interval's total thickness (nx, ny)."""
    from .grid.builder import _cell_thickness
    ks = np.nonzero(np.asarray(layer_interval) == a)[0]
    h = _cell_thickness(grid.zcorn, grid.nx, grid.ny, grid.nz)[:, :, ks]
    h = np.where(np.isfinite(h) & (h > 0), h, 0.0)
    cum = np.cumsum(h, axis=2) - 0.5 * h
    return ks, h, cum, h.sum(axis=2)


def _positions(curve_mode, cum, total):
    if curve_mode == "relative":
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(total[:, :, None] > 0, cum / total[:, :, None], 0.0)
    return cum


# -- from wells ----------------------------------------------------------------------------------------------------------

@dataclass
class IntervalResult:
    interval: str
    source: str
    curve: Curve | None = None
    problems: list = field(default_factory=list)
    global_curve: tuple | None = None      # thickness-weighted mean over the interval's active cells
    global_wells: tuple | None = None      # thickness-weighted mean over its hard cells
    conflicts: int = 0                     # hard cells whose facies has zero proportion in the curve at their position
    cells: int = 0
    wells: int = 0


@dataclass
class ProportionResult:
    scheme: tuple
    setup: ProportionSetup
    intervals: list = field(default_factory=list)

    @property
    def ok(self):
        return all(not r.problems and r.conflicts == 0 and r.curve is not None for r in self.intervals)

    def curves(self):
        return {r.interval: r.curve for r in self.intervals if r.curve is not None}

    def summary(self) -> str:
        names = {f.code: f.name for f in self.scheme}
        s = self.setup.settings
        pos = (f"relative position, {s.nbins} bins" if s.mode == "relative" else f"true depth, {s.bin_size:g} m bins")
        out = [f"Vertical proportion curves ({pos}; smoothing window {s.window}; at least {s.min_wells} well(s) and "
               f"{s.min_cells} hard cell(s) per bin):"]
        for r in self.intervals:
            head = f"\n{r.interval}: source {r.source}"
            if r.source == "wells":
                head += f", {r.cells} hard cell(s) from {r.wells} well(s)"
            out.append(head)
            for p in r.problems:
                out.append(f"  PROBLEM: {p}")
            if r.curve is None:
                continue
            if r.global_curve is not None:
                out.append("  interval mean from the curve: " + ", ".join(f"{names[c]} {100 * v:.1f}%" for c, v in
                                                                         zip(r.curve.codes, r.global_curve)))
            if r.global_wells is not None:
                out.append("  interval mean from the hard cells: " + ", ".join(f"{names[c]} {100 * v:.1f}%" for c, v in
                                                                              zip(r.curve.codes, r.global_wells)))
                diff = max(abs(a - b) for a, b in zip(r.global_curve, r.global_wells)) * 100
                out.append(f"  largest difference: {diff:.1f} percentage points")
            if r.conflicts:
                out.append(f"  CONFLICT: {r.conflicts} hard cell(s) have a facies the curve gives zero proportion at its position")
        return "\n".join(out)


def _smooth_sum(a, window):
    """Moving sum over ``window`` bins along axis 0, window truncated at the ends."""
    r = window // 2
    out = np.zeros_like(a)
    for b in range(a.shape[0]):
        out[b] = a[max(0, b - r):b + r + 1].sum(axis=0)
    return out


def _from_wells(name, codes, st: Settings, grid, layer_interval, a, hard, ok_rows):
    res = IntervalResult(name, "wells")
    ks, h, cum, total = interval_geometry(grid, layer_interval, a)
    if not ks.size:
        res.problems.append("the interval has no layers")
        return res
    pos = _positions(st.mode, cum, total)
    sub = hard.cells[:, :, ks]
    I, J, L = np.nonzero(sub > 0)
    res.cells = int(len(I))
    rows = ok_rows & (np.isin(hard.k, ks))
    res.wells = len(set(hard.well[rows]))
    if not len(I):
        res.problems.append("no hard cells in this interval")
        return res
    if res.wells < st.min_wells:
        res.problems.append(f"{res.wells} well(s) with hard cells, {st.min_wells} needed")
    if st.mode == "relative":
        nb = st.nbins
        size = 1.0 / nb
    else:
        size = st.bin_size
        act_total = total[grid.actnum[:, :, ks].any(axis=2)] if grid.actnum is not None else total
        nb = max(1, int(math.ceil(float(act_total.max()) / size - 1e-9)))
    bins = np.clip(np.floor(pos[I, J, L] / size + 1e-9).astype(int), 0, nb - 1)
    code_idx = {c: n for n, c in enumerate(codes)}
    cnt = np.zeros((nb, len(codes)))
    wsum = np.zeros((nb, len(codes)))
    for i, j, l, b in zip(I, J, L, bins):
        f = code_idx.get(int(sub[i, j, l]))
        if f is None:
            continue
        cnt[b, f] += 1
        wsum[b, f] += h[i, j, l]
    cs, ws = _smooth_sum(cnt, st.window), _smooth_sum(wsum, st.window)
    support = cs.sum(axis=1)
    weak = [b + 1 for b in range(nb) if support[b] < st.min_cells]
    if weak:
        res.problems.append(f"bin(s) {', '.join(map(str, weak))} have fewer than {st.min_cells} hard cell(s) within the "
                            f"smoothing window; use fewer or wider bins, a wider window, or type the curve")
    if res.problems:
        return res
    tot = ws.sum(axis=1, keepdims=True)
    values = tuple(tuple(float(x) for x in r) for r in ws / tot)
    res.curve = Curve(name, st.mode, size, tuple(codes), values, "wells", tuple(int(x) for x in support), res.wells)
    return res


def build_curves(project, model, setup: ProportionSetup | None = None) -> ProportionResult:
    """Curves for every interval of ``model`` (a ModelBuild), from the user's choice per interval; and the checks."""
    from .facies import FaciesError, classify_all, upscale
    setup = setup or project.proportions
    fs = project.facies_setup
    if setup is None:
        raise ProportionError("set up the proportions first")
    if fs is None:
        raise ProportionError("set up the facies first (Facies menu)")
    if fs.min_coverage is None or fs.conflict is None:
        raise ProportionError("choose the minimum classified share and the conflict policy in the facies setup first")
    names = list(model.interval_names)
    missing = [n for n in names if setup.source_of(n) is None]
    if missing:
        raise ProportionError("choose a proportion source for: " + ", ".join(missing))
    extra = [n for n, _ in setup.choices if n not in names]
    if extra:
        raise ProportionError("sources are given for intervals that do not exist: " + ", ".join(extra))
    codes = tuple(f.code for f in fs.scheme)
    grid = model.grid
    li = np.asarray(model.report.layer_interval)
    try:
        wf, skipped = classify_all(project, fs)
        if not wf:
            raise ProportionError("no well could be classified: " + "; ".join(f"{w}: {r}" for w, r in skipped.items()))
        hard = upscale(wf, grid, fs.scheme, min_coverage=fs.min_coverage, conflict=fs.conflict, skipped=skipped)
    except FaciesError as exc:
        raise ProportionError(str(exc)) from None
    ok_rows = (hard.code > 0) & (hard.status == "ok") if hard.k.size else np.zeros(0, bool)
    result = ProportionResult(fs.scheme, setup)
    for a, name in enumerate(names):
        if setup.source_of(name) == "wells":
            r = _from_wells(name, codes, setup.settings, grid, li, a, hard, ok_rows)
        else:
            r = IntervalResult(name, "user")
            c = setup.curve_of(name)
            if tuple(c.codes) != codes:
                r.problems.append(f"the typed curve has facies codes {list(c.codes)} but the scheme has {list(codes)}")
            else:
                r.curve = c
        if r.curve is not None:
            _check_against_data(r, grid, li, a, hard, setup.settings, codes)
        result.intervals.append(r)
    return result


def _check_against_data(r: IntervalResult, grid, li, a, hard, st, codes):
    ks, h, cum, total = interval_geometry(grid, li, a)
    if not ks.size:
        return
    pos = _positions(st.mode, cum, total)
    act = grid.actnum[:, :, ks].astype(bool) & (h > 0)
    try:
        if act.any():
            r.global_curve = tuple(float(x) for x in r.curve.global_proportions(pos[act], h[act]))
        sub = hard.cells[:, :, ks]
        I, J, L = np.nonzero(sub > 0)
        if len(I):
            p = r.curve.proportions(pos[I, J, L])
            idx = np.array([codes.index(int(sub[i, j, l])) if int(sub[i, j, l]) in codes else -1 for i, j, l in zip(I, J, L)])
            ok = idx >= 0
            r.conflicts = int((p[np.arange(len(I))[ok], idx[ok]] <= 0).sum())
            wts = h[I, J, L][ok]
            oh = np.zeros(len(codes))
            for f in range(len(codes)):
                oh[f] = wts[idx[ok] == f].sum()
            if oh.sum() > 0:
                r.global_wells = tuple(float(x) for x in oh / oh.sum())
    except ProportionError as exc:
        r.problems.append(str(exc))
        r.curve = None


def layer_proportions(model, result: ProportionResult) -> dict:
    """{interval: array (nx, ny, nlayers, nfacies)} of proportions for every cell; refuses if the result has problems."""
    if not result.ok:
        raise ProportionError("the proportion curves have problems or conflicts; resolve them first")
    grid = model.grid
    li = np.asarray(model.report.layer_interval)
    st = result.setup.settings
    out = {}
    for a, r in enumerate(result.intervals):
        ks, h, cum, total = interval_geometry(grid, li, a)
        pos = _positions(st.mode, cum, total)
        out[r.interval] = r.curve.proportions(pos.reshape(-1)).reshape(pos.shape + (len(r.curve.codes),))
    return out


def combine_with_maps(model, result: ProportionResult, maps: dict, outside_map: str, max_incompatible: float) -> tuple:
    """Per cell p_k = vpc_k(position) * map_k(i, j) / G_k, renormalised to sum 1, for the intervals that have a map
    (G_k = thickness-weighted interval mean of the curve, ``IntervalResult.global_curve``). Intervals without a map keep
    their curve. Every decision is an argument, nothing is defaulted:
      outside_map      'curve' = columns without map data keep the curve alone; 'stop' = refuse if any such column exists.
      max_incompatible largest share (0-1) of cells where curve and map leave nothing (the renormalising sum is 0); those
                       cells keep the curve alone and are counted in the report; more than this share stops the build.
    -> ({interval: (nx, ny, nlayers, nfac)}, report text)."""
    if outside_map not in ("curve", "stop"):
        raise ProportionError("outside_map must be 'curve' or 'stop'")
    if not (isinstance(max_incompatible, (int, float)) and 0 <= max_incompatible <= 1):
        raise ProportionError("max_incompatible must be a number between 0 and 1")
    base = layer_proportions(model, result)
    lines = []
    for r in result.intervals:
        pm = maps.get(r.interval)
        if pm is None:
            continue
        if tuple(pm.codes) != tuple(r.curve.codes):
            raise ProportionError(f"interval {r.interval!r}: the map and the curve do not have the same facies")
        p = base[r.interval]
        nx, ny = p.shape[:2]
        if pm.fractions.shape[:2] != (nx, ny):
            raise ProportionError(f"interval {r.interval!r}: the proportion map was made for a {pm.fractions.shape[0]} x "
                                  f"{pm.fractions.shape[1]} grid, the model is {nx} x {ny}; import the map again")
        G = np.asarray(r.global_curve, float)
        m = pm.fractions[:, :, None, :]
        nodata = np.isnan(pm.fractions[:, :, 0])
        active = np.asarray(model.grid.actnum) > 0                      # (nx, ny, nz)
        ks = np.nonzero(np.asarray(model.report.layer_interval) == [x.interval for x in result.intervals].index(r.interval))[0]
        colactive = active[:, :, ks].any(axis=2)
        miss = nodata & colactive
        if miss.any() and outside_map == "stop":
            raise ProportionError(f"interval {r.interval!r}: {int(miss.sum())} active column(s) have no map data")
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where(G > 0, p * np.nan_to_num(m) / G, 0.0)
        s = w.sum(axis=-1, keepdims=True)
        bad = (s[..., 0] <= 0) & ~nodata[:, :, None]
        cells = int((colactive[:, :, None] & np.ones_like(bad)).sum())
        nbad = int((bad & colactive[:, :, None]).sum())
        share = nbad / cells if cells else 0.0
        if share > max_incompatible:
            raise ProportionError(f"interval {r.interval!r}: curve and map leave nothing in {nbad} of {cells} cells "
                                  f"({100 * share:.1f}%), more than the {100 * max_incompatible:.1f}% allowed")
        with np.errstate(invalid="ignore", divide="ignore"):
            new = w / s
        keep = nodata[:, :, None] | bad
        out = np.where(keep[..., None], p, new)
        base[r.interval] = out
        lines.append(f"{r.interval}: map applied; {int(miss.sum())} active column(s) without map data kept the curve; "
                     f"{nbad} incompatible cell(s) ({100 * share:.1f}%) kept the curve")
    return base, "\n".join(lines)

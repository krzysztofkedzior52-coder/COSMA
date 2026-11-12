"""Facies from well logs: the hard data for facies simulation.

Everything is explicit; nothing is guessed:
  * a facies scheme: codes (whole numbers >= 1; 0 means "no facies") with names;
  * how a log becomes facies: either ordered RULES (first rule whose conditions all hold wins; a sample no rule
    matches stays undefined) on one or more curves of one log source, or a CODE MAP when the curve already holds
    facies codes (a code with no mapping stays undefined and is reported);
  * user FLAGS: depth intervals of a well that are a given facies, overriding the log; a well with flags but no log
    is sampled every ``flag_step`` metres;
  * upscaling to cells: each cell takes the facies with the greatest length of well path in it, provided the
    classified length is at least ``min_coverage`` of the path length in the cell. An exact tie leaves the cell
    undefined. Two wells in one cell that disagree are a conflict, resolved only by the policy the caller names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .wells.io import WellDataError, _depth_to_position
from .wells.trajectory import TrajectoryError

OPS = ("<", "<=", ">", ">=")
DEPTH_TYPES = ("md", "tvd", "tvdss")
CONFLICT_POLICIES = ("longest", "drop")
_CMP = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}


class FaciesError(ValueError):
    pass


def _finite(x, what):
    if isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating)) or not math.isfinite(x):
        raise FaciesError(f"{what} must be a number")
    return float(x)


@dataclass(frozen=True)
class Facies:
    code: int
    name: str

    def __post_init__(self):
        if isinstance(self.code, bool) or not isinstance(self.code, (int, np.integer)) or self.code < 1:
            raise FaciesError("a facies code must be a whole number >= 1 (0 means no facies)")
        if not isinstance(self.name, str) or not self.name.strip():
            raise FaciesError(f"facies {self.code} needs a name")


@dataclass(frozen=True)
class Condition:
    curve: str
    op: str
    value: float

    def __post_init__(self):
        if not isinstance(self.curve, str) or not self.curve.strip():
            raise FaciesError("a condition needs a curve name")
        if self.op not in OPS:
            raise FaciesError(f"condition operator {self.op!r} is not one of {', '.join(OPS)}")
        _finite(self.value, "a condition value")


@dataclass(frozen=True)
class Rule:
    facies: int
    conditions: tuple

    def __post_init__(self):
        if not self.conditions or not all(isinstance(c, Condition) for c in self.conditions):
            raise FaciesError("a rule needs at least one condition")


@dataclass(frozen=True)
class CodeMap:
    curve: str
    mapping: dict          # curve value (whole number) -> scheme code

    def __post_init__(self):
        if not self.curve or not self.mapping:
            raise FaciesError("a code map needs a curve and at least one mapping")
        for k, v in self.mapping.items():
            if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
                raise FaciesError("code map keys must be whole-number curve values")


@dataclass(frozen=True)
class Flag:
    well: str
    top: float
    base: float
    depth_type: str
    facies: int

    def __post_init__(self):
        _finite(self.top, "a flag top")
        _finite(self.base, "a flag base")
        if self.base <= self.top:
            raise FaciesError(f"flag in {self.well}: base must be below top")
        if self.depth_type not in DEPTH_TYPES:
            raise FaciesError(f"flag depth type {self.depth_type!r} is not one of {', '.join(DEPTH_TYPES)}")


@dataclass(frozen=True)
class FaciesSetup:
    scheme: tuple
    rules: tuple = ()
    code_map: CodeMap | None = None
    source: object = None            # log source name for all wells, or {well: source}; needed only where ambiguous
    flags: tuple = ()
    flag_step: float | None = None   # m; needed when a flagged well has no log
    min_coverage: float | None = None   # upscaling: share of a cell's well path that must be classified (None = not yet chosen)
    conflict: str | None = None         # upscaling: how wells that disagree in a cell are resolved (None = not yet chosen)

    def __post_init__(self):
        if not self.scheme:
            raise FaciesError("define at least one facies")
        codes = [f.code for f in self.scheme]
        names = [f.name for f in self.scheme]
        if len(set(codes)) != len(codes):
            raise FaciesError("facies codes must be unique")
        if len({n.strip().lower() for n in names}) != len(names):
            raise FaciesError("facies names must be unique")
        if self.rules and self.code_map is not None:
            raise FaciesError("use rules or a code map, not both")
        if not (self.rules or self.code_map is not None or self.flags):
            raise FaciesError("say how facies are found: rules, a code map or flags")
        for r in self.rules:
            if r.facies not in codes:
                raise FaciesError(f"a rule gives facies {r.facies}, which is not in the scheme")
        if self.code_map is not None:
            bad = sorted(set(self.code_map.mapping.values()) - set(codes))
            if bad:
                raise FaciesError(f"the code map gives facies {bad}, which are not in the scheme")
        for f in self.flags:
            if f.facies not in codes:
                raise FaciesError(f"a flag in {f.well} gives facies {f.facies}, which is not in the scheme")
        if self.min_coverage is not None and not (isinstance(self.min_coverage, (int, float)) and 0 < self.min_coverage <= 1):
            raise FaciesError("min_coverage must be in (0, 1]")
        if self.conflict is not None and self.conflict not in CONFLICT_POLICIES:
            raise FaciesError(f"conflict policy must be one of {', '.join(CONFLICT_POLICIES)}")
        if self.flag_step is not None:
            _finite(self.flag_step, "the flag step")
            if self.flag_step <= 0:
                raise FaciesError("the flag step must be > 0")

    def name_of(self, code: int) -> str:
        return next((f.name for f in self.scheme if f.code == code), f"code {code}")


# -- persistence ------------------------------------------------------------------------------------------------------

def to_state(s: FaciesSetup | None):
    if s is None:
        return None
    return dict(
        scheme=[[f.code, f.name] for f in s.scheme],
        rules=[dict(facies=r.facies, conditions=[[c.curve, c.op, c.value] for c in r.conditions]) for r in s.rules],
        code_map=None if s.code_map is None else dict(curve=s.code_map.curve,
                                                        mapping=[[int(k), int(v)] for k, v in s.code_map.mapping.items()]),
        source=s.source, flags=[[f.well, f.top, f.base, f.depth_type, f.facies] for f in s.flags], flag_step=s.flag_step,
        min_coverage=s.min_coverage, conflict=s.conflict)


def from_state(d) -> FaciesSetup | None:
    if d is None:
        return None
    try:
        cm = d.get("code_map")
        return FaciesSetup(
            tuple(Facies(int(c), n) for c, n in d["scheme"]),
            tuple(Rule(int(r["facies"]), tuple(Condition(c, o, v) for c, o, v in r["conditions"])) for r in d.get("rules", [])),
            None if cm is None else CodeMap(cm["curve"], {int(k): int(v) for k, v in cm["mapping"]}),
            d.get("source"), tuple(Flag(w, t, b, dt, int(f)) for w, t, b, dt, f in d.get("flags", [])), d.get("flag_step"),
            d.get("min_coverage"), d.get("conflict"))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise FaciesError(f"bad facies setup: {exc}") from None


# -- classifying a well -------------------------------------------------------------------------------------------------

@dataclass
class WellFacies:
    well: str
    source: str | None                  # log source used, None for a flags-only well
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray                       # TVDSS
    length: np.ndarray                  # path length each sample stands for (m)
    code: np.ndarray                    # int, 0 = undefined
    unmapped: dict = field(default_factory=dict)    # curve value -> samples (code map only)
    flagged: int = 0                    # samples set by a flag
    outside_path: int = 0               # log samples beyond the trajectory, dropped
    src_index: np.ndarray | None = None  # positions of these samples in the log source's arrays (None for flags-only wells)


def _lengths(md: np.ndarray) -> np.ndarray:
    if md.size == 1:
        return np.ones(1)
    mid = 0.5 * (md[1:] + md[:-1])
    edges = np.concatenate([[md[0] - (mid[0] - md[0])], mid, [md[-1] + (md[-1] - mid[-1])]])
    return np.diff(edges)


def _pick_source(well: str, logs: dict, want) -> str | None:
    if isinstance(want, dict):
        want = want.get(well)
    if want is not None:
        if want not in logs:
            raise FaciesError(f"no log source {want!r}")
        return want
    if len(logs) > 1:
        raise FaciesError(f"several log sources ({', '.join(sorted(logs))}); name the one to use")
    return next(iter(logs), None)


def _flag_range(well: str, tr, f: Flag):
    try:
        a = _depth_to_position(well, tr, f.top, f.depth_type)[0]
        b = _depth_to_position(well, tr, f.base, f.depth_type)[0]
    except (WellDataError, TrajectoryError) as exc:
        raise FaciesError(f"flag in {well}: {exc}") from None
    if b <= a:
        raise FaciesError(f"flag in {well}: base is not below top along the well")
    return a, b


def classify_well(rec, setup: FaciesSetup) -> WellFacies:
    tr = rec.trajectory
    if tr is None:
        raise FaciesError("needs a header and a deviation survey")
    flags = [f for f in setup.flags if f.well == rec.name]
    uses_log = bool(setup.rules) or setup.code_map is not None
    src = _pick_source(rec.name, rec.logs, setup.source) if uses_log and rec.logs else None
    if uses_log and src is None and not flags:
        raise FaciesError("no log")
    lo, hi = tr.md_range
    unmapped, outside = {}, 0
    if src is not None:
        ls = rec.logs[src]
        md_all = np.asarray(ls.md, float)
        keep = (md_all >= lo) & (md_all <= hi)
        outside = int((~keep).sum())
        md = md_all[keep]
        src_index = np.nonzero(keep)[0]
        if md.size == 0:
            raise FaciesError("the log lies outside the well trajectory")
        cur = {k: np.asarray(v, float)[keep] for k, v in ls.curves.items()}
        code = np.zeros(md.size, int)
        if setup.rules:
            need = {c.curve for r in setup.rules for c in r.conditions}
            miss = sorted(need - set(cur))
            if miss:
                raise FaciesError(f"log source {src!r} has no curve {', '.join(miss)}")
            for r in setup.rules:
                m = np.ones(md.size, bool)
                for c in r.conditions:
                    v = cur[c.curve]
                    m &= np.isfinite(v) & _CMP[c.op](v, c.value)
                code[m & (code == 0)] = r.facies
        else:
            cm = setup.code_map
            if cm.curve not in cur:
                raise FaciesError(f"log source {src!r} has no curve {cm.curve!r}")
            v = cur[cm.curve]
            ok = np.isfinite(v)
            if np.any(np.abs(v[ok] - np.round(v[ok])) > 1e-9):
                raise FaciesError(f"curve {cm.curve!r} does not hold whole-number codes")
            vi = np.where(ok, np.round(np.where(ok, v, 0.0)), 0).astype(int)
            for k in np.unique(vi[ok]):
                sel = ok & (vi == k)
                if int(k) in cm.mapping:
                    code[sel] = cm.mapping[int(k)]
                else:
                    unmapped[int(k)] = int(sel.sum())
        length = _lengths(md)
    else:
        md, code, length = np.empty(0), np.empty(0, int), np.empty(0)
        src_index = None
    n_flagged = 0
    if flags:
        ranges = [(_flag_range(rec.name, tr, f), f) for f in flags]
        if src is None:
            if setup.flag_step is None:
                raise FaciesError("this well has flags but no log: set the flag step (sampling distance along the well)")
            mds, lens = [], []
            for (a, b), _ in ranges:
                a, b = max(a, lo), min(b, hi)
                if b > a:
                    n = max(1, int(round((b - a) / setup.flag_step)))
                    mds.append(a + (np.arange(n) + 0.5) * (b - a) / n)
                    lens.append(np.full(n, (b - a) / n))
            if mds:
                md, length = np.concatenate(mds), np.concatenate(lens)
                order = np.argsort(md, kind="stable")
                md, length = md[order], length[order]
            else:
                md, length = np.empty(0), np.empty(0)
            code = np.zeros(md.size, int)
        for (a, b), f in ranges:
            m = (md >= a) & (md <= b)
            code[m] = f.facies
            n_flagged += int(m.sum())
    if md.size == 0:
        raise FaciesError("no samples to classify")
    x, y, z = tr.at_md(md)
    return WellFacies(rec.name, src, md, np.asarray(x, float), np.asarray(y, float), np.asarray(z, float),
                      length, code, unmapped, n_flagged, outside, src_index)


def classify_all(project, setup: FaciesSetup, wells=None) -> tuple:
    """({well: WellFacies}, {well: reason it was skipped})."""
    out, skipped = {}, {}
    for name, rec in project.wells.items():
        if wells is not None and name not in wells:
            continue
        if not (setup.rules or setup.code_map is not None) and not any(f.well == name for f in setup.flags):
            skipped[name] = "no flags for this well and no rules or code map"
            continue
        try:
            out[name] = classify_well(rec, setup)
        except FaciesError as exc:
            skipped[name] = str(exc)
    unknown = sorted({f.well for f in setup.flags} - set(project.wells))
    for w in unknown:
        skipped[w] = "flags given for a well that does not exist"
    return out, skipped


# -- locating samples in the grid ---------------------------------------------------------------------------------------

def locate_cells(grid, X, Y, Z, chunk: int = 20000):
    """Cell (i, j, k) of points given in map coordinates and TVDSS depth. Returns int arrays i, j, k (-1 where the point is
    outside the grid footprint or in no active cell) and a reason array: 0 found, 1 outside footprint, 2 not in an active
    cell. Pillars must be vertical (as COSMA builds them)."""
    c = grid.coord
    if not (np.allclose(c[..., 0], c[..., 3]) and np.allclose(c[..., 1], c[..., 4])):
        raise FaciesError("cell lookup needs vertical pillars")
    lx, ly = grid.frame.to_local(np.asarray(X, float), np.asarray(Y, float))
    xe, ye = c[0, :, 0], c[:, 0, 1]
    Z = np.asarray(Z, float)
    n = lx.size
    I = np.full(n, -1)
    J = np.full(n, -1)
    K = np.full(n, -1)
    why = np.zeros(n, int)
    i = np.searchsorted(xe, lx, side="right") - 1
    j = np.searchsorted(ye, ly, side="right") - 1
    inside = (i >= 0) & (i < grid.nx) & (j >= 0) & (j < grid.ny) & np.isfinite(lx) & np.isfinite(ly) & np.isfinite(Z)
    why[~inside] = 1
    idx = np.nonzero(inside)[0]
    act = grid.actnum.astype(bool)
    for s in range(0, idx.size, chunk):
        p = idx[s:s + chunk]
        ii, jj = i[p], j[p]
        u = (lx[p] - xe[ii]) / (xe[ii + 1] - xe[ii])
        v = (ly[p] - ye[jj]) / (ye[jj + 1] - ye[jj])
        zc = grid.zcorn[:, :, jj, :, ii, :]                 # (npts, nz, 2[t], 2[j], 2[i]) after fancy-index reordering
        w00, w10, w01, w11 = (1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v
        z = (zc[..., 0, 0] * w00[:, None, None] + zc[..., 0, 1] * w10[:, None, None]
             + zc[..., 1, 0] * w01[:, None, None] + zc[..., 1, 1] * w11[:, None, None])      # (npts, nz, 2)
        top, bot = z[..., 0], z[..., 1]
        hit = (top <= Z[p][:, None]) & (Z[p][:, None] < bot) & act[ii, jj, :]
        found = hit.any(axis=1)
        k = np.argmax(hit, axis=1)
        I[p[found]], J[p[found]], K[p[found]] = ii[found], jj[found], k[found]
        why[p[~found]] = 2
    return I, J, K, why


# -- upscaling to cells -------------------------------------------------------------------------------------------------

@dataclass
class HardFacies:
    well: np.ndarray          # (n,) object
    i: np.ndarray
    j: np.ndarray
    k: np.ndarray
    code: np.ndarray          # 0 = undefined (no hard datum)
    coverage: np.ndarray      # classified length / path length in the cell
    path: np.ndarray          # path length in the cell (m)
    status: np.ndarray        # 'ok' | 'low_coverage' | 'tie'
    cells: np.ndarray         # (nx, ny, nz) int: facies code of each hard-data cell, 0 = none
    conflicts: list           # [(i, j, k, {well: code})]
    samples: dict             # counts: total, in_cells, outside_footprint, not_in_active_cell
    unmapped: dict            # well -> {curve value: samples}
    skipped: dict
    scheme: tuple
    min_coverage: float
    conflict_policy: str

    def counts(self) -> dict:
        """{code: number of hard-data cells}"""
        c, n = np.unique(self.cells[self.cells > 0], return_counts=True)
        return {int(a): int(b) for a, b in zip(c, n)}

    def summary(self) -> str:
        names = {f.code: f.name for f in self.scheme}
        s = self.samples
        lines = [f"{s['total']:,} samples from {s['wells']} well(s), {len(set(self.well))} reaching active cells: "
                 f"{s['in_cells']:,} fall in active cells, "
                 f"{s['outside_footprint']:,} outside the grid footprint, {s['not_in_active_cell']:,} above/below/in inactive cells."]
        ok = int(((self.status == "ok") & (self.code > 0)).sum())
        lines.append(f"{len(self.code):,} well-cell pairs: {ok:,} give a facies; "
                     f"{int((self.status == 'low_coverage').sum())} left undefined for coverage below {self.min_coverage:g}; "
                     f"{int((self.status == 'tie').sum())} left undefined by an exact tie.")
        cnt = self.counts()
        tot = sum(cnt.values())
        for code, n in sorted(cnt.items()):
            lines.append(f"  {names.get(code, code)} (code {code}): {n:,} hard cells ({100 * n / tot:.1f}%)")
        for f in self.scheme:
            if f.code not in cnt:
                lines.append(f"  {f.name} (code {f.code}): NO hard cells")
        if self.conflicts:
            lines.append(f"{len(self.conflicts)} cell(s) are met by wells that disagree; policy: "
                         + ("the facies with the longest path wins" if self.conflict_policy == "longest"
                            else "such cells carry no hard datum"))
        for w, m in self.unmapped.items():
            if m:
                lines.append(f"{w}: curve values with no mapping, left undefined: "
                             + ", ".join(f"{k} ({v} samples)" for k, v in sorted(m.items())))
        for w, r in self.skipped.items():
            lines.append(f"skipped {w}: {r}")
        return "\n".join(lines)


def upscale(wf: dict, grid, scheme: tuple, *, min_coverage: float, conflict: str, skipped=None) -> HardFacies:
    """Hard facies per cell from classified wells (see the module docstring for the rules)."""
    if not (isinstance(min_coverage, (int, float)) and 0 < min_coverage <= 1):
        raise FaciesError("min_coverage must be in (0, 1]: the share of the path in a cell that must be classified")
    if conflict not in CONFLICT_POLICIES:
        raise FaciesError(f"conflict policy must be one of {', '.join(CONFLICT_POLICIES)}")
    codes = [f.code for f in scheme]
    ncode = max(codes) + 1
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    rows = []                                   # (well, i, j, k, code, coverage, path, status, per-code lengths)
    counts = dict(total=0, in_cells=0, outside_footprint=0, not_in_active_cell=0, wells=len(wf))
    for w, d in wf.items():
        I, J, K, why = locate_cells(grid, d.x, d.y, d.z)
        counts["total"] += d.md.size
        counts["outside_footprint"] += int((why == 1).sum())
        counts["not_in_active_cell"] += int((why == 2).sum())
        ok = why == 0
        counts["in_cells"] += int(ok.sum())
        key = (I[ok] * ny + J[ok]) * nz + K[ok]
        uk, inv = np.unique(key, return_inverse=True)
        lens = np.zeros((uk.size, ncode))
        np.add.at(lens, (inv, d.code[ok]), d.length[ok])
        for g, kk in enumerate(uk):
            path = float(lens[g].sum())
            classified = float(lens[g, 1:].sum())
            cov = classified / path if path > 0 else 0.0
            top = lens[g, 1:].max() if classified > 0 else 0.0
            winners = np.nonzero(np.isclose(lens[g, 1:], top, rtol=1e-9, atol=0.0))[0] + 1 if top > 0 else []
            if classified == 0 or cov + 1e-12 < min_coverage:
                status, code = "low_coverage", 0
            elif len(winners) > 1:
                status, code = "tie", 0
            else:
                status, code = "ok", int(winners[0])
            rows.append((w, int(kk // (ny * nz)), int((kk // nz) % ny), int(kk % nz), code, cov, path, status, lens[g]))
    n = len(rows)
    well = np.array([r[0] for r in rows], object)
    arr = lambda t, dt: np.array([r[t] for r in rows], dt) if n else np.empty(0, dt)    # noqa: E731
    I, J, K = arr(1, int), arr(2, int), arr(3, int)
    code, cov, path, status = arr(4, int), arr(5, float), arr(6, float), arr(7, object)
    cells = np.zeros((nx, ny, nz), int)
    conflicts = []
    bycell: dict = {}
    for r in rows:
        if r[4] > 0:
            bycell.setdefault((r[1], r[2], r[3]), []).append(r)
    for cell, rs in bycell.items():
        votes = {r[0]: r[4] for r in rs}
        if len(set(votes.values())) > 1:
            conflicts.append((*cell, votes))
            if conflict == "longest":
                tot = {}
                for r in rs:
                    tot[r[4]] = tot.get(r[4], 0.0) + r[8][r[4]]
                best = max(tot.values())
                win = [c for c, v in tot.items() if np.isclose(v, best, rtol=1e-9)]
                if len(win) == 1:
                    cells[cell] = win[0]
        else:
            cells[cell] = rs[0][4]
    return HardFacies(well, I, J, K, code, cov, path, status, cells, conflicts, counts,
                      {w: d.unmapped for w, d in wf.items()}, dict(skipped or {}), scheme, float(min_coverage), conflict)


def hard_facies(project, setup: FaciesSetup, grid, *, min_coverage: float | None = None, conflict: str | None = None,
                wells=None) -> HardFacies:
    """Upscaled hard facies. The coverage threshold and the conflict policy come from the arguments, else from the setup;
    if neither has them, the user has not chosen yet and this refuses."""
    min_coverage = setup.min_coverage if min_coverage is None else min_coverage
    conflict = setup.conflict if conflict is None else conflict
    if min_coverage is None or conflict is None:
        raise FaciesError("choose the minimum coverage and the conflict policy for upscaling to cells first")
    wf, skipped = classify_all(project, setup, wells)
    if not wf:
        raise FaciesError("no well could be classified: " + "; ".join(f"{w}: {r}" for w, r in skipped.items()))
    return upscale(wf, grid, setup.scheme, min_coverage=min_coverage, conflict=conflict, skipped=skipped)

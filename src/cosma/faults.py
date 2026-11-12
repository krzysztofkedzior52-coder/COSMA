"""Fault sticks.

A fault stick is a 3D polyline digitised along a fault on one seismic section (or one
depth slice). A fault is a set of sticks. COSMA v1 models vertical faults, so what the
grid builder needs from the sticks is a plan-view trace *at a depth the user chooses*;
what the user needs to be told is how far from vertical the fault really is.

Rules, same as everywhere in COSMA: nothing is guessed.

* Units and the vertical convention (depth down / elevation up) come from the user.
  Time-domain sticks are not accepted: depth-convert them first.
* Which points form a stick comes from a stick-id column or from separators in the
  file. When the file has neither, each fault is read as ONE polyline and the preview
  says so. One stick lies in one section, so its plane is the seismic section, not the
  fault: the dip is then reported as undetermined, never as a number.
* A stick that crosses the requested depth more than once, or not at all, contributes
  nothing to the trace at that depth, and is listed as skipped with the reason.

Internally: metres, x/y in the project CRS, z = depth positive down (like surfaces).
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from .units import unit_factor


class FaultError(ValueError):
    pass


# -- data -------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Fault:
    name: str
    sticks: tuple                          # tuple of (n, 3) float arrays: x, y (m), depth (m, positive down)
    source: str = ""
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise FaultError("a fault needs a name")
        if not self.sticks:
            raise FaultError(f"{self.name}: a fault needs at least one stick")
        for k, s in enumerate(self.sticks):
            a = np.asarray(s, float)
            if a.ndim != 2 or a.shape[1] != 3 or len(a) < 2:
                raise FaultError(f"{self.name}: stick {k + 1} needs at least 2 points of (x, y, depth)")
            if not np.all(np.isfinite(a)):
                raise FaultError(f"{self.name}: stick {k + 1} has non-finite coordinates")
            if np.all(a == a[0]):
                raise FaultError(f"{self.name}: stick {k + 1} has zero length (all points identical)")

    @property
    def n_sticks(self) -> int:
        return len(self.sticks)

    @property
    def n_points(self) -> int:
        return int(sum(len(s) for s in self.sticks))

    def points(self) -> np.ndarray:
        return np.vstack([np.asarray(s, float) for s in self.sticks])

    def bounds(self):
        p = self.points()
        return tuple(float(v) for v in (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max(),
                                        p[:, 2].min(), p[:, 2].max()))

    def same_geometry(self, other: "Fault") -> bool:
        return (self.n_sticks == other.n_sticks
                and all(a.shape == b.shape and np.array_equal(a, b) for a, b in zip(self.sticks, other.sticks)))

    @cached_property
    def strike_direction(self) -> np.ndarray:
        """Unit vector (x, y) along the fault in plan view, the same for every depth, so that traces taken at different
        depths run the same way. Its sign is arbitrary but fixed: x component positive (y positive if x is ~0)."""
        xy = self.points()[:, :2]
        _, _, vt = np.linalg.svd(xy - xy.mean(axis=0), full_matrices=False)
        u = vt[0].copy()
        if u[0] < -1e-12 or (abs(u[0]) <= 1e-12 and u[1] < 0):
            u = -u
        return u

    def fit(self) -> "FaultFit":
        return fit_plane(self)

    def trace_at(self, depth: float) -> "Trace":
        return trace_at(self, depth)

    def label(self) -> str:
        f = self.fit()
        dip = "dip undetermined" if f.dip_deg is None else f"dip {f.dip_deg:.0f} deg"
        return f"{self.n_sticks} stick(s), {self.n_points} points, {dip}"


# -- plane fit ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FaultFit:
    n_sticks: int
    n_points: int
    centroid: tuple
    dip_deg: float | None                  # 0 = horizontal, 90 = vertical; None = not determined
    dip_direction_deg: float | None        # azimuth (clockwise from north) of steepest descent; None for vertical/undetermined
    axis_azimuth_deg: float | None         # azimuth of the horizontal line in the plane, 0-180 (undirected)
    rms: float | None                      # RMS distance of the points from the plane, metres
    max_residual: float | None
    conditioning: float | None             # 2nd/1st singular value: ~0 means the points are nearly on a line
    reason: str = ""                       # why the plane is not determined
    warnings: tuple = ()

    def horizontal_shift(self, dz: float) -> float | None:
        """How far the fault trace moves sideways over a depth change `dz` (m), if the dip is known."""
        if self.dip_deg is None:
            return None
        if self.dip_deg >= 90.0 - 1e-9:
            return 0.0
        if self.dip_deg <= 1e-9:
            return math.inf
        return abs(dz) / math.tan(math.radians(self.dip_deg))


VERTICAL_WARN_BELOW = 80.0       # degrees: v1 treats faults as vertical; below this the error is reported
THIN_RATIO = 0.1


def fit_plane(fault: Fault) -> FaultFit:
    P = fault.points()
    n = len(P)
    c = P.mean(axis=0)
    base = dict(n_sticks=fault.n_sticks, n_points=n, centroid=tuple(float(v) for v in c))
    if fault.n_sticks < 2:
        return FaultFit(**base, dip_deg=None, dip_direction_deg=None, axis_azimuth_deg=None, rms=None, max_residual=None,
                        conditioning=None,
                        reason="one stick lies in one seismic section: the plane through it is the section, not the fault. "
                               "At least two sticks are needed for a dip.")
    U, S, Vt = np.linalg.svd(P - c, full_matrices=False)
    if S[0] <= 0 or S[1] <= 1e-6 * S[0]:
        return FaultFit(**base, dip_deg=None, dip_direction_deg=None, axis_azimuth_deg=None, rms=None, max_residual=None,
                        conditioning=0.0, reason="all points lie on one line: the plane is not determined")
    nrm = Vt[2].copy()
    if nrm[2] < 0:
        nrm = -nrm
    resid = (P - c) @ nrm
    rms = float(math.sqrt(np.mean(resid ** 2)))
    nz = float(min(max(nrm[2], 0.0), 1.0))
    dip = math.degrees(math.acos(nz))
    dipdir = None
    if dip > 1e-6 and nz > 1e-9 and dip < 90.0 - 1e-2:
        dipdir = math.degrees(math.atan2(-nrm[0] / nrm[2], -nrm[1] / nrm[2])) % 360.0
    axis = None
    if math.hypot(nrm[0], nrm[1]) > 1e-12:
        axis = math.degrees(math.atan2(-nrm[1], nrm[0])) % 180.0
    cond = float(S[1] / S[0])
    warns = []
    if cond < THIN_RATIO:
        warns.append(f"the points are nearly on a line (ratio {cond:.3f}): the dip is poorly constrained")
    return FaultFit(**base, dip_deg=dip, dip_direction_deg=dipdir, axis_azimuth_deg=axis, rms=rms,
                    max_residual=float(np.abs(resid).max()), conditioning=cond, warnings=tuple(warns))


# -- trace at a depth ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Trace:
    depth: float
    xy: np.ndarray                          # (m, 2) ordered along the fault
    used: tuple                             # stick indices that contributed
    skipped: dict                           # stick index -> reason
    warnings: tuple = ()

    @property
    def length(self) -> float:
        return 0.0 if len(self.xy) < 2 else float(np.hypot(*np.diff(self.xy, axis=0).T).sum())


FOLD_WARN = 1.5


def trace_at(fault: Fault, depth: float) -> Trace:
    d = float(depth)
    if not math.isfinite(d):
        raise FaultError("the depth of the trace must be a number")
    pts, used, skipped = [], [], {}
    for k, s in enumerate(fault.sticks):
        a = np.asarray(s, float)
        sg = a[:, 2] - d
        if np.all(sg == 0.0) or np.any((sg[:-1] == 0.0) & (sg[1:] == 0.0)):
            skipped[k] = "runs along this depth"
            continue
        lo = (sg[:-1] < 0) & (sg[1:] >= 0)
        hi = (sg[:-1] >= 0) & (sg[1:] < 0)
        events = [("seg", int(i)) for i in np.flatnonzero(lo | hi)]
        for j in np.flatnonzero(sg == 0.0):               # a vertex exactly at d that no segment above already counts
            if not ((j > 0 and sg[j - 1] < 0) or (j < len(sg) - 1 and sg[j + 1] < 0)):
                events.append(("vertex", int(j)))
        if not events:
            skipped[k] = f"does not reach this depth (its depths are {a[:, 2].min():.6g} to {a[:, 2].max():.6g} m)"
            continue
        if len(events) > 1:
            skipped[k] = f"crosses or touches this depth {len(events)} times: the fault position is ambiguous"
            continue
        kind, i = events[0]
        if kind == "vertex":
            pts.append(a[i, :2].copy())
        else:
            t = (d - a[i, 2]) / (a[i + 1, 2] - a[i, 2])
            pts.append(a[i, :2] + t * (a[i + 1, :2] - a[i, :2]))
        used.append(k)
    warns = []
    if len(pts) == 0:
        return Trace(d, np.empty((0, 2)), (), skipped, ("no stick reaches this depth",))
    xy = np.array(pts)
    if len(xy) == 1:
        return Trace(d, xy, tuple(used), skipped, ("only one stick reaches this depth: a single point, not a trace",))
    c = xy.mean(axis=0)
    _, S, Vt = np.linalg.svd(xy - c, full_matrices=False)
    if S[0] <= 1e-12:
        return Trace(d, xy[:1], tuple(used[:1]), skipped, ("all sticks cross this depth at one point: no trace",))
    u = Vt[0]
    if u @ fault.strike_direction < 0:                  # run every depth's trace the same way (the sign of Vt[0] is arbitrary)
        u = -u
    order = np.argsort((xy - c) @ u, kind="stable")
    xy = xy[order]
    used = tuple(used[i] for i in order)
    ext = float(((xy - c) @ u).max() - ((xy - c) @ u).min())
    length = float(np.hypot(*np.diff(xy, axis=0).T).sum())
    if ext > 0 and length / ext > FOLD_WARN:
        warns.append(f"the trace is strongly curved or folds back (length {length:.4g} m over {ext:.4g} m): "
                     "ordering the sticks along their main direction may be wrong - check it in the plot")
    if skipped:
        warns.append(f"{len(skipped)} stick(s) do not contribute at this depth")
    return Trace(d, xy, used, skipped, tuple(warns))


# -- reading -------------------------------------------------------------------------------------------
_ROLES = {
    "x": ("x", "easting", "east", "xcoord", "x_coord", "x_coordinate", "xutm"),
    "y": ("y", "northing", "north", "ycoord", "y_coord", "y_coordinate", "yutm"),
    "z": ("z", "depth", "tvdss", "tvd", "elevation", "elev", "z_value", "twt", "time", "msl"),
    "fault": ("fault", "fault_name", "faultname", "fault_id", "faultid", "name", "fault_no"),
    "stick": ("stick", "stick_id", "stickid", "stick_no", "stick_nr", "segment", "segment_id", "line", "line_id", "polyline"),
}
_TIME_NAMES = ("twt", "time", "ms", "owt")
_SENTINELS = (999.0, -999.0)
_COMMENT = re.compile(r"^\s*(#|--|//)")


@dataclass
class ParsedSticks:
    path: Path
    layout: str                            # "table" | "separated"
    delimiter: str
    columns: dict                          # role -> header text in the file ("" when absent)
    faults: dict                           # name -> list of (n, 3) arrays, raw (file units and sign)
    n_points: int
    stick_source: str                      # "column" | "separators" | "none"
    ignored_columns: tuple = ()
    notes: list = field(default_factory=list)
    has_fault_column: bool = False


def _norm_header(h: str) -> str:
    h = re.sub(r"[\[\(].*?[\]\)]", "", h.strip().strip("\"'"))
    return re.sub(r"[\s\-\.]+", "_", h.strip()).strip("_").lower()


def _tokens(line: str, delim: str) -> list:
    if delim == "ws":
        return line.split()
    return next(csv.reader([line], delimiter=delim, skipinitialspace=True))


def _is_number(t: str) -> bool:
    try:
        float(t)
        return True
    except ValueError:
        return False


def _is_sentinel(vals) -> bool:
    return len(vals) == 3 and all(any(abs(v - s) < 1e-9 for s in _SENTINELS) for v in vals) and len(set(vals)) == 1


def parse_sticks(path) -> ParsedSticks:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise FaultError(f"{path.name}: not a text file in UTF-8: {exc}") from None
    except OSError as exc:
        raise FaultError(f"{path.name}: cannot read: {exc}") from None
    lines = text.splitlines()
    first = next((ln for ln in lines if ln.strip() and not _COMMENT.match(ln)), None)
    if first is None:
        raise FaultError(f"{path.name}: no data found")
    delim = "ws"
    for cand in ("\t", ";", ","):
        if cand in first:
            delim = cand
            break
    toks0 = _tokens(first, delim)
    headered = not all(_is_number(t) for t in toks0)

    if headered:
        keys = [_norm_header(t) for t in toks0]
        cols, ignored = {}, []
        for role, names in _ROLES.items():
            hits = [i for i, k in enumerate(keys) if k in names]
            if len(hits) > 1:
                raise FaultError(f"{path.name}: more than one column could be '{role}': "
                                 f"{', '.join(toks0[i] for i in hits)}. Remove or rename all but one.")
            if hits:
                cols[role] = hits[0]
        for need in ("x", "y", "z"):
            if need not in cols:
                raise FaultError(f"{path.name}: no column for {need}; header is: {', '.join(toks0)}")
        used_cols = set(cols.values())
        ignored = tuple(toks0[i] for i in range(len(toks0)) if i not in used_cols)
        names_by_role = {r: toks0[i] for r, i in cols.items()}
        layout = "table"
    else:
        cols = None
        names_by_role = {}
        ignored = ()
        layout = "separated"

    faults: dict = {}                      # name -> {stick key -> list of rows}
    order_sticks: dict = {}
    cur_stick = 0
    stick_in_file = False
    seen_first = False
    n_points = 0
    sep_used = False
    for ln_no, ln in enumerate(lines, 1):
        if _COMMENT.match(ln):
            continue
        if not ln.strip():
            if seen_first:
                cur_stick += 1
                sep_used = True
            continue
        if headered and not seen_first:
            seen_first = True
            continue
        seen_first = True
        t = _tokens(ln, delim)
        if headered:
            if len(t) != len(toks0):
                raise FaultError(f"{path.name} line {ln_no}: {len(t)} values but the header has {len(toks0)} columns"
                                 + (" (names with spaces need a comma, semicolon or tab delimited file)" if delim == "ws" else ""))
            try:
                v = [float(t[cols[r]]) for r in ("x", "y", "z")]
            except ValueError:
                raise FaultError(f"{path.name} line {ln_no}: x, y or z is not a number: {ln.strip()!r}") from None
            name = t[cols["fault"]].strip() if "fault" in cols else ""
            stick = t[cols["stick"]].strip() if "stick" in cols else None
            if "fault" in cols and not name:
                raise FaultError(f"{path.name} line {ln_no}: the fault name is empty")
            if stick is not None and stick == "":
                raise FaultError(f"{path.name} line {ln_no}: the stick id is empty")
            key = stick if stick is not None else cur_stick
            if stick is not None:
                stick_in_file = True
        else:
            if len(t) != 3:
                raise FaultError(f"{path.name} line {ln_no}: {len(t)} values on a line of a file without a header. A header "
                                 "row is needed to say what extra columns are; without one only 'x y z' lines are read.")
            try:
                v = [float(x) for x in t]
            except ValueError:
                raise FaultError(f"{path.name} line {ln_no}: not a number: {ln.strip()!r}") from None
            if _is_sentinel(v):
                cur_stick += 1
                sep_used = True
                continue
            name, key = "", cur_stick
        if not all(math.isfinite(x) for x in v):
            raise FaultError(f"{path.name} line {ln_no}: x, y or z is not finite")
        faults.setdefault(name, {}).setdefault(key, []).append(v)
        n_points += 1
    if n_points == 0:
        raise FaultError(f"{path.name}: no points found")

    out = {}
    for name, sticks in faults.items():
        out[name] = [np.array(rows, float) for rows in sticks.values() if rows]
    stick_source = "column" if stick_in_file else ("separators" if sep_used else "none")
    notes = []
    if stick_source == "none":
        notes.append("The file has no stick id column and no separators (blank lines or 999 999 999 lines): "
                     "each fault is read as ONE polyline, i.e. one stick.")
    if headered and "z" in names_by_role and _norm_header(names_by_role["z"]) in _TIME_NAMES:
        notes.append(f"The z column is called '{names_by_role['z']}': time-domain sticks are not accepted, depth-convert them first.")
    if stick_in_file and sep_used:
        notes.append("Blank lines were found as well as a stick id column: the stick id column is used.")
    return ParsedSticks(path, layout, "whitespace" if delim == "ws" else {"\t": "tab"}.get(delim, delim), names_by_role,
                        out, n_points, stick_source, ignored, notes, has_fault_column=bool(headered and "fault" in cols))


def build_faults(parsed: ParsedSticks, *, xy_unit: str, z_unit: str, z_is: str, fault_name: str | None = None,
                 source: str = "") -> list:
    """Raw parsed sticks -> metres, depth positive down. `z_is`: 'depth' (positive down) or 'elevation' (positive up)."""
    if z_is not in ("depth", "elevation"):
        raise FaultError(f"z_is must be 'depth' or 'elevation', not {z_is!r}")
    fx, fz = unit_factor(xy_unit), unit_factor(z_unit)
    sign = 1.0 if z_is == "depth" else -1.0
    out = []
    for raw_name, sticks in parsed.faults.items():
        name = raw_name or (fault_name or "").strip()
        if not name:
            raise FaultError("the file has no fault names: give the fault a name")
        conv = tuple(np.column_stack([s[:, 0] * fx, s[:, 1] * fx, s[:, 2] * fz * sign]) for s in sticks)
        info = dict(xy_unit=xy_unit, z_unit=z_unit, z_is=z_is, stick_source=parsed.stick_source,
                    file=parsed.path.name)
        out.append(Fault(name, conv, source or parsed.path.stem, info))
    lower = {}
    for f in out:
        lower.setdefault(f.name.lower(), []).append(f.name)
    clash = [v for v in lower.values() if len(v) > 1]
    if clash:
        raise FaultError(f"fault names that differ only by case: {clash}")
    return out


def read_fault_sticks(path, *, xy_unit, z_unit, z_is, fault_name=None) -> list:
    return build_faults(parse_sticks(path), xy_unit=xy_unit, z_unit=z_unit, z_is=z_is, fault_name=fault_name)


def export_sticks(faults, path) -> int:
    """Write sticks as CSV: x, y (m), depth (m, positive down), fault, stick. Returns the number of points."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["x", "y", "depth", "fault", "stick"])
    n = 0
    for f in faults:
        for k, s in enumerate(f.sticks, 1):
            for x, y, z in np.asarray(s, float):
                w.writerow([float(x), float(y), float(z), f.name, k])
                n += 1
    Path(path).write_text(buf.getvalue(), encoding="utf-8")
    return n

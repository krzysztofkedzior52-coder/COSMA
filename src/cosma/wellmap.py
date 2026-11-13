"""Plan-view picture of a project: where the wells are and what else is there (headless data; the dialog draws it).

Everything is in map coordinates, metres. A well is drawn at its wellhead with its path in plan view (a vertical well
is a point) and the bottom-hole position when a deviation survey exists. A well without a header has no position and
is listed as not drawn; nothing is placed by guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid.frame import GridFrame
from .project import Project


@dataclass
class WellPoint:
    name: str
    x: float
    y: float
    kb: float
    td: float | None
    path: np.ndarray | None          # (n, 2) plan-view path wellhead -> bottom, or None without a survey
    bottom: tuple | None             # (x, y) at the end of the survey
    deviated: bool
    n_tops: int
    n_curves: int
    has_core: bool
    is_type_well: bool

    @property
    def data(self) -> str:
        return "core and logs" if self.has_core and self.n_curves else "core" if self.has_core else "logs" if self.n_curves else "no logs or core"

    def describe(self) -> str:
        s = f"{self.name}{' (type well)' if self.is_type_well else ''}: x {self.x:,.1f}, y {self.y:,.1f}, KB {self.kb:g} m"
        if self.td is not None:
            s += f", TD {self.td:g} m MD"
        s += f"; {self.n_tops} tops, {self.n_curves} log curves, {'core' if self.has_core else 'no core'}"
        s += "; deviated, bottom hole %.0f m from the wellhead" % float(np.hypot(self.bottom[0] - self.x, self.bottom[1] - self.y)) \
            if self.deviated and self.bottom else ("; vertical" if self.path is not None else "; no deviation survey")
        return s


@dataclass
class MapData:
    wells: list = field(default_factory=list)
    not_drawn: dict = field(default_factory=dict)          # well -> reason
    fault_sticks: dict = field(default_factory=dict)       # fault -> list of (n, 2) arrays
    fault_traces: dict = field(default_factory=dict)       # fault -> (m, 2) trace at the chosen depth
    fault_depths: dict = field(default_factory=dict)
    boundary: np.ndarray | None = None
    grid_outline: np.ndarray | None = None
    surface: dict | None = None                           # {"label", "X", "Y", "Z"} node arrays, decimated
    notes: list = field(default_factory=list)

    def extent(self):
        """(xmin, xmax, ymin, ymax) of everything drawn except the background surface, or None."""
        pts = []
        for w in self.wells:
            pts.append([(w.x, w.y)] + ([w.bottom] if w.bottom else []))
        for arrs in self.fault_sticks.values():
            pts += [a[:, :2] for a in arrs]
        for a in (self.boundary, self.grid_outline):
            if a is not None:
                pts.append(a)
        if not pts:
            return None
        P = np.vstack([np.asarray(p, float).reshape(-1, 2) for p in pts])
        return float(P[:, 0].min()), float(P[:, 0].max()), float(P[:, 1].min()), float(P[:, 1].max())


def well_points(project: Project, step: float = 10.0):
    pts, not_drawn = [], {}
    for n, rec in project.wells.items():
        h = rec.header
        if h is None:
            not_drawn[n] = "no header: its position is unknown"
            continue
        path, bottom, dev = None, None, False
        tr = rec.trajectory
        if tr is not None:
            lo, hi = tr.md_range
            md = np.unique(np.append(np.arange(lo, hi, step), hi))
            x, y, _ = tr.at_md(md)
            path = np.column_stack([x, y])
            bottom = (float(x[-1]), float(y[-1]))
            dev = bool(np.hypot(x[-1] - x[0], y[-1] - y[0]) > 1.0)
        pts.append(WellPoint(n, float(h["x"]), float(h["y"]), float(h["kb"]), h.get("td"), path, bottom, dev, len(rec.tops),
                             sum(len(l.curves) for l in rec.logs.values()), bool(rec.core), n == project.type_well))
    return pts, not_drawn


def grid_outline(project: Project):
    g = project.grid_def
    if g is None:
        return None
    frame = GridFrame(g.origin_x, g.origin_y, g.azimuth_deg)
    lx, ly = g.nx * g.dx, g.ny * g.dy
    X, Y = frame.to_map(np.array([0, lx, lx, 0, 0.0]), np.array([0, 0, ly, ly, 0.0]))
    return np.column_stack([X, Y])


def surface_background(project: Project, horizon_uid: str, max_nodes: int = 120):
    sf = project.surfaces.get(horizon_uid, {})
    lab = project.primary.get(horizon_uid) or next(iter(sf), None)
    if lab is None:
        return None
    s = sf[lab]
    X, Y = s.node_xy()
    k = max(1, int(np.ceil(max(s.ncol, s.nrow) / max_nodes)))
    return dict(label=lab, X=X[::k, ::k], Y=Y[::k, ::k], Z=np.asarray(s.values)[::k, ::k])


def collect(project: Project, *, horizon_uid: str | None = None) -> MapData:
    d = MapData()
    d.wells, d.not_drawn = well_points(project)
    for n, f in project.faults.items():
        d.fault_sticks[n] = [np.asarray(s, float)[:, :2] for s in f.sticks]
        depth = project.fault_depths.get(n)
        if depth is not None:
            tr = f.trace_at(depth)
            if len(tr.xy) >= 2:
                d.fault_traces[n] = np.asarray(tr.xy, float)
                d.fault_depths[n] = float(depth)
    if project.boundary is not None:
        d.boundary = np.asarray(project.boundary.exterior.coords, float)
    d.grid_outline = grid_outline(project)
    if horizon_uid is not None:
        d.surface = surface_background(project, horizon_uid)
        if d.surface is None:
            d.notes.append("that horizon has no surface yet")
    for w, why in d.not_drawn.items():
        d.notes.append(f"{w}: not drawn ({why})")
    return d

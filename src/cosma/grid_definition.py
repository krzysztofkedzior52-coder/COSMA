"""The user's grid definition and the project-level grid build.

Everything the build needs is explicit: where the grid sits and how big its cells are, one layering per
interval, erosional or onlap for every unconformity, the minimum cell thickness, and (through the faults) a
trace depth per fault and confirmed ages. A missing item is a refusal that names it, never a default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .grid.boundary import grid_spec_from_cells
from .grid.boundary_options import (BoundaryOptionError, BoundaryOptions, aqudims_text, aquifer_text, boundary_layout,
                                    combination_warnings, connection_runs, edge_columns, multpv_cube, multpv_text)
from .grid.builder import DEFAULT_MIN_THICKNESS, BuildReport, Layering, build_grid, sample_horizons
from .grid.frame import GridFrame


class GridDefinitionError(ValueError):
    pass


@dataclass
class GridDefinition:
    origin_x: float
    origin_y: float
    azimuth_deg: float
    nx: int
    ny: int
    dx: float
    dy: float
    min_thickness: float = DEFAULT_MIN_THICKNESS
    layerings: dict = field(default_factory=dict)          # interval uid -> Layering
    unconformity: dict = field(default_factory=dict)       # interval uid -> "erosional" | "onlap"
    boundary: BoundaryOptions = field(default_factory=BoundaryOptions)

    def __post_init__(self):
        for n, v in (("origin x", self.origin_x), ("origin y", self.origin_y), ("azimuth", self.azimuth_deg)):
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise GridDefinitionError(f"{n} must be a number")
        for n, v in (("nx", self.nx), ("ny", self.ny)):
            if int(v) != v or v < 1:
                raise GridDefinitionError(f"{n} must be a whole number >= 1")
        self.nx, self.ny = int(self.nx), int(self.ny)
        for n, v in (("dx", self.dx), ("dy", self.dy)):
            if not isinstance(v, (int, float)) or not (math.isfinite(v) and v > 0):
                raise GridDefinitionError(f"{n} must be a positive cell size in metres")
        if not (isinstance(self.min_thickness, (int, float)) and math.isfinite(self.min_thickness)
                and self.min_thickness >= 0):
            raise GridDefinitionError("the minimum cell thickness must be >= 0")
        for uid, v in self.unconformity.items():
            if v not in ("erosional", "onlap"):
                raise GridDefinitionError(f"unconformity type {v!r} must be 'erosional' or 'onlap'")

    def spec(self, polygon=None):
        """The areal grid. With a boundary polygon, columns outside it (and outside any buffer) are inactive."""
        sp = grid_spec_from_cells(self.origin_x, self.origin_y, self.azimuth_deg, self.nx, self.ny, self.dx, self.dy)
        self.region = None
        self.layout_warnings = []
        if polygon is not None:
            sp.active_xy, self.region, self.layout_warnings = boundary_layout(sp, polygon, self.boundary)
            sp.boundary = polygon
        return sp

    def to_state(self) -> dict:
        return dict(origin_x=self.origin_x, origin_y=self.origin_y, azimuth_deg=self.azimuth_deg, nx=self.nx,
                    ny=self.ny, dx=self.dx, dy=self.dy, min_thickness=self.min_thickness,
                    layerings={u: dict(mode=l.mode, n=l.n, dz=l.dz) for u, l in self.layerings.items()},
                    unconformity=dict(self.unconformity), boundary=self.boundary.to_state())

    @classmethod
    def from_state(cls, st: dict) -> "GridDefinition":
        try:
            lay = {u: Layering(d["mode"], d.get("n"), d.get("dz")) for u, d in st["layerings"].items()}
            return cls(st["origin_x"], st["origin_y"], st["azimuth_deg"], st["nx"], st["ny"], st["dx"], st["dy"],
                       st["min_thickness"], lay, dict(st["unconformity"]),
                       BoundaryOptions.from_state(st["boundary"]) if st.get("boundary") else BoundaryOptions())
        except (KeyError, TypeError, ValueError) as exc:
            raise GridDefinitionError(f"bad grid definition: {exc}") from None


@dataclass
class ModelBuild:
    grid: object
    report: BuildReport
    fault_reports: list
    age_order: list
    interval_names: list
    options: BoundaryOptions = field(default_factory=BoundaryOptions)
    region: object = None            # (nx, ny) 1 = inside the polygon, 2 = buffer, 0 = outside
    notes: list = field(default_factory=list)
    fault_choices: dict = field(default_factory=dict)

    @property
    def active_columns(self):
        return self.grid.actnum.any(axis=2)

    def edge_count(self) -> int:
        return int(edge_columns(self.active_columns, self.options.pv.rings).sum()) if self.options.pv else 0

    def summary(self) -> str:
        g, r = self.grid, self.report
        lines = [f"Grid {g.nx} x {g.ny} x {g.nz} = {g.nx * g.ny * g.nz:,} cells, {int(g.actnum.sum()):,} active."]
        for name, n in zip(self.interval_names, r.layers_per_interval):
            lines.append(f"  {name}: {n} layer(s)")
        if r.thin_cells:
            lines.append(f"{r.thin_cells:,} cells thinner than the minimum were made inactive "
                         f"({r.lost_volume:,.0f} m3 of {r.total_volume:,.0f} m3).")
        if any(r.clipped_corners):
            lines.append("Corners moved by unconformity clipping (per horizon): " + ", ".join(map(str, r.clipped_corners)))
        for fr in self.fault_reports:
            if fr.has_throw:
                worst = max((abs(t) for t in fr.median_throw if np.isfinite(t)), default=float("nan"))
                lines.append(f"Fault {fr.name} (trace at {fr.depth:g} m): {fr.n_faces} faces, median throw up to "
                             f"{worst:.1f} m; faces lie up to {fr.snap_distance_max:.0f} m from the true trace "
                             f"(mean {fr.snap_distance_mean:.0f} m).")
            else:
                lines.append(f"Fault {fr.name}: throw negligible, transmissibility only "
                             f"({fr.n_faces} faces would be involved).")
        if self.age_order and len(self.age_order) > 1:
            lines.append("Fault ages, oldest first: " + ", ".join(self.age_order))
        lines += self.boundary_lines()
        lines += [f"Warning: {w}" for w in r.warnings]
        for fr in self.fault_reports:
            lines += [f"Warning ({fr.name}): {w}" for w in fr.warnings]
        return "\n".join(lines)


def _boundary_lines(self) -> list:
    o, g = self.options, self.grid
    act = self.active_columns
    out = []
    if self.region is not None:
        n_in = int(((self.region == 1) & act).sum())
        n_buf = int(((self.region == 2) & act).sum())
        out.append(f"Boundary: {n_in} columns inside the polygon" + (f", {n_buf} in the buffer zone" if o.buffer else "")
                   + f"; {int((~act).sum())} columns inactive.")
    out.append("Boundary treatment: " + ("closed (no-flow edge)" if o.closed else
               ", ".join(x for x in (f"buffer {o.buffer:g} m" if o.buffer else None,
                                     f"pore-volume multiplier {o.pv.factor:g} on {self.edge_count()} edge columns" if o.pv else None,
                                     f"{o.aquifer.kind.replace('_', '-')} aquifer on {', '.join(o.aquifer.faces)} "
                                     f"({len(connection_runs(act, o.aquifer.faces))} connection records)" if o.aquifer else None) if x)) + ".")
    out += [f"Warning: {w}" for w in self.notes]
    return out


ModelBuild.boundary_lines = _boundary_lines


def write_model(model: "ModelBuild", path, *, props: dict | None = None, fault_props: dict | None = None) -> list:
    """Write the GRDECL and, when chosen, the pore-volume multiplier and aquifer include files next to it.
    ``props`` {KEYWORD: (nx, ny, nz) array} are written into the GRDECL; ``fault_props`` are cell properties used only to compute
    the fault transmissibility (for example VSHALE) and are NOT written. Returns the paths written."""
    g = model.grid
    orig = dict(g.props)
    try:
        g.props.update({k: np.asarray(v) for k, v in (props or {}).items()})
        for k, v in (fault_props or {}).items():
            g.props.setdefault(k, np.asarray(v))
        return _write_model(model, path, set((fault_props or {}).keys()) - set((props or {}).keys()))
    finally:
        g.props.clear()
        g.props.update(orig)


def _write_model(model: "ModelBuild", path, fault_only: set) -> list:
    from pathlib import Path
    from .io.grdecl import write_grdecl
    path = Path(path)
    g, o = model.grid, model.options
    faults_inc = None
    if model.fault_reports:                 # built first: a refusal must not leave half an export behind
        from .fault_transmissibility import PendingProperties, TransmissibilityError, faults_text
        try:
            faults_inc = faults_text(model, model.fault_choices)
        except PendingProperties as exc:       # not an error: the geometry is complete, the faults wait for properties
            model.notes = [n for n in model.notes if not n.startswith("fault transmissibility include")]
            model.notes.append("fault transmissibility include NOT written (no MULTFLT for any fault yet): " + str(exc))
        except TransmissibilityError as exc:
            raise GridDefinitionError(str(exc)) from None
    for k in fault_only:
        g.props.pop(k, None)
    saved = dict(g.props)
    try:
        if model.region is not None and o.buffer is not None:
            reg = np.where(model.region > 0, model.region, 1)
            g.props["FIPNUM"] = np.repeat(reg[:, :, None], g.nz, axis=2).astype(int)
        extra = sorted(k for k in g.props if k != "FIPNUM")
        write_grdecl(g, path, comment=("COSMA grid geometry" + (" and properties " + ", ".join(extra) if extra else " (no properties yet)"))
                     + ("\nFIPNUM: 1 = inside the boundary polygon, 2 = buffer zone" if "FIPNUM" in g.props else ""))
    finally:
        g.props.clear()
        g.props.update(saved)
    out = [path]
    if faults_inc is not None:
        f = path.with_name(path.stem + "_faults.inc")
        f.write_text(faults_inc, encoding="ascii")
        out.append(f)
    if o.pv is not None:
        f = path.with_name(path.stem + "_grid_boundary.inc")
        f.write_text(multpv_text(multpv_cube(model.active_columns, o.pv, g.nz), o.pv), encoding="ascii")
        out.append(f)
    if o.aquifer is not None:
        f = path.with_name(path.stem + "_solution_boundary.inc")
        f.write_text(aquifer_text(o.aquifer, model.active_columns, g.nz), encoding="ascii")
        out.append(f)
        f = path.with_name(path.stem + "_runspec_boundary.inc")
        f.write_text(aqudims_text(o.aquifer, model.active_columns), encoding="ascii")
        out.append(f)
    return out


def required_items(project, gd: GridDefinition | None) -> list:
    """Everything still missing before a build; empty when the build can run."""
    out = []
    if project.chart is None:
        return ["define the stratigraphic chart"]
    if gd is None:
        return ["define the grid (origin, azimuth, cells)"]
    for h in project.chart.horizons():
        lab = project.primary.get(h.uid)
        if lab is None or lab not in project.surfaces.get(h.uid, {}):
            out.append(f"attach a depth surface to horizon {h.name!r} and make it primary")
    for iv in project.chart.intervals:
        if iv.uid not in gd.layerings:
            out.append(f"choose the layering of interval {iv.name!r}")
    for i, iv in enumerate(project.chart.intervals):
        if i > 0 and iv.contact_above == "unconformity" and iv.uid not in gd.unconformity:
            out.append(f"say whether the unconformity at the top of {iv.name!r} is erosional or onlap")
    if gd.boundary.buffer is not None and getattr(project, "boundary", None) is None:
        out.append("import a boundary polygon (the buffer zone is measured from it)")
    if project.faults:
        for n in project.faults:
            if n not in project.fault_transmissibility:
                out.append(f"choose the transmissibility of fault {n!r} (open, fixed or calculated)")
        lacking = [n for n in project.faults if n not in project.fault_depths]
        if lacking:
            out.append("choose a trace depth for fault(s): " + ", ".join(lacking))
        elif len(project.faults) > 1 and project.fault_reach is None:
            out.append("check the fault interactions")
        else:
            from .fault_interactions import unconfirmed
            for i in unconfirmed(project.fault_interactions):
                out.append("confirm: " + i.describe())
    return out


def build_model(project, gd: GridDefinition) -> ModelBuild:
    missing = required_items(project, gd)
    if missing:
        raise GridDefinitionError("cannot build yet:\n- " + "\n- ".join(missing))
    chart = project.chart
    horizons = chart.horizons()
    surfaces = [project.surfaces[h.uid][project.primary[h.uid]] for h in horizons]
    contacts = []
    for i, iv in enumerate(chart.intervals):
        c = iv.contact_above
        contacts.append(gd.unconformity[iv.uid] if (i > 0 and c == "unconformity") else "conformable")
    layerings = [gd.layerings[iv.uid] for iv in chart.intervals]
    spec = gd.spec(getattr(project, 'boundary', None))
    reports, order = [], []
    if project.faults:
        from .grid.fault_sampling import corner_depths
        order = project.fault_age_order()
        # ages decide who claims shared corners only when the user's answers actually order faults;
        # otherwise the order is arbitrary and the nearest fault is used (with a warning)
        decided = any(i.answer in (i.a, i.b) for i in project.fault_interactions)
        deps, reports = corner_depths(spec, surfaces, dict(project.faults), dict(project.fault_depths),
                                      age_order=order if decided else None)
    else:
        deps = sample_horizons(spec, surfaces)
    grid, rep = build_grid(spec, deps, contacts, layerings, gd.min_thickness)
    choices = dict(project.fault_transmissibility)
    project.fault_throw_seen = {fr.name: fr.has_throw for fr in reports}      # lets the dialogs offer only what is possible
    for fr in reports:
        c = choices.get(fr.name)
        if c is not None and c.mode == "calculated" and not fr.has_throw:
            raise GridDefinitionError(f"fault {fr.name!r} has negligible throw, so its transmissibility cannot be "
                                      "calculated; choose open or fixed")
    mb = ModelBuild(grid, rep, reports, order, [iv.name for iv in chart.intervals], gd.boundary, gd.region)
    mb.fault_choices = choices
    mb.notes = list(gd.layout_warnings) + combination_warnings(gd.boundary)
    if gd.region is not None and gd.boundary.buffer is not None:
        lost = int(((gd.region == 2) & ~mb.active_columns).sum())
        if lost:
            mb.notes.append(f"{lost} buffer columns are inactive because a horizon is undefined there")
    if gd.boundary.aquifer is not None and not connection_runs(mb.active_columns, gd.boundary.aquifer.faces):
        mb.notes.append("the aquifer has no outer faces on the chosen sides")
    return mb


# -- suggestions --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class OrientationSuggestion:
    label: str
    azimuth_deg: float           # azimuth (clockwise from north) of the grid J axis
    cells: int | None            # cells needed to cover the footprint, None when no footprint is known
    cost_ratio: float | None     # cells relative to a north-aligned grid


def _footprint_points(project):
    """Valid node positions of the top surface (map coordinates), subsampled."""
    if project.chart is None:
        return None
    h = project.chart.horizons()[0]
    lab = project.primary.get(h.uid)
    s = project.surfaces.get(h.uid, {}).get(lab) if lab else None
    if s is None:
        return None
    X, Y = s.node_xy()
    ok = np.isfinite(s.values)
    if not ok.any():
        return None
    P = np.column_stack([X[ok], Y[ok]])
    if len(P) > 20000:
        P = P[:: len(P) // 20000 + 1]
    return P


def footprint_extent(points, azimuth_deg: float):
    """Bounding rectangle of ``points`` in a frame rotated by ``azimuth_deg``: (xmin, xmax, ymin, ymax) local."""
    fr = GridFrame(0.0, 0.0, azimuth_deg)
    x, y = fr.to_local(points[:, 0], points[:, 1])
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def suggest_extent(project, azimuth_deg: float, dx: float, dy: float):
    """Origin and cell counts of a grid at this azimuth and cell size that covers the primary top surface.
    Returns (origin_x, origin_y, nx, ny) or None when there is no surface. A suggestion to edit, not a decision."""
    P = _footprint_points(project)
    if P is None:
        return None
    x0, x1, y0, y1 = footprint_extent(P, azimuth_deg)
    nx, ny = max(1, math.ceil((x1 - x0) / dx - 1e-9)), max(1, math.ceil((y1 - y0) / dy - 1e-9))
    fr = GridFrame(0.0, 0.0, azimuth_deg)
    ox, oy = fr.to_map(x0, y0)
    return float(ox), float(oy), nx, ny


def _axial_mean(angles_deg, weights):
    a = np.radians(2 * np.asarray(angles_deg, float))
    w = np.asarray(weights, float)
    return (math.degrees(math.atan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())) / 2) % 180.0


def structural_dip(project):
    """(dip_deg, down-dip azimuth) of a plane fitted to the primary top surface, or None."""
    P = _footprint_points(project)
    if P is None or len(P) < 3:
        return None
    h = project.chart.horizons()[0]
    s = project.surfaces[h.uid][project.primary[h.uid]]
    X, Y = s.node_xy()
    ok = np.isfinite(s.values)
    sub = np.arange(ok.sum())
    sub = sub[:: len(sub) // 20000 + 1]
    A = np.column_stack([np.ones(len(sub)), X[ok][sub], Y[ok][sub]])
    coef, *_ = np.linalg.lstsq(A, s.values[ok][sub], rcond=None)
    gx, gy = coef[1], coef[2]
    dip = math.degrees(math.atan(math.hypot(gx, gy)))
    return dip, math.degrees(math.atan2(gx, gy)) % 360.0


def orientation_suggestions(project, dx: float, dy: float, min_dip_deg: float = 0.5) -> list:
    """Candidate J-axis azimuths: along each fault's strike, along the length-weighted mean strike, and with the
    I axis along the structural dip direction. Each carries the cell count needed to cover the top surface
    and its cost relative to a north-aligned grid."""
    P = _footprint_points(project)
    cands = []
    strikes = []
    for n, f in project.faults.items():
        fit = f.fit()
        if fit.axis_azimuth_deg is not None:
            span = float(np.hypot(*np.ptp(f.points()[:, :2], axis=0)))
            strikes.append((n, fit.axis_azimuth_deg, span))
    for n, a, _ in strikes:
        cands.append((f"along the strike of fault {n} ({a:.0f} deg)", a % 360.0))
    if len(strikes) > 1:
        m = _axial_mean([a for _, a, _ in strikes], [w for *_, w in strikes])
        cands.append((f"along the mean fault strike ({m:.0f} deg)", m))
    sd = structural_dip(project)
    if sd is not None and sd[0] >= min_dip_deg:
        cands.append((f"I axis down the structural dip (dip {sd[0]:.1f} deg towards {sd[1]:.0f} deg)", (sd[1] - 90.0) % 360.0))

    def cells(az):
        if P is None:
            return None
        x0, x1, y0, y1 = footprint_extent(P, az)
        return max(1, math.ceil((x1 - x0) / dx)) * max(1, math.ceil((y1 - y0) / dy))

    base = cells(0.0)
    out = [OrientationSuggestion("north-aligned (no rotation)", 0.0, base, 1.0 if base else None)]
    for label, az in cands:
        c = cells(az)
        out.append(OrientationSuggestion(label, float(az), c, (c / base) if (c and base) else None))
    return out

"""Import confirmation layer.

Every importer has an `inspect_*` step that reads the file cheaply, WITHOUT interpreting
units, and returns an `ImportPreview`: what was found in the file, plus the parameters
the import needs (units, depth type, CRS ...). The user must confirm that preview before
the real reader runs. This module is headless; the Qt dialog (`cosma.gui.import_dialog`)
is a thin view over `ImportPreview`, so the same rules apply in scripts and tests:

    values = confirm(preview)          # dict of param values, or None = cancelled
    import_file(kind, path, confirm)   # inspect -> confirm -> validate -> read

A parameter the file cannot supply (a unit, a depth type) has no value and must be
chosen; it is never pre-filled with a guess. A value taken from the file is shown as
such; a documented default is shown as a default so it still gets looked at.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import crs_catalog as _crs_cat
from .io import polygons as _poly
from .project import Action, ImportResult, Project, WellLink, check_action, suggest_links
from .surfaces import SurfaceError, _sniff, read_surface
from .wells import io as _wio

UNIT_CHOICES = ("m", "km", "ft_intl", "ft_us")
MAP_UNITS = ("fraction", "percent", "mD", "m", "other")


class ImportCancelled(Exception):
    """The user closed the confirmation without accepting."""


class ImportRejected(ValueError):
    """Confirmed values are incomplete or invalid."""


@dataclass
class Param:
    key: str
    label: str
    kind: str = "choice"                 # "choice" | "text" | "number" | "crs"
    choices: tuple = ()
    value: Any = None                    # None = must be supplied
    source: str = "required"             # "file" | "default" | "required"
    required: bool = True
    required_if: tuple | None = None     # (other_key, other_value): only needed/shown then
    help: str = ""
    choice_labels: dict = field(default_factory=dict)   # value -> text shown instead of the value
    suggestions: list = field(default_factory=list)     # kind "crs": [crs_catalog.Suggestion] shown first in the list

    def active(self, values: dict) -> bool:
        return self.required_if is None or values.get(self.required_if[0]) == self.required_if[1]


@dataclass
class ImportPreview:
    kind: str
    path: Path
    found: list = field(default_factory=list)      # [(label, text)] read from the file
    params: list = field(default_factory=list)     # [Param]
    warnings: list = field(default_factory=list)
    file_wells: list = field(default_factory=list)   # well names the file refers to (well data only)
    existing: list = field(default_factory=list)     # wells already in the project
    links: dict = field(default_factory=dict)        # file well -> (WellLink suggested, note)
    conflict_finder: Callable | None = None          # (clean params incl. well_links) -> [Conflict]
    project: Any = None
    suggester: Callable | None = None                # (values) -> Placement | None   (surfaces)
    advisor: Callable | None = None                  # (clean params) -> [str] extra warnings
    top_names: list = field(default_factory=list)    # distinct pick names in the file (chart present)
    top_choices: dict = field(default_factory=dict)  # horizon uid | "ignore" -> label
    top_suggestions: dict = field(default_factory=dict)   # pick name -> (uid | None, note)
    crs_hints: Callable | None = None                # (values) -> {param key: ([Suggestion], note)}  (polygon CRS lists)

    def initial_values(self) -> dict:
        v = {p.key: p.value for p in self.params}
        if self.file_wells:
            v["well_links"] = {fw: l for fw, (l, _) in self.links.items()}
        if self.file_wells or self.conflict_finder is not None:
            v["conflict_actions"] = {}
        if self.top_names:
            v["top_map"] = {n: u for n, (u, _) in self.top_suggestions.items() if u}
        return v


def _unit(key, label, value=None, source="required", help=""):
    return Param(key, label, "choice", UNIT_CHOICES, value, source if value is not None else "required", help=help)


def _check_top_map(preview, raw, problems) -> dict:
    out = {}
    for n in preview.top_names:
        v = raw.get(n)
        if v is None:
            problems.append(f"top {n!r}: map it to a horizon of the chart or choose 'ignore'")
        elif v not in preview.top_choices:
            problems.append(f"top {n!r}: {v!r} is not a horizon of the chart")
        else:
            out[n] = v
    extra = sorted(set(raw) - set(preview.top_names))
    if extra:
        problems.append(f"mapping given for picks not in the file: {extra}")
    return out


def _as_link(v):
    if isinstance(v, WellLink):
        return v
    if isinstance(v, dict):
        return WellLink(v.get("mode"), (v.get("name") or "").strip())
    mode, name = v
    return WellLink(mode, (name or "").strip())


def _check_links(preview, raw, problems) -> dict:
    out, used = {}, {}
    for fw in preview.file_wells:
        if fw not in raw:
            problems.append(f"well {fw!r}: connect it to an existing well or import it as a new well")
            continue
        l = _as_link(raw[fw])
        if l.mode == "existing":
            if l.name not in preview.existing:
                problems.append(f"well {fw!r}: {l.name!r} is not a well in the project")
                continue
        elif l.mode == "new":
            if not l.name:
                problems.append(f"well {fw!r}: the new well needs a name")
                continue
            if l.name in preview.existing:
                problems.append(f"well {fw!r}: a well called {l.name!r} already exists - connect to it or rename the new one")
                continue
        else:
            problems.append(f"well {fw!r}: choose 'existing' or 'new'")
            continue
        if l.name in used:
            problems.append(f"wells {used[l.name]!r} and {fw!r} would both become {l.name!r}")
            continue
        used[l.name] = fw
        out[fw] = l
    extra = sorted(set(raw) - set(preview.file_wells))
    if extra:
        problems.append(f"links given for wells not in the file: {extra}")
    return out


def finalize(preview: ImportPreview, values: dict, *, conflicts: bool = True) -> dict:
    """Validate the user's values against the preview; return the clean parameter dict."""
    out, problems = {}, []
    if preview.file_wells:
        out["well_links"] = _check_links(preview, values.get("well_links", preview.initial_values()["well_links"]), problems)
    if preview.top_names:
        out["top_map"] = _check_top_map(preview, {**preview.initial_values()["top_map"], **values.get("top_map", {})}, problems)
    effective = {**preview.initial_values(), **values}
    known = ({p.key for p in preview.params} | ({"well_links"} if preview.file_wells else set())
             | ({"conflict_actions"} if preview.file_wells or preview.conflict_finder else set())
             | ({"top_map"} if preview.top_names else set()))
    for extra in sorted(set(values) - known):
        problems.append(f"unknown parameter {extra!r}")
    for p in preview.params:
        if not p.active(effective):
            continue
        v = values[p.key] if p.key in values else p.value     # omitted = the prefilled (file/default) value
        if isinstance(v, str):
            v = v.strip() or None
        if v is None:
            if p.required:
                problems.append(f"{p.label} must be chosen")
            continue
        if p.kind == "crs":
            try:
                v = _crs_cat.normalise(v)
            except ValueError as exc:
                problems.append(f"{p.label}: {exc}")
                continue
            if v is None:
                if p.required:
                    problems.append(f"{p.label} must be chosen")
                continue
        elif p.kind == "choice" and v not in p.choices:
            problems.append(f"{p.label}: {v!r} is not one of {list(p.choices)}")
        elif p.kind == "number":
            try:
                v = float(v)
            except (TypeError, ValueError):
                problems.append(f"{p.label}: {v!r} is not a number")
                continue
        out[p.key] = v
    if conflicts and not problems and preview.conflict_finder is not None:
        try:
            conflicts = preview.conflict_finder(dict(out))
        except (ValueError, KeyError) as exc:        # unreadable data is reported, not hidden
            problems.append(str(exc))
            conflicts = []
        given = values.get("conflict_actions", {})
        acts = {}
        for c in conflicts:
            a = given.get(c.key)
            if isinstance(a, (tuple, dict)):
                a = Action(*a) if isinstance(a, tuple) else Action(a.get("mode"), a.get("suffix", ""))
            problem = check_action(c, a)
            if problem:
                problems.append(problem)
            else:
                acts[c.key] = a
        extra = sorted(set(given) - {c.key for c in conflicts})
        if extra:
            problems.append(f"actions given for things that are not in conflict: {extra}")
        out["conflict_actions"] = acts
    if problems:
        raise ImportRejected("; ".join(problems))
    return out


def conflicts_for(preview: ImportPreview, values: dict):
    """Conflicts the current (valid) parameters and well links would cause, or None while the
    parameters are still incomplete. Used by the dialog to show them as soon as they are known."""
    if preview.conflict_finder is None:
        return []
    try:
        return preview.conflict_finder(finalize(preview, values, conflicts=False))
    except (ImportRejected, ValueError, KeyError):
        return None


def placement_for(preview: ImportPreview, values: dict):
    """Placement suggestions for a surface once its units are chosen, else None."""
    if preview.suggester is None or not all(values.get(k) for k in ("xy_unit", "z_unit")):
        return None
    return preview.suggester(values)


def advice_for(preview: ImportPreview, values: dict) -> list:
    """Extra warnings (e.g. picks out of chart order) for the current, valid parameters."""
    if preview.advisor is None:
        return []
    try:
        return preview.advisor(finalize(preview, values, conflicts=False))
    except (ImportRejected, ValueError, KeyError):
        return []


def scripted(values: dict) -> Callable:
    """A `confirm` callable for scripts and tests: returns fixed values. Still validated by
    `import_file`, so a missing unit is an error, not a default."""
    return lambda preview: dict(values)


# -- inspectors ---------------------------------------------------------------------------------
def _fmt(v):
    return f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v)


def inspect_surface(path) -> ImportPreview:
    import xtgeo

    path = Path(path)
    try:
        fmt = _sniff(path)
        s = xtgeo.surface_from_file(str(path), fformat=fmt)
    except SurfaceError:
        raise
    except Exception as exc:
        raise SurfaceError(f"{path.name}: cannot read surface: {exc}") from exc
    v = np.ma.filled(s.values.astype(float), np.nan)
    ok = np.isfinite(v)
    found = [
        ("Format", fmt),
        ("Nodes (columns x rows)", f"{s.ncol} x {s.nrow}"),
        ("Origin (x, y) as in file", f"{_fmt(s.xori)}, {_fmt(s.yori)}"),
        ("Increment (x, y) as in file", f"{_fmt(s.xinc)}, {_fmt(s.yinc)}"),
        ("Rotation (deg, counter-clockwise)", _fmt(s.rotation)),
        ("Y axis", "normal" if s.yflip == 1 else "REVERSED (will be refused)"),
        ("Defined nodes", f"{int(ok.sum())} of {ok.size}"),
        ("Value range as in file", f"{_fmt(np.nanmin(v))} to {_fmt(np.nanmax(v))}" if ok.any() else "none"),
    ]
    warn = []
    if not ok.any():
        warn.append("The surface has no defined nodes.")
    if s.yflip != 1:
        warn.append("Reversed y axis: this import will be refused.")
    return ImportPreview("surface", path, found, [
        _unit("xy_unit", "Horizontal unit of the file"),
        _unit("z_unit", "Vertical unit of the file"),
        Param("positive", "Values are", "choice", ("down", "up"), "down", "default",
              help="'down' = depth, positive downwards; 'up' = elevation, positive upwards"),
    ], warn)


def inspect_polygon(path) -> ImportPreview:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in _poly._TEXT_EXT:
        parsed = _poly._parse_text(path)
    elif ext in _poly._GEOJSON_EXT:
        parsed = _poly._parse_geojson(path)
    elif ext in _poly._KML_EXT:
        parsed = _poly._parse_kml(path)
    elif ext in _poly._SHP_EXT:
        parsed = _poly._parse_shapefile(path)
    else:
        raise _poly.BoundaryError(f"unsupported polygon format {ext!r}; supported: {_poly._SUPPORTED}")
    file_crs = _poly._describe_crs(parsed.file_crs) if parsed.file_crs is not None else None
    found = [("Format", ext.lstrip(".")), ("Polygons", str(len(parsed.raws))),
             ("Coordinate system in file", file_crs or ("none found" if parsed.carries_crs else "not carried by this format"))]
    for n, r in enumerate(parsed.raws):
        xy = r.shell
        found.append((f"  [{n}] {r.name or '(unnamed)'}",
                      f"{len(xy)} vertices, x {_fmt(xy[:, 0].min())}..{_fmt(xy[:, 0].max())}, "
                      f"y {_fmt(xy[:, 1].min())}..{_fmt(xy[:, 1].max())}, {len(r.holes)} hole(s)"))
    params = []
    if len(parsed.raws) > 1:
        names = tuple(str(i) for i in range(len(parsed.raws))) + ("largest",)
        params.append(Param("which", "Polygon to use (index or 'largest')", "choice", names, None, "required",
                            help="The file holds several polygons"))
    xy = np.vstack([r.shell for r in parsed.raws])
    if parsed.carries_crs:
        params.append(Param("source_crs", "Source CRS", "crs", (), None, "file" if parsed.file_crs else "required",
                            required=parsed.file_crs is None,
                            help="Leave empty to use the CRS in the file" if parsed.file_crs else "File has no CRS: required, e.g. EPSG:4326"))
        params.append(Param("target_crs", "Model CRS (target)", "crs", (), None, "required",
                            help="The CRS of the model: wells, surfaces and the grid use it"))
    else:
        params.append(Param("source_crs", "Source CRS (only to reproject)", "crs", (), None, "default", required=False,
                            help="Empty = coordinates are already in the model CRS"))
        params.append(Param("target_crs", "Model CRS (only to reproject)", "crs", (), None, "default", required=False))
    prev = ImportPreview("polygon", path, found, params,
                         [f"Ignored geometries: {parsed.ignored}"] if parsed.ignored else [])
    file_crs_text = _crs_cat.normalise(parsed.file_crs) if parsed.file_crs is not None else None

    def hints(values, prev=prev):
        """{param key: ([Suggestion], note)} for the CRS lists, from the polygon's coordinates and the project's CRS."""
        pc = getattr(prev.project, "crs", None)
        typed = values.get("source_crs")
        try:
            src = _crs_cat.normalise(typed) or file_crs_text
        except ValueError:
            src = file_crs_text
        out = {}
        if parsed.carries_crs:
            sg, note = _crs_cat.source_suggestions(xy, parsed.file_crs, pc)
            out["source_crs"] = (sg, note)
            out["target_crs"] = (_crs_cat.target_suggestions(xy, src, pc) if src else
                                 ([_crs_cat.Suggestion(pc, _crs_cat.label_of(pc), "the project's CRS (set by an earlier import)")] if pc else []),
                                 "" if src else "Choose the source CRS first; the target suggestion depends on it.")
        else:
            out["source_crs"] = ([], "")
            out["target_crs"] = ([_crs_cat.Suggestion(pc, _crs_cat.label_of(pc), "the project's CRS")] if pc else [], "")
        return out

    prev.crs_hints = hints
    prev.advisor = lambda p: ([f"The project's CRS is {_crs_cat.label_of(prev.project.crs)} but this boundary is brought into "
                               f"{_crs_cat.label_of(p['target_crs'])}: wells and surfaces already in the project would not match it."]
                              if p.get("target_crs") and getattr(prev.project, "crs", None) and p["target_crs"] != prev.project.crs else [])
    return prev


def _table_found(path, rows, cm):
    return [("Rows", str(len(rows))),
            ("Columns recognised", ", ".join(f"{k} <- '{v}'" for k, v in cm.items())),
            ("All columns", ", ".join(rows[0].keys()))]


def _num_range(rows, col):
    vals = []
    for r in rows:
        try:
            vals.append(float(r[col]))
        except ValueError:
            pass
    return f"{_fmt(min(vals))} to {_fmt(max(vals))}" if vals else "none"


def inspect_well_headers(path) -> ImportPreview:
    rows, cm = _wio._read_table(path, {"well": 1, "x": 1, "y": 1, "kb": 1}, ("td",))
    found = _table_found(path, rows, cm) + [
        ("Wells", ", ".join(r[cm["well"]] for r in rows)),
        ("x range as in file", _num_range(rows, cm["x"])),
        ("y range as in file", _num_range(rows, cm["y"])),
        ("KB range as in file", _num_range(rows, cm["kb"]))]
    return ImportPreview("well_headers", Path(path), found, [
        _unit("xy_unit", "Unit of x, y"), _unit("z_unit", "Unit of KB and TD")],
        ["KB is read as elevation of the depth reference ABOVE mean sea level, not height above ground."])


def inspect_deviation(path) -> ImportPreview:
    rows, cm = _wio._read_table(path, {"well": 1, "md": 1, "incl": 1, "azim": 1})
    per = {}
    for r in rows:
        per[r[cm["well"]]] = per.get(r[cm["well"]], 0) + 1
    found = _table_found(path, rows, cm) + [
        ("Stations per well", ", ".join(f"{w}: {n}" for w, n in per.items())),
        ("MD range as in file", _num_range(rows, cm["md"])),
        ("Inclination range (deg)", _num_range(rows, cm["incl"])),
        ("Azimuth range (deg)", _num_range(rows, cm["azim"]))]
    return ImportPreview("deviation", Path(path), found, [
        _unit("z_unit", "Unit of MD"),
        Param("azimuth_reference", "Azimuth is relative to", "choice", ("grid", "true"), None, "required"),
        Param("convergence_deg", "Grid convergence (deg)", "number", (), None, "required",
              required_if=("azimuth_reference", "true"),
              help="grid north minus true north, positive when grid north is east of true north")])


def inspect_tops(path) -> ImportPreview:
    rows, cm = _wio._read_table(path, {"well": 1, "top": 1, "depth": 1})
    found = _table_found(path, rows, cm) + [
        ("Wells", ", ".join(sorted({r[cm["well"]] for r in rows}))),
        ("Tops", ", ".join(sorted({r[cm["top"]] for r in rows}))),
        ("Depth range as in file", _num_range(rows, cm["depth"]))]
    return ImportPreview("tops", Path(path), found, [
        _unit("z_unit", "Unit of depth"),
        Param("depth_type", "Depth type", "choice", ("md", "tvd", "tvdss"), None, "required",
              help="md = measured; tvd = true vertical below KB; tvdss = below sea level")])


def inspect_core(path) -> ImportPreview:
    rows, _ = _wio._read_table(path, {"well": 1}, ())
    header = list(rows[0])
    depth_col = next((h for h in header if h in _wio._ALIASES["md"] or h in _wio._ALIASES["depth"]), None)
    if depth_col is None:
        raise _wio.WellDataError(f"{Path(path).name}: no depth column (md/depth); found {header}")
    wcol = next(h for h in header if h in _wio._ALIASES["well"])
    data = [h for h in header if h not in (wcol, depth_col)]
    poro = [h for h in data if h in _wio._POROSITY]
    found = [("Rows", str(len(rows))), ("Wells", ", ".join(sorted({r[wcol] for r in rows}))),
             ("Depth column", depth_col), ("Depth range as in file", _num_range(rows, depth_col)),
             ("Data columns", ", ".join(data)),
             ("Porosity columns", ", ".join(poro) or "none")]
    for c in poro:
        found.append((f"  {c} range as in file", _num_range(rows, c)))
    return ImportPreview("core", Path(path), found, [
        _unit("z_unit", "Unit of depth"),
        Param("depth_type", "Depth type", "choice", ("md", "tvd", "tvdss"), "md" if depth_col in _wio._ALIASES["md"] else None,
              "file" if depth_col in _wio._ALIASES["md"] else "required",
              help=f"column is named '{depth_col}'"),
        Param("porosity_unit", "Porosity is in", "choice", ("frac", "percent"), None, "required")],
        ["Permeability columns are taken as mD."])


def inspect_las(path) -> ImportPreview:
    import lasio

    path = Path(path)
    try:
        las = lasio.read(str(path))
    except Exception as exc:
        raise _wio.WellDataError(f"{path.name}: cannot read LAS: {exc}") from exc
    idx = las.curves[0]
    unit = (idx.unit or "").strip().upper()
    file_m = unit in ("M", "METRE", "METRES", "METER", "METERS")
    d = np.asarray(las.index, float)
    found = [("LAS version", str(las.version.VERS.value) if "VERS" in las.version else "?"),
             ("Well", str(las.well.WELL.value) if "WELL" in las.well else "?"),
             ("Index curve / unit in file", f"{idx.mnemonic} / {idx.unit or '(none)'}"),
             ("Depth range as in file", f"{_fmt(d.min())} to {_fmt(d.max())}" if d.size else "none"),
             ("Samples", str(d.size)),
             ("Null value", str(las.well.NULL.value) if "NULL" in las.well else "?"),
             ("Curves", ", ".join(f"{c.mnemonic} [{c.unit or ''}]" for c in las.curves[1:]))]
    return ImportPreview("las", path, found, [
        _unit("depth_unit", "Unit of the depth index", "m" if file_m else None, "file" if file_m else "required",
              help="File says metres" if file_m else f"File says {idx.unit!r}: international or US survey feet?")])


def inspect_fault_sticks(path) -> ImportPreview:
    from . import faults as F
    path = Path(path)
    r = F.parse_sticks(path)
    found = [("Layout", "table with a header row" if r.layout == "table" else "x y z lines without a header"),
             ("Delimiter", r.delimiter)]
    if r.columns:
        found.append(("Columns recognised", ", ".join(f"{k} <- '{v}'" for k, v in r.columns.items())))
    if r.ignored_columns:
        found.append(("Columns ignored", ", ".join(r.ignored_columns)))
    found.append(("Sticks are identified by", {"column": "a stick id column", "separators": "blank / 999 999 999 lines",
                                               "none": "nothing: each fault is ONE polyline"}[r.stick_source]))
    found.append(("Faults / points", f"{len(r.faults)} / {r.n_points}"))
    for name, sticks in r.faults.items():
        a = np.vstack(sticks)
        found.append((f"  {name or '(no name in file)'}",
                      f"{len(sticks)} stick(s), {len(a)} points; x {_fmt(a[:, 0].min())}..{_fmt(a[:, 0].max())}, "
                      f"y {_fmt(a[:, 1].min())}..{_fmt(a[:, 1].max())}, z {_fmt(a[:, 2].min())}..{_fmt(a[:, 2].max())} (as in file)"))
    params = [_unit("xy_unit", "Horizontal unit of the file"), _unit("z_unit", "Vertical unit of the file"),
              Param("z_is", "Vertical values are", "choice", ("depth", "elevation"), None, "required",
                    choice_labels={"depth": "depth, positive downwards", "elevation": "elevation, positive upwards"},
                    help="Time-domain sticks (TWT) are not accepted: depth-convert them first")]
    if not r.has_fault_column:
        params.append(Param("fault_name", "Name of the fault in this file", "text", (), path.stem, "default",
                            help="The file carries no fault name"))
    return ImportPreview("fault_sticks", path, found, params, list(r.notes))


def _read_faults(path, p):
    from . import faults as F
    return F.read_fault_sticks(path, xy_unit=p["xy_unit"], z_unit=p["z_unit"], z_is=p["z_is"], fault_name=p.get("fault_name"))


def _fault_advice(path, p, project):
    from . import faults as F
    try:
        fs = _read_faults(path, p)
    except Exception:
        return []
    out = []
    for f in fs:
        fit = f.fit()
        z0, z1 = f.bounds()[4:]
        if fit.dip_deg is None:
            out.append(f"{f.name}: dip not determined - {fit.reason}")
            continue
        out += [f"{f.name}: {w}" for w in fit.warnings]
        if fit.dip_deg < F.VERTICAL_WARN_BELOW:
            out.append(f"{f.name}: dip is {fit.dip_deg:.0f} deg but v1 treats faults as vertical; over its stick depth range "
                       f"({z1 - z0:.0f} m) the fault moves sideways by {fit.horizontal_shift(z1 - z0):.0f} m. "
                       "The trace will be taken at one depth you choose.")
    if project is not None and project.surfaces:
        zs = [np.nanmin(s.values) for b in project.surfaces.values() for s in b.values()] + \
             [np.nanmax(s.values) for b in project.surfaces.values() for s in b.values()]
        zlo, zhi = float(min(zs)), float(max(zs))
        boxes = []
        for b in project.surfaces.values():
            for s in b.values():
                X, Y = s.node_xy()
                boxes.append((X.min(), X.max(), Y.min(), Y.max()))
        for f in fs:
            x0, x1, y0, y1, a, b = f.bounds()
            if not any(x0 <= bx1 and x1 >= bx0 and y0 <= by1 and y1 >= by0 for bx0, bx1, by0, by1 in boxes):
                out.append(f"{f.name}: lies outside every surface in the project horizontally - check the horizontal unit and CRS")
            if b < zlo or a > zhi:
                out.append(f"{f.name}: its depths ({a:.0f} to {b:.0f} m) do not overlap the project's surfaces ({zlo:.0f} to "
                           f"{zhi:.0f} m) - check depth/elevation and the vertical unit")
    return out


def _read_variogram_file(path):
    from . import variogram as V
    import json
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="strict")
    if path.suffix.lower() == ".json":
        try:
            return V.from_dict(json.loads(text)), "json"
        except json.JSONDecodeError as exc:
            raise V.VariogramError(f"{path.name}: not valid JSON: {exc}") from None
    return V.from_gslib(text), "gslib"


def inspect_property_map(path) -> ImportPreview:
    prev = inspect_surface(path)
    prev.kind = "property_map"
    prev.params = [
        _unit("xy_unit", "Horizontal unit of the file"),
        Param("variable", "Property this map describes", "text", value=None, source="required", help="e.g. PHI, PERMX, NTG"),
        Param("value_unit", "Unit of the map values", "choice", MAP_UNITS, None, "required",
              choice_labels={"fraction": "fraction (0-1)", "percent": "percent (converted to fraction)", "mD": "mD",
                             "m": "metres", "other": "other (kept as is)"},
              help="Nothing is assumed: a porosity map in percent must say so, or it is read as fractions > 1"),
    ]
    return prev


def _property_map_advice(path, p):
    try:
        s = _read_map_values(path, p["xy_unit"])
    except Exception:
        return []
    mx = float(np.nanmax(s.values))
    out = []
    if p["value_unit"] == "fraction" and mx > 1.0 + 1e-9:
        out.append(f"Declared as a fraction but the largest value is {mx:.4g}: this looks like percent.")
    if p["value_unit"] == "percent" and mx <= 1.0:
        out.append(f"Declared as percent but the largest value is {mx:.4g}: this looks like a fraction.")
    return out


def _read_map_values(path, xy_unit):
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        return read_surface(path, xy_unit=xy_unit, z_unit="m", positive="down")


def inspect_variogram(path) -> ImportPreview:
    path = Path(path)
    m, fmt = _read_variogram_file(path)
    found = [("Format", "JSON" if fmt == "json" else "GSLIB parameter block"),
             ("Nugget", _fmt(m.nugget)), ("Total sill", _fmt(m.sill)),
             ("Major axis azimuth (deg from north)", _fmt(m.azimuth))]
    for k, s in enumerate(m.structures, 1):
        found.append((f"Structure {k}", f"{s.type}, contribution {_fmt(s.contribution)}, ranges major/minor/vertical "
                                         f"{_fmt(s.range_major)} / {_fmt(s.range_minor)} / {_fmt(s.range_vert)}"))
    warn = ["GSLIB files carry no units: the ranges are taken as metres. Check them."] if fmt == "gslib" else []
    return ImportPreview("variogram", path, found, [
        Param("variable", "Property this variogram describes", "text", value=m.variable or None,
              source="file" if m.variable else "required"),
        Param("transform", "Data the variogram was computed on", "choice", ("none", "normal_score", "indicator"),
              m.transform if fmt == "json" else None, "file" if fmt == "json" else "required",
              choice_labels={"none": "raw values", "normal_score": "normal-score transformed",
                             "indicator": "0/1 indicator of one facies"},
              help="A variogram of normal scores has sill ~1 and must be used with a normal-score transform; an indicator "
                   "variogram has sill ~ p(1-p) and belongs to a facies, named like CURVE[code]"),
        Param("range_unit", "Unit of the ranges", "choice", ("m", "km", "ft_intl", "ft_us"), "m" if fmt == "json" else None,
              "file" if fmt == "json" else "required"),
    ], warn)


_INSPECT = {"fault_sticks": inspect_fault_sticks, "property_map": inspect_property_map, "variogram": inspect_variogram, "surface": inspect_surface, "polygon": inspect_polygon, "well_headers": inspect_well_headers,
            "deviation": inspect_deviation, "tops": inspect_tops, "core": inspect_core, "las": inspect_las}


def _file_wells(kind, path) -> list:
    if kind in ("surface", "polygon", "variogram", "property_map", "fault_sticks"):
        return []
    if kind == "las":
        import lasio
        las = lasio.read(str(path))
        n = str(las.well.WELL.value).strip() if "WELL" in las.well else ""
        return [n or Path(path).stem]
    rows, cm = _wio._read_table(path, {"well": 1}, ())
    col = next(h for h in rows[0] if h in _wio._ALIASES["well"])
    seen = []
    for r in rows:
        if r[col] not in seen:
            seen.append(r[col])
    return seen


def inspect(kind: str, path, project: Project | None = None) -> ImportPreview:
    if kind not in _INSPECT:
        raise ImportRejected(f"unknown import kind {kind!r}; known: {sorted(_INSPECT)}")
    prev = _INSPECT[kind](path)
    prev.project = project
    chart = project.chart if project is not None else None
    if kind == "surface" and chart is not None:
        labels = {h.uid: h.label for h in chart.horizons()}
        prev.params.append(Param("horizon", "Attach to horizon", "choice", tuple(labels), None, "required",
                                 choice_labels=labels,
                                 help="'Base of A' and 'Top of B' are the same horizon: one choice serves both"))
        prev.suggester = lambda values: _placement(path, values, project)
        prev.conflict_finder = lambda p: project.conflicts_surface(p["horizon"], Path(path).stem)
    if kind == "fault_sticks":
        if project is not None:
            prev.conflict_finder = lambda p: project.conflicts_faults(_read_faults(path, p))
        prev.advisor = lambda p: _fault_advice(path, p, project)
    if kind == "property_map":
        if chart is None:
            raise ImportRejected("define the stratigraphic chart first; maps are attached to its intervals")
        labels = {iv.uid: iv.name for iv in chart.intervals}
        prev.params.append(Param("interval", "Attach to interval", "choice", tuple(labels), None, "required",
                                 choice_labels=labels))
        prev.conflict_finder = lambda p: project.conflicts_map(p["interval"], p["variable"], Path(path).stem)
        prev.advisor = lambda p: _property_map_advice(path, p)
    if kind == "variogram":
        if chart is None:
            raise ImportRejected("define the stratigraphic chart first; variograms are attached to its intervals")
        labels = {iv.uid: iv.name for iv in chart.intervals}
        prev.params.append(Param("interval", "Attach to interval", "choice", tuple(labels), None, "required",
                                 choice_labels=labels))
        prev.conflict_finder = lambda p: project.conflicts_variogram(p["interval"], p["variable"], Path(path).stem)
    prev.file_wells = _file_wells(kind, path)
    if prev.file_wells:
        prev.existing = project.names() if project else []
        prev.links = suggest_links(prev.file_wells, prev.existing)
        if kind == "well_headers":
            prev.warnings.append("If a header is connected to a well that already has one, you choose whether to replace "
                                 "or keep it; a replaced header moves the trajectory and pick positions.")
        if project is not None:
            prev.conflict_finder = lambda p: _find_conflicts(kind, path, p, project)
        if kind == "tops" and chart is not None:
            _prepare_top_mapping(prev, path, chart)
            prev.advisor = lambda p: _advice(path, p, project)
    return prev


def _placement(path, values, project):
    from .placement import suggest_horizons
    try:
        s = read_surface(path, xy_unit=values.get("xy_unit"), z_unit=values.get("z_unit"),
                         positive=values.get("positive") or "down")
    except Exception:
        return None
    import warnings as _w
    return suggest_horizons(project, s, Path(path).stem)


def _prepare_top_mapping(prev, path, chart):
    rows, cm = _wio._read_table(path, {"well": 1, "top": 1, "depth": 1})
    names = []
    for r in rows:
        if r[cm["top"]] not in names:
            names.append(r[cm["top"]])
    prev.top_names = names
    prev.top_choices = {h.uid: h.label for h in chart.horizons()}
    prev.top_choices["ignore"] = "Ignore (do not import)"
    for n in names:
        uid = chart.match_name(n)
        prev.top_suggestions[n] = ((uid, "matches " + chart.horizon(uid).label) if uid else (None, "not in the chart: choose"))


def _advice(path, p, project):
    q = {k: v for k, v in p.items() if k in ("depth_type", "z_unit")}
    mapped = project.map_picks(_wio.parse_tops(path, **q), p["top_map"])
    order = {h.name: h.index for h in project.chart.horizons()}
    out = []
    import copy as _copy
    from .project import WellRecord
    for fw, picks in mapped.items():
        l = p["well_links"][fw]
        rec = _copy.deepcopy(project.wells[l.name]) if l.mode == "existing" else WellRecord(l.name)
        for t in picks:
            rec.tops[t.name] = t
        out += project.rec_order_issues(rec, order)
    return out


def _blank(v):
    return None if v in (None, "") else v


def _find_conflicts(kind, path, p, project):
    """What the confirmed parameters would collide with in the project."""
    q = {k: v for k, v in p.items() if k not in ("well_links", "conflict_actions", "top_map")}
    links = p["well_links"]
    if kind == "well_headers":
        return project.conflicts_headers(_wio.read_well_headers(path, **q), links)
    if kind == "deviation":
        return project.conflicts_surveys(_wio.parse_surveys(path, **q), links)
    if kind == "tops":
        return project.conflicts_tops(project.map_picks(_wio.parse_tops(path, **q), p.get("top_map", {})), links)
    if kind == "core":
        return project.conflicts_core(_wio.parse_core(path, **q), links)
    if kind == "las":
        (fw, link), = links.items()
        return project.conflicts_logs(_wio.read_las(path, **q), fw, link)
    return []


_WELL_KINDS = ("well_headers", "deviation", "tops", "core", "las")


def _run(kind, path, p, project):
    if kind == "surface":
        horizon = p.pop("horizon", None)
        acts = p.pop("conflict_actions", {})
        surf = read_surface(path, **p)
        if project is not None and project.chart is not None:
            return project.attach_surface(horizon, surf, Path(path).stem, acts)
        return surf
    if kind == "polygon":
        which = p.get("which")
        which = int(which) if which not in (None, "largest") else which
        poly = _poly.read_boundary(path, which=which, source_crs=_blank(p.get("source_crs")),
                                   target_crs=_blank(p.get("target_crs")))
        if project is None:
            return poly
        replaced = project.boundary is not None
        project.set_boundary(poly)
        res = ImportResult("polygon")
        tcrs = _blank(p.get("target_crs"))
        if tcrs and project.crs is None:
            project.crs = tcrs
            res.notes.append(f"the project's CRS is now {_crs_cat.label_of(tcrs)}")
        res.applied.append(f"boundary: {len(poly.exterior.coords) - 1} vertices, {poly.area:,.0f} m2"
                           + (" (replaced the previous boundary)" if replaced else ""))
        return res
    if kind == "fault_sticks":
        fs = _read_faults(path, p)
        if project is None:
            return fs
        return project.attach_faults(fs, p.get("conflict_actions", {}))
    if kind == "property_map":
        from .maps import MapGrid, PropertyMap, grid_from_surface
        sf = _read_map_values(path, p["xy_unit"])
        f = 0.01 if p["value_unit"] == "percent" else 1.0
        unit = "fraction" if p["value_unit"] == "percent" else p["value_unit"]
        pm = PropertyMap(grid_from_surface(sf), sf.values * f, p["variable"].strip(), unit, "imported", None,
                         dict(source=Path(path).name, declared_unit=p["value_unit"]))
        return project.attach_map(p["interval"], pm, Path(path).stem, p.get("conflict_actions", {}))
    if kind == "variogram":
        from dataclasses import replace as _rep, fields as _f
        from .units import unit_factor
        from .variogram import Structure
        m, _ = _read_variogram_file(path)
        f = unit_factor(p["range_unit"])
        m = _rep(m, structures=tuple(Structure(s.type, s.contribution, s.range_major * f, s.range_minor * f, s.range_vert * f)
                                      for s in m.structures), transform=p["transform"], variable=p["variable"])
        return project.attach_variogram(p["interval"], p["variable"], m, Path(path).stem, p.get("conflict_actions", {}))
    links = p.pop("well_links")
    acts = p.pop("conflict_actions", {})
    top_map = p.pop("top_map", None)
    if kind == "well_headers":
        return project.apply_headers(_wio.read_well_headers(path, **p), links, acts)
    if kind == "deviation":
        return project.apply_surveys(_wio.parse_surveys(path, **p), links, acts)
    if kind == "tops":
        parsed = _wio.parse_tops(path, **p)
        if top_map is not None:
            parsed = project.map_picks(parsed, top_map)
        res = project.apply_tops(parsed, links, acts)
        if top_map is not None:
            project.remember_aliases(top_map)
            res.notes += [f"order: {m}" for m in project.order_issues()]
        return res
    if kind == "core":
        return project.apply_core(_wio.parse_core(path, **p), links, acts, source=Path(path).stem)
    if kind == "las":
        (fw, link), = links.items()
        return project.apply_logs(Path(path).stem, _wio.read_las(path, **p), fw, link, acts)
    raise ImportRejected(kind)


def import_file(kind: str, path, confirm: Callable, *, project: Project | None = None):
    """inspect -> confirm -> validate -> read. `confirm(preview)` returns the values the
    user accepted (parameters plus `well_links` for well data), or None to cancel.
    Well data (headers, deviation, tops, core, LAS) is applied to `project` and an
    `ImportResult` is returned; surfaces and polygons are returned directly."""
    if kind in ("variogram", "property_map") and (project is None or project.chart is None):
        raise ImportRejected("define the stratigraphic chart first; "
                             + ("variograms" if kind == "variogram" else "maps") + " are attached to its intervals")
    if kind in _WELL_KINDS and project is None:
        raise ImportRejected(f"importing {kind!r} needs project= (the wells it connects to)")
    if kind == "surface" and project is not None and project.chart is None:
        raise ImportRejected("define the stratigraphic chart first (build it from the type well's tops); "
                             "surfaces are attached to its horizons")
    prev = inspect(kind, path, project)
    values = confirm(prev)
    if values is None:
        raise ImportCancelled(f"{Path(path).name}: import cancelled")
    return _run(kind, path, finalize(prev, values), project)


# -- several LAS files in one go -------------------------------------------------------------------------------------------
@dataclass
class BatchResult:
    kind: str
    files: list = field(default_factory=list)       # [(file name, ImportResult, "asked" | "same choices as <first file>")]

    @property
    def applied(self) -> list:
        return [f"{n}: {', '.join(r.applied) or 'nothing new'}" for n, r, _ in self.files]

    @property
    def notes(self) -> list:
        return [x for _, r, _ in self.files for x in r.notes]


def las_overview(paths) -> list:
    """[(label, text)] for the first confirmation of a batch: one line per file, so nothing is a surprise."""
    import lasio
    out = []
    for p in paths:
        p = Path(p)
        try:
            las = lasio.read(str(p))
            w = str(las.well.WELL.value).strip() if "WELL" in las.well else ""
            u = (las.curves[0].unit or "").strip() or "(none)"
            out.append((p.name, f"well {w or '(no name; file name is used)'}, index unit {u}, {len(las.index)} samples, "
                                f"{len(las.curves) - 1} curves"))
        except Exception as exc:
            out.append((p.name, f"CANNOT READ: {exc}"))
    return out


def _same_choices(prev, shared):
    """Values reusing the choices already made for the batch, or None when this file needs its own confirmation:
    its unit differs from the file's own declaration, a well link is a guess (note) or ambiguous, or anything is left
    to choose (conflicts with data already there)."""
    values = prev.initial_values()
    if dict(prev.found).get("Index curve / unit in file") != shared.get("_declared"):
        return None                               # a different index curve or unit label than the one the choice was made for
    for p in prev.params:
        if p.key in shared:
            values[p.key] = shared[p.key]
    for fw, (link, _) in prev.links.items():
        if link.name != fw:                       # matched on a similar name only: the user has to look at it
            return None
    try:
        finalize(prev, values)
    except ImportRejected:
        return None
    return values


def import_las_batch(paths, confirm: Callable, *, project: Project):
    """Import several LAS files. The first file (and any later one that differs: another index unit label than the first file, a well
    link that is only a guess, a clash with data already there) goes through `confirm(preview)` like a single import;
    the choices made there are reused, visibly, for the files that fit them. All or nothing: if one file is refused,
    cancelled or fails, the project is left as it was."""
    paths = [Path(p) for p in paths]
    if not paths:
        raise ImportRejected("no LAS file chosen")
    if len({p.resolve() for p in paths}) != len(paths):
        raise ImportRejected("the same file is chosen twice")
    snapshot = copy.deepcopy(project.wells)
    res, shared, first = BatchResult("las"), None, None
    try:
        for k, p in enumerate(paths):
            prev = inspect("las", p, project)
            values = _same_choices(prev, shared) if shared is not None else None
            how = f"same choices as {first}"
            if values is None:
                if len(paths) > 1:
                    prev.found = [(f"File {k + 1} of {len(paths)}", p.name)] + (las_overview(paths) if k == 0 else []) + prev.found
                    prev.warnings = prev.warnings + ([f"Your choices here are reused for the other {len(paths) - 1} file(s) that "
                                                      "fit them; a file with a different unit, an uncertain well match or "
                                                      "a clash with existing data asks again."] if k == 0 else [])
                values = confirm(prev)
                if values is None:
                    raise ImportCancelled(f"{p.name}: import cancelled; nothing was imported")
                how = "asked"
                if shared is None:
                    shared = {q.key: values.get(q.key, q.value) for q in prev.params}
                    shared["_declared"] = dict(prev.found).get("Index curve / unit in file")
                    first = p.name
            res.files.append((p.name, _run("las", p, finalize(prev, values), project), how))
    except BaseException:
        project.wells.clear()
        project.wells.update(snapshot)
        raise
    return res

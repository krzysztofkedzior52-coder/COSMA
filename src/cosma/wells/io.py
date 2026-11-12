"""Well data importers: headers, deviation surveys, LAS logs, tops and core (CSV).

No guessing: horizontal and vertical units, the depth type of tops/core (MD, TVD from
KB, or TVDSS), the north reference of azimuths and the unit of porosity are all
arguments. Column names are matched case-insensitively against a small alias list;
anything unrecognised is an error that lists what was found.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

from ..units import unit_factor
from .model import CoreSamples, LogSet, RawCore, RawTop, Survey, Top, Well
from .trajectory import Trajectory, TrajectoryError


class WellDataError(ValueError):
    pass


_ALIASES = {
    "well": ("well", "well_name", "wellname", "uwi", "name"),
    "x": ("x", "easting", "x_coord", "xcoord", "east"),
    "y": ("y", "northing", "y_coord", "ycoord", "north"),
    "kb": ("kb", "rkb", "kb_elev", "kb_elevation", "kelly_bushing", "rt"),
    "td": ("td", "total_depth", "td_md"),
    "md": ("md", "mdkb", "measured_depth", "depth_md"),
    "incl": ("incl", "inclination", "inc", "dip"),
    "azim": ("azim", "azimuth", "azi", "az"),
    "top": ("top", "horizon", "marker", "formation", "pick", "surface"),
    "depth": ("depth", "pick_depth", "top_depth"),
}


def _read_table(path, required: dict, optional: tuple = ()):
    """Parse a delimited text file. Returns (rows, colmap): rows is a list of dicts keyed
    by the lower-case file header, colmap maps our key to the file's column name."""
    path = Path(path)
    if not path.exists():
        raise WellDataError(f"{path}: no such file")
    text = path.read_bytes().decode("utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t ")
        delim = dialect.delimiter
    except csv.Error:
        delim = ","
    rdr = csv.reader(io.StringIO(text), delimiter=delim)
    rows = [r for r in rdr if r and any(c.strip() for c in r)]
    rows = [r for r in rows if not r[0].lstrip().startswith("#")]
    if len(rows) < 2:
        raise WellDataError(f"{path.name}: needs a header line and at least one data row")
    header = [h.strip().lower() for h in rows[0]]
    colmap = {}
    for key in list(required) + list(optional):
        found = [h for h in header if h in _ALIASES.get(key, (key,))]
        if len(found) > 1:
            raise WellDataError(f"{path.name}: several columns could be '{key}': {found}")
        if found:
            colmap[key] = found[0]
        elif key in required:
            raise WellDataError(f"{path.name}: no '{key}' column (accepted: {_ALIASES.get(key, (key,))}); "
                                f"found: {header}")
    out = []
    for ln, r in enumerate(rows[1:], start=2):
        if len(r) != len(header):
            raise WellDataError(f"{path.name}, line {ln}: {len(r)} fields, header has {len(header)}")
        out.append({h: v.strip() for h, v in zip(header, r)})
    return out, colmap


def _num(path, ln, field, text):
    try:
        v = float(text)
    except ValueError:
        raise WellDataError(f"{Path(path).name}, row {ln}: {field} = {text!r} is not a number") from None
    if not np.isfinite(v):
        raise WellDataError(f"{Path(path).name}, row {ln}: {field} is not finite")
    return v


def read_well_headers(path, *, xy_unit: str, z_unit: str) -> dict:
    """CSV with well, x, y, kb, [td]. `kb` = elevation of the depth reference (KB/RT)
    above the datum used by the surfaces (mean sea level), NOT height above ground.
    Returns {name: dict(x, y, kb, td or None)} in metres."""
    fx, fz = unit_factor(xy_unit), unit_factor(z_unit)
    rows, cm = _read_table(path, {"well": 1, "x": 1, "y": 1, "kb": 1}, ("td",))
    out = {}
    for ln, r in enumerate(rows, start=2):
        name = r[cm["well"]]
        if name in out:
            raise WellDataError(f"{Path(path).name}, row {ln}: duplicate well {name!r}")
        out[name] = dict(
            x=_num(path, ln, "x", r[cm["x"]]) * fx,
            y=_num(path, ln, "y", r[cm["y"]]) * fx,
            kb=_num(path, ln, "kb", r[cm["kb"]]) * fz,
            td=(_num(path, ln, "td", r[cm["td"]]) * fz) if "td" in cm and r[cm["td"]] else None,
        )
    return out


def parse_surveys(path, *, z_unit: str, azimuth_reference: str, convergence_deg: float | None = None) -> dict:
    """CSV with well, md, incl, azim (degrees) -> {well: Survey}, validated but not yet placed
    (placing needs the well header). The survey must start at MD 0: the wellhead position
    is the first station, and nothing is extrapolated or guessed."""
    fz = unit_factor(z_unit)
    rows, cm = _read_table(path, {"well": 1, "md": 1, "incl": 1, "azim": 1})
    per = {}
    for ln, r in enumerate(rows, start=2):
        per.setdefault(r[cm["well"]], []).append((_num(path, ln, "md", r[cm["md"]]) * fz,
                                                   _num(path, ln, "incl", r[cm["incl"]]),
                                                   _num(path, ln, "azim", r[cm["azim"]])))
    out = {}
    for w, st in per.items():
        a = np.array(st)
        if a[0, 0] != 0.0:
            raise WellDataError(f"{w}: survey starts at MD {a[0,0]:g}, not 0. Add the surface station "
                                "(MD 0, incl 0) or supply the TVD at the first station; refusing to guess.")
        try:   # throwaway placement: validates ordering, ranges, NaN and the azimuth reference
            Trajectory.from_survey(a[:, 0], a[:, 1], a[:, 2], x0=0, y0=0, kb=0,
                                   azimuth_reference=azimuth_reference, convergence_deg=convergence_deg)
        except TrajectoryError as exc:
            raise WellDataError(f"{w}: {exc}") from exc
        out[w] = Survey(a[:, 0], a[:, 1], a[:, 2], azimuth_reference, convergence_deg)
    return out


def place_survey(survey: Survey, header: dict) -> Trajectory:
    return Trajectory.from_survey(survey.md, survey.incl, survey.azim, x0=header["x"], y0=header["y"],
                                  kb=header["kb"], azimuth_reference=survey.azimuth_reference,
                                  convergence_deg=survey.convergence_deg)


def read_deviation(path, headers: dict, *, z_unit: str, azimuth_reference: str,
                   convergence_deg: float | None = None) -> dict:
    """parse_surveys + placement at each well's header. Wells in the file but not in
    `headers` are an error."""
    surveys = parse_surveys(path, z_unit=z_unit, azimuth_reference=azimuth_reference,
                            convergence_deg=convergence_deg)
    unknown = sorted(set(surveys) - set(headers))
    if unknown:
        raise WellDataError(f"{Path(path).name}: wells without a header: {unknown}")
    return {w: place_survey(sv, headers[w]) for w, sv in surveys.items()}


def read_las(path, *, depth_unit: str | None = None) -> LogSet:
    """LAS 1.2/2.0 via lasio. The index must be measured depth. `depth_unit` is required
    unless the file states metres ('M'); feet are never assumed (international or survey?).
    Null values (the file's NULL) become NaN."""
    import lasio

    try:
        las = lasio.read(str(path))
    except Exception as exc:
        raise WellDataError(f"{Path(path).name}: cannot read LAS: {exc}") from exc
    idx = las.curves[0]
    if idx.mnemonic.upper() not in ("DEPT", "DEPTH", "MD"):
        raise WellDataError(f"{Path(path).name}: first curve is {idx.mnemonic!r}, expected DEPT/DEPTH/MD (time-indexed logs unsupported)")
    unit = depth_unit
    if unit is None:
        file_unit = (idx.unit or "").strip().upper()
        if file_unit in ("M", "METRE", "METRES", "METER", "METERS"):
            unit = "m"
        else:
            raise WellDataError(f"{Path(path).name}: depth unit in file is {idx.unit!r}; pass depth_unit= explicitly")
    md = np.asarray(las.index, float) * unit_factor(unit)
    if np.any(np.diff(md) <= 0):
        raise WellDataError(f"{Path(path).name}: depth index is not strictly increasing")
    curves, units = {}, {}
    for c in las.curves[1:]:
        curves[c.mnemonic] = np.asarray(c.data, float)
        units[c.mnemonic] = c.unit or ""
    return LogSet(md=md, curves=curves, units=units)


def _depth_to_position(w: str, traj: Trajectory, depth: float, depth_type: str):
    """(md, tvdss, x, y) for a pick given as MD, TVD (below KB) or TVDSS."""
    try:
        if depth_type == "md":
            md = depth
        else:
            tvdss = depth - traj.kb if depth_type == "tvd" else depth
            md = float(traj.md_at_tvdss(tvdss)[0])
        x, y, z = traj.at_md(md)
    except TrajectoryError as exc:
        raise WellDataError(f"{w}: {exc}") from exc
    return md, float(z[0]), float(x[0]), float(y[0])


def resolve_top(well: str, traj: Trajectory, t: RawTop) -> Top:
    md, tvdss, x, y = _depth_to_position(well, traj, t.depth, t.depth_type)
    return Top(well, t.name, md, tvdss, x, y)


def resolve_core(well: str, traj: Trajectory, rc: RawCore) -> CoreSamples:
    mds = np.array([_depth_to_position(well, traj, float(d), rc.depth_type)[0] for d in rc.depth])
    order = np.argsort(mds, kind="stable")
    return CoreSamples(well=well, md=mds[order], columns={k: np.asarray(v)[order] for k, v in rc.columns.items()})


def parse_tops(path, *, depth_type: str, z_unit: str) -> dict:
    """CSV with well, top, depth -> {well: [RawTop]}; positions are derived later from the
    trajectory, so a corrected survey never leaves stale pick coordinates behind."""
    if depth_type not in ("md", "tvd", "tvdss"):
        raise WellDataError("depth_type must be 'md', 'tvd' or 'tvdss'")
    fz = unit_factor(z_unit)
    rows, cm = _read_table(path, {"well": 1, "top": 1, "depth": 1})
    out = {}
    for ln, r in enumerate(rows, start=2):
        out.setdefault(r[cm["well"]], []).append(
            RawTop(r[cm["top"]], _num(path, ln, "depth", r[cm["depth"]]) * fz, depth_type))
    return out


def read_tops(path, trajectories: dict, *, depth_type: str, z_unit: str) -> list:
    """parse_tops + positions from each well's trajectory. Returns a list of Top."""
    parsed = parse_tops(path, depth_type=depth_type, z_unit=z_unit)
    tops = []
    for w, picks in parsed.items():
        if w not in trajectories:
            raise WellDataError(f"{Path(path).name}: no trajectory for well {w!r}")
        tops += [resolve_top(w, trajectories[w], t) for t in picks]
    return tops


def parse_core(path, *, depth_type: str, z_unit: str, porosity_unit: str) -> dict:
    """CSV with well, md (or depth per depth_type), plus data columns -> {well: RawCore}.
    Porosity columns (poro/phi/porosity) are converted with porosity_unit ('frac' or
    'percent') and must land in [0, 1]. Other columns are kept as given (permeability: mD)."""
    if porosity_unit not in ("frac", "percent"):
        raise WellDataError("porosity_unit must be 'frac' or 'percent'")
    if depth_type not in ("md", "tvd", "tvdss"):
        raise WellDataError("depth_type must be 'md', 'tvd' or 'tvdss'")
    fz = unit_factor(z_unit)
    key = "md" if depth_type == "md" else "depth"
    rows, cm = _read_table(path, {"well": 1, key: 1})
    data_cols = [h for h in rows[0] if h not in (cm["well"], cm[key])]
    by = {}
    for ln, r in enumerate(rows, start=2):
        d = _num(path, ln, "depth", r[cm[key]]) * fz
        vals = {}
        for c in data_cols:
            if r[c] == "":
                vals[c] = np.nan
                continue
            v = _num(path, ln, c, r[c])
            if c in _POROSITY:
                v = v / 100.0 if porosity_unit == "percent" else v
                if not 0.0 <= v <= 1.0:
                    raise WellDataError(f"{Path(path).name}, row {ln}: porosity {r[c]} is outside [0, 1] "
                                        f"after interpreting it as {porosity_unit!r}")
            vals[c] = v
        by.setdefault(r[cm["well"]], []).append((d, vals))
    return {w: RawCore(np.array([d for d, _ in it]), depth_type,
                       {c: np.array([v[c] for _, v in it]) for c in data_cols}) for w, it in by.items()}


def read_core(path, trajectories: dict, *, depth_type: str, z_unit: str, porosity_unit: str) -> dict:
    """parse_core + MD from each well's trajectory. Returns {well: CoreSamples} (MD, metres)."""
    parsed = parse_core(path, depth_type=depth_type, z_unit=z_unit, porosity_unit=porosity_unit)
    out = {}
    for w, rc in parsed.items():
        if w not in trajectories:
            raise WellDataError(f"{Path(path).name}: no trajectory for well {w!r}")
        out[w] = resolve_core(w, trajectories[w], rc)
    return out


_POROSITY = ("poro", "phi", "porosity", "phit", "phie")


def build_wells(headers: dict, trajectories: dict, *, logs: dict | None = None, tops: list | None = None,
                core: dict | None = None) -> list:
    """Assemble Well objects; every well needs an explicit trajectory (use
    Trajectory.vertical to say a well is vertical - it is never assumed)."""
    missing = sorted(set(headers) - set(trajectories))
    if missing:
        raise WellDataError(f"wells with a header but no trajectory (supply a deviation survey or "
                            f"Trajectory.vertical explicitly): {missing}")
    wells = []
    for name in headers:
        w = Well(name=name, trajectory=trajectories[name], logs=(logs or {}).get(name),
                 core=(core or {}).get(name))
        w.tops = [t for t in (tops or []) if t.well == name]
        wells.append(w)
    return wells

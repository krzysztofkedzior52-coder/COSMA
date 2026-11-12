"""Save and load a project: one `.cosma` file (a zip) holding

    manifest.json   format name, format version, writer version, SHA-256 of the other two members
    project.json    everything except the numbers (names, settings, chart, variogram models)
    arrays.npz      the numbers (well logs, surveys, core, surface grids)

No pickle is used anywhere: a project file cannot run code when opened, and stays readable by
other tools (any zip tool, numpy). Floats round-trip exactly. Writing is atomic (a temporary
file next to the target, then a rename), so a crash or a full disk never leaves a half-written
project in place of a good one. Loading checks the format version (a file from a newer COSMA is
refused, never guessed at), the checksums, and every cross-reference (surfaces and variograms
must belong to chart horizons, primaries must exist, ...) before returning a project.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from . import __version__ as COSMA_VERSION
from .faults import Fault, FaultError
from .maps import MapError, MapGrid, PropertyMap
from .project import Project, WellRecord
from .strat import StratChart, StratError
from .surfaces import Surface, SurfaceError
from .variogram import VariogramError, from_dict, to_dict
from .wells.model import LogSet, RawCore, RawTop, Survey

FORMAT = "cosma-project"
FORMAT_VERSION = 13         # 13: adds the realization plan; 12: adds the facies simulation setup; 11: adds proportion maps; 10: adds the proportion setup; 9: fault transmissibility may use SGR relations and the user's own SGR methods; 8: adds the data-check setup; 7: adds the facies setup; 6: adds fault transmissibility choices; 5: adds the boundary polygon and boundary options; 4: the grid definition; 3: fault depths and interactions; 2: faults; older files still load
EXTENSION = ".cosma"


class PersistError(ValueError):
    pass


class _Arrays:
    def __init__(self, data: dict | None = None):
        self.data = data if data is not None else {}

    def put(self, arr) -> dict:
        a = np.asarray(np.ma.filled(arr, np.nan) if isinstance(arr, np.ma.MaskedArray) else arr)
        if a.dtype.kind not in "fiub":
            raise PersistError(f"cannot store an array of type {a.dtype} (only numbers; no pickling)")
        key = f"a{len(self.data)}"
        self.data[key] = a
        return {"$array": key}

    def get(self, ref) -> np.ndarray:
        try:
            a = self.data[ref["$array"]]
        except (KeyError, TypeError):
            raise PersistError(f"missing array reference {ref!r}") from None
        return np.array(a)


def _np_default(o):
    if isinstance(o, (np.generic,)):
        return o.item()
    raise TypeError(f"cannot store {type(o).__name__} in a project file")


def _pairs(d: dict) -> list:
    return [[list(k) if isinstance(k, tuple) else k, v] for k, v in d.items()]


# -- project -> document ------------------------------------------------------------------------------------------------------
def _doc(pr: Project, ar: _Arrays) -> dict:
    wells = {}
    for n, r in pr.wells.items():
        wells[n] = dict(
            header=None if r.header is None else dict(r.header),
            survey=None if r.survey is None else dict(
                md=ar.put(r.survey.md), incl=ar.put(r.survey.incl), azim=ar.put(r.survey.azim),
                azimuth_reference=r.survey.azimuth_reference, convergence_deg=r.survey.convergence_deg),
            logs={s: dict(md=ar.put(l.md), units=dict(l.units), curves={c: ar.put(v) for c, v in l.curves.items()})
                  for s, l in r.logs.items()},
            tops={k: dict(name=t.name, depth=t.depth, depth_type=t.depth_type) for k, t in r.tops.items()},
            core={s: dict(depth=ar.put(c.depth), depth_type=c.depth_type, columns={k: ar.put(v) for k, v in c.columns.items()})
                  for s, c in r.core.items()})
    surfaces = {uid: {lab: dict(xori=s.xori, yori=s.yori, xinc=s.xinc, yinc=s.yinc, rotation=s.rotation, yflip=s.yflip,
                                name=s.name, values=ar.put(s.values)) for lab, s in b.items()}
                for uid, b in pr.surfaces.items()}
    variograms = {uid: {var: {lab: to_dict(m) for lab, m in b.items()} for var, b in vs.items()}
                  for uid, vs in pr.variograms.items()}
    maps = {uid: {var: {lab: dict(grid=dict(xori=m.grid.xori, yori=m.grid.yori, xinc=m.grid.xinc, yinc=m.grid.yinc, ncol=m.grid.ncol,
                                            nrow=m.grid.nrow, rotation=m.grid.rotation),
                                  values=ar.put(m.values), variance=None if m.variance is None else ar.put(m.variance),
                                  variable=m.variable, unit=m.unit, method=m.method, info=m.info)
                        for lab, m in b.items()} for var, b in vs.items()} for uid, vs in pr.maps.items()}
    faults = {n: dict(sticks=[ar.put(np.asarray(k, float)) for k in f.sticks], source=f.source, info=f.info)
              for n, f in pr.faults.items()}
    from .fault_interactions import to_state as _ints
    from .fault_transmissibility import to_state as _tm_state
    from .fault_relations import methods_to_state as _methods_state
    from .facies import to_state as _facies_state
    from .datacheck import to_state as _dc_state
    from .proportions import to_state as _prop_state
    from .proportion_maps import ProportionMap  # noqa: F401
    pmaps = {k: dict(interval=m.interval, codes=list(m.codes), fractions=ar.put(m.fractions), recipe=m.recipe_state, report=m.report)
             for k, m in pr.proportion_maps.items()}
    from .facies_sim import to_state as _fs_state
    from .realizations import to_state as _rp_state
    bnd = None
    if pr.boundary is not None:
        bnd = dict(shell=ar.put(np.asarray(pr.boundary.exterior.coords, float)),
                   holes=[ar.put(np.asarray(h.coords, float)) for h in pr.boundary.interiors])
    return dict(boundary=bnd, faults=faults, fault_depths=dict(pr.fault_depths), fault_reach=pr.fault_reach,
                fault_transmissibility=_tm_state(pr.fault_transmissibility), sgr_methods=_methods_state(pr.sgr_methods), facies=_facies_state(pr.facies_setup), data_checks=_dc_state(pr.data_checks), proportions=_prop_state(pr.proportions), proportion_maps=pmaps, facies_sim=_fs_state(pr.facies_sim), realization_plan=_rp_state(pr.realization_plan),
                fault_interactions=_ints(pr.fault_interactions),
                grid_def=None if pr.grid_def is None else pr.grid_def.to_state(), maps=maps, map_primary=_pairs(pr.map_primary), name=pr.name, type_well=pr.type_well, chart=None if pr.chart is None else pr.chart.to_state(),
                wells=wells, surfaces=surfaces, primary=dict(pr.primary), variograms=variograms,
                var_primary=_pairs(pr.var_primary), var_meta=_pairs(pr.var_meta))


def _serialize(pr: Project):
    ar = _Arrays()
    doc = _doc(pr, ar)
    pj = json.dumps(doc, indent=1, allow_nan=True, default=_np_default).encode("utf-8")
    buf = io.BytesIO()
    np.savez(buf, **ar.data)
    return pj, buf.getvalue()


def fingerprint(pr: Project) -> str:
    """Hash of everything that would be saved: equal fingerprints = nothing to save."""
    pj, ab = _serialize(pr)
    return hashlib.sha256(pj + hashlib.sha256(ab).digest()).hexdigest()


def save_project(pr: Project, path) -> Path:
    path = Path(path)
    if path.suffix.lower() != EXTENSION:
        path = path.with_name(path.name + EXTENSION)
    pj, ab = _serialize(pr)
    manifest = json.dumps(dict(format=FORMAT, format_version=FORMAT_VERSION, cosma_version=COSMA_VERSION,
                               sha256=dict(project=hashlib.sha256(pj).hexdigest(), arrays=hashlib.sha256(ab).hexdigest())),
                          indent=1).encode()
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh, zipfile.ZipFile(fh, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", manifest)
            z.writestr("project.json", pj)
            z.writestr("arrays.npz", ab)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


# -- document -> project ---------------------------------------------------------------------------------------------------------
def _read_members(path: Path):
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            need = {"manifest.json", "project.json", "arrays.npz"}
            if not need <= names:
                raise PersistError(f"{path.name} is not a COSMA project (missing {sorted(need - names)})")
            return z.read("manifest.json"), z.read("project.json"), z.read("arrays.npz")
    except zipfile.BadZipFile:
        raise PersistError(f"{path.name} is not a COSMA project file (not a zip archive, or damaged)") from None


def load_project(path) -> Project:
    path = Path(path)
    if not path.is_file():
        raise PersistError(f"{path}: no such file")
    mb, pj, ab = _read_members(path)
    try:
        man = json.loads(mb)
        fmt, ver, sha = man["format"], man["format_version"], man["sha256"]
    except (json.JSONDecodeError, KeyError, TypeError):
        raise PersistError(f"{path.name}: manifest is unreadable") from None
    if fmt != FORMAT:
        raise PersistError(f"{path.name}: not a COSMA project (format {fmt!r})")
    if not isinstance(ver, int) or ver < 1:
        raise PersistError(f"{path.name}: invalid format version {ver!r}")
    if ver > FORMAT_VERSION:
        raise PersistError(f"{path.name} was written by a newer COSMA (project format {ver}; this version reads up to "
                           f"{FORMAT_VERSION}). Update COSMA to open it.")
    if hashlib.sha256(pj).hexdigest() != sha.get("project") or hashlib.sha256(ab).hexdigest() != sha.get("arrays"):
        raise PersistError(f"{path.name}: checksum mismatch - the file is damaged or was edited")
    try:
        doc = json.loads(pj)
        with np.load(io.BytesIO(ab), allow_pickle=False) as npz:
            ar = _Arrays({k: npz[k] for k in npz.files})
        return _build(doc, ar)
    except PersistError:
        raise
    except (KeyError, TypeError, ValueError, StratError, SurfaceError, VariogramError, MapError, AttributeError) as exc:
        raise PersistError(f"{path.name}: project content is inconsistent: {exc}") from exc


def _build(doc: dict, ar: _Arrays) -> Project:
    pr = Project(doc["name"])
    for n, w in doc["wells"].items():
        r = WellRecord(n)
        r.header = None if w["header"] is None else dict(w["header"])
        sv = w["survey"]
        if sv is not None:
            r.survey = Survey(ar.get(sv["md"]), ar.get(sv["incl"]), ar.get(sv["azim"]), sv["azimuth_reference"],
                              sv["convergence_deg"])
        for s, l in w["logs"].items():
            r.logs[s] = LogSet(ar.get(l["md"]), {c: ar.get(v) for c, v in l["curves"].items()}, dict(l["units"]))
        for k, t in w["tops"].items():
            r.tops[k] = RawTop(t["name"], float(t["depth"]), t["depth_type"])
        for s, c in w["core"].items():
            r.core[s] = RawCore(ar.get(c["depth"]), c["depth_type"], {k: ar.get(v) for k, v in c["columns"].items()})
        pr.wells[n] = r
    if doc["type_well"] is not None and doc["type_well"] not in pr.wells:
        raise PersistError(f"type well {doc['type_well']!r} is not among the wells")
    pr.type_well = doc["type_well"]
    pr.chart = None if doc["chart"] is None else StratChart.from_state(doc["chart"])
    uids = set() if pr.chart is None else {h.uid for h in pr.chart.horizons()}
    ivs = set() if pr.chart is None else {i.uid for i in pr.chart.intervals}
    for uid, b in doc["surfaces"].items():
        if uid not in uids:
            raise PersistError(f"surface attached to unknown horizon {uid!r}")
        pr.surfaces[uid] = {lab: Surface(s["xori"], s["yori"], s["xinc"], s["yinc"], s["rotation"], s["yflip"], ar.get(s["values"]),
                                         s["name"]) for lab, s in b.items()}
    pr.primary = dict(doc["primary"])
    for uid, lab in pr.primary.items():
        if lab not in pr.surfaces.get(uid, {}):
            raise PersistError(f"primary surface {lab!r} of horizon {uid!r} does not exist")
    if set(pr.surfaces) - set(pr.primary):
        raise PersistError("a horizon has surfaces but no primary one")
    for uid, vs in doc["variograms"].items():
        if uid not in ivs:
            raise PersistError(f"variogram attached to unknown interval {uid!r}")
        pr.variograms[uid] = {var: {lab: from_dict(m) for lab, m in b.items()} for var, b in vs.items()}
    pr.var_primary = {tuple(k): v for k, v in doc["var_primary"]}
    for (uid, var), lab in pr.var_primary.items():
        if lab not in pr.variograms.get(uid, {}).get(var, {}):
            raise PersistError(f"primary variogram {lab!r} of {var!r} does not exist")
    for uid, vs in pr.variograms.items():
        for var, b in vs.items():
            if b and (uid, var) not in pr.var_primary:
                raise PersistError(f"{var!r} has variograms but no primary one")
    for uid, vs in doc.get("maps", {}).items():
        if uid not in ivs:
            raise PersistError(f"map attached to unknown interval {uid!r}")
        pr.maps[uid] = {var: {lab: PropertyMap(MapGrid(**m["grid"]), ar.get(m["values"]),
                                               m["variable"], m["unit"], m["method"],
                                               None if m["variance"] is None else ar.get(m["variance"]), dict(m["info"]))
                              for lab, m in b.items()} for var, b in vs.items()}
    pr.map_primary = {tuple(k): v for k, v in doc.get("map_primary", [])}
    for (uid, var), lab in pr.map_primary.items():
        if lab not in pr.maps.get(uid, {}).get(var, {}):
            raise PersistError(f"primary map {lab!r} of {var!r} does not exist")
    for uid, vs in pr.maps.items():
        for var, b in vs.items():
            if b and (uid, var) not in pr.map_primary:
                raise PersistError(f"{var!r} has maps but no primary one")
    for n, f in doc.get("faults", {}).items():
        try:
            pr.faults[n] = Fault(n, tuple(np.asarray(ar.get(k), float) for k in f["sticks"]), f.get("source", ""), dict(f.get("info", {})))
        except FaultError as exc:
            raise PersistError(f"fault {n!r}: {exc}") from None
    from .fault_interactions import InteractionError, from_state as _from_ints
    for n, d in doc.get("fault_depths", {}).items():
        if n not in pr.faults:
            raise PersistError(f"trace depth for a fault that does not exist: {n!r}")
        if not isinstance(d, (int, float)) or d != d or abs(d) == float("inf"):
            raise PersistError(f"fault {n!r}: bad trace depth {d!r}")
        pr.fault_depths[n] = float(d)
    from .fault_transmissibility import TransmissibilityError, from_state as _from_tm
    from .fault_relations import RelationError, methods_from_state
    try:
        pr.sgr_methods = methods_from_state(doc.get("sgr_methods", []))
    except RelationError as exc:
        raise PersistError(str(exc)) from None
    try:
        pr.fault_transmissibility = _from_tm(doc.get("fault_transmissibility", {}))
    except TransmissibilityError as exc:
        raise PersistError(str(exc)) from None
    for n in pr.fault_transmissibility:
        if n not in pr.faults:
            raise PersistError(f"transmissibility choice for a fault that does not exist: {n!r}")
    from .facies import FaciesError, from_state as _facies_from
    try:
        pr.facies_setup = _facies_from(doc.get("facies"))
    except FaciesError as exc:
        raise PersistError(str(exc)) from None
    from .datacheck import DataCheckError, from_state as _dc_from
    try:
        pr.data_checks = _dc_from(doc.get("data_checks"))
    except DataCheckError as exc:
        raise PersistError(str(exc)) from None
    from .proportions import ProportionError, from_state as _prop_from
    try:
        pr.proportions = _prop_from(doc.get("proportions"))
    except ProportionError as exc:
        raise PersistError(str(exc)) from None
    from .proportion_maps import MapImportError, ProportionMap, recipe_from_state
    pr.proportion_maps = {}
    try:
        for k, m in (doc.get("proportion_maps") or {}).items():
            fr = ar.get(m["fractions"])
            codes = tuple(m["codes"])
            if fr.ndim != 3 or fr.shape[2] != len(codes) or m["interval"] != k:
                raise PersistError(f"bad proportion map {k!r}")
            recipe_from_state(m["recipe"])                      # validates the stored recipe
            pr.proportion_maps[k] = ProportionMap(k, codes, fr, m["recipe"], str(m["report"]))
    except (MapImportError, KeyError, TypeError) as exc:
        raise PersistError(f"bad proportion map: {exc}") from None
    from .facies_sim import FaciesSimError, from_state as _fs_from
    try:
        pr.facies_sim = _fs_from(doc.get("facies_sim"))
    except FaciesSimError as exc:
        raise PersistError(str(exc)) from None
    from .realizations import RealizationError, from_state as _rp_from
    try:
        pr.realization_plan = _rp_from(doc.get("realization_plan"))
    except RealizationError as exc:
        raise PersistError(str(exc)) from None
    reach = doc.get("fault_reach")
    if reach is not None and not (isinstance(reach, (int, float)) and reach > 0 and reach != float("inf")):
        raise PersistError(f"bad fault reach {reach!r}")
    pr.fault_reach = None if reach is None else float(reach)
    try:
        pr.fault_interactions = _from_ints(doc.get("fault_interactions", []))
    except InteractionError as exc:
        raise PersistError(str(exc)) from None
    for i in pr.fault_interactions:
        if i.a not in pr.faults or i.b not in pr.faults:
            raise PersistError(f"interaction between faults that do not exist: {i.a!r}/{i.b!r}")
    if doc.get("boundary") is not None:
        from shapely.geometry import Polygon
        from .grid.boundary import BoundaryError
        try:
            b = doc["boundary"]
            pr.set_boundary(Polygon(np.asarray(ar.get(b["shell"]), float), [np.asarray(ar.get(h), float) for h in b["holes"]]))
        except (BoundaryError, KeyError, ValueError, TypeError) as exc:
            raise PersistError(f"boundary polygon: {exc}") from None
    if doc.get("grid_def") is not None:
        from .grid_definition import GridDefinition, GridDefinitionError
        try:
            pr.grid_def = GridDefinition.from_state(doc["grid_def"])
        except GridDefinitionError as exc:
            raise PersistError(str(exc)) from None
    pr.var_meta = {tuple(k): v for k, v in doc["var_meta"]}
    for (uid, var, lab) in pr.var_meta:
        if lab not in pr.variograms.get(uid, {}).get(var, {}):
            raise PersistError(f"how-it-was-made record for a variogram that does not exist: {var!r}/{lab!r}")
    return pr

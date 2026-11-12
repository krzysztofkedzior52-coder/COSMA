"""Polygon import for model boundaries.

Formats: delimited text (CSV / whitespace XYZ, IRAP/RMS-style ASCII polygons
with 999 separators), GeoJSON, ESRI shapefile (.shp, with .prj / .dbf when
present) and KML / KMZ.

Coordinates are never guessed. Formats that carry a coordinate system (GeoJSON,
KML, shapefile) require ``target_crs``: the model CRS the polygon is brought
into. Text files carry no CRS, so they are assumed to be in the model CRS
already, and a CRS pair is optional but must be complete if given.
"""

from __future__ import annotations

import json
import math
import re
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree.ElementTree import ParseError

import numpy as np
from shapely.geometry import Polygon

from ..grid.boundary import BoundaryError, validate_polygon

_TEXT_EXT = {".csv", ".txt", ".xyz", ".dat", ".pol", ".ascii", ".asc"}
_GEOJSON_EXT = {".geojson", ".json"}
_KML_EXT = {".kml", ".kmz"}
_SHP_EXT = {".shp"}
_SUPPORTED = "CSV/XYZ text, GeoJSON, shapefile (.shp) and KML/KMZ"

MAX_XML_BYTES = 50 * 1024 * 1024  # KML / KMZ payload limit (also bounds KMZ decompression)
_POLYGON_SHAPE_TYPES = (5, 15, 25)  # POLYGON, POLYGONZ, POLYGONM
_NULL_SHAPE = 0


@dataclass(frozen=True)
class PolygonRecord:
    """One polygon from a file, in the target CRS (validated, CCW)."""

    polygon: Polygon
    index: int  # position in the returned list: what ``which=<int>`` selects
    record: int | None  # record number inside the file where meaningful (shapefile)
    name: str | None  # KML Placemark name, GeoJSON/shapefile "name" attribute
    attributes: dict = field(default_factory=dict)


@dataclass
class _Raw:
    shell: np.ndarray
    holes: list[np.ndarray] = field(default_factory=list)
    name: str | None = None
    attributes: dict = field(default_factory=dict)
    record: int | None = None


@dataclass
class _Parsed:
    raws: list[_Raw]
    ignored: dict[str, int]  # geometry kind -> count skipped (not polygons)
    file_crs: object | None  # CRS of the stored coordinates, when the format states one
    carries_crs: bool  # True: target_crs is mandatory


# ---------------------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------------------
def read_polygon_records(path: str | Path, source_crs=None, target_crs=None) -> list[PolygonRecord]:
    """Read every polygon in ``path``, reprojected to ``target_crs``.

    ``source_crs`` / ``target_crs`` are anything pyproj accepts (4326,
    "EPSG:32616", WKT...). ``source_crs`` overrides the CRS stored in the file.
    Datum changes (e.g. NAD83 -> WGS84) raise a warning, because PROJ may apply
    a null or low-accuracy shift when no grid files are available.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in _TEXT_EXT:
        parsed = _parse_text(path)
    elif ext in _GEOJSON_EXT:
        parsed = _parse_geojson(path)
    elif ext in _KML_EXT:
        parsed = _parse_kml(path)
    elif ext in _SHP_EXT:
        parsed = _parse_shapefile(path)
    else:
        raise BoundaryError(f"unsupported polygon format {ext!r}; supported: {_SUPPORTED}")

    src = source_crs if source_crs is not None else parsed.file_crs
    if parsed.carries_crs:
        if src is None:
            raise BoundaryError(
                f"{path.name} has no coordinate system information (no .prj found); "
                "pass source_crs, e.g. source_crs='EPSG:32616'"
            )
        if target_crs is None:
            raise BoundaryError(
                f"{path.name} coordinates are {_describe_crs(src)}; pass target_crs (the model CRS, "
                "e.g. 'EPSG:32616'). If the model already uses this CRS, pass it as target_crs too."
            )
    else:
        if (source_crs is None) != (target_crs is None):
            raise BoundaryError("give both source_crs and target_crs, or neither")
        src = source_crs
    transform = _make_transform(src, target_crs)

    records: list[PolygonRecord] = []
    for n, raw in enumerate(parsed.raws, start=1):
        try:
            poly = _build(raw.shell, raw.holes, transform)
        except BoundaryError as e:
            raise BoundaryError(f"{path.name}, polygon {n}: {e}") from None
        records.append(PolygonRecord(poly, len(records), raw.record, raw.name, dict(raw.attributes)))
    if not records:
        detail = ""
        if parsed.ignored:
            detail = " (skipped " + ", ".join(f"{n} {k}" for k, n in parsed.ignored.items()) + "; only polygons are used)"
        raise BoundaryError(f"{path.name}: no polygons found{detail}")
    return records


def read_polygons(path: str | Path, source_crs=None, target_crs=None) -> list[Polygon]:
    """Like :func:`read_polygon_records` but returns just the polygons."""
    return [r.polygon for r in read_polygon_records(path, source_crs, target_crs)]


def read_boundary(path: str | Path, which: int | str | None = None, **crs) -> Polygon:
    """Read exactly one boundary polygon. A file with several polygons needs an
    explicit choice: ``which`` = 0-based position in the file's polygon list,
    ``"largest"``, or a name (KML Placemark name / "name" attribute, matched
    case-insensitively and required to be unique)."""
    recs = read_polygon_records(path, **crs)
    fname = Path(path).name
    if which is None:
        if len(recs) == 1:
            return recs[0].polygon
        raise BoundaryError(
            f"{fname} holds {len(recs)} polygons; choose one with which=<index>, which='largest' or which=<name>"
        )
    if isinstance(which, str):
        if which == "largest":
            return max(recs, key=lambda r: r.polygon.area).polygon
        hits = [r for r in recs if r.name is not None and r.name.casefold() == which.casefold()]
        if len(hits) == 1:
            return hits[0].polygon
        if not hits:
            names = sorted({r.name for r in recs if r.name})
            raise BoundaryError(f"no polygon named {which!r} in {fname}; names found: {names or 'none'}")
        raise BoundaryError(f"{len(hits)} polygons are named {which!r} in {fname}; use which=<index>")
    if isinstance(which, int) and not isinstance(which, bool) and 0 <= which < len(recs):
        return recs[which].polygon
    raise BoundaryError(f"which={which!r} is not valid for {len(recs)} polygons")


# ---------------------------------------------------------------------------------------
# CRS handling
# ---------------------------------------------------------------------------------------
def _crs(value):
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    try:
        return CRS.from_user_input(value)
    except CRSError as e:
        raise BoundaryError(f"unknown coordinate system {value!r}: {e}") from None


def _describe_crs(value) -> str:
    crs = _crs(value)
    epsg = crs.to_epsg()
    tag = f"EPSG:{epsg}" if epsg else "a custom CRS"
    return f"{'lon/lat in ' if crs.is_geographic else ''}{tag} ({crs.name})"


def _make_transform(source_crs, target_crs):
    if source_crs is None:
        return lambda a: np.asarray(a, dtype=float)
    from pyproj import Transformer
    from pyproj.exceptions import ProjError

    src, dst = _crs(source_crs), _crs(target_crs)
    t = Transformer.from_crs(src, dst, always_xy=True)
    identical = src == dst
    # Geographic CRS in degrees: compare the unit's radian factor, not its name
    # (EPSG says "degree", ESRI .prj files say "Degree").
    degrees = bool(
        src.is_geographic
        and src.axis_info
        and math.isclose(src.axis_info[0].unit_conversion_factor, math.pi / 180.0, rel_tol=1e-9)
    )
    warned = False

    def _datum(c) -> str:
        name = c.datum.name if c.datum is not None else ""
        return re.sub(r"\s*ensemble\s*$", "", name.casefold().replace("_", " ")).strip()

    def fn(a):
        nonlocal warned
        a = np.asarray(a, dtype=float)
        if degrees and a.size and (np.abs(a[:, 0]).max() > 180.0 or np.abs(a[:, 1]).max() > 90.0):
            raise BoundaryError(
                f"coordinates fall outside lon/lat limits (|lon| <= 180, |lat| <= 90) but the source CRS is "
                f"{_describe_crs(src)}; the data is probably projected: pass the right source_crs"
            )
        if identical:
            return a.copy()
        try:
            x, y = t.transform(a[:, 0], a[:, 1], errcheck=True)
        except ProjError as e:
            raise BoundaryError(f"coordinate transformation failed: {e}") from None
        if not warned:
            warned = True
            acc = t.accuracy  # known only after the first transform; -1 = unknown
            names_differ = _datum(src) != _datum(dst)
            inexact = acc is not None and acc > 0
            unknown = acc is None or acc < 0
            if inexact or (unknown and names_differ):
                shown = "unknown" if unknown else f"{acc:g} m"
                what = (
                    f"datum change {src.datum.name} -> {dst.datum.name}"
                    if names_differ
                    else "inexact coordinate transformation"
                )
                warnings.warn(
                    f"{what}: PROJ used '{t.description}' (stated accuracy {shown}); shifts of a few "
                    "metres are possible and no grid-shift files are applied",
                    UserWarning,
                    stacklevel=2,
                )
        return np.column_stack([x, y])

    return fn


def _build(shell: np.ndarray, holes: list[np.ndarray], transform) -> Polygon:
    shell_t = transform(shell)
    if not holes:
        return validate_polygon(shell_t)  # cleans duplicates and the closing vertex
    try:
        poly = Polygon(shell_t, [transform(h) for h in holes])
    except ValueError as e:  # e.g. a ring with fewer than 3 vertices
        raise BoundaryError(str(e)) from None
    return validate_polygon(poly)


# ---------------------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------------------
_SPLIT = re.compile(r"[,\s;]+")


def _parse_text(path: Path) -> _Parsed:
    rings: list[list[list[float]]] = []
    cur: list[list[float]] = []
    seen_data = False

    def flush():
        nonlocal cur
        if cur:
            rings.append(cur)
        cur = []

    for lineno, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        s = raw.strip()
        if not s:
            flush()
            continue
        if s.startswith(("#", "--", "//")):
            continue
        toks = [t for t in _SPLIT.split(s) if t]
        try:
            vals = [float(t) for t in toks]
        except ValueError:
            if not seen_data:  # a header row such as "x,y" or "X Y Z ID"
                continue
            raise BoundaryError(f"{path.name}, line {lineno}: cannot read coordinates from {s!r}") from None
        if len(vals) < 2:
            raise BoundaryError(f"{path.name}, line {lineno}: need at least x and y, got {s!r}")
        seen_data = True
        if all(v == 999.0 for v in vals[: min(3, len(vals))]):  # IRAP/RMS polygon separator
            flush()
            continue
        cur.append(vals[:2])
    flush()
    return _Parsed([_Raw(np.array(r, dtype=float)) for r in rings], {}, None, carries_crs=False)


# ---------------------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------------------
def _parse_geojson(path: Path) -> _Parsed:
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        raise BoundaryError(f"{path.name}: not valid JSON ({e})") from None
    raws: list[_Raw] = []
    ignored: dict[str, int] = {}

    def ring(coords) -> np.ndarray:
        return np.array([c[:2] for c in coords], dtype=float)

    def props_of(feature) -> tuple[str | None, dict]:
        p = feature.get("properties") if isinstance(feature, dict) else None
        p = dict(p) if isinstance(p, dict) else {}
        name = next((str(v) for k, v in p.items() if k.casefold() == "name" and v is not None), None)
        return name, p

    def visit(g, name=None, attrs=None) -> None:
        t = g.get("type") if isinstance(g, dict) else None
        if t == "FeatureCollection":
            for f in g.get("features", []):
                visit(f)
        elif t == "Feature":
            n, a = props_of(g)
            if g.get("geometry"):
                visit(g["geometry"], n, a)
        elif t == "GeometryCollection":
            for sub in g.get("geometries", []):
                visit(sub, name, attrs)
        elif t == "Polygon":
            rings = g["coordinates"]
            raws.append(_Raw(ring(rings[0]), [ring(r) for r in rings[1:]], name, attrs or {}))
        elif t == "MultiPolygon":
            for rings in g["coordinates"]:
                raws.append(_Raw(ring(rings[0]), [ring(r) for r in rings[1:]], name, attrs or {}))
        elif t:
            ignored[t] = ignored.get(t, 0) + 1

    visit(obj)
    # RFC 7946 is always WGS84 lon/lat; older files may state another CRS in a "crs" member.
    crs = 4326
    member = obj.get("crs") if isinstance(obj, dict) else None
    if isinstance(member, dict) and isinstance(member.get("properties"), dict):
        stated = member["properties"].get("name")
        if stated:
            crs = _crs(stated)
    return _Parsed(raws, ignored, crs, carries_crs=True)


# ---------------------------------------------------------------------------------------
# KML / KMZ
# ---------------------------------------------------------------------------------------
def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _read_kml_bytes(path: Path) -> bytes:
    if path.suffix.lower() == ".kmz":
        try:
            zf = zipfile.ZipFile(path)
        except zipfile.BadZipFile:
            raise BoundaryError(f"{path.name}: not a valid KMZ (zip) file") from None
        with zf:
            kmls = [i for i in zf.infolist() if not i.is_dir() and i.filename.lower().endswith(".kml")]
            if not kmls:
                raise BoundaryError(f"{path.name}: no .kml file inside the KMZ")
            info = next((i for i in kmls if i.filename.lower() == "doc.kml"), kmls[0])
            if info.file_size > MAX_XML_BYTES:
                raise BoundaryError(f"{path.name}: KML inside the KMZ is larger than {MAX_XML_BYTES // 2**20} MB")
            with zf.open(info) as f:
                data = f.read(MAX_XML_BYTES + 1)  # the size in the header can lie
        if len(data) > MAX_XML_BYTES:
            raise BoundaryError(f"{path.name}: KML inside the KMZ is larger than {MAX_XML_BYTES // 2**20} MB")
        return data
    if path.stat().st_size > MAX_XML_BYTES:
        raise BoundaryError(f"{path.name}: file is larger than {MAX_XML_BYTES // 2**20} MB")
    return path.read_bytes()


def _kml_coords(text: str | None, where: str) -> np.ndarray:
    pts = []
    # tolerate "lon, lat, alt" with spaces after commas
    for tok in re.sub(r"\s*,\s*", ",", (text or "").strip()).split():
        parts = tok.split(",")
        try:
            if len(parts) < 2:
                raise ValueError
            pts.append((float(parts[0]), float(parts[1])))  # KML order is lon, lat[, alt]
        except ValueError:
            raise BoundaryError(f"{where}: cannot read KML coordinate {tok!r}") from None
    if not pts:
        raise BoundaryError(f"{where}: empty <coordinates>")
    return np.array(pts, dtype=float)


def _parse_kml(path: Path) -> _Parsed:
    from defusedxml import ElementTree as SafeET
    from defusedxml.common import DefusedXmlException

    data = _read_kml_bytes(path)
    try:
        root = SafeET.fromstring(data, forbid_dtd=True)
    except DefusedXmlException as e:
        raise BoundaryError(f"{path.name}: refused for safety ({type(e).__name__}); KML must not contain a DTD or entities") from None
    except ParseError as e:
        raise BoundaryError(f"{path.name}: not a readable KML file ({e})") from None

    raws: list[_Raw] = []
    ignored: dict[str, int] = {}
    for pm in root.iter():
        if _local(pm.tag) != "Placemark":
            continue
        name = next((e.text.strip() for e in pm if _local(e.tag) == "name" and e.text and e.text.strip()), None)
        for el in pm.iter():
            kind = _local(el.tag)
            if kind == "Polygon":
                where = f"{path.name}, Placemark {name!r}" if name else path.name
                outer = [c for b in el if _local(b.tag) == "outerBoundaryIs" for c in b.iter() if _local(c.tag) == "coordinates"]
                if not outer:
                    raise BoundaryError(f"{where}: Polygon without an outer boundary")
                inner = [c for b in el if _local(b.tag) == "innerBoundaryIs" for c in b.iter() if _local(c.tag) == "coordinates"]
                raws.append(
                    _Raw(
                        _kml_coords(outer[0].text, where),
                        [_kml_coords(c.text, where) for c in inner],
                        name,
                    )
                )
            elif kind in ("LineString", "Point", "Track"):
                ignored[kind] = ignored.get(kind, 0) + 1
    return _Parsed(raws, ignored, 4326, carries_crs=True)  # KML coordinates are WGS84 lon/lat by definition


# ---------------------------------------------------------------------------------------
# shapefile
# ---------------------------------------------------------------------------------------
def _sibling(path: Path, suffix: str) -> Path | None:
    for s in (suffix, suffix.upper()):
        p = path.with_suffix(s)
        if p.is_file():
            return p
    return None


def _shoelace(r: np.ndarray) -> float:
    x, y = r[:, 0], r[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _group_rings(rings: list[np.ndarray]) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Assign rings to (shell, holes) by nesting depth, ignoring orientation.

    The shapefile spec says outer rings are clockwise and holes counter-
    clockwise, but plenty of writers get this wrong. Nesting is unambiguous:
    a ring inside an even number of larger rings is a shell, inside an odd
    number it is a hole of its smallest container (so an island inside a hole
    is a shell again).
    """
    items = []
    for r in rings:
        try:
            poly = Polygon(r)
            rep = poly.representative_point() if len(r) >= 3 and poly.is_valid else None
        except ValueError:  # fewer than 3 vertices: reported later by validation
            poly, rep = None, None
        items.append((_shoelace(r) if len(r) >= 3 else 0.0, r, poly, rep))
    items.sort(key=lambda t: -t[0])
    shells: list[tuple[np.ndarray, list[np.ndarray]]] = []
    shell_of: dict[int, int] = {}  # item index -> index in shells
    for k, (_, r, poly, rep) in enumerate(items):
        parents = [j for j in range(k) if rep is not None and items[j][2] is not None and items[j][2].contains(rep)]
        if len(parents) % 2 == 0:
            shell_of[k] = len(shells)
            shells.append((r, []))
        else:
            shells[shell_of[parents[-1]]][1].append(r)
    return shells


def _parse_shapefile(path: Path) -> _Parsed:
    try:
        import shapefile  # pyshp
    except ImportError as e:  # pragma: no cover
        raise BoundaryError("reading shapefiles needs pyshp (pip install pyshp)") from e

    file_crs = None
    prj = _sibling(path, ".prj")
    if prj is not None:
        from pyproj import CRS
        from pyproj.exceptions import CRSError

        try:
            file_crs = CRS.from_wkt(prj.read_text(encoding="utf-8", errors="replace"))
        except CRSError as e:
            raise BoundaryError(f"{prj.name}: cannot interpret the coordinate system ({e}); pass source_crs") from None

    raws: list[_Raw] = []
    ignored: dict[str, int] = {}
    try:
        reader = shapefile.Reader(str(path), encoding="utf-8", encodingErrors="replace")
    except Exception as e:  # pyshp raises several unrelated types on damaged input
        raise BoundaryError(f"{path.name}: cannot read shapefile ({type(e).__name__}: {e})") from None
    try:
        with reader:
            if reader.shapeType not in _POLYGON_SHAPE_TYPES and reader.shapeType != _NULL_SHAPE:
                ignored[reader.shapeTypeName] = len(reader)
                return _Parsed(raws, ignored, file_crs, carries_crs=True)
            try:
                records = [r.as_dict() for r in reader.records()]
            except Exception:  # missing or unreadable .dbf: attributes are optional
                records = []
            for i, shp in enumerate(reader.iterShapes()):
                if shp.shapeType == _NULL_SHAPE:
                    continue
                pts = np.asarray(shp.points, dtype=float).reshape(-1, 2) if len(shp.points) else np.empty((0, 2))
                bounds = list(shp.parts) + [len(pts)]
                rings = [pts[bounds[k] : bounds[k + 1]] for k in range(len(shp.parts))]
                attrs = records[i] if i < len(records) else {}
                name = next((str(v).strip() or None for k, v in attrs.items() if k.casefold() == "name" and v is not None), None)
                for shell, holes in _group_rings(rings):
                    raws.append(_Raw(shell, holes, name, attrs, record=i))
    except BoundaryError:
        raise
    except Exception as e:
        raise BoundaryError(f"{path.name}: cannot read shapefile ({type(e).__name__}: {e})") from None
    return _Parsed(raws, ignored, file_crs, carries_crs=True)

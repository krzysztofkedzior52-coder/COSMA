"""Built-in list of coordinate reference systems and suggestions for a polygon file.

The list is a curated set of EPSG codes (the datums and projections used for CO2 storage and oil-and-gas work in
North America, Europe, Australia, South America, plus the world UTM zones). Names and areas of use come from the
EPSG data bundled with pyproj, so the list works offline. Anything else can still be typed as an EPSG code or WKT.

Suggestions never claim more than the data supports: each one carries the reason it was made, and a CRS is only
preselected when the file or the project states it. Numbers alone cannot name a UTM zone, and the datum of lon/lat
values is not in the numbers either; those cases give a short list with that said, not a guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

GEOGRAPHIC = (4326, 4269, 4267, 4258, 4230, 4283, 7844, 4674, 4277)
NATIONAL = (3857, 27700, 28992, 2154, 3035, 31467, 3067, 3006, 3347, 5070, 3395)
UTM_FAMILIES = {                       # name: (first EPSG code, zones, southern first code or None)
    "WGS 84 / UTM": (32601, range(1, 61), 32701),
    "NAD83 / UTM": (26900, range(1, 24), None),
    "NAD27 / UTM": (26700, range(3, 23), None),
    "ETRS89 / UTM": (25800, range(28, 39), None),
    "ED50 / UTM": (23000, range(28, 39), None),
    "GDA94 / MGA": (28300, range(48, 59), None),
    "GDA2020 / MGA": (7800, range(46, 60), None),
}


@dataclass(frozen=True)
class Entry:
    code: str            # "EPSG:32616"
    name: str
    geographic: bool
    bounds: tuple | None  # (west, south, east, north) of the area of use, degrees

    @property
    def label(self) -> str:
        return f"{self.code} - {self.name}"


@dataclass(frozen=True)
class Suggestion:
    code: str
    label: str
    reason: str
    strong: bool = False     # True: stated by the file or the project, safe to prefill; False: listed for the user to choose


def _codes():
    out = list(GEOGRAPHIC) + list(NATIONAL)
    for name, (first, zones, south) in UTM_FAMILIES.items():
        for z in zones:
            if name == "WGS 84 / UTM":
                out += [32600 + z, 32700 + z]
            elif name == "NAD83 / UTM":
                out.append(26900 + z)
            elif name == "NAD27 / UTM":
                out.append(26700 + z)
            elif name == "ETRS89 / UTM":
                out.append(25800 + z)
            elif name == "ED50 / UTM":
                out.append(23000 + z)
            elif name == "GDA94 / MGA":
                out.append(28300 + z)
            elif name == "GDA2020 / MGA":
                out.append(7800 + z)
    return list(dict.fromkeys(out))


@lru_cache(maxsize=1)
def catalog() -> tuple:
    from pyproj import CRS
    out = []
    for c in _codes():
        try:
            crs = CRS.from_epsg(c)
        except Exception:
            continue                                  # a code the bundled EPSG database does not have is left out, not faked
        a = crs.area_of_use
        out.append(Entry(f"EPSG:{c}", crs.name, crs.is_geographic, tuple(a.bounds) if a else None))
    return tuple(out)


def entry(code: str):
    return next((e for e in catalog() if e.code == code), None)


def label_of(code: str) -> str:
    e = entry(code)
    if e:
        return e.label
    try:
        from pyproj import CRS
        c = CRS.from_user_input(code)
        return f"{('EPSG:%d' % c.to_epsg()) if c.to_epsg() else 'custom'} - {c.name}"
    except Exception:
        return str(code)


def normalise(text) -> str | None:
    """The CRS as a canonical 'EPSG:n' (or WKT when it has no code); None for blank. Raises ValueError if unknown."""
    if text is None or not str(text).strip():
        return None
    from pyproj import CRS
    from pyproj.exceptions import CRSError
    t = str(text).split("   [suggested]")[0].strip()
    if " - " in t and t.upper().startswith("EPSG:"):          # a label picked from the list
        t = t.split(" - ", 1)[0].strip()
    try:
        c = CRS.from_user_input(int(t) if t.isdigit() else t)
    except (CRSError, ValueError) as exc:
        raise ValueError(f"{text!r} is not a coordinate system pyproj/EPSG knows ({exc})") from None
    e = c.to_epsg()
    return f"EPSG:{e}" if e else c.to_wkt()


def _datum_family(crs) -> str:
    """'wgs84' | 'nad83' | 'etrs89' | 'gda2020' | 'gda94' | 'other', from the CRS's datum."""
    d = (crs.datum.name if crs.datum else crs.name).lower()
    for key, fam in (("world geodetic system 1984", "wgs84"), ("north american datum 1983", "nad83"),
                     ("european terrestrial reference system 1989", "etrs89"), ("geocentric datum of australia 2020", "gda2020"),
                     ("geocentric datum of australia 1994", "gda94")):
        if key in d:
            return fam
    return "other"


def utm_zone(lon: float, lat: float) -> tuple:
    zone = int((lon + 180.0) // 6.0) % 60 + 1
    return zone, ("N" if lat >= 0 else "S")


def _utm_code(family: str, zone: int, hemi: str):
    if family == "nad83" and hemi == "N" and 1 <= zone <= 23:
        return 26900 + zone
    if family == "etrs89" and hemi == "N" and 28 <= zone <= 38:
        return 25800 + zone
    if family == "gda2020" and hemi == "S" and 46 <= zone <= 59:
        return 7800 + zone
    if family == "gda94" and hemi == "S" and 48 <= zone <= 58:
        return 28300 + zone
    return (32600 if hemi == "N" else 32700) + zone


def lonlat_of(xy, source_crs):
    """(lon, lat) arrays of the coordinates in `source_crs` (a CRS-like), or None if not possible."""
    import numpy as np
    from pyproj import CRS, Transformer
    try:
        src = CRS.from_user_input(source_crs)
        t = Transformer.from_crs(src, 4326, always_xy=True)
        lon, lat = t.transform(xy[:, 0], xy[:, 1])
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        return (lon, lat) if np.isfinite(lon).all() and np.isfinite(lat).all() else None
    except Exception:
        return None


def looks_geographic(xy) -> bool:
    return bool((abs(xy[:, 0]) <= 180).all() and (abs(xy[:, 1]) <= 90).all())


def source_suggestions(xy, file_crs, project_crs=None) -> tuple:
    """(suggestions, note) for the CRS the polygon coordinates are in, when the file does not say."""
    out, note = [], ""
    if file_crs is not None:
        return out, note
    if looks_geographic(xy):
        lon, lat = float(xy[:, 0].mean()), float(xy[:, 1].mean())
        out.append(Suggestion("EPSG:4326", label_of("EPSG:4326"),
                              "the values look like longitude/latitude; WGS 84 is the usual datum but the file does not state it"))
        if -170 <= lon <= -50 and 15 <= lat <= 75:
            out.append(Suggestion("EPSG:4269", label_of("EPSG:4269"), "also plausible in North America (NAD83)"))
        elif -12 <= lon <= 35 and 34 <= lat <= 72:
            out.append(Suggestion("EPSG:4258", label_of("EPSG:4258"), "also plausible in Europe (ETRS89)"))
        elif 110 <= lon <= 160 and -50 <= lat <= -8:
            out.append(Suggestion("EPSG:7844", label_of("EPSG:7844"), "also plausible in Australia (GDA2020)"))
        note = "Modern datums differ by about a metre or two; older ones (NAD27, ED50) by tens to hundreds of metres."
    else:
        pc = project_crs
        if pc:
            ll = lonlat_of(xy, pc)
            e = entry(pc)
            from pyproj import CRS
            ok = False
            if ll is not None:
                a = CRS.from_user_input(pc).area_of_use
                ok = a is None or (a.west <= ll[0].min() and ll[0].max() <= a.east and a.south <= ll[1].min() and ll[1].max() <= a.north)
            if ok:
                out.append(Suggestion(pc, label_of(pc), "the project's CRS, and these coordinates fall inside its area of use (other zones would too: check)", False))
        if not out:
            note = ("The values are projected coordinates. A zone or projection cannot be read from the numbers: "
                    "choose the CRS the data was made in.")
    return out, note


def target_suggestions(xy, source_crs, project_crs=None) -> list:
    """Suggestions for the model CRS (what the polygon is brought into), given the polygon's own CRS."""
    from pyproj import CRS
    out = []
    if project_crs:
        out.append(Suggestion(project_crs, label_of(project_crs), "the project's CRS (set by an earlier import)", True))
    if source_crs is None:
        return out
    try:
        src = CRS.from_user_input(source_crs)
    except Exception:
        return out
    if not src.is_geographic:
        if not out or out[0].code != normalise(source_crs):
            out.append(Suggestion(normalise(source_crs), label_of(normalise(source_crs)),
                                  "the file's own projected CRS: no reprojection needed", not out))
        return out
    ll = lonlat_of(xy, source_crs)
    if ll is None:
        return out
    lon, lat = ll
    z0, h0 = utm_zone(float(lon.mean()), float(lat.mean()))
    zs = {utm_zone(float(a), float(b)) for a, b in ((lon.min(), lat.min()), (lon.max(), lat.max()), (lon.min(), lat.max()), (lon.max(), lat.min()))}
    fam = _datum_family(src)
    code = _utm_code(fam, z0, h0)
    seen = {s.code for s in out}
    extra = "" if len(zs) == 1 else f" (the polygon touches {len(zs)} zones; the zone of its centre is suggested)"
    reasons = [f"UTM zone {z0}{h0} covers the polygon's centre ({float(lon.mean()):.3f}, {float(lat.mean()):.3f}){extra}"]
    cands = [(code, reasons[0] + ("; same datum as the file" if fam != "other" and code not in (32600 + z0, 32700 + z0) else ""))]
    wgs = (32600 if h0 == "N" else 32700) + z0
    if wgs != code:
        cands.append((wgs, "the same zone on WGS 84"))
    for c, why in cands:
        k = f"EPSG:{c}"
        if k not in seen:
            out.append(Suggestion(k, label_of(k), why, False))
            seen.add(k)
    return out

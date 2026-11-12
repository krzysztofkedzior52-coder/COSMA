"""Shapefile and KML/KMZ polygon import.

Files are written by the tests with an independent writer (pyshp for shapefiles,
literal XML for KML). Areas and positions are hand-computed from squares in UTM
metres; the one external CRS anchor is that lon -87, lat 0 is exactly easting
500000, northing 0 in UTM zone 16N (the zone's central meridian).
"""

from __future__ import annotations

import warnings
import zipfile

import numpy as np
import pytest

shapefile = pytest.importorskip("shapefile")
from pyproj import CRS, Transformer  # noqa: E402

import cosma.io.polygons as polygons  # noqa: E402
from cosma.grid.boundary import BoundaryError, grid_spec_from_polygon  # noqa: E402
from cosma.io.polygons import read_boundary, read_polygon_records, read_polygons  # noqa: E402

X0, Y0 = 683450.0, 4701234.0


def sq(x, y, size):
    return [(x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y)]  # CCW, closed


def cw(ring):
    return ring if _signed_area(ring) < 0 else ring[::-1]


def ccw(ring):
    return ring if _signed_area(ring) > 0 else ring[::-1]


def _signed_area(r):
    a = np.asarray(r, float)
    return 0.5 * float(np.dot(a[:, 0], np.roll(a[:, 1], -1)) - np.dot(a[:, 1], np.roll(a[:, 0], -1)))


def write_shp(base, rings_per_record, names=None, shape_type=None, prj_epsg=None, prj_wkt=None, extra_fields=True):
    """rings_per_record: list of records, each a list of rings (exactly as written)."""
    shape_type = shape_type or shapefile.POLYGON
    with shapefile.Writer(str(base), shapeType=shape_type) as w:
        w.field("NAME", "C", 40)
        w.field("ID", "N", 10)
        for n, rings in enumerate(rings_per_record):
            if rings is None:
                w.null()
            elif shape_type == shapefile.POLYLINE:
                w.line(rings)
            elif shape_type == shapefile.POLYGONZ:
                w.polyz(rings)
            else:
                w.poly(rings)
            w.record((names[n] if names else f"rec{n}"), n + 1)
    if prj_epsg is not None:
        base.with_suffix(".prj").write_text(CRS.from_epsg(prj_epsg).to_wkt(version="WKT1_ESRI"))
    if prj_wkt is not None:
        base.with_suffix(".prj").write_text(prj_wkt)
    return base.with_suffix(".shp")


# ============================== shapefile ====================================================
def test_projected_shapefile_with_esri_prj_passes_through(tmp_path):
    p = write_shp(tmp_path / "aoi", [[cw(sq(X0, Y0, 1000))]], prj_epsg=32616)
    (poly,) = read_polygons(p, target_crs="EPSG:32616")
    assert poly.area == pytest.approx(1.0e6, rel=1e-9)
    assert poly.exterior.is_ccw
    xy = np.array(poly.exterior.coords)
    assert (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()) == pytest.approx((X0, Y0, X0 + 1000, Y0 + 1000), abs=1e-6)


def test_missing_prj_needs_source_crs(tmp_path):
    p = write_shp(tmp_path / "noprj", [[cw(sq(X0, Y0, 1000))]])
    with pytest.raises(BoundaryError, match="no coordinate system information"):
        read_polygons(p, target_crs=32616)
    assert read_polygons(p, source_crs=32616, target_crs=32616)[0].area == pytest.approx(1.0e6)


def test_prj_present_but_no_target_names_the_crs(tmp_path):
    p = write_shp(tmp_path / "t", [[cw(sq(X0, Y0, 1000))]], prj_epsg=32616)
    with pytest.raises(BoundaryError, match=r"EPSG:32616.*pass target_crs"):
        read_polygons(p)


def test_lonlat_shapefile_hits_utm_anchor(tmp_path):
    ring = cw([(-87.0, 0.0), (-86.99, 0.0), (-86.99, 0.01), (-87.0, 0.01), (-87.0, 0.0)])
    p = write_shp(tmp_path / "ll", [[ring]], prj_epsg=4326)
    (poly,) = read_polygons(p, target_crs="EPSG:32616")
    xy = np.array(poly.exterior.coords)
    assert np.hypot(xy[:, 0] - 500000.0, xy[:, 1]).min() < 1e-3


def test_wrong_prj_is_caught_by_range_guard_and_can_be_overridden(tmp_path):
    # UTM coordinates but a .prj claiming WGS84 lon/lat (a common data-prep mistake)
    p = write_shp(tmp_path / "wrong", [[cw(sq(X0, Y0, 1000))]], prj_epsg=4326)
    with pytest.raises(BoundaryError, match="outside lon/lat limits"):
        read_polygons(p, target_crs=32616)
    (poly,) = read_polygons(p, source_crs=32616, target_crs=32616)  # explicit CRS wins over .prj
    assert poly.area == pytest.approx(1.0e6)


def test_garbage_prj_is_reported(tmp_path):
    p = write_shp(tmp_path / "g", [[cw(sq(X0, Y0, 1000))]], prj_wkt="this is not a CRS")
    with pytest.raises(BoundaryError, match=r"g\.prj: cannot interpret"):
        read_polygons(p, target_crs=32616)


def test_datum_change_warns_but_same_datum_does_not(tmp_path):
    nad83 = write_shp(tmp_path / "nad83", [[cw(sq(X0, Y0, 1000))]], prj_epsg=26916)
    with pytest.warns(UserWarning, match="datum change"):
        (poly,) = read_polygons(nad83, target_crs="EPSG:32616")
    # whatever shift PROJ applies, it must be small (metres), never kilometres
    assert np.array(poly.exterior.coords)[:, 0].min() == pytest.approx(X0, abs=5.0)
    # WGS84 lon/lat -> WGS84 UTM is a projection change only: no warning
    ring = cw([(-87.0, 0.0), (-86.99, 0.0), (-86.99, 0.01), (-87.0, 0.01), (-87.0, 0.0)])
    ll = write_shp(tmp_path / "ll", [[ring]], prj_epsg=4326)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        read_polygons(ll, target_crs="EPSG:32616")


@pytest.mark.parametrize("orient", [cw, ccw], ids=["spec_cw_outer", "nonspec_ccw_outer"])
def test_outer_ring_orientation_does_not_matter(tmp_path, orient):
    p = write_shp(tmp_path / "o", [[orient(sq(X0, Y0, 1000))]], prj_epsg=32616)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (poly,) = read_polygons(p, target_crs=32616)
    assert poly.area == pytest.approx(1.0e6, rel=1e-9) and not poly.interiors


@pytest.mark.parametrize("outer_orient, hole_orient", [(cw, ccw), (ccw, cw), (ccw, ccw), (cw, cw)])
def test_hole_found_by_nesting_whatever_the_orientation(tmp_path, outer_orient, hole_orient):
    outer, hole = outer_orient(sq(X0, Y0, 1000)), hole_orient(sq(X0 + 400, Y0 + 400, 200))
    p = write_shp(tmp_path / "h", [[outer, hole]], prj_epsg=32616)
    (poly,) = read_polygons(p, target_crs=32616)
    assert len(poly.interiors) == 1
    assert poly.area == pytest.approx(1000 * 1000 - 200 * 200, rel=1e-9)


def test_island_inside_hole_is_its_own_polygon(tmp_path):
    outer = cw(sq(X0, Y0, 1000))
    hole = ccw(sq(X0 + 200, Y0 + 200, 600))
    island = cw(sq(X0 + 400, Y0 + 400, 200))
    p = write_shp(tmp_path / "i", [[outer, hole, island]], prj_epsg=32616)
    polys = read_polygons(p, target_crs=32616)
    assert sorted(round(q.area) for q in polys) == [40000, 1000 * 1000 - 600 * 600]


def test_multipart_record_gives_one_polygon_per_part(tmp_path):
    p = write_shp(tmp_path / "m", [[cw(sq(X0, Y0, 1000)), cw(sq(X0 + 5000, Y0, 500))]], names=["Two parts"], prj_epsg=32616)
    recs = read_polygon_records(p, target_crs=32616)
    assert sorted(round(r.polygon.area) for r in recs) == [250000, 1000000]
    assert {r.name for r in recs} == {"Two parts"} and {r.record for r in recs} == {0}
    assert [r.index for r in recs] == [0, 1]


def test_names_attributes_and_selection(tmp_path):
    p = write_shp(
        tmp_path / "multi",
        [[cw(sq(X0, Y0, 1000))], [cw(sq(X0 + 5000, Y0, 500))], [cw(sq(X0 + 9000, Y0, 300))]],
        names=["Area A", "Area B", "Area B"],
        prj_epsg=32616,
    )
    recs = read_polygon_records(p, target_crs=32616)
    assert [r.attributes["ID"] for r in recs] == [1, 2, 3]
    assert [r.attributes["NAME"] for r in recs] == ["Area A", "Area B", "Area B"]
    assert read_boundary(p, which="area a", target_crs=32616).area == pytest.approx(1.0e6)  # case-insensitive
    assert read_boundary(p, which="largest", target_crs=32616).area == pytest.approx(1.0e6)
    assert read_boundary(p, which=1, target_crs=32616).area == pytest.approx(250000)
    with pytest.raises(BoundaryError, match="2 polygons are named 'Area B'"):
        read_boundary(p, which="Area B", target_crs=32616)
    with pytest.raises(BoundaryError, match=r"no polygon named 'Zed'.*Area A"):
        read_boundary(p, which="Zed", target_crs=32616)
    with pytest.raises(BoundaryError, match="holds 3 polygons"):
        read_boundary(p, target_crs=32616)


def test_null_shape_is_skipped_and_record_number_kept(tmp_path):
    p = write_shp(tmp_path / "n", [None, [cw(sq(X0, Y0, 1000))]], names=["gone", "kept"], prj_epsg=32616)
    (rec,) = read_polygon_records(p, target_crs=32616)
    assert (rec.record, rec.name) == (1, "kept")


def test_polygonz_is_read(tmp_path):
    ring = [(x, y, 1500.0) for x, y in cw(sq(X0, Y0, 1000))]
    p = write_shp(tmp_path / "z", [[ring]], shape_type=shapefile.POLYGONZ, prj_epsg=32616)
    assert read_polygons(p, target_crs=32616)[0].area == pytest.approx(1.0e6)


def test_missing_dbf_still_reads_geometry(tmp_path):
    p = write_shp(tmp_path / "nodbf", [[cw(sq(X0, Y0, 1000))]], prj_epsg=32616)
    p.with_suffix(".dbf").unlink()
    (rec,) = read_polygon_records(p, target_crs=32616)
    assert rec.attributes == {} and rec.name is None and rec.polygon.area == pytest.approx(1.0e6)


def test_polyline_shapefile_is_rejected_with_a_hint(tmp_path):
    p = write_shp(tmp_path / "lines", [[[(0, 0), (1, 1)]], [[(2, 2), (3, 3)]]], shape_type=shapefile.POLYLINE, prj_epsg=32616)
    with pytest.raises(BoundaryError, match=r"no polygons found \(skipped 2 POLYLINE"):
        read_polygons(p, target_crs=32616)


@pytest.mark.filterwarnings("ignore:Declared file size")  # pyshp's own note about the corrupt header
def test_damaged_shapefile(tmp_path):
    p = tmp_path / "bad.shp"
    p.write_bytes(b"\x00" * 7 + b"not a shapefile at all" * 20)
    with pytest.raises(BoundaryError, match=r"bad\.shp: cannot read shapefile"):
        read_polygons(p, source_crs=32616, target_crs=32616)


def test_uppercase_extensions(tmp_path):
    p = write_shp(tmp_path / "up", [[cw(sq(X0, Y0, 1000))]], prj_epsg=32616)
    for ext in (".shp", ".dbf", ".shx", ".prj"):
        p.with_suffix(ext).rename(tmp_path / ("UP" + ext.upper()))
    assert read_polygons(tmp_path / "UP.SHP", target_crs=32616)[0].area == pytest.approx(1.0e6)


def test_shapefile_lonlat_to_grid_end_to_end(tmp_path):
    """A lon/lat shapefile around a Michigan-sized area, projected, then fitted."""
    back = Transformer.from_crs(32616, 4326, always_xy=True)
    ring_utm = [(X0, Y0), (X0 + 12000, Y0 + 1500), (X0 + 14000, Y0 + 9000), (X0 + 3000, Y0 + 11000)]
    lon, lat = back.transform(*np.array(ring_utm).T)
    p = write_shp(tmp_path / "mi", [[cw(list(zip(lon, lat)) + [(lon[0], lat[0])])]], prj_epsg=4326)
    poly = read_boundary(p, target_crs="EPSG:32616")
    got = np.array(poly.exterior.coords)[:-1]
    want = np.array(ring_utm)
    # same four corners (order may differ): each wanted corner has a match within 1 mm
    assert max(np.hypot(*(got - c).T).min() for c in want) < 1e-3
    spec = grid_spec_from_polygon(poly, 250.0, 250.0)
    assert abs(spec.area_ratio - 1.0) < 0.05 and -45.0 <= spec.frame.azimuth_deg < 45.0


# ============================== KML / KMZ ====================================================
NS22 = "http://www.opengis.net/kml/2.2"
_to_ll = Transformer.from_crs(32616, 4326, always_xy=True)


def lonlat(ring_utm):
    x, y = _to_ll.transform(*np.asarray(ring_utm, float).T)
    return list(zip(x.tolist(), y.tolist()))


def coords(ring, alt=True, sep=" "):
    return sep.join(f"{lo!r},{la!r}" + (",0" if alt else "") for lo, la in ring)


def polygon_xml(outer, inners=()):
    s = f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords(outer)}</coordinates></LinearRing></outerBoundaryIs>"
    for i in inners:
        s += f"<innerBoundaryIs><LinearRing><coordinates>{coords(i)}</coordinates></LinearRing></innerBoundaryIs>"
    return s + "</Polygon>"


def placemark(name, geometry):
    return f"<Placemark>{'<name>' + name + '</name>' if name else ''}{geometry}</Placemark>"


def kml(body, ns=NS22, decl='<?xml version="1.0" encoding="UTF-8"?>'):
    root = f'<kml xmlns="{ns}">' if ns else "<kml>"
    return f"{decl}\n{root}<Document>{body}</Document></kml>"


def write_kml(tmp_path, text, name="a.kml", encoding="utf-8"):
    p = tmp_path / name
    p.write_bytes(text.encode(encoding))
    return p


def test_kml_roundtrip_positions_and_anchor(tmp_path):
    p = write_kml(tmp_path, kml(placemark("AOI", polygon_xml(lonlat(sq(X0, Y0, 1000))))))
    (poly,) = read_polygons(p, target_crs="EPSG:32616")
    got = np.array(poly.exterior.coords)[:-1]
    for c in sq(X0, Y0, 1000)[:-1]:
        assert np.hypot(*(got - c).T).min() < 1e-3
    assert poly.area == pytest.approx(1.0e6, rel=1e-8)
    # independent anchor: lon -87 / lat 0 is exactly (500000, 0) in UTM 16N
    ring = [(-87.0, 0.0), (-86.99, 0.0), (-86.99, 0.01), (-87.0, 0.01), (-87.0, 0.0)]
    a = write_kml(tmp_path, kml(placemark("anchor", polygon_xml(ring))), "anchor.kml")
    xy = np.array(read_polygons(a, target_crs="EPSG:32616")[0].exterior.coords)
    assert np.hypot(xy[:, 0] - 500000.0, xy[:, 1]).min() < 1e-3


def test_kml_requires_target_crs_and_says_why(tmp_path):
    p = write_kml(tmp_path, kml(placemark("x", polygon_xml(lonlat(sq(X0, Y0, 1000))))))
    with pytest.raises(BoundaryError, match="lon/lat in EPSG:4326.*pass target_crs"):
        read_polygons(p)


@pytest.mark.parametrize("ns", ["http://earth.google.com/kml/2.1", "http://earth.google.com/kml/2.0", NS22, None])
def test_kml_namespace_variants(tmp_path, ns):
    p = write_kml(tmp_path, kml(placemark("x", polygon_xml(lonlat(sq(X0, Y0, 1000)))), ns=ns))
    assert read_polygons(p, target_crs=32616)[0].area == pytest.approx(1.0e6, rel=1e-8)


def test_kml_hole_multigeometry_folders_and_names(tmp_path):
    a = polygon_xml(lonlat(sq(X0, Y0, 1000)), [lonlat(sq(X0 + 400, Y0 + 400, 200))])
    b = polygon_xml(lonlat(sq(X0 + 5000, Y0, 500)))
    c = polygon_xml(lonlat(sq(X0 + 9000, Y0, 300)))
    body = (
        "<Folder><name>f</name>"
        + placemark("Lease", f"<MultiGeometry>{a}{b}</MultiGeometry>")
        + "<Folder>" + placemark("Other", c) + "</Folder></Folder>"
    )
    p = write_kml(tmp_path, kml(body))
    recs = read_polygon_records(p, target_crs=32616)
    assert [r.name for r in recs] == ["Lease", "Lease", "Other"]
    assert sorted(round(r.polygon.area) for r in recs) == [90000, 250000, 1000 * 1000 - 200 * 200]
    assert read_boundary(p, which="other", target_crs=32616).area == pytest.approx(90000, rel=1e-6)
    assert len(next(r for r in recs if r.name == "Lease" and r.polygon.interiors).polygon.interiors) == 1


def test_kml_coordinate_formatting_variants(tmp_path):
    ring = lonlat(sq(X0, Y0, 1000))
    messy = "\n\t  " + "\n\t  ".join(f"{lo!r}, {la!r}, 12.5" for lo, la in ring) + "\n"
    xml = kml(f"<Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>{messy}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>")
    assert read_polygons(write_kml(tmp_path, xml), target_crs=32616)[0].area == pytest.approx(1.0e6, rel=1e-8)
    no_alt = kml(placemark(None, f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords(ring, alt=False)}</coordinates></LinearRing></outerBoundaryIs></Polygon>"))
    assert read_polygons(write_kml(tmp_path, no_alt, "b.kml"), target_crs=32616)[0].area == pytest.approx(1.0e6, rel=1e-8)


def test_kml_latin1_name(tmp_path):
    xml = kml(placemark("Zakład Śląsk é", polygon_xml(lonlat(sq(X0, Y0, 1000)))), decl='<?xml version="1.0" encoding="UTF-8"?>')
    (rec,) = read_polygon_records(write_kml(tmp_path, xml), target_crs=32616)
    assert rec.name == "Zakład Śląsk é"
    latin = kml(placemark("Café", polygon_xml(lonlat(sq(X0, Y0, 1000)))), decl='<?xml version="1.0" encoding="ISO-8859-1"?>')
    (rec,) = read_polygon_records(write_kml(tmp_path, latin, "l.kml", encoding="latin-1"), target_crs=32616)
    assert rec.name == "Café"


def test_kml_lines_and_points_are_not_polygons(tmp_path):
    body = (
        placemark("path", "<LineString><coordinates>-87,0,0 -86.99,0.01,0</coordinates></LineString>")
        + placemark("pt", "<Point><coordinates>-87,0,0</coordinates></Point>")
        + placemark("pt2", "<Point><coordinates>-87,0.1,0</coordinates></Point>")
    )
    with pytest.raises(BoundaryError, match=r"no polygons found \(skipped 1 LineString, 2 Point"):
        read_polygons(write_kml(tmp_path, kml(body)), target_crs=32616)


def test_kml_projected_numbers_are_caught(tmp_path):
    xml = kml(placemark("oops", f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords(sq(X0, Y0, 1000))}</coordinates></LinearRing></outerBoundaryIs></Polygon>"))
    with pytest.raises(BoundaryError, match="outside lon/lat limits"):
        read_polygons(write_kml(tmp_path, xml), target_crs=32616)


@pytest.mark.parametrize(
    "body, match",
    [
        ("<Placemark><Polygon></Polygon></Placemark>", "without an outer boundary"),
        ("<Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates> </coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>", "empty <coordinates>"),
        ("<Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>-87,0 abc,def -86,1</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>", "cannot read KML coordinate 'abc,def'"),
        ("<Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>-87,0 -86,0 -85,0 -87,0</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>", "zero area|invalid polygon"),
    ],
)
def test_kml_bad_polygons(tmp_path, body, match):
    with pytest.raises(BoundaryError, match=match):
        read_polygons(write_kml(tmp_path, kml(body)), target_crs=32616)


def test_kml_malformed_xml(tmp_path):
    with pytest.raises(BoundaryError, match="not a readable KML"):
        read_polygons(write_kml(tmp_path, "<kml><Document><Placemark>"), target_crs=32616)


def test_kml_hostile_xml_is_refused(tmp_path):
    bomb = (
        '<?xml version="1.0"?><!DOCTYPE kml [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]>'
        "<kml><Document><Placemark><name>&b;</name></Placemark></Document></kml>"
    )
    with pytest.raises(BoundaryError, match="refused for safety"):
        read_polygons(write_kml(tmp_path, bomb), target_crs=32616)
    xxe = '<?xml version="1.0"?><!DOCTYPE kml [<!ENTITY x SYSTEM "file:///etc/passwd">]><kml><Document><Placemark><name>&x;</name></Placemark></Document></kml>'
    with pytest.raises(BoundaryError, match="refused for safety"):
        read_polygons(write_kml(tmp_path, xxe, "x.kml"), target_crs=32616)


def _kmz(path, files):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in files.items():
            z.writestr(name, text)
    return path


def test_kmz_roundtrip_and_doc_kml_preferred(tmp_path):
    good = kml(placemark("good", polygon_xml(lonlat(sq(X0, Y0, 1000)))))
    other = kml(placemark("other", polygon_xml(lonlat(sq(X0, Y0, 500)))))
    p = _kmz(tmp_path / "a.kmz", {"zzz.kml": other, "doc.kml": good, "files/icon.png": "x"})
    (rec,) = read_polygon_records(p, target_crs=32616)
    assert rec.name == "good" and rec.polygon.area == pytest.approx(1.0e6, rel=1e-8)
    q = _kmz(tmp_path / "b.kmz", {"only.kml": other})
    assert read_polygons(q, target_crs=32616)[0].area == pytest.approx(250000, rel=1e-8)


def test_kmz_errors(tmp_path):
    with pytest.raises(BoundaryError, match="no .kml file inside"):
        read_polygons(_kmz(tmp_path / "e.kmz", {"x.txt": "hi"}), target_crs=32616)
    (tmp_path / "n.kmz").write_bytes(b"not a zip")
    with pytest.raises(BoundaryError, match="not a valid KMZ"):
        read_polygons(tmp_path / "n.kmz", target_crs=32616)


def test_size_limits(tmp_path, monkeypatch):
    text = kml(placemark("x", polygon_xml(lonlat(sq(X0, Y0, 1000)))))
    kml_path = write_kml(tmp_path, text)
    kmz_path = _kmz(tmp_path / "s.kmz", {"doc.kml": text + " " * 10000})  # compresses to almost nothing
    monkeypatch.setattr(polygons, "MAX_XML_BYTES", 100)
    with pytest.raises(BoundaryError, match="larger than"):
        read_polygons(kml_path, target_crs=32616)
    with pytest.raises(BoundaryError, match="larger than"):
        read_polygons(kmz_path, target_crs=32616)  # small on disk, big once decompressed


# ============================== CRS policy ===================================================
def test_geojson_crs_member_is_honoured(tmp_path):
    import json

    ring = [[X0, Y0], [X0 + 1000, Y0], [X0 + 1000, Y0 + 1000], [X0, Y0 + 1000], [X0, Y0]]
    gj = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32616"}},
        "features": [{"type": "Feature", "properties": {"Name": "Lease 7"}, "geometry": {"type": "Polygon", "coordinates": [ring]}}],
    }
    p = tmp_path / "u.geojson"
    p.write_text(json.dumps(gj))
    (rec,) = read_polygon_records(p, target_crs=32616)
    assert rec.polygon.area == pytest.approx(1.0e6) and rec.name == "Lease 7"


def test_unknown_crs_code(tmp_path):
    p = write_shp(tmp_path / "u", [[cw(sq(X0, Y0, 1000))]])
    with pytest.raises(BoundaryError, match="unknown coordinate system"):
        read_polygons(p, source_crs="EPSG:99999999", target_crs=32616)

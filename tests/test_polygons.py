"""Polygon import, and polygon -> grid -> GRDECL -> xtgeo end to end."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
import xtgeo
from shapely.geometry import Point, Polygon

from cosma.grid.boundary import BoundaryError, grid_spec_from_polygon
from cosma.grid.synthetic import make_faulted_box
from cosma.io.grdecl import write_grdecl
from cosma.io.polygons import read_boundary, read_polygons

resdata_grid = pytest.importorskip("resdata.grid")

X0, Y0 = 683450.0, 4701234.0
SQ = [(X0, Y0), (X0 + 1000, Y0), (X0 + 1000, Y0 + 1000), (X0, Y0 + 1000)]  # 1.0e6 m2
SQ_SMALL = [(X0 + 5000, Y0), (X0 + 5500, Y0), (X0 + 5500, Y0 + 500), (X0 + 5000, Y0 + 500)]  # 2.5e5 m2


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


# -- text formats --------------------------------------------------------------------------
def test_csv_with_header_and_closing_vertex(tmp_path):
    rows = "\n".join(f"{x},{y}" for x, y in SQ + [SQ[0]])
    p = _write(tmp_path, "aoi.csv", "x,y\n" + rows + "\n")
    (poly,) = read_polygons(p)
    assert poly.area == pytest.approx(1.0e6)
    assert poly.exterior.is_ccw
    assert len(poly.exterior.coords) == 5


def test_whitespace_xyz_with_z_and_id_columns(tmp_path):
    rows = "\n".join(f"{x} {y} 1500.0 7" for x, y in SQ)
    p = _write(tmp_path, "aoi.xyz", "# exported polygon\n" + rows + "\n")
    assert read_boundary(p).area == pytest.approx(1.0e6)


def test_irap_style_999_separators_and_choice(tmp_path):
    def block(pts):
        return "\n".join(f"{x} {y} 1500.0" for x, y in pts) + "\n999 999 999\n"

    p = _write(tmp_path, "two.pol", block(SQ) + block(SQ_SMALL))
    polys = read_polygons(p)
    assert sorted(round(q.area) for q in polys) == [250000, 1000000]
    with pytest.raises(BoundaryError, match="holds 2 polygons"):
        read_boundary(p)
    assert read_boundary(p, which="largest").area == pytest.approx(1.0e6)
    assert read_boundary(p, which=1).area == pytest.approx(250000.0)
    with pytest.raises(BoundaryError, match="not valid"):
        read_boundary(p, which=5)


def test_blank_line_separated_polygons(tmp_path):
    txt = "\n".join(f"{x},{y}" for x, y in SQ) + "\n\n" + "\n".join(f"{x},{y}" for x, y in SQ_SMALL) + "\n"
    assert len(read_polygons(_write(tmp_path, "b.csv", txt))) == 2


def test_semicolon_and_utf8_bom(tmp_path):
    p = tmp_path / "s.csv"
    p.write_bytes(b"\xef\xbb\xbf" + "\n".join(f"{x};{y}" for x, y in SQ).encode())
    assert read_boundary(p).area == pytest.approx(1.0e6)


def test_bad_line_reports_line_number(tmp_path):
    p = _write(tmp_path, "bad.csv", "x,y\n1,2\n3,4\n5,oops\n")
    with pytest.raises(BoundaryError, match=r"line 4"):
        read_polygons(p)


def test_single_column_line(tmp_path):
    with pytest.raises(BoundaryError, match="x and y"):
        read_polygons(_write(tmp_path, "c.csv", "1,2\n3\n"))


def test_invalid_polygon_names_file_and_index(tmp_path):
    bow = [(0, 0), (10, 10), (10, 0), (0, 10)]
    p = _write(tmp_path, "bow.csv", "\n".join(f"{x},{y}" for x, y in bow))
    with pytest.raises(BoundaryError, match=r"bow\.csv, polygon 1: invalid polygon: Self-intersection"):
        read_polygons(p)


def test_empty_and_unsupported(tmp_path):
    with pytest.raises(BoundaryError, match="no polygons"):
        read_polygons(_write(tmp_path, "h.csv", "x,y\n"))
    with pytest.raises(BoundaryError, match="unsupported"):
        read_polygons(tmp_path / "area.gpkg")


def test_crs_pair_must_be_complete(tmp_path):
    p = _write(tmp_path, "a.csv", "\n".join(f"{x},{y}" for x, y in SQ))
    with pytest.raises(BoundaryError, match="both source_crs and target_crs"):
        read_polygons(p, source_crs=4326)


# -- GeoJSON + CRS ----------------------------------------------------------------------------------
def _gj(geometry):
    return {"type": "Feature", "properties": {}, "geometry": geometry}


def test_geojson_requires_target_crs(tmp_path):
    ring = [[-87.0, 0.0], [-86.99, 0.0], [-86.99, 0.01], [-87.0, 0.01], [-87.0, 0.0]]
    p = _write(tmp_path, "a.geojson", json.dumps(_gj({"type": "Polygon", "coordinates": [ring]})))
    with pytest.raises(BoundaryError, match="lon/lat in EPSG:4326.*pass target_crs"):
        read_polygons(p)


def test_geojson_reprojects_to_utm_anchor_point(tmp_path):
    """UTM zone 16N has its central meridian at 87 W: lon -87, lat 0 is exactly
    easting 500000, northing 0 by definition of the projection."""
    ring = [[-87.0, 0.0], [-86.99, 0.0], [-86.99, 0.01], [-87.0, 0.01], [-87.0, 0.0]]
    p = _write(tmp_path, "a.geojson", json.dumps(_gj({"type": "Polygon", "coordinates": [ring]})))
    (poly,) = read_polygons(p, target_crs="EPSG:32616")
    d = np.hypot(np.array(poly.exterior.coords)[:, 0] - 500000.0, np.array(poly.exterior.coords)[:, 1] - 0.0)
    assert d.min() < 1e-3
    # 0.01 degrees is about 1.1 km, so the polygon is roughly 1.1 km x 1.1 km
    assert 1.0e6 < poly.area < 1.5e6


def test_geojson_projected_passthrough_with_hole_and_multi(tmp_path):
    outer = [[X0, Y0], [X0 + 1000, Y0], [X0 + 1000, Y0 + 800], [X0, Y0 + 800], [X0, Y0]]
    hole = [[X0 + 400, Y0 + 300], [X0 + 600, Y0 + 300], [X0 + 600, Y0 + 500], [X0 + 400, Y0 + 500], [X0 + 400, Y0 + 300]]
    small = [[X0 + 5000, Y0], [X0 + 5500, Y0], [X0 + 5500, Y0 + 500], [X0 + 5000, Y0 + 500], [X0 + 5000, Y0]]
    fc = {
        "type": "FeatureCollection",
        "features": [
            _gj({"type": "Polygon", "coordinates": [outer, hole]}),
            _gj({"type": "MultiPolygon", "coordinates": [[small]]}),
            _gj({"type": "Point", "coordinates": [X0, Y0]}),  # ignored
        ],
    }
    p = _write(tmp_path, "p.geojson", json.dumps(fc))
    polys = read_polygons(p, source_crs="EPSG:32616", target_crs="EPSG:32616")
    assert sorted(round(q.area) for q in polys) == [250000, 1000 * 800 - 200 * 200]
    assert max(polys, key=lambda q: q.area).interiors  # hole survived


def test_geojson_bad_json(tmp_path):
    with pytest.raises(BoundaryError, match="not valid JSON"):
        read_polygons(_write(tmp_path, "x.geojson", "{nope"), target_crs=32616)


def test_text_file_lonlat_with_crs_pair(tmp_path):
    rows = "\n".join(f"{x},{y}" for x, y in [(-87.0, 0.0), (-86.99, 0.0), (-86.99, 0.01), (-87.0, 0.01)])
    p = _write(tmp_path, "ll.csv", "lon,lat\n" + rows)
    (poly,) = read_polygons(p, source_crs=4326, target_crs="EPSG:32616")
    xy = np.array(poly.exterior.coords)
    assert np.hypot(xy[:, 0] - 500000.0, xy[:, 1]).min() < 1e-3


# -- end to end ------------------------------------------------------------------------------------------
PENTAGON = [(X0, Y0), (X0 + 3200, Y0 + 300), (X0 + 3900, Y0 + 2100), (X0 + 1800, Y0 + 3300), (X0 - 400, Y0 + 1700)]


@pytest.mark.parametrize("az", ["auto", 20.0])
def test_polygon_file_to_grdecl_footprint(tmp_path, az):
    """Import a polygon, fit a grid, build, export with MAPAXES, read back with
    xtgeo, and compare footprint and fault against the polygon itself."""
    pf = _write(tmp_path, "aoi.csv", "x,y\n" + "\n".join(f"{x},{y}" for x, y in PENTAGON))
    poly = read_boundary(pf)
    spec = grid_spec_from_polygon(poly, 100.0, 100.0, azimuth=az, margin=50.0)
    g = make_faulted_box(spec=spec, nz=4, dz=10.0, top=1800.0, fault_i=spec.nx // 2, throw=30.0, dip_x=0.01)
    path = write_grdecl(g, tmp_path / "aoi.grdecl")

    xg = xtgeo.grid_from_file(path, fformat="grdecl")
    assert tuple(xg.dimensions) == (spec.nx, spec.ny, 4)

    # 1. active mask survives the round trip
    act = xg.get_actnum().values
    np.testing.assert_array_equal(act[:, :, 0].astype(bool), spec.active_xy)
    assert spec.active_xy.sum() > 0.5 * spec.nx * spec.ny * 0.4

    # 2. independent footprint check: the centre of every column, taken from
    #    xtgeo's MAP-coordinate corners, is inside the polygon iff active. A
    #    centre within 1 mm of the polygon edge is ambiguous and allowed either
    #    way (the code's own rule for that case is tested separately).
    bad = []
    for i in range(spec.nx):
        for j in range(spec.ny):
            c = np.array(xg.get_xyz_cell_corners(ijk=(i + 1, j + 1, 1), activeonly=False)).reshape(8, 3)
            centre = Point(c[:, 0].mean(), c[:, 1].mean())
            if poly.exterior.distance(centre) < 1e-3:
                continue
            if poly.contains(centre) != bool(spec.active_xy[i, j]):
                bad.append((i, j))
    assert bad == []

    # 3. active bulk volume = active columns * nz * 100 * 100 * 10
    bulk = xg.get_bulk_volume().values
    assert float(np.ma.filled(bulk, 0.0).sum()) == pytest.approx(spec.active_xy.sum() * 4 * 100.0 * 100.0 * 10.0, rel=1e-6)

    # 4. staircase error is small for this resolution and reported honestly
    assert abs(spec.area_ratio - 1.0) < 0.06

    # 5. the fault is still a clean 30 m offset between adjacent columns, in a rotated grid
    fi = spec.nx // 2
    for j in (0, spec.ny // 2, spec.ny - 1):
        a = xg.get_xyz_cell_corners(ijk=(fi, j + 1, 1), activeonly=False)  # east edge of column fi-1 (1-based fi)
        b = xg.get_xyz_cell_corners(ijk=(fi + 1, j + 1, 1), activeonly=False)  # west edge of column fi
        z_east, z_west = a[3 * 1 + 2], b[3 * 0 + 2]
        assert z_west - z_east == pytest.approx(30.0, abs=1e-6)

    # 6. resdata agrees on the number of active cells
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rg = resdata_grid.Grid.load_from_grdecl(str(path))
        assert rg.get_num_active() == int(spec.active_xy.sum()) * 4

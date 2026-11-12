"""Grid frame, MAPAXES export and UTM-scale precision.

Anchors: map positions are computed from the frame definition (origin, azimuth)
by hand-written trigonometry here, not by calling the code under test.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import xtgeo

from cosma.grid.frame import GridFrame
from cosma.grid.synthetic import make_faulted_box
from cosma.io.grdecl import float32_coordinate_error, scan_keywords, write_grdecl

resdata_grid = pytest.importorskip("resdata.grid")

X0, Y0 = 683450.25, 4701234.75  # UTM-16N-scale origin (7-digit northing)


def _expected_map(x, y, az_deg, ox=X0, oy=Y0):
    a = math.radians(az_deg)
    return ox + x * math.cos(a) + y * math.sin(a), oy - x * math.sin(a) + y * math.cos(a)


# -- frame maths -----------------------------------------------------------------------
@pytest.mark.parametrize("az", [0.0, 30.0, -17.5, 90.0, 123.4])
def test_frame_roundtrip_and_axes(az):
    f = GridFrame(X0, Y0, az)
    x, y = np.array([0.0, 250.0, -40.0]), np.array([0.0, 75.0, 310.0])
    X, Y = f.to_map(x, y)
    xb, yb = f.to_local(X, Y)
    np.testing.assert_allclose(xb, x, atol=1e-7)
    np.testing.assert_allclose(yb, y, atol=1e-7)
    for xi, yi, Xi, Yi in zip(x, y, X, Y):
        ex, ey = _expected_map(xi, yi, az)
        assert (Xi, Yi) == pytest.approx((ex, ey), abs=1e-7)


def test_azimuth_zero_is_i_east_j_north():
    f = GridFrame(1000.0, 2000.0, 0.0)
    assert f.to_map(1.0, 0.0) == pytest.approx((1001.0, 2000.0))  # +x east
    assert f.to_map(0.0, 1.0) == pytest.approx((1000.0, 2001.0))  # +y north


def test_azimuth_is_clockwise_from_north():
    # J axis pointing east (azimuth 90): local +y -> map +x; local +x -> map -y (south)
    f = GridFrame(0.0, 0.0, 90.0)
    assert f.to_map(0.0, 1.0) == pytest.approx((1.0, 0.0), abs=1e-12)
    assert f.to_map(1.0, 0.0) == pytest.approx((0.0, -1.0), abs=1e-12)


# -- precision ----------------------------------------------------------------------------
def test_float32_error_is_zero_locally_and_large_in_utm():
    g = make_faulted_box(frame=GridFrame(X0, Y0, 30.0), nx=40, ny=40, dx=97.31, dy=103.77)
    assert float32_coordinate_error(g, "local") < 1e-3
    assert float32_coordinate_error(g, "absolute") > 0.1


def test_absolute_export_warns_local_does_not(tmp_path):
    g = make_faulted_box(frame=GridFrame(X0, Y0, 30.0))
    with pytest.warns(UserWarning, match="float32"):
        write_grdecl(g, tmp_path / "abs.grdecl", coordinates="absolute")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_grdecl(g, tmp_path / "loc.grdecl", coordinates="local")


def test_bad_coordinates_option(tmp_path):
    with pytest.raises(ValueError):
        write_grdecl(make_faulted_box(), tmp_path / "x.grdecl", coordinates="utm")


def test_identity_frame_writes_no_mapaxes(tmp_path):
    p = write_grdecl(make_faulted_box(), tmp_path / "plain.grdecl")
    assert "MAPAXES" not in scan_keywords(p)


@pytest.mark.parametrize("az", [0.0, 30.0, -25.0])
def test_mapaxes_roundtrip_xtgeo_sub_millimetre(tmp_path, az):
    """Local coordinates + MAPAXES: xtgeo must put every corner at the exact
    map position, to well under a millimetre, at UTM scale."""
    g = make_faulted_box(frame=GridFrame(X0, Y0, az), throw=25.0, dip_x=0.02, dx=97.31, dy=103.77)
    p = write_grdecl(g, tmp_path / "m.grdecl")
    assert scan_keywords(p)["MAPAXES"] == 6
    xg = xtgeo.grid_from_file(p, fformat="grdecl")
    worst = 0.0
    for k in range(g.nz):
        for j in range(g.ny):
            for i in range(g.nx):
                theirs = np.array(xg.get_xyz_cell_corners(ijk=(i + 1, j + 1, k + 1))).reshape(8, 3)
                worst = max(worst, float(np.abs(theirs - g.cell_corners_map(i, j, k)).max()))
    assert worst < 1e-3


def test_mapaxes_resdata_within_float32_limit(tmp_path):
    """resdata applies MAPAXES too, but returns float32-limited map coordinates:
    at a northing of 4.7 million that is ~0.3 m. This pins the reader's limit,
    not a problem in the file (xtgeo above reads the same file to < 1 mm)."""
    g = make_faulted_box(frame=GridFrame(X0, Y0, 30.0))
    p = write_grdecl(g, tmp_path / "m.grdecl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rg = resdata_grid.Grid.load_from_grdecl(str(p))
        for (i, j, k) in [(0, 0, 0), (5, 3, 4), (2, 1, 3)]:
            ours = g.cell_corners_map(i, j, k)
            for n in range(8):
                theirs = rg.get_cell_corner(n, ijk=(i, j, k))
                assert theirs == pytest.approx(tuple(ours[n]), abs=0.6)


def test_opm_parser_accepts_mapaxes(tmp_path):
    opm_parser = pytest.importorskip("opm.io.parser")
    f = GridFrame(X0, Y0, 30.0)
    p = write_grdecl(make_faulted_box(frame=f), tmp_path / "m.grdecl")
    ctx = opm_parser.ParseContext(
        [("PARSE_RANDOM_SLASH", opm_parser.action.ignore), ("PARSE_UNKNOWN_KEYWORD", opm_parser.action.ignore)]
    )
    deck = opm_parser.Parser().parse(str(p), ctx)
    got = np.array(deck["MAPAXES"][0][0].get_raw_data_list())
    np.testing.assert_allclose(got, f.mapaxes(100.0), atol=1e-6)


def test_absolute_export_matches_mapaxes_export_in_xtgeo(tmp_path):
    """Both export modes describe the same grid; only the float32 step differs."""
    g = make_faulted_box(frame=GridFrame(X0, Y0, 30.0), throw=25.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pa = write_grdecl(g, tmp_path / "abs.grdecl", coordinates="absolute")
    pl = write_grdecl(g, tmp_path / "loc.grdecl", coordinates="local")
    assert "MAPAXES" not in scan_keywords(pa)
    xa = xtgeo.grid_from_file(pa, fformat="grdecl")
    xl = xtgeo.grid_from_file(pl, fformat="grdecl")
    ca = np.array(xa.get_xyz_cell_corners(ijk=(3, 2, 2)))
    cl = np.array(xl.get_xyz_cell_corners(ijk=(3, 2, 2)))
    assert np.abs(ca - cl).max() <= 0.26  # only float32 rounding of the absolute file

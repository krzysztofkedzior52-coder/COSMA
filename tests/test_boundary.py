"""Model boundary: manual definitions, polygon fitting, validation.

Expected values come from hand geometry (known areas, hand-placed rectangles,
half-plane tests written out here), not from the code under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from cosma.grid.boundary import (
    BoundaryError,
    auto_azimuth,
    grid_spec_from_cells,
    grid_spec_from_extent,
    grid_spec_from_polygon,
    grid_spec_from_rectangle,
    validate_polygon,
)

X0, Y0 = 683450.25, 4701234.75


def _rect(cx, cy, length, width, long_side_az_deg):
    """Rectangle centred at (cx, cy); the `length` side runs along azimuth
    long_side_az_deg (clockwise from north)."""
    a = math.radians(long_side_az_deg)
    u = np.array([math.sin(a), math.cos(a)])  # along the long side
    v = np.array([math.cos(a), -math.sin(a)])  # along the short side
    c = np.array([cx, cy])
    return np.array([c - u * length / 2 - v * width / 2, c + u * length / 2 - v * width / 2,
                     c + u * length / 2 + v * width / 2, c - u * length / 2 + v * width / 2])


# -- manual definitions ----------------------------------------------------------------
def test_from_cells():
    s = grid_spec_from_cells(X0, Y0, 0.0, nx=12, ny=7, dx=50.0, dy=40.0)
    assert (s.nx, s.ny) == (12, 7)
    assert s.active_xy.all() and s.boundary is None and s.area_ratio is None
    assert s.extent_map() == pytest.approx((X0, Y0, X0 + 600.0, Y0 + 280.0))
    assert s.active_area == pytest.approx(600.0 * 280.0)


def test_from_cells_rotated_extent():
    s = grid_spec_from_cells(0.0, 0.0, 90.0, nx=10, ny=4, dx=100.0, dy=100.0)
    # azimuth 90: local +y -> map +x (4*100 east), local +x -> map -y (10*100 south)
    assert s.extent_map() == pytest.approx((0.0, -1000.0, 400.0, 0.0), abs=1e-9)


@pytest.mark.parametrize("bad", [(0, 5, 10.0, 10.0), (5, 0, 10.0, 10.0), (5, 5, 0.0, 10.0), (5, 5, 10.0, -1.0)])
def test_from_cells_rejects_bad_sizes(bad):
    with pytest.raises(BoundaryError):
        grid_spec_from_cells(0, 0, 0, *bad)


def test_rectangle_expand_and_adjust():
    e = grid_spec_from_rectangle(0, 0, 0, length_x=1050.0, length_y=400.0, dx=100.0, dy=100.0, fit="expand")
    assert (e.nx, e.ny) == (11, 4)
    assert e.x_edges[-1] == pytest.approx(1100.0)  # extends past the request
    a = grid_spec_from_rectangle(0, 0, 0, length_x=1050.0, length_y=400.0, dx=100.0, dy=100.0, fit="adjust")
    assert (a.nx, a.ny) == (11, 4)
    assert a.x_edges[-1] == pytest.approx(1050.0)  # honours the request
    assert np.diff(a.x_edges) == pytest.approx(1050.0 / 11)


def test_rectangle_exact_multiple_does_not_add_a_cell():
    # 3 * 33.3 is not exactly 99.9 in floating point; must still give 3 cells
    s = grid_spec_from_rectangle(0, 0, 0, length_x=99.9, length_y=33.3, dx=33.3, dy=33.3)
    assert (s.nx, s.ny) == (3, 1)


def test_rectangle_bad_fit():
    with pytest.raises(BoundaryError):
        grid_spec_from_rectangle(0, 0, 0, 100, 100, 10, 10, fit="shrink")


def test_extent():
    s = grid_spec_from_extent(X0, X0 + 2000.0, Y0, Y0 + 1000.0, 100.0, 100.0)
    assert (s.nx, s.ny) == (20, 10)
    assert s.frame.azimuth_deg == 0.0
    assert s.extent_map() == pytest.approx((X0, Y0, X0 + 2000.0, Y0 + 1000.0))
    with pytest.raises(BoundaryError):
        grid_spec_from_extent(10, 5, 0, 1, 1, 1)


# -- polygon validation -----------------------------------------------------------------------
def test_validate_cleans_and_orients():
    cw = [(0, 0), (0, 10), (10, 10), (10, 0), (10, 0), (0, 0)]  # clockwise, dup vertex, closed
    p = validate_polygon(cw)
    assert p.exterior.is_ccw
    assert len(p.exterior.coords) == 5  # 4 vertices + closing
    assert p.area == pytest.approx(100.0)


def test_validate_accepts_xyz_columns():
    p = validate_polygon([(0, 0, 5), (10, 0, 5), (10, 10, 5), (0, 10, 5)])
    assert p.area == pytest.approx(100.0)


@pytest.mark.parametrize(
    "pts, text",
    [
        ([(0, 0), (10, 10), (10, 0), (0, 10)], "Self-intersection"),  # bow-tie
        ([(0, 0), (1, 1)], "at least 3"),
        ([(0, 0), (1, 1), (1, 1), (0, 0)], "at least 3"),
        ([(0, 0), (1, 1), (2, 2)], None),  # collinear: zero area / invalid
        ([(0, 0), (1, 0), (float("nan"), 1)], "NaN"),
    ],
)
def test_validate_rejects(pts, text):
    with pytest.raises(BoundaryError, match=text):
        validate_polygon(pts)


# -- orientation ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "long_az, expected",
    # Edge azimuths are long_az and long_az + 90 (and +180, +270 for the
    # reversed edges); J takes the representative in [-45, 45). Worked by hand:
    #   75  -> 75 - 90  = -15      120 -> 120 - 90 = 30      -20 -> -20
    [(0.0, 0.0), (30.0, 30.0), (75.0, -15.0), (120.0, 30.0), (-20.0, -20.0), (200.0, 20.0)],
)
def test_auto_azimuth_folds_into_pm45(long_az, expected):
    p = validate_polygon(_rect(X0, Y0, 3000.0, 2000.0, long_az))
    assert auto_azimuth(p) == pytest.approx(expected, abs=1e-6)
    assert -45.0 <= auto_azimuth(p) < 45.0


@pytest.mark.parametrize("long_az", [0.0, 30.0, 75.0, 110.0, -20.0])
def test_rotated_rectangle_fits_exactly(long_az):
    poly = _rect(X0, Y0, 3000.0, 2000.0, long_az)
    s = grid_spec_from_polygon(poly, 50.0, 50.0)
    # every column active, perfect fit, 60 x 40 cells in some order
    assert s.active_xy.all()
    assert sorted((s.nx, s.ny)) == [40, 60]
    assert s.area_ratio == pytest.approx(1.0, abs=1e-9)
    # the polygon touches all four sides of the local grid rectangle
    lx, ly = s.frame.to_local(poly[:, 0], poly[:, 1])
    assert (lx.min(), ly.min()) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert (lx.max(), ly.max()) == pytest.approx((s.x_edges[-1], s.y_edges[-1]), abs=1e-6)
    # J axis is the one nearest to north
    assert -45.0 <= s.frame.azimuth_deg < 45.0


def test_rectangle_long_side_becomes_j_when_closest_to_north():
    s = grid_spec_from_polygon(_rect(X0, Y0, 3000.0, 2000.0, 30.0), 50.0, 50.0)
    assert s.frame.azimuth_deg == pytest.approx(30.0, abs=1e-6)
    assert (s.nx, s.ny) == (40, 60)  # long side along J
    s = grid_spec_from_polygon(_rect(X0, Y0, 3000.0, 2000.0, 75.0), 50.0, 50.0)
    assert s.frame.azimuth_deg == pytest.approx(-15.0, abs=1e-6)
    assert (s.nx, s.ny) == (60, 40)  # long side along I


def test_explicit_azimuth_overrides_auto():
    poly = _rect(X0, Y0, 3000.0, 2000.0, 30.0)
    s = grid_spec_from_polygon(poly, 50.0, 50.0, azimuth=0.0)
    assert s.frame.azimuth_deg == 0.0
    assert not s.active_xy.all()  # axis-aligned grid around a tilted rectangle wastes columns
    assert s.area_ratio < 1.0 + 0.2


# -- active-column mask against hand geometry ---------------------------------------------------------
TRI = [(X0, Y0), (X0 + 1000.0, Y0), (X0, Y0 + 600.0)]  # legs 1000 and 600
TRI_AREA = 0.5 * 1000.0 * 600.0


def test_triangle_centre_mask_matches_half_plane_test():
    dx = dy = 17.0
    s = grid_spec_from_polygon(TRI, dx, dy, azimuth=0.0, inside="center")
    assert (s.nx, s.ny) == (math.ceil(1000 / dx), math.ceil(600 / dy))
    assert s.frame.origin_x == pytest.approx(X0) and s.frame.origin_y == pytest.approx(Y0)
    expected = np.zeros((s.nx, s.ny), dtype=bool)
    for i in range(s.nx):
        for j in range(s.ny):
            x, y = (i + 0.5) * dx, (j + 0.5) * dy
            expected[i, j] = x / 1000.0 + y / 600.0 < 1.0  # hand half-plane test
    np.testing.assert_array_equal(s.active_xy, expected)
    assert abs(s.area_ratio - 1.0) < 0.05


@pytest.mark.parametrize("az", [0.0, 33.3, "auto"])
def test_fraction_mode_conserves_polygon_area_exactly(az):
    s = grid_spec_from_polygon(TRI, 17.0, 23.0, azimuth=az, inside="fraction", min_fraction=0.5)
    covered = float((s.coverage_xy * s.column_area).sum())
    assert covered == pytest.approx(TRI_AREA, rel=1e-9)  # exact intersection areas
    assert s.coverage_xy.min() >= 0.0 and s.coverage_xy.max() <= 1.0 + 1e-12


def test_fraction_threshold_changes_active_set():
    lo = grid_spec_from_polygon(TRI, 40.0, 40.0, azimuth=0.0, inside="fraction", min_fraction=0.05)
    hi = grid_spec_from_polygon(TRI, 40.0, 40.0, azimuth=0.0, inside="fraction", min_fraction=0.95)
    assert lo.active_xy.sum() > hi.active_xy.sum()
    assert (hi.active_xy <= lo.active_xy).all()  # hi is a subset of lo
    full = grid_spec_from_polygon(TRI, 40.0, 40.0, azimuth=0.0, inside="fraction", min_fraction=1.0)
    assert (full.coverage_xy[full.active_xy] >= 1.0 - 1e-9).all()


def test_polygon_with_hole():
    shell = [(X0, Y0), (X0 + 1000, Y0), (X0 + 1000, Y0 + 800), (X0, Y0 + 800)]
    hole = [(X0 + 400, Y0 + 300), (X0 + 600, Y0 + 300), (X0 + 600, Y0 + 500), (X0 + 400, Y0 + 500)]
    poly = Polygon(shell, [hole])
    s = grid_spec_from_polygon(poly, 25.0, 25.0, azimuth=0.0, inside="fraction")
    assert float((s.coverage_xy * s.column_area).sum()) == pytest.approx(1000 * 800 - 200 * 200, rel=1e-9)
    # hole columns are inactive: a column whose centre is at the hole centre
    ci, cj = int(500 / 25), int(400 / 25)
    assert not s.active_xy[ci, cj]
    assert s.active_xy[2, 2]


def test_margin_expands_and_shifts_origin():
    base = grid_spec_from_polygon(TRI, 20.0, 20.0, azimuth=0.0)
    m = grid_spec_from_polygon(TRI, 20.0, 20.0, azimuth=0.0, margin=100.0)
    assert (m.nx, m.ny) == (base.nx + 10, base.ny + 10)
    assert (m.frame.origin_x, m.frame.origin_y) == pytest.approx((X0 - 100.0, Y0 - 100.0))
    assert m.active_xy.sum() == base.active_xy.sum()  # extra margin columns are all outside


def test_polygon_errors():
    tiny = [(X0, Y0), (X0 + 10, Y0), (X0 + 10, Y0 + 10), (X0, Y0 + 10)]
    with pytest.raises(BoundaryError, match="no grid columns"):
        grid_spec_from_polygon(tiny, 1000.0, 1000.0, azimuth=0.0)
    with pytest.raises(BoundaryError, match="no grid columns"):
        grid_spec_from_polygon(tiny, 1000.0, 1000.0, azimuth=0.0, inside="fraction", min_fraction=0.5)
    with pytest.raises(BoundaryError, match="azimuth"):
        grid_spec_from_polygon(TRI, 10, 10, azimuth="north")
    with pytest.raises(BoundaryError, match="inside"):
        grid_spec_from_polygon(TRI, 10, 10, inside="touching")
    with pytest.raises(BoundaryError, match="margin"):
        grid_spec_from_polygon(TRI, 10, 10, margin=-1.0)
    with pytest.raises(BoundaryError, match="min_fraction"):
        grid_spec_from_polygon(TRI, 10, 10, inside="fraction", min_fraction=0.0)
    with pytest.raises(BoundaryError, match="cell sizes"):
        grid_spec_from_polygon(TRI, 0.0, 10)


@pytest.mark.parametrize("az", [0.0, 33.3, -12.5])
def test_centres_exactly_on_the_boundary_count_as_inside(az):
    """A polygon edge that passes through column centres (half-cell margin) must
    include those columns deterministically, even with rotation and UTM-scale
    coordinates, where floating-point noise is ~1e-9 m."""
    a = math.radians(az)
    u = np.array([math.sin(a), math.cos(a)])  # local +y in map
    v = np.array([math.cos(a), -math.sin(a)])  # local +x in map
    o = np.array([X0, Y0])
    corners = [o, o + 1000 * v, o + 1000 * v + 1000 * u, o + 1000 * u]
    s = grid_spec_from_polygon(corners, 100.0, 100.0, azimuth=az, margin=50.0)
    # cell centres at local 0, 100, ..., 1000 in both directions: all on or inside
    assert (s.nx, s.ny) == (11, 11)
    assert s.active_xy.all()
    # 1 mm is well beyond the tolerance: shrink the polygon by 1 mm per side and grow
    # the margin by the same amount so the cell centres stay exactly where they were
    shrunk = [o + 0.001 * (v + u), o + (1000 - 0.001) * v + 0.001 * u,
              o + (1000 - 0.001) * (v + u), o + 0.001 * v + (1000 - 0.001) * u]
    s2 = grid_spec_from_polygon(shrunk, 100.0, 100.0, azimuth=az, margin=50.0 + 0.001)
    assert s2.active_xy.sum() == 81  # only the 9 x 9 interior centres remain

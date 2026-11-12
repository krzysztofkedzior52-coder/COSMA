"""Surface import against hand-written files whose truth is an analytic plane."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xtgeo

from cosma.surfaces import Surface, SurfaceError, read_surface
from cosma.units import UnitError


def plane(x, y):
    return 1500.0 + 0.05 * (x - 1000.0) - 0.02 * (y - 2000.0)


def write_irap(path, ncol, nrow, xori, yori, xinc, yinc, rot, fn, undefined=()):
    """IRAP ASCII written from the format description, node positions computed here."""
    r = np.radians(rot)
    vals = []
    for j in range(nrow):
        for i in range(ncol):
            x = xori + i * xinc * np.cos(r) - j * yinc * np.sin(r)
            y = yori + i * xinc * np.sin(r) + j * yinc * np.cos(r)
            vals.append(9999900.0 if (i, j) in undefined else fn(x, y))
    txt = f"-996 {nrow} {xinc} {yinc}\n"
    txt += f"{xori} {xori + (ncol - 1) * xinc} {yori} {yori + (nrow - 1) * yinc}\n"
    txt += f"{ncol} {rot} {xori} {yori}\n0 0 0 0 0 0 0\n"
    for k in range(0, len(vals), 6):
        txt += " ".join(f"{v:.6f}" for v in vals[k:k + 6]) + "\n"
    path.write_text(txt)


KW = dict(xy_unit="m", z_unit="m")


def test_irap_unrotated_nodes_and_bilinear(tmp_path):
    p = tmp_path / "a.irap"
    write_irap(p, 5, 4, 1000.0, 2000.0, 10.0, 20.0, 0.0, plane)
    s = read_surface(p, **KW)
    assert (s.ncol, s.nrow) == (5, 4)
    X, Y = s.node_xy()
    np.testing.assert_allclose(s.values, plane(X, Y), atol=1e-4)
    # bilinear reproduces a plane exactly anywhere inside
    px, py = np.array([1003.3, 1037.9, 1000.0, 1040.0]), np.array([2001.7, 2058.1, 2000.0, 2060.0])
    np.testing.assert_allclose(s.sample(px, py), plane(px, py), atol=1e-4)


def test_irap_rotated_30deg_matches_plane_and_xtgeo(tmp_path):
    p = tmp_path / "r.irap"
    write_irap(p, 6, 5, 683450.0, 4701234.0, 25.0, 40.0, 30.0, lambda x, y: 1800.0 + 0.01 * (x - 683450.0) + 0.03 * (y - 4701234.0))
    s = read_surface(p, **KW)
    assert s.rotation == pytest.approx(30.0)
    # points inside, defined independently of the Surface code: along its rotated axes
    r = np.radians(30.0)
    u, v = np.array([12.5, 60.0, 124.0]), np.array([7.0, 80.0, 159.0])
    px = 683450.0 + u * np.cos(r) - v * np.sin(r)
    py = 4701234.0 + u * np.sin(r) + v * np.cos(r)
    truth = 1800.0 + 0.01 * (px - 683450.0) + 0.03 * (py - 4701234.0)
    np.testing.assert_allclose(s.sample(px, py), truth, atol=1e-4)
    xs = xtgeo.surface_from_file(str(p), fformat="irap_ascii")
    xv = [xs.get_value_from_xy((a, b)) for a, b in zip(px, py)]
    np.testing.assert_allclose(s.sample(px, py), xv, atol=1e-3)
    # a point that is inside the bounding box of the rotated grid but outside the grid itself
    assert np.isnan(s.sample(683450.0 + 140.0, 4701234.0 + 2.0))


def test_outside_is_nan_and_edges_are_inside(tmp_path):
    p = tmp_path / "a.irap"
    write_irap(p, 3, 3, 0.0, 0.0, 10.0, 10.0, 0.0, lambda x, y: 100.0 + x)
    s = read_surface(p, **KW)
    assert np.isnan(s.sample([-0.01, 20.01, 5.0], [5.0, 5.0, 20.01])).all()
    assert s.sample(20.0, 20.0) == pytest.approx(120.0)
    assert s.sample(0.0, 0.0) == pytest.approx(100.0)


def test_undefined_nodes_poison_only_their_cells(tmp_path):
    p = tmp_path / "u.irap"
    write_irap(p, 4, 4, 0.0, 0.0, 10.0, 10.0, 0.0, lambda x, y: 100.0, undefined={(1, 1)})
    s = read_surface(p, **KW)
    assert np.isnan(s.values[1, 1]) and np.isfinite(s.values).sum() == 15
    assert np.isnan(s.sample(12.0, 12.0))      # cell touching the hole
    assert s.sample(25.0, 25.0) == pytest.approx(100.0)   # cell away from it


def test_zmap_orientation_from_spec(tmp_path):
    """ZMAP+ lists columns west to east, each north to south."""
    def z(x, y):
        return 100 + (x - 1000) / 10 + 10 * (y - 2000) / 20

    cols = [z(x, y) for x in (1000, 1010, 1020) for y in (2020, 2000)]
    txt = "! hand written\n@hand, GRID, 5\n 15, 1.0E30, , 7, 1\n 2, 3, 1000.0, 1020.0, 2000.0, 2020.0\n 0.0, 0.0, 0.0\n@\n"
    for k in range(0, len(cols), 5):
        txt += " ".join(f"{v:15.7E}" for v in cols[k:k + 5]) + "\n"
    p = tmp_path / "h.zmap"
    p.write_text(txt)
    s = read_surface(p, **KW)
    assert (s.xori, s.yori, s.xinc, s.yinc) == (1000.0, 2000.0, 10.0, 20.0)
    assert s.sample(1010.0, 2020.0) == pytest.approx(z(1010, 2020))
    assert s.sample(1020.0, 2000.0) == pytest.approx(z(1020, 2000))
    assert s.sample(1005.0, 2010.0) == pytest.approx(z(1005, 2010))


def test_irap_binary_roundtrip_and_sniff(tmp_path):
    p = tmp_path / "b.gri"
    write_irap(tmp_path / "t.irap", 4, 3, 500.0, 600.0, 10.0, 15.0, 0.0, lambda x, y: 2000.0 + x - y)
    xtgeo.surface_from_file(str(tmp_path / "t.irap"), fformat="irap_ascii").to_file(str(p), fformat="irap_binary")
    s = read_surface(p, **KW)
    assert s.sample(512.5, 607.5) == pytest.approx(2000.0 + 512.5 - 607.5, abs=1e-3)


def test_units_and_sign(tmp_path):
    p = tmp_path / "a.irap"
    write_irap(p, 3, 3, 0.0, 0.0, 100.0, 100.0, 0.0, lambda x, y: 5000.0)
    s = read_surface(p, xy_unit="ft_intl", z_unit="ft_intl")
    assert s.xinc == pytest.approx(30.48)
    assert s.values[0, 0] == pytest.approx(1524.0)
    su = read_surface(p, xy_unit="ft_us", z_unit="ft_us")
    assert su.xinc == pytest.approx(100 * 1200 / 3937)
    assert su.xinc != pytest.approx(s.xinc, rel=1e-7)          # the two feet are different
    up = read_surface(p, positive="up", **KW)
    assert up.values[0, 0] == -5000.0
    with pytest.raises(UnitError, match="ambiguous"):
        read_surface(p, xy_unit="m", z_unit="ft")


def test_all_negative_depths_warn(tmp_path):
    p = tmp_path / "n.irap"
    write_irap(p, 3, 3, 0.0, 0.0, 10.0, 10.0, 0.0, lambda x, y: -1500.0)
    with pytest.warns(UserWarning, match="positive='up'"):
        read_surface(p, **KW)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        read_surface(p, positive="up", **KW)


def test_errors(tmp_path):
    with pytest.raises(SurfaceError, match="no such file"):
        read_surface(tmp_path / "nope.irap", **KW)
    bad = tmp_path / "bad.irap"
    bad.write_text("-996 3 10 10\nnot a surface\n")
    with pytest.raises(SurfaceError):
        read_surface(bad, **KW)
    junk = tmp_path / "junk.txt"
    junk.write_text("hello world\n")
    with pytest.raises(SurfaceError, match="recognise"):
        read_surface(junk, **KW)
    with pytest.raises(SurfaceError, match="positive"):
        read_surface(bad, positive="sideways", **KW)
    with pytest.raises(SurfaceError, match="at least 2"):
        Surface(0, 0, 1, 1, 0, 1, np.zeros((1, 3)))


def test_node_positions_rotated_independent_formula(tmp_path):
    """node_xy against hand-computed corners of a 30 deg rotated grid."""
    p = tmp_path / "r.irap"
    write_irap(p, 3, 3, 100.0, 200.0, 10.0, 10.0, 30.0, lambda x, y: 1.0)
    s = read_surface(p, **KW)
    X, Y = s.node_xy()
    c, sn = np.cos(np.radians(30)), np.sin(np.radians(30))
    assert (X[2, 0], Y[2, 0]) == pytest.approx((100 + 20 * c, 200 + 20 * sn))     # 2 steps along rotated x
    assert (X[0, 2], Y[0, 2]) == pytest.approx((100 - 20 * sn, 200 + 20 * c))     # 2 steps along rotated y
    fp = s.footprint()
    assert fp[2] == pytest.approx([100 + 20 * c - 20 * sn, 200 + 20 * sn + 20 * c])


def test_sampling_is_consistent_with_node_positions_when_y_reversed():
    vals = np.arange(12, dtype=float).reshape(3, 4)
    for flip in (1, -1):
        s = Surface(500.0, 800.0, 10.0, 25.0, 20.0, flip, vals)
        X, Y = s.node_xy()
        np.testing.assert_allclose(s.sample(X, Y), vals, atol=1e-9)


def test_reversed_y_file_is_refused(tmp_path):
    p = tmp_path / "f.irap"
    p.write_text("-996 2 10.0 -20.0\n1000.0 1020.0 2000.0 2020.0\n3 0.0 1000.0 2020.0\n0 0 0 0 0 0 0\n1 2 3 4 5 6\n")
    with pytest.raises(SurfaceError, match="yflip"):
        read_surface(p, **KW)

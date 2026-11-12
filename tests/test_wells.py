"""Trajectories, units and well importers, checked against analytic geometry."""

from __future__ import annotations

import numpy as np
import pytest

from cosma.units import UnitError, to_metres, unit_factor
from cosma.wells.io import (WellDataError, build_wells, read_core, read_deviation, read_las,
                            read_tops, read_well_headers)
from cosma.wells.trajectory import Trajectory, TrajectoryError

G = dict(azimuth_reference="grid")


# -- units ---------------------------------------------------------------------------------
def test_units():
    assert unit_factor("m") == 1.0
    assert unit_factor("ft_intl") == 0.3048
    assert unit_factor("ft_us") == pytest.approx(1200 / 3937)
    assert to_metres([1.0, np.nan], "km")[0] == 1000.0
    for bad in ("ft", "feet", "furlong"):
        with pytest.raises(UnitError):
            unit_factor(bad)


# -- trajectory ---------------------------------------------------------------------------------
def test_vertical_well_with_kb():
    t = Trajectory.vertical(1000.0, 2000.0, kb=25.0, td=1500.0)
    x, y, z = t.at_md([0.0, 750.0, 1500.0])
    np.testing.assert_allclose(x, 1000.0)
    np.testing.assert_allclose(y, 2000.0)
    np.testing.assert_allclose(z, [-25.0, 725.0, 1475.0])     # TVDSS = MD - KB


def test_tangent_section_straight_line():
    md = [0.0, 500.0, 1000.0]
    t = Trajectory.from_survey(md, [30.0] * 3, [90.0] * 3, x0=0, y0=0, kb=0, **G)
    x, y, z = t.at_md(750.0)
    assert x[0] == pytest.approx(750 * np.sin(np.radians(30)))
    assert y[0] == pytest.approx(0.0, abs=1e-9)
    assert z[0] == pytest.approx(750 * np.cos(np.radians(30)))


def test_azimuth_clockwise_from_north():
    for az, (ex, ey) in {0: (0, 1), 90: (1, 0), 180: (0, -1), 270: (-1, 0)}.items():
        t = Trajectory.from_survey([0.0, 100.0], [45.0, 45.0], [az, az], x0=0, y0=0, kb=0, **G)
        assert t.x[-1] == pytest.approx(100 * np.sin(np.pi / 4) * ex, abs=1e-9)
        assert t.y[-1] == pytest.approx(100 * np.sin(np.pi / 4) * ey, abs=1e-9)


def test_build_section_is_an_exact_circle():
    R = 1000.0
    th = np.radians(np.arange(0.0, 91.0, 30.0))
    t = Trajectory.from_survey(R * th, np.degrees(th), np.full(th.shape, 90.0), x0=0, y0=0, kb=0, **G)
    m = np.linspace(0.0, R * np.pi / 2, 13)
    x, y, z = t.at_md(m)
    a = m / R
    np.testing.assert_allclose(x, R * (1 - np.cos(a)), atol=1e-9)
    np.testing.assert_allclose(z, R * np.sin(a), atol=1e-9)
    np.testing.assert_allclose(y, 0.0, atol=1e-9)


def test_true_north_convergence():
    # true azimuth 90 with grid north 2 deg east of true north -> grid azimuth 88
    t = Trajectory.from_survey([0.0, 100.0], [90.0, 90.0], [90.0, 90.0], x0=0, y0=0, kb=0,
                               azimuth_reference="true", convergence_deg=2.0)
    assert np.degrees(np.arctan2(t.x[-1], t.y[-1])) == pytest.approx(88.0)
    with pytest.raises(TrajectoryError, match="convergence_deg"):
        Trajectory.from_survey([0.0, 1.0], [0, 0], [0, 0], x0=0, y0=0, kb=0, azimuth_reference="true")
    with pytest.raises(TrajectoryError, match="azimuth_reference"):
        Trajectory.from_survey([0.0, 1.0], [0, 0], [0, 0], x0=0, y0=0, kb=0, azimuth_reference="magnetic")
    with pytest.raises(TrajectoryError, match="convergence_deg given"):
        Trajectory.from_survey([0.0, 1.0], [0, 0], [0, 0], x0=0, y0=0, kb=0, azimuth_reference="grid", convergence_deg=1.0)


def test_md_at_tvdss_roundtrip_on_arc():
    R = 800.0
    th = np.radians(np.arange(0.0, 61.0, 20.0))
    t = Trajectory.from_survey(R * th, np.degrees(th), np.full(th.shape, 45.0), x0=0, y0=0, kb=30.0, **G)
    m = np.array([10.0, 123.0, 400.0, 800.0])
    z = t.tvdss_at_md(m)
    np.testing.assert_allclose(t.md_at_tvdss(z), m, atol=1e-6)


def test_trajectory_errors():
    with pytest.raises(TrajectoryError, match="increasing"):
        Trajectory.from_survey([0, 10, 10], [0, 0, 0], [0, 0, 0], x0=0, y0=0, kb=0, **G)
    with pytest.raises(TrajectoryError, match="inclination"):
        Trajectory.from_survey([0, 10], [0, 181], [0, 0], x0=0, y0=0, kb=0, **G)
    with pytest.raises(TrajectoryError, match="NaN"):
        Trajectory.from_survey([0, 10], [0, np.nan], [0, 0], x0=0, y0=0, kb=0, **G)
    t = Trajectory.vertical(0, 0, kb=0, td=100)
    with pytest.raises(TrajectoryError, match="outside surveyed"):
        t.at_md(101.0)
    with pytest.raises(TrajectoryError, match="outside surveyed"):
        t.md_at_tvdss(101.0)
    horiz = Trajectory.from_survey([0, 100, 200], [0, 90, 90], [0, 0, 0], x0=0, y0=0, kb=0, **G)
    with pytest.raises(TrajectoryError, match="ambiguous"):
        horiz.md_at_tvdss(50.0)


# -- CSV importers ---------------------------------------------------------------------------------
HEADERS = "Well,X,Y,KB,TD\nW-1,1000,2000,25,1500\nW-2,3000,2000,30,\n"


def _w(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_headers_units_and_errors(tmp_path):
    h = read_well_headers(_w(tmp_path, "h.csv", HEADERS), xy_unit="ft_us", z_unit="ft_intl")
    assert h["W-1"]["x"] == pytest.approx(1000 * 1200 / 3937)
    assert h["W-1"]["kb"] == pytest.approx(25 * 0.3048)
    assert h["W-2"]["td"] is None
    with pytest.raises(WellDataError, match="duplicate"):
        read_well_headers(_w(tmp_path, "d.csv", HEADERS + "W-1,1,1,1,1\n"), xy_unit="m", z_unit="m")
    with pytest.raises(WellDataError, match="no 'kb' column"):
        read_well_headers(_w(tmp_path, "k.csv", "well,x,y\nA,1,2\n"), xy_unit="m", z_unit="m")
    with pytest.raises(WellDataError, match="row 3.*not a number"):
        read_well_headers(_w(tmp_path, "n.csv", "well,x,y,kb\nA,1,2,3\nB,1,oops,3\n"), xy_unit="m", z_unit="m")
    with pytest.raises(WellDataError, match="not finite"):
        read_well_headers(_w(tmp_path, "i.csv", "well,x,y,kb\nA,1,2,nan\n"), xy_unit="m", z_unit="m")


def _wells(tmp_path):
    h = read_well_headers(_w(tmp_path, "h.csv", HEADERS), xy_unit="m", z_unit="m")
    dev = _w(tmp_path, "dev.csv", "well,md,incl,azim\nW-1,0,30,90\nW-1,500,30,90\nW-1,1500,30,90\n")
    tr = read_deviation(dev, h, z_unit="m", **G)
    tr["W-2"] = Trajectory.vertical(h["W-2"]["x"], h["W-2"]["y"], kb=h["W-2"]["kb"], td=1000.0)
    return h, tr


def test_deviation_import_and_guards(tmp_path):
    h, tr = _wells(tmp_path)
    x, y, z = tr["W-1"].at_md(1000.0)
    assert x[0] == pytest.approx(1000 + 1000 * 0.5)
    assert y[0] == pytest.approx(2000.0)
    assert z[0] == pytest.approx(1000 * np.cos(np.radians(30)) - 25.0)
    with pytest.raises(WellDataError, match="starts at MD 100"):
        read_deviation(_w(tmp_path, "s.csv", "well,md,incl,azim\nW-1,100,0,0\nW-1,200,0,0\n"), h, z_unit="m", **G)
    with pytest.raises(WellDataError, match="without a header"):
        read_deviation(_w(tmp_path, "u.csv", "well,md,incl,azim\nZZ,0,0,0\nZZ,10,0,0\n"), h, z_unit="m", **G)
    with pytest.raises(WellDataError, match="W-1.*increasing"):
        read_deviation(_w(tmp_path, "m.csv", "well,md,incl,azim\nW-1,0,0,0\nW-1,0,0,0\n"), h, z_unit="m", **G)


def test_tops_same_pick_three_depth_types(tmp_path):
    h, tr = _wells(tmp_path)
    md = 1000.0
    tvd = md * np.cos(np.radians(30))          # below KB, analytic for a straight tangent
    tvdss = tvd - 25.0
    truth_x = 1000 + md * 0.5
    got = []
    for dtype, d in (("md", md), ("tvd", tvd), ("tvdss", tvdss)):
        (t,) = read_tops(_w(tmp_path, f"t_{dtype}.csv", f"well,top,depth\nW-1,Seal,{float(d)!r}\n"), tr, depth_type=dtype, z_unit="m")
        assert (t.md, t.tvdss, t.x) == pytest.approx((md, tvdss, truth_x), abs=1e-6)
        got.append(t)
    assert got[0].y == pytest.approx(2000.0)


def test_tops_errors(tmp_path):
    h, tr = _wells(tmp_path)
    with pytest.raises(WellDataError, match="no trajectory for well 'W-9'"):
        read_tops(_w(tmp_path, "a.csv", "well,top,depth\nW-9,S,10\n"), tr, depth_type="md", z_unit="m")
    with pytest.raises(WellDataError, match="outside surveyed"):
        read_tops(_w(tmp_path, "b.csv", "well,top,depth\nW-2,S,5000\n"), tr, depth_type="md", z_unit="m")
    with pytest.raises(WellDataError, match="depth_type"):
        read_tops(_w(tmp_path, "c.csv", "well,top,depth\nW-2,S,5\n"), tr, depth_type="kb", z_unit="m")


CORE = "well,md,porosity,perm_mD\nW-2,600,12.5,50\nW-2,500,20,300\nW-1,1000,,1.5\n"


def test_core_percent_sorted_and_blank(tmp_path):
    h, tr = _wells(tmp_path)
    c = read_core(_w(tmp_path, "core.csv", CORE), tr, depth_type="md", z_unit="m", porosity_unit="percent")
    np.testing.assert_allclose(c["W-2"].md, [500.0, 600.0])           # sorted
    np.testing.assert_allclose(c["W-2"].columns["porosity"], [0.20, 0.125])
    np.testing.assert_allclose(c["W-2"].columns["perm_md"], [300.0, 50.0])
    assert np.isnan(c["W-1"].columns["porosity"][0])
    with pytest.raises(WellDataError, match="outside \\[0, 1\\].*'frac'"):
        read_core(_w(tmp_path, "core2.csv", CORE), tr, depth_type="md", z_unit="m", porosity_unit="frac")
    with pytest.raises(WellDataError, match="porosity_unit"):
        read_core(_w(tmp_path, "core3.csv", CORE), tr, depth_type="md", z_unit="m", porosity_unit="percentage")


def test_core_tvdss_depth_converted_to_md(tmp_path):
    h, tr = _wells(tmp_path)
    c = read_core(_w(tmp_path, "c.csv", "well,depth,porosity\nW-2,470,0.1\n"), tr, depth_type="tvd", z_unit="m", porosity_unit="frac")
    assert c["W-2"].md[0] == pytest.approx(470.0)                  # vertical: TVD from KB = MD


def test_build_wells_requires_trajectory(tmp_path):
    h, tr = _wells(tmp_path)
    ws = build_wells(h, tr)
    assert [w.name for w in ws] == ["W-1", "W-2"] and ws[0].kb == 25.0 and ws[1].x == 3000.0
    del tr["W-2"]
    with pytest.raises(WellDataError, match="W-2"):
        build_wells(h, tr)


# -- LAS -------------------------------------------------------------------------------------------------
LAS = """~VERSION INFORMATION
 VERS.   2.0 : CWLS
 WRAP.   NO  : One line per depth step
~WELL INFORMATION
 STRT.{u}  {s} :
 STOP.{u}  {e} :
 STEP.{u}  {st} :
 NULL.  -999.25 :
 WELL.  W-1 :
~CURVE INFORMATION
 DEPT.{u}   : Depth
 GR  .GAPI  : Gamma
 PHIT.V/V   : Porosity
~ASCII
{rows}
"""


def _las(tmp_path, unit, rows=((1000.0, 55.0, 0.21), (1000.5, -999.25, 0.22), (1001.0, 60.0, 0.23)), name="a.las"):
    txt = LAS.format(u=unit, s=rows[0][0], e=rows[-1][0], st=0.5,
                     rows="\n".join(f"{d} {g} {p}" for d, g, p in rows))
    return _w(tmp_path, name, txt)


def test_las_metres_and_nulls(tmp_path):
    ls = read_las(_las(tmp_path, "M"))
    np.testing.assert_allclose(ls.md, [1000.0, 1000.5, 1001.0])
    assert np.isnan(ls.curves["GR"][1]) and ls.curves["GR"][0] == 55.0
    np.testing.assert_allclose(ls.curves["PHIT"], [0.21, 0.22, 0.23])
    assert ls.units["GR"] == "GAPI"


def test_las_feet_need_a_choice(tmp_path):
    p = _las(tmp_path, "FT")
    with pytest.raises(WellDataError, match="pass depth_unit"):
        read_las(p)
    ls_i = read_las(p, depth_unit="ft_intl")
    ls_u = read_las(p, depth_unit="ft_us")
    assert ls_i.md[0] == pytest.approx(1000 * 0.3048)
    assert ls_u.md[0] == pytest.approx(1000 * 1200 / 3937) and ls_u.md[0] != ls_i.md[0]


def test_las_rejects_bad_index(tmp_path):
    with pytest.raises(WellDataError, match="strictly increasing"):
        read_las(_las(tmp_path, "M", rows=((1000.0, 1, 0.1), (1000.0, 2, 0.1), (999.0, 3, 0.1))))
    p = _las(tmp_path, "M").read_text().replace(" DEPT.M", " TIME.M")
    q = tmp_path / "t.las"
    q.write_text(p)
    with pytest.raises(WellDataError, match="expected DEPT"):
        read_las(q)

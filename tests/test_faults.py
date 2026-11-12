import math

import numpy as np
import pytest

from cosma.faults import (Fault, FaultError, build_faults, export_sticks, fit_plane, parse_sticks, read_fault_sticks,
                          trace_at)


def plane_sticks(dip, dipdir, offsets, length=1000.0, n=6, origin=(0.0, 0.0)):
    """Sticks on the plane through `origin` at depth 0 with the given dip/dip direction; each stick runs down the dip
    direction, starting `offsets` metres along the strike from the first."""
    dd = math.radians(dipdir)
    ex, ny = math.sin(dd), math.cos(dd)                       # unit vector of the dip direction (east, north)
    t = math.tan(math.radians(dip))
    sticks = []
    for o in offsets:
        r = np.linspace(0.0, length, n)
        sticks.append(np.column_stack([origin[0] + r * ex + o * ny, origin[1] + r * ny - o * ex, r * t]))
    return tuple(sticks)


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# -- reading --------------------------------------------------------------------------------------
def test_table_with_header_stick_column_and_extra_columns(tmp_path):
    p = write(tmp_path, "f.csv", "X [m],Y,Depth,Fault_Name,Stick_ID,Comment\n"
                                 "0,0,100,F1,a,x\n10,0,200,F1,a,x\n0,50,100,F1,b,x\n10,50,200,F1,b,x\n"
                                 "500,0,100,F2,1,x\n510,0,200,F2,1,x\n")
    r = parse_sticks(p)
    assert r.layout == "table" and r.delimiter == "," and r.stick_source == "column" and r.has_fault_column
    assert list(r.faults) == ["F1", "F2"] and [len(v) for v in r.faults.values()] == [2, 1]
    assert r.faults["F1"][1].tolist() == [[0, 50, 100], [10, 50, 200]]
    assert r.ignored_columns == ("Comment",) and r.n_points == 6
    assert r.columns == {"x": "X [m]", "y": "Y", "z": "Depth", "fault": "Fault_Name", "stick": "Stick_ID"}


def test_whitespace_and_tab_and_semicolon_delimiters(tmp_path):
    body = [(0, 0, 10), (1, 0, 20), (0, 5, 10), (1, 5, 20)]
    for name, d in (("a.txt", " "), ("b.tsv", "\t"), ("c.csv", ";")):
        rows = [d.join(["x", "y", "z", "fault", "stick"])] + [d.join(map(str, b)) + d + "F" + d + str(k // 2) for k, b in enumerate(body)]
        r = parse_sticks(write(tmp_path, name, "\n".join(rows) + "\n"))
        assert len(r.faults["F"]) == 2 and r.stick_source == "column"
        assert r.delimiter == {" ": "whitespace", "\t": "tab", ";": ";"}[d]


def test_headerless_sticks_separated_by_blank_lines_and_999(tmp_path):
    a = parse_sticks(write(tmp_path, "a.txt", "# comment\n0 0 10\n1 0 20\n\n0 5 10\n1 5 20\n"))
    assert a.layout == "separated" and a.stick_source == "separators" and list(a.faults) == [""]
    assert [len(s) for s in a.faults[""]] == [2, 2]
    b = parse_sticks(write(tmp_path, "b.txt", "0 0 10\n1 0 20\n999.0 999.0 999.0\n0 5 10\n1 5 20\n-- note\n999 999 999\n"))
    assert b.stick_source == "separators" and [len(s) for s in b.faults[""]] == [2, 2]
    c = parse_sticks(write(tmp_path, "c.txt", "0 0 10\n1 0 20\n0 5 10\n1 5 20\n"))
    assert c.stick_source == "none" and len(c.faults[""]) == 1 and "ONE polyline" in c.notes[0]


def test_no_separators_in_a_table_means_one_stick_per_fault_and_says_so(tmp_path):
    r = parse_sticks(write(tmp_path, "t.csv", "x,y,z,fault\n0,0,1,A\n1,0,2,A\n0,9,1,B\n1,9,2,B\n"))
    assert r.stick_source == "none" and [len(v) for v in r.faults.values()] == [1, 1] and any("ONE polyline" in n for n in r.notes)


def test_blank_lines_in_a_table_without_stick_column_split_sticks(tmp_path):
    r = parse_sticks(write(tmp_path, "t.csv", "x,y,z,fault\n0,0,1,A\n1,0,2,A\n\n0,9,1,A\n1,9,2,A\n"))
    assert r.stick_source == "separators" and len(r.faults["A"]) == 2


def test_time_named_column_is_flagged(tmp_path):
    r = parse_sticks(write(tmp_path, "t.csv", "x,y,twt,fault\n0,0,1,A\n1,0,2,A\n"))
    assert any("time-domain" in n for n in r.notes)


@pytest.mark.parametrize("text,match", [
    ("x,y,fault\n1,2,A\n", "no column for z"),
    ("x,y,z,depth\n1,2,3,4\n", "more than one column could be 'z'"),
    ("x,y,z,fault\n1,2,3\n", "3 values but the header has 4"),
    ("x,y,z,fault\n1,2,abc,A\n", "not a number"),
    ("x,y,z,fault\n1,2,3,\n", "fault name is empty"),
    ("x,y,z,stick\n1,2,3,\n", "stick id is empty"),
    ("x,y,z,fault\n1,2,nan,A\n", "not finite"),
    ("1 2 3 4\n", "without a header"),
    ("1 2\n", "without a header"),
    ("1 2 3\n4 5 x\n", "not a number"),
    ("# only a comment\n\n", "no data"),
    ("x,y,z\n", "no points"),
])
def test_reader_refusals(tmp_path, text, match):
    with pytest.raises(FaultError, match=match):
        parse_sticks(write(tmp_path, "bad.txt", text))


def test_binary_and_missing_files_are_refused(tmp_path):
    b = tmp_path / "bin.dat"
    b.write_bytes(b"\xff\xfe\x00\x80\x81")
    with pytest.raises(FaultError, match="UTF-8"):
        parse_sticks(b)
    with pytest.raises(FaultError, match="cannot read"):
        parse_sticks(tmp_path / "nope.txt")


def test_names_with_spaces_need_a_real_delimiter(tmp_path):
    with pytest.raises(FaultError, match="comma, semicolon or tab"):
        parse_sticks(write(tmp_path, "s.txt", "x y z fault\n1 2 3 North Fault\n"))


# -- building ---------------------------------------------------------------------------------------
def raw(tmp_path, text="x,y,z,fault,stick\n1000,2000,100,A,1\n1000,2000,300,A,1\n1100,2000,100,A,2\n1100,2000,300,A,2\n"):
    return parse_sticks(write(tmp_path, "r.csv", text))


def test_units_and_sign_conversion(tmp_path):
    r = raw(tmp_path)
    (f,) = build_faults(r, xy_unit="ft_intl", z_unit="ft_us", z_is="depth")
    assert f.sticks[0][0] == pytest.approx([1000 * 0.3048, 2000 * 0.3048, 100 * 1200 / 3937])
    (g,) = build_faults(r, xy_unit="m", z_unit="m", z_is="elevation")
    assert g.sticks[0][:, 2].tolist() == [-100.0, -300.0]
    assert g.info == dict(xy_unit="m", z_unit="m", z_is="elevation", stick_source="column", file="r.csv")
    with pytest.raises(Exception, match="ambiguous"):
        build_faults(r, xy_unit="ft", z_unit="m", z_is="depth")
    with pytest.raises(FaultError, match="z_is"):
        build_faults(r, xy_unit="m", z_unit="m", z_is="time")


def test_unnamed_fault_takes_the_given_name_or_is_refused(tmp_path):
    r = parse_sticks(write(tmp_path, "h.txt", "0 0 1\n1 0 2\n"))
    with pytest.raises(FaultError, match="give the fault a name"):
        build_faults(r, xy_unit="m", z_unit="m", z_is="depth")
    (f,) = build_faults(r, xy_unit="m", z_unit="m", z_is="depth", fault_name="  Main ")
    assert f.name == "Main"


def test_names_differing_only_by_case_are_refused(tmp_path):
    r = raw(tmp_path, "x,y,z,fault\n0,0,1,Fa\n1,0,2,Fa\n0,5,1,fa\n1,5,2,fa\n")
    with pytest.raises(FaultError, match="only by case"):
        build_faults(r, xy_unit="m", z_unit="m", z_is="depth")


def test_fault_validation():
    ok = np.array([[0, 0, 0], [1, 1, 1.0]])
    with pytest.raises(FaultError, match="name"):
        Fault(" ", (ok,))
    with pytest.raises(FaultError, match="at least one stick"):
        Fault("F", ())
    with pytest.raises(FaultError, match="at least 2 points"):
        Fault("F", (ok[:1],))
    with pytest.raises(FaultError, match="non-finite"):
        Fault("F", (np.array([[0, 0, 0], [np.inf, 1, 1.0]]),))
    with pytest.raises(FaultError, match="zero length"):
        Fault("F", (np.array([[1, 1, 1], [1, 1, 1.0]]),))
    with pytest.raises(FaultError, match="at least 2 points"):
        Fault("F", (np.zeros((3, 2)),))


# -- plane fit ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("dip,dipdir", [(60, 90), (45, 0), (30, 200), (75, 315), (10, 45), (89, 130)])
def test_fit_recovers_dip_direction_and_strike(dip, dipdir):
    f = Fault("F", plane_sticks(dip, dipdir, [0, 300, 700]))
    fit = f.fit()
    assert fit.dip_deg == pytest.approx(dip, abs=1e-6)
    assert fit.dip_direction_deg == pytest.approx(dipdir, abs=1e-6)
    assert fit.axis_azimuth_deg == pytest.approx((dipdir - 90) % 180, abs=1e-6)
    assert fit.rms == pytest.approx(0, abs=1e-8) and fit.max_residual == pytest.approx(0, abs=1e-8)
    assert fit.n_sticks == 3 and fit.n_points == 18


def test_vertical_fault_has_dip_90_and_no_dip_direction():
    sticks = tuple(np.array([[100 + k * 50.0, 200 + k * 50.0 * 0.5, 0.0], [100 + k * 50.0, 200 + k * 50.0 * 0.5, 800.0]]) for k in range(4))
    fit = fit_plane(Fault("V", sticks))
    assert fit.dip_deg == pytest.approx(90, abs=1e-6) and fit.dip_direction_deg is None
    assert fit.axis_azimuth_deg == pytest.approx(math.degrees(math.atan2(50, 25)) % 180, abs=1e-6)
    assert fit.horizontal_shift(1000) == 0.0


def test_single_stick_dip_is_undetermined_never_a_number():
    fit = Fault("F", plane_sticks(60, 90, [0])).fit()
    assert fit.dip_deg is None and fit.axis_azimuth_deg is None and "seismic section" in fit.reason
    assert fit.horizontal_shift(500) is None


def test_collinear_points_are_undetermined():
    s = (np.array([[0, 0, 0], [1, 1, 1.0]]), np.array([[2, 2, 2], [3, 3, 3.0]]))
    fit = Fault("F", s).fit()
    assert fit.dip_deg is None and "one line" in fit.reason and fit.conditioning == 0.0


def test_noise_gives_rms_and_ill_conditioning_is_warned():
    rng = np.random.default_rng(1)
    sticks = tuple(s + rng.normal(0, 5.0, s.shape) for s in plane_sticks(70, 120, [0, 400, 800]))
    fit = Fault("F", sticks).fit()
    assert 1.0 < fit.rms < 10.0 and fit.dip_deg == pytest.approx(70, abs=3)
    thin = Fault("T", plane_sticks(70, 120, [0, 1, 2], length=1000.0)).fit()           # three sticks 1 m apart: almost a line
    assert thin.conditioning < 0.01 and any("poorly constrained" in w for w in thin.warnings)
    assert not Fault("F", plane_sticks(70, 120, [0, 400, 800])).fit().warnings


def test_horizontal_shift():
    fit = Fault("F", plane_sticks(45, 0, [0, 500])).fit()
    assert fit.horizontal_shift(1000) == pytest.approx(1000) and fit.horizontal_shift(-1000) == pytest.approx(1000)
    flat = Fault("H", (np.array([[0, 0, 5.0], [10, 0, 5.0]]), np.array([[0, 10, 5.0], [10, 10, 5.0]]))).fit()
    assert flat.dip_deg == pytest.approx(0, abs=1e-9) and flat.dip_direction_deg is None and flat.horizontal_shift(10) == math.inf
    assert flat.axis_azimuth_deg is None


# -- trace -----------------------------------------------------------------------------------------------
def test_trace_of_a_dipping_fault_moves_with_depth_exactly():
    f = Fault("F", plane_sticks(60, 90, [0, 300, 700, 1000]))
    for d in (0.0, 500.0, 1000.0):
        tr = trace_at(f, d)
        assert len(tr.xy) == 4 and not tr.skipped and not tr.warnings
        assert tr.xy[:, 0] == pytest.approx(d / math.tan(math.radians(60)))
        assert sorted(tr.xy[:, 1]) == pytest.approx([-1000, -700, -300, 0])
        assert tr.length == pytest.approx(1000.0)


def test_trace_order_is_along_the_fault_not_the_file_order():
    sticks = list(plane_sticks(90.0 - 1e-9, 90, [0, 300, 700, 1000]))
    shuffled = Fault("F", tuple(sticks[i] for i in (2, 0, 3, 1)))
    tr = trace_at(shuffled, 400.0)
    ys = tr.xy[:, 1]
    assert np.all(np.diff(ys) > 0) or np.all(np.diff(ys) < 0)               # monotonic along the strike, whatever the file order
    assert sorted(tr.used) == [0, 1, 2, 3] and tr.used in ((2, 0, 3, 1), (1, 3, 0, 2))   # offsets 1000, 700, 300, 0 or the reverse


def test_sticks_that_miss_or_cross_twice_are_skipped_with_reasons():
    good = plane_sticks(90.0 - 1e-9, 90, [0, 500])
    short = np.array([[0, 250.0, 0.0], [0, 250.0, 100.0]])                       # depths 0-100 only
    hook = np.array([[0, 750.0, 0.0], [0, 750.0, 600.0], [0, 750.0, 300.0], [0, 750.0, 900.0]])   # crosses 400 three times
    flat = np.array([[0, 900.0, 400.0], [5, 900.0, 400.0]])
    f = Fault("F", (good[0], short, good[1], hook, flat))
    tr = trace_at(f, 400.0)
    assert tr.used == (0, 2) or tr.used == (2, 0)
    assert "does not reach" in tr.skipped[1] and "100" in tr.skipped[1]
    assert "3 times" in tr.skipped[3] and "runs along this depth" in tr.skipped[4]
    assert any("3 stick(s)" in w for w in tr.warnings)


def test_depth_equal_to_a_vertex_is_counted_once_and_endpoints_work():
    s = np.array([[0, 0, 0.0], [10, 0, 100.0], [20, 0, 200.0]])
    s2 = s + np.array([0, 50.0, 0])
    f = Fault("F", (s, s2))
    for d, x in ((100.0, 10.0), (0.0, 0.0), (150.0, 15.0)):
        tr = trace_at(f, d)
        assert len(tr.xy) == 2 and tr.xy[:, 0] == pytest.approx(x), d
    assert trace_at(f, 200.0).xy[:, 0] == pytest.approx([20.0, 20.0])          # the last vertex is on the depth: it counts
    assert trace_at(f, 200.001).xy.shape[0] == 0
    assert trace_at(f, -1.0).xy.shape[0] == 0


def test_no_trace_cases_and_fold_warning():
    f = Fault("F", plane_sticks(60, 90, [0, 300]))
    assert trace_at(f, 1e6).xy.shape == (0, 2) and trace_at(f, 1e6).warnings == ("no stick reaches this depth",)
    one = Fault("F", (f.sticks[0], np.array([[0, 300.0, 0.0], [0, 300.0, 10.0]])))
    t1 = trace_at(one, 500.0)
    assert t1.xy.shape == (1, 2) and "only one stick" in t1.warnings[0] and 1 in t1.skipped
    same = Fault("F", (np.array([[1, 1, 0.0], [1, 1, 10.0]]), np.array([[1, 1, 0.0], [1, 1, 20.0]])))
    assert "one point" in trace_at(same, 5.0).warnings[0]
    # a U-shaped trace: ordering along the main axis would zig-zag
    pts = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 50.01), (50, 50.01)]
    u = Fault("U", tuple(np.array([[x, y, 0.0], [x, y, 100.0]]) for x, y in [(0, 0), (100, 0), (100, 100), (0, 100)]))
    assert any("folds back" in w or "strongly curved" in w for w in trace_at(u, 50.0).warnings)
    with pytest.raises(FaultError, match="number"):
        trace_at(f, float("nan"))


def test_label_and_bounds_and_same_geometry():
    f = Fault("F", plane_sticks(60, 90, [0, 300]))
    assert "2 stick(s), 12 points, dip 60 deg" in f.label()
    assert "dip undetermined" in Fault("G", f.sticks[:1]).label()
    b = f.bounds()
    assert b[0] == pytest.approx(0) and b[4] == pytest.approx(0) and b[5] == pytest.approx(1000 * math.tan(math.radians(60)))
    g = Fault("F", tuple(s.copy() for s in f.sticks))
    assert f.same_geometry(g) and not f.same_geometry(Fault("F", f.sticks[:1]))
    h = Fault("F", (f.sticks[0] + 1e-9, f.sticks[1]))
    assert not f.same_geometry(h)


def test_export_roundtrip(tmp_path):
    f = Fault("North fault, main", plane_sticks(63.3, 71.0, [0, 123.4567, 700]))
    g = Fault("G", plane_sticks(80, 10, [0, 50]))
    p = tmp_path / "out.csv"
    assert export_sticks([f, g], p) == f.n_points + g.n_points
    back = {x.name: x for x in read_fault_sticks(p, xy_unit="m", z_unit="m", z_is="depth")}
    assert set(back) == {"North fault, main", "G"} and back["G"].same_geometry(g) and back["North fault, main"].same_geometry(f)


# -- cases found by deliberately breaking the code ----------------------------------------------------------------
def test_nearly_collinear_points_are_refused_but_a_thin_fault_only_warned():
    a = np.array([[0, 0, 0.0], [0, 0, 1000.0]])
    near = (a, a + np.array([1e-5, 1e-5, 0.0]) )                              # a second stick 1e-5 m away: ratio ~1e-8
    assert "one line" in Fault("F", near).fit().reason
    ratio = Fault("F", plane_sticks(70, 120, [0, 10, 20], length=1000)).fit().conditioning
    assert 0.001 < ratio < 0.1
    assert any("poorly constrained" in w for w in Fault("F", plane_sticks(70, 120, [0, 10, 20], length=1000)).fit().warnings)


def test_rms_is_root_mean_square_of_distances_to_the_fitted_plane():
    rng = np.random.default_rng(5)
    sticks = tuple(s + np.column_stack([np.zeros(6), np.zeros(6), rng.standard_cauchy(6) * 3]) for s in plane_sticks(65, 40, [0, 400, 900]))
    f = Fault("F", sticks)
    fit = f.fit()
    d, dd = math.radians(fit.dip_deg), math.radians(fit.dip_direction_deg)
    n = np.array([-math.sin(d) * math.sin(dd), -math.sin(d) * math.cos(dd), math.cos(d)])
    r = (f.points() - np.array(fit.centroid)) @ n
    assert fit.rms == pytest.approx(math.sqrt(np.mean(r ** 2)), rel=1e-9) and fit.max_residual == pytest.approx(np.abs(r).max())
    assert not math.isclose(fit.rms, float(np.mean(np.abs(r))), rel_tol=1e-3)


def test_dip_direction_is_not_reported_for_almost_vertical_planes():
    assert Fault("F", plane_sticks(89.995, 120, [0, 300], length=0.1)).fit().dip_direction_deg is None
    assert Fault("F", plane_sticks(89.0, 120, [0, 300], length=0.1)).fit().dip_direction_deg == pytest.approx(120, abs=1e-6)


def test_horizontal_shift_uses_cotangent():
    fit = Fault("F", plane_sticks(60, 0, [0, 500])).fit()
    assert fit.horizontal_shift(1000) == pytest.approx(1000 / math.tan(math.radians(60)))
    assert Fault("F", plane_sticks(20, 0, [0, 500])).fit().horizontal_shift(100) == pytest.approx(100 / math.tan(math.radians(20)))


def test_stick_rising_from_the_depth_and_other_edge_cases_of_trace():
    up = np.array([[0, 0, 100.0], [10, 0, 0.0]])                              # starts exactly at 100 m and goes up
    up2 = up + np.array([0, 50.0, 0])
    tr = trace_at(Fault("F", (up, up2)), 100.0)
    assert len(tr.xy) == 2 and tr.xy[:, 0] == pytest.approx(0.0)
    down_up = np.array([[0, 0, 0.0], [10, 0, 200.0], [20, 0, 0.0]])           # crosses 100 m exactly twice
    tr = trace_at(Fault("F", (down_up, up2)), 100.0)
    assert 0 in tr.skipped and "2 times" in tr.skipped[0]
    flat_run = np.array([[0, 0, 0.0], [10, 0, 100.0], [20, 0, 100.0], [30, 0, 200.0]])
    tr = trace_at(Fault("F", (flat_run, up2)), 100.0)
    assert tr.skipped[0] == "runs along this depth"


@pytest.mark.parametrize("dip,dipdir", [(60, 90), (60, 270), (75, 0), (75, 180), (50, 45), (50, 225), (88, 135), (60, 100)])
def test_traces_at_different_depths_run_the_same_way(dip, dipdir):
    f = Fault("F", plane_sticks(dip, dipdir, [0, 200, 450, 800], length=300.0))
    z0, z1 = f.bounds()[4:]
    for d in np.linspace(z0 + 1, z1 - 1, 9):
        tr = trace_at(f, float(d))
        assert len(tr.xy) == 4
        assert np.all(np.diff(tr.xy, axis=0) @ f.strike_direction > 0)
    assert abs(np.hypot(*f.strike_direction) - 1.0) < 1e-12


@pytest.mark.parametrize("dipdir", [0, 30, 90, 135, 180, 270, 315])
def test_strike_direction_sign_convention(dipdir):
    u = Fault("F", plane_sticks(70, dipdir, [0, 300, 700])).strike_direction
    assert u[0] > 1e-9 or (abs(u[0]) <= 1e-12 and u[1] > 0)

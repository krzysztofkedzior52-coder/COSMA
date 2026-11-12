"""Facies from logs: scheme, rules, flags, locating samples in the grid, upscaling to hard cells."""

import numpy as np
import pytest

from cosma.facies import (CodeMap, Condition, Facies, FaciesError, FaciesSetup, Flag, Rule, classify_all, classify_well,
                          from_state, hard_facies, locate_cells, to_state, upscale)
from cosma.grid.boundary import grid_spec_from_cells
from cosma.grid.builder import Layering, build_grid
from cosma.project import Project, WellRecord
from cosma.wells.model import LogSet, Survey

SAND, SILT = Facies(1, "Sand"), Facies(2, "Silt")
SCHEME = (SAND, SILT)


def make_grid(nx=4, ny=3, dx=100.0, dy=100.0, top=1000.0, base=1100.0, nlay=10, slope=0.0, azimuth=0.0, origin=(0.0, 0.0)):
    spec = grid_spec_from_cells(origin[0], origin[1], azimuth, nx, ny, dx, dy)
    X, Y = np.meshgrid(spec.x_edges, spec.y_edges)           # (ny+1, nx+1) local coordinates
    h = [top + slope * X, base + slope * X]
    grid, _ = build_grid(spec, h, ["conformable"], [Layering("fixed_count", n=nlay)])
    return grid


def well(name, x, y, curves=None, md=None, kb=0.0, td=1200.0, source="lg", survey=None):
    rec = WellRecord(name)
    rec.header = dict(x=x, y=y, kb=kb, td=td)
    rec.survey = survey or Survey(np.array([0.0, td]), np.zeros(2), np.zeros(2), "grid")
    if curves is not None:
        rec.logs[source] = LogSet(np.asarray(md, float), {k: np.asarray(v, float) for k, v in curves.items()})
    return rec


def gr_log(step=0.5):
    """Gamma ray: 20 for depth [1000,1030), 70 for [1030,1065), 130 elsewhere (from 990 to 1110 m)."""
    md = np.arange(990.0, 1110.0 + 1e-9, step)
    gr = np.where((md >= 1000) & (md < 1030), 20.0, np.where((md >= 1030) & (md < 1065), 70.0, 130.0))
    return md, gr


def project_with(*recs):
    pr = Project("p")
    for r in recs:
        pr.wells[r.name] = r
    return pr


RULES = (Rule(1, (Condition("GR", "<", 50.0),)), Rule(2, (Condition("GR", ">=", 50.0), Condition("GR", "<", 100.0))))


# -- the setup ----------------------------------------------------------------------------------------------------------

def test_scheme_validation():
    for bad in (dict(code=0, name="x"), dict(code=-1, name="x"), dict(code=1.5, name="x"), dict(code=True, name="x"),
                dict(code=1, name=""), dict(code=1, name="  ")):
        with pytest.raises(FaciesError):
            Facies(**bad)
    r = (Rule(1, (Condition("GR", "<", 1.0),)),)
    with pytest.raises(FaciesError, match="at least one facies"):
        FaciesSetup((), r)
    with pytest.raises(FaciesError, match="codes must be unique"):
        FaciesSetup((SAND, Facies(1, "Other")), r)
    with pytest.raises(FaciesError, match="names must be unique"):
        FaciesSetup((SAND, Facies(2, " sand ")), r)
    with pytest.raises(FaciesError, match="not both"):
        FaciesSetup(SCHEME, r, CodeMap("F", {1: 1}))
    with pytest.raises(FaciesError, match="say how facies"):
        FaciesSetup(SCHEME)
    with pytest.raises(FaciesError, match="not in the scheme"):
        FaciesSetup(SCHEME, (Rule(3, (Condition("GR", "<", 1.0),)),))
    with pytest.raises(FaciesError, match="not in the scheme"):
        FaciesSetup(SCHEME, code_map=CodeMap("F", {1: 9}))
    with pytest.raises(FaciesError, match="not in the scheme"):
        FaciesSetup(SCHEME, flags=(Flag("W", 1.0, 2.0, "md", 7),))
    with pytest.raises(FaciesError):
        FaciesSetup(SCHEME, r, flag_step=0.0)
    for bad in (lambda: Condition("GR", "==", 1.0), lambda: Condition("", "<", 1.0), lambda: Condition("GR", "<", float("nan")),
                lambda: Rule(1, ()), lambda: CodeMap("F", {}), lambda: CodeMap("F", {1.5: 1}),
                lambda: Flag("W", 5.0, 5.0, "md", 1), lambda: Flag("W", 1.0, 5.0, "depth", 1)):
        with pytest.raises(FaciesError):
            bad()


def test_state_roundtrip():
    s = FaciesSetup(SCHEME, RULES, source={"A": "lg"}, flags=(Flag("A", 1.0, 2.5, "tvdss", 2),), flag_step=0.25)
    assert from_state(to_state(s)) == s
    s2 = FaciesSetup(SCHEME, code_map=CodeMap("FAC", {3: 1, 4: 2}), source="lg")
    assert from_state(to_state(s2)) == s2
    assert from_state(None) is None and to_state(None) is None
    with pytest.raises(FaciesError):
        from_state({"scheme": [[1, "A"]], "rules": [{"facies": 5, "conditions": [["GR", "<", 1]]}]})
    with pytest.raises(FaciesError, match="bad facies setup"):
        from_state({})


# -- classifying a well -------------------------------------------------------------------------------------------------

def test_rules_first_match_wins_and_unmatched_stays_undefined():
    md, gr = gr_log()
    wf = classify_well(well("A", 50, 50, {"GR": gr}, md), FaciesSetup(SCHEME, RULES))
    assert wf.source == "lg"
    expect = np.where(gr < 50, 1, np.where(gr < 100, 2, 0))
    assert (wf.code == expect).all()
    # an earlier broad rule shadows a later narrow one
    both = (Rule(2, (Condition("GR", "<", 100.0),)), Rule(1, (Condition("GR", "<", 50.0),)))
    wf2 = classify_well(well("A", 50, 50, {"GR": gr}, md), FaciesSetup(SCHEME, both))
    assert set(np.unique(wf2.code)) == {0, 2}


def test_operator_edges_and_nan():
    md = np.array([1000.0, 1001.0, 1002.0, 1003.0])
    gr = np.array([49.999, 50.0, np.nan, 50.001])
    for op, want in (("<", [1, 0, 0, 0]), ("<=", [1, 1, 0, 0]), (">", [0, 0, 0, 1]), (">=", [0, 1, 0, 1])):
        s = FaciesSetup(SCHEME, (Rule(1, (Condition("GR", op, 50.0),)),))
        assert list(classify_well(well("A", 0, 0, {"GR": gr}, md), s).code) == want, op


def test_rules_on_two_curves_must_all_hold():
    md = np.arange(1000.0, 1010.0)
    gr = np.array([20.0] * 5 + [20.0] * 5)
    rho = np.array([2.3] * 5 + [2.7] * 5)
    s = FaciesSetup(SCHEME, (Rule(1, (Condition("GR", "<", 50.0), Condition("RHOB", "<", 2.5))),))
    assert list(classify_well(well("A", 0, 0, {"GR": gr, "RHOB": rho}, md), s).code) == [1] * 5 + [0] * 5


def test_missing_curve_log_survey_and_source_are_reported():
    md, gr = gr_log()
    s = FaciesSetup(SCHEME, RULES)
    with pytest.raises(FaciesError, match="no curve GR"):
        classify_well(well("A", 0, 0, {"XX": gr}, md), s)
    with pytest.raises(FaciesError, match="no log"):
        classify_well(well("A", 0, 0), s)
    r = well("A", 0, 0, {"GR": gr}, md)
    r.logs["other"] = r.logs["lg"]
    with pytest.raises(FaciesError, match="several log sources"):
        classify_well(r, s)
    assert classify_well(r, FaciesSetup(SCHEME, RULES, source="other")).source == "other"
    assert classify_well(r, FaciesSetup(SCHEME, RULES, source={"A": "lg"})).source == "lg"
    with pytest.raises(FaciesError, match="no log source 'zzz'"):
        classify_well(r, FaciesSetup(SCHEME, RULES, source="zzz"))
    r2 = well("B", 0, 0, {"GR": gr}, md)
    r2.survey = None
    with pytest.raises(FaciesError, match="deviation survey"):
        classify_well(r2, s)


def test_log_beyond_the_trajectory_is_dropped_and_counted():
    md = np.arange(0.0, 300.0, 1.0)
    r = well("A", 0, 0, {"GR": np.full(md.size, 20.0)}, md, td=200.0)
    wf = classify_well(r, FaciesSetup(SCHEME, RULES))
    assert wf.md.max() <= 200.0 and wf.outside_path == 99
    r2 = well("B", 0, 0, {"GR": np.full(5, 20.0)}, np.arange(500.0, 505.0), td=200.0)
    with pytest.raises(FaciesError, match="outside the well trajectory"):
        classify_well(r2, FaciesSetup(SCHEME, RULES))


def test_code_map():
    md = np.arange(1000.0, 1008.0)
    fac = np.array([10, 10, 20, 20, 30, np.nan, 30, 10], float)
    s = FaciesSetup(SCHEME, code_map=CodeMap("FAC", {10: 1, 20: 2}))
    wf = classify_well(well("A", 0, 0, {"FAC": fac}, md), s)
    assert list(wf.code) == [1, 1, 2, 2, 0, 0, 0, 1]
    assert wf.unmapped == {30: 2}                                   # reported, not guessed
    with pytest.raises(FaciesError, match="whole-number"):
        classify_well(well("A", 0, 0, {"FAC": fac + 0.5}, md), s)
    with pytest.raises(FaciesError, match="no curve 'FAC'"):
        classify_well(well("A", 0, 0, {"GR": fac}, md), s)


def test_flags_override_the_log_and_depth_types_are_resolved():
    md, gr = gr_log()
    flag = Flag("A", 1000.0, 1010.0, "md", 2)                      # the log says sand there
    wf = classify_well(well("A", 0, 0, {"GR": gr}, md), FaciesSetup(SCHEME, RULES, flags=(flag,)))
    inside = (wf.md >= 1000.0) & (wf.md <= 1010.0)
    assert (wf.code[inside] == 2).all() and wf.flagged == int(inside.sum())
    assert (wf.code[wf.md > 1010.0][:20] == 1).all()
    # tvd counts from KB, tvdss from the datum: a KB of 25 m shifts the MD of a given TVDSS
    r = well("A", 0, 0, {"GR": gr}, md, kb=25.0)
    by_tvd = classify_well(r, FaciesSetup(SCHEME, RULES, flags=(Flag("A", 1000.0, 1010.0, "tvd", 2),)))
    by_tvdss = classify_well(r, FaciesSetup(SCHEME, RULES, flags=(Flag("A", 1000.0, 1010.0, "tvdss", 2),)))
    assert by_tvd.md[by_tvd.code == 2].min() == pytest.approx(1000.0, abs=0.5)
    assert by_tvdss.md[by_tvdss.code == 2].min() == pytest.approx(1025.0, abs=0.5)


def test_flags_only_wells_are_sampled_at_the_flag_step():
    r = well("F", 50, 50)
    s = FaciesSetup(SCHEME, flags=(Flag("F", 1000.0, 1010.0, "md", 1), Flag("F", 1040.0, 1050.0, "md", 2)), flag_step=0.5)
    wf = classify_well(r, s)
    assert wf.source is None and wf.md.size == 40 and wf.length.sum() == pytest.approx(20.0)
    assert (wf.code[wf.md < 1020] == 1).all() and (wf.code[wf.md > 1020] == 2).all()
    with pytest.raises(FaciesError, match="flag step"):
        classify_well(r, FaciesSetup(SCHEME, flags=s.flags))
    # a flag base above its top is caught when the flag is made; one that has no extent along the well is caught here
    with pytest.raises(FaciesError, match="flag in F"):
        classify_well(r, FaciesSetup(SCHEME, flags=(Flag("F", 5000.0, 5100.0, "tvdss", 1),), flag_step=1.0))


def test_classify_all_reports_who_was_skipped_and_why():
    md, gr = gr_log()
    a = well("A", 0, 0, {"GR": gr}, md)
    b = well("B", 0, 0)                                           # no log, no flags
    c = well("C", 0, 0, {"GR": gr}, md)
    c.survey = None
    pr = project_with(a, b, c)
    wf, skipped = classify_all(pr, FaciesSetup(SCHEME, RULES, flags=(Flag("Ghost", 1.0, 2.0, "md", 1),)))
    assert list(wf) == ["A"] and "no log" in skipped["B"] and "survey" in skipped["C"] and "does not exist" in skipped["Ghost"]
    wf2, sk2 = classify_all(pr, FaciesSetup(SCHEME, RULES), wells=["A"])
    assert list(wf2) == ["A"] and sk2 == {}
    # without rules, only flagged wells take part
    wf3, sk3 = classify_all(pr, FaciesSetup(SCHEME, flags=(Flag("A", 1000.0, 1001.0, "md", 1),), flag_step=0.5))
    assert list(wf3) == ["A"] and "no flags" in sk3["B"]


# -- finding cells ------------------------------------------------------------------------------------------------------

def test_locate_cells_flat_grid():
    g = make_grid()                                                # cells 100 m, layers 10 m from 1000 m
    X = np.array([50.0, 399.9, 250.0, 250.0, 250.0, 250.0, -1.0, 400.0, 50.0])
    Y = np.array([50.0, 299.9, 150.0, 150.0, 150.0, 150.0, 50.0, 50.0, 300.0])
    Z = np.array([1000.0, 1099.9, 1055.0, 1010.0, 999.9, 1100.0, 1050.0, 1050.0, 1050.0])
    i, j, k, why = locate_cells(g, X, Y, Z)
    assert list(i[:4]) == [0, 3, 2, 2] and list(j[:4]) == [0, 2, 1, 1] and list(k[:4]) == [0, 9, 5, 1]   # 1010 -> layer 1
    assert list(why) == [0, 0, 0, 0, 2, 2, 1, 1, 1] and (i[4:] == -1).all()                         # above, below, off-grid


def test_locate_cells_dipping_rotated_and_offset_grid():
    az, org = 30.0, (5000.0, 7000.0)
    g = make_grid(slope=0.1, azimuth=az, origin=org)              # layer tops at 1000 + 0.1 x + 10 k  (x local)
    rng = np.random.default_rng(1)
    lx, ly = rng.uniform(0.0, 400.0, 400), rng.uniform(0.0, 300.0, 400)
    z = rng.uniform(995.0, 1160.0, 400)
    X, Y = g.frame.to_map(lx, ly)
    i, j, k, why = locate_cells(g, X, Y, z)
    kk = np.floor((z - 1000.0 - 0.1 * lx) / 10.0).astype(int)
    ok = (kk >= 0) & (kk <= 9)
    assert (why[ok] == 0).all() and (why[~ok] == 2).all()
    assert (k[ok] == kk[ok]).all() and (i[ok] == np.floor(lx[ok] / 100).astype(int)).all() \
        and (j[ok] == np.floor(ly[ok] / 100).astype(int)).all()


def test_locate_cells_skips_inactive_and_collapsed_cells():
    g = make_grid()
    g.actnum[2, 1, 5] = 0
    i, j, k, why = locate_cells(g, np.array([250.0, 250.0]), np.array([150.0, 150.0]), np.array([1055.0, 1045.0]))
    assert list(why) == [2, 0] and k[1] == 4
    gc = make_grid(nlay=10)
    gc.zcorn[3:, :, 0, :, 0, :] = gc.zcorn[3, 0, 0, 0, 0, 0]       # collapse layers 3-9 in column (0,0) onto the layer-3 top: zero thickness
    i, j, k, why = locate_cells(gc, np.array([50.0]), np.array([50.0]), np.array([1065.0]))
    assert why[0] == 2


def test_locate_cells_refuses_slanted_pillars():
    g = make_grid()
    g.coord[..., 3] += 5.0
    with pytest.raises(FaciesError, match="vertical pillars"):
        locate_cells(g, np.array([50.0]), np.array([50.0]), np.array([1050.0]))


def test_a_deviated_well_is_located_where_its_path_goes():
    g = make_grid(nx=8, ny=3, top=600.0, base=700.0)
    survey = Survey(np.array([0.0, 1200.0]), np.array([45.0, 45.0]), np.array([90.0, 90.0]), "grid")   # 45 deg, heading east
    md = np.arange(800.0, 1100.0, 0.5)
    r = well("D", 10.0, 150.0, {"GR": np.full(md.size, 20.0)}, md, survey=survey)
    wf = classify_well(r, FaciesSetup(SCHEME, RULES))
    assert wf.x.max() > wf.x.min() + 100.0
    i, j, k, why = locate_cells(g, wf.x, wf.y, wf.z)
    ok = why == 0
    assert ok.sum() > 100 and (j[ok] == 1).all()
    assert (i[ok] == np.floor(wf.x[ok] / 100.0).astype(int)).all()
    assert (k[ok] == np.floor((wf.z[ok] - 600.0) / 10.0).astype(int)).all()
    assert i[ok].max() > i[ok].min()                              # it really crosses columns
    # path length per cell is MD length: a 10 m thick layer is crossed along ~14.1 m of well at 45 degrees
    h = upscale({"D": wf}, g, SCHEME, min_coverage=0.5, conflict="drop")
    mid = [(p, c) for p, c, kk in zip(h.path, h.coverage, h.k) if 2 <= kk <= 7]
    assert all(abs(p - 10.0 * np.sqrt(2.0)) < 1.0 for p, _ in mid) and all(c == 1.0 for _, c in mid)


# -- upscaling ----------------------------------------------------------------------------------------------------------

def test_upscale_majority_coverage_and_status():
    g = make_grid()
    md, gr = gr_log()
    pr = project_with(well("A", 50.0, 50.0, {"GR": gr}, md))
    s = FaciesSetup(SCHEME, RULES)
    h = hard_facies(pr, s, g, min_coverage=0.5, conflict="drop")
    col = h.cells[0, 0, :]
    assert list(col) == [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]            # layer 6: half silt, half unclassified, 0.5 >= 0.5
    assert h.cells.sum() == col.sum()                            # nothing leaks into other columns
    row = {int(k): (st, c) for k, st, c in zip(h.k, h.status, h.coverage)}
    assert row[6][0] == "ok" and row[6][1] == pytest.approx(0.5)
    assert row[7][0] == "low_coverage" and row[7][1] == 0.0
    h2 = hard_facies(pr, s, g, min_coverage=0.6, conflict="drop")
    assert list(h2.cells[0, 0, :]) == [1, 1, 1, 2, 2, 2, 0, 0, 0, 0]
    assert h2.counts() == {1: 3, 2: 3} and h.counts() == {1: 3, 2: 4}
    assert (h.path[h.k < 10] > 0).all() and h.samples["total"] == md.size
    assert h.samples["outside_footprint"] == 0 and h.samples["not_in_active_cell"] == int(((md < 1000) | (md >= 1100)).sum())


def test_majority_is_by_length_not_by_sample_count():
    g = make_grid()
    md = np.concatenate([np.arange(1000.0, 1004.0, 1.0), np.arange(1004.0, 1010.0, 3.0)])        # 4 fine samples, 2 coarse
    gr = np.array([20.0] * 4 + [70.0] * 2)
    pr = project_with(well("A", 50.0, 50.0, {"GR": gr}, md))
    h = hard_facies(pr, FaciesSetup(SCHEME, RULES), g, min_coverage=0.5, conflict="drop")
    # sand: samples 1000..1003 cover 1000-1004 (4 m); silt: samples at 1004 and 1007 cover 1004-1010 (6 m)
    assert h.cells[0, 0, 0] == 2


def test_exact_ties_leave_the_cell_undefined():
    g = make_grid()
    md = np.arange(1000.0, 1010.0, 1.0)
    gr = np.array([20.0] * 5 + [70.0] * 5)
    pr = project_with(well("A", 50.0, 50.0, {"GR": gr}, md))
    h = hard_facies(pr, FaciesSetup(SCHEME, RULES), g, min_coverage=0.5, conflict="drop")
    assert h.cells[0, 0, 0] == 0 and list(h.status) == ["tie"] and "exact tie" in h.summary()


def test_wells_that_disagree_in_a_cell_are_a_conflict_with_an_explicit_policy():
    g = make_grid()
    md = np.arange(1000.0, 1010.0, 1.0)
    a = well("A", 40.0, 50.0, {"GR": np.full(10, 20.0)}, md)
    b = well("B", 60.0, 50.0, {"GR": np.full(10, 70.0)}, md)
    c = well("C", 70.0, 60.0, {"GR": np.full(10, 20.0)}, md)
    s = FaciesSetup(SCHEME, RULES)
    h = hard_facies(project_with(a, b), s, g, min_coverage=0.5, conflict="drop")
    assert h.cells[0, 0, 0] == 0 and h.conflicts == [(0, 0, 0, {"A": 1, "B": 2})]
    assert "disagree" in h.summary() and "no hard datum" in h.summary()
    h = hard_facies(project_with(a, b, c), s, g, min_coverage=0.5, conflict="longest")
    assert h.cells[0, 0, 0] == 1 and len(h.conflicts) == 1                     # sand has 20 m of path, silt 10 m
    h = hard_facies(project_with(a, b), s, g, min_coverage=0.5, conflict="longest")
    assert h.cells[0, 0, 0] == 0                                                # equal lengths: still no guess
    h = hard_facies(project_with(a, c), s, g, min_coverage=0.5, conflict="drop")
    assert h.cells[0, 0, 0] == 1 and h.conflicts == []                          # agreeing wells are no conflict


def test_upscale_argument_checks_and_empty_results():
    g = make_grid()
    for mc in (0, -0.1, 1.5, "x"):
        with pytest.raises(FaciesError, match="min_coverage"):
            upscale({}, g, SCHEME, min_coverage=mc, conflict="drop")
    with pytest.raises(FaciesError, match="policy"):
        upscale({}, g, SCHEME, min_coverage=0.5, conflict="whatever")
    pr = project_with(well("B", 0, 0))
    with pytest.raises(FaciesError, match="no well could be classified"):
        hard_facies(pr, FaciesSetup(SCHEME, RULES), g, min_coverage=0.5, conflict="drop")
    md, gr = gr_log()
    far = project_with(well("A", 5000.0, 5000.0, {"GR": gr}, md))                # well outside the grid
    h = hard_facies(far, FaciesSetup(SCHEME, RULES), g, min_coverage=0.5, conflict="drop")
    assert h.cells.sum() == 0 and h.samples["outside_footprint"] == md.size and "NO hard cells" in h.summary()


def test_summary_lists_proportions_unmapped_codes_and_skips():
    g = make_grid()
    md = np.arange(1000.0, 1040.0, 1.0)
    fac = np.array([10.0] * 20 + [30.0] * 10 + [20.0] * 10)
    pr = project_with(well("A", 50.0, 50.0, {"FAC": fac}, md), well("B", 0, 0))
    h = hard_facies(pr, FaciesSetup(SCHEME, code_map=CodeMap("FAC", {10: 1, 20: 2}), flags=()), g, min_coverage=0.5, conflict="drop")
    t = h.summary()
    assert "Sand (code 1): 2 hard cells (66.7%)" in t and "Silt (code 2): 1 hard cells (33.3%)" in t
    assert "30 (10 samples)" in t and "skipped B" in t


def test_upscaling_choices_live_in_the_setup_and_are_required():
    g = make_grid()
    md, gr = gr_log()
    pr = project_with(well("A", 50.0, 50.0, {"GR": gr}, md))
    bare = FaciesSetup(SCHEME, RULES)
    with pytest.raises(FaciesError, match="choose the minimum coverage and the conflict policy"):
        hard_facies(pr, bare, g)
    with pytest.raises(FaciesError, match="choose the minimum"):
        hard_facies(pr, FaciesSetup(SCHEME, RULES, min_coverage=0.5), g)
    chosen = FaciesSetup(SCHEME, RULES, min_coverage=0.6, conflict="drop")
    assert hard_facies(pr, chosen, g).counts() == {1: 3, 2: 3}
    assert hard_facies(pr, chosen, g, min_coverage=0.5).counts() == {1: 3, 2: 4}          # an argument overrides the setup
    for bad in (dict(min_coverage=0.0), dict(min_coverage=1.5), dict(min_coverage="x"), dict(conflict="coin toss")):
        with pytest.raises(FaciesError):
            FaciesSetup(SCHEME, RULES, **bad)
    assert from_state(to_state(chosen)) == chosen


def test_project_only_accepts_a_facies_setup():
    from cosma.project import ProjectError
    pr = Project("p")
    pr.set_facies_setup(FaciesSetup(SCHEME, RULES))
    assert pr.facies_setup is not None
    with pytest.raises(ProjectError):
        pr.set_facies_setup({"scheme": 1})
    pr.set_facies_setup(None)
    assert pr.facies_setup is None

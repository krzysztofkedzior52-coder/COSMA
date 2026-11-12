"""Stratigraphic chart, type well, top mapping, and attaching surfaces to horizons."""

from __future__ import annotations

import numpy as np
import pytest

from cosma.imports import (ImportRejected, advice_for, import_file, inspect, placement_for, scripted)
from cosma.placement import name_match_uid, parse_hint, suggest_horizons
from cosma.project import Action, Project, ProjectError, WellLink
from cosma.strat import BASE_UID, StratChart, StratError
from test_imports import UM, _w
from test_surfaces import write_irap

MD = {"z_unit": "m", "depth_type": "md"}
REPL, SKIP = Action("replace"), Action("skip")
TOPS1 = "well,top,depth\nW-1,Basement,2000\nW-1,Seal,1000\nW-1,Res,1200\nW-1,Mt Simon,1500\n"   # deliberately unordered
SURV = "well,md,incl,azim\n{w},0,0,0\n{w},3000,0,0\n"


def _typewell(tmp_path, tops=TOPS1):
    """W-1 (KB 25, vertical) with tops; chart built from it."""
    pr = Project("t")
    import_file("well_headers", _w(tmp_path, "h.csv", "well,x,y,kb\nW-1,1000,2000,25\nW-2,3000,2000,25\n"), scripted(UM), project=pr)
    import_file("deviation", _w(tmp_path, "d.csv", SURV.format(w="W-1") + SURV.format(w="W-2").split("\n", 1)[1]
                                .replace("0,0,0\n0,", "W-2,0,0,0\nW-2,", 1)), scripted({"z_unit": "m", "azimuth_reference": "grid"}), project=pr) \
        if False else None
    dev = "well,md,incl,azim\nW-1,0,0,0\nW-1,3000,0,0\nW-2,0,0,0\nW-2,3000,0,0\n"
    import_file("deviation", _w(tmp_path, "d.csv", dev), scripted({"z_unit": "m", "azimuth_reference": "grid"}), project=pr)
    import_file("tops", _w(tmp_path, "t.csv", tops), scripted(MD), project=pr)
    pr.set_type_well("W-1")
    pr.build_chart()
    return pr


def _plane(tmp_path, name, depth, x0=0.0):
    p = tmp_path / name
    write_irap(p, 5, 5, x0, 0.0, 1000.0, 1000.0, 0.0, lambda x, y: depth)
    return p


# -- chart model -----------------------------------------------------------------------------------------
def test_chart_from_picks_orders_by_depth_and_shares_horizons():
    c = StratChart.from_picks([("Basement", 2000), ("Seal", 1000), ("Res", 1200), ("Mt Simon", 1500)])
    assert [i.name for i in c.intervals] == ["Seal", "Res", "Mt Simon"] and c.base_name == "Basement"
    hs = c.horizons()
    assert [h.label for h in hs] == ["Top of Seal", "Base of Seal / Top of Res", "Base of Res / Top of Mt Simon",
                                      "Base of Mt Simon (Basement)"]
    assert hs[1].uid == c.intervals[1].uid and hs[3].uid == BASE_UID          # one object per boundary


def test_chart_refuses_ties_and_too_few_picks_and_duplicate_names():
    with pytest.raises(StratError, match="same depth"):
        StratChart.from_picks([("A", 1), ("B", 1), ("C", 2)])
    with pytest.raises(StratError, match="at least two"):
        StratChart.from_picks([("A", 1)])
    c = StratChart(["A", "B"], "Base")
    with pytest.raises(StratError, match="already the name or alias"):
        c.rename(c.intervals[0].uid, "b")
    with pytest.raises(StratError, match="already exists"):
        c.add_interval("A ", 1)
    with pytest.raises(StratError, match="empty"):
        c.rename(c.intervals[0].uid, "  ")


def test_rename_keeps_alias_and_reorder_keeps_uids():
    c = StratChart(["A", "B", "C"], "Base")
    a, b, cc = (i.uid for i in c.intervals)
    c.rename(b, "Reservoir")
    assert c.match_name("B") == b and c.match_name("reservoir") == b
    c.move(cc, 0)
    assert [i.uid for i in c.intervals] == [cc, a, b]
    assert c.horizon(a).label == "Base of C / Top of A"          # horizon follows its interval
    c.set_contact(a, "unconformity")
    assert c.horizon(a).contact == "unconformity"
    with pytest.raises(StratError, match="contact"):
        c.set_contact(a, "faulted")
    with pytest.raises(StratError, match="outside"):
        c.move(a, 7)


# -- project: type well and chart ---------------------------------------------------------------------------
def test_chart_built_from_type_well_tops(tmp_path):
    pr = _typewell(tmp_path)
    assert pr.type_well == "W-1" and pr.chart.names() == ["Seal", "Res", "Mt Simon", "Basement"]
    with pytest.raises(ProjectError, match="already has a stratigraphic chart"):
        pr.build_chart()
    with pytest.raises(ProjectError, match="choose the well"):
        Project().build_chart()


def test_chart_needs_two_tops_and_orders_mixed_depth_types_through_the_trajectory(tmp_path):
    pr = Project()
    import_file("well_headers", _w(tmp_path, "h.csv", "well,x,y,kb\nW-1,0,0,25\n"), scripted(UM), project=pr)
    import_file("tops", _w(tmp_path, "t.csv", "well,top,depth\nW-1,A,100\n"), scripted(MD), project=pr)
    with pytest.raises(ProjectError, match="at least two"):
        pr.build_chart("W-1")
    import_file("tops", _w(tmp_path, "t2.csv", "well,top,depth\nW-1,B,375\n"), scripted({"z_unit": "m", "depth_type": "tvdss"}), project=pr)
    with pytest.raises(ProjectError, match="different depth types"):
        pr.build_chart("W-1")                                   # no survey: cannot order md against tvdss
    import_file("deviation", _w(tmp_path, "d.csv", "well,md,incl,azim\nW-1,0,0,0\nW-1,1000,0,0\n"),
                scripted({"z_unit": "m", "azimuth_reference": "grid"}), project=pr)
    c = pr.build_chart("W-1")                                   # A: MD 100 (tvdss 75); B: tvdss 375 (MD 400)
    assert c.names() == ["A", "B"]


def test_rename_horizon_renames_picks_in_every_well(tmp_path):
    pr = _typewell(tmp_path)
    import_file("tops", _w(tmp_path, "t2.csv", "well,top,depth\nW-2,Seal,1010\n"), scripted(MD), project=pr)
    uid = pr.chart.match_name("Seal")
    pr.rename_horizon(uid, "Eau Claire")
    assert "Eau Claire" in pr.wells["W-1"].tops and "Seal" not in pr.wells["W-1"].tops
    assert pr.wells["W-2"].tops["Eau Claire"].depth == 1010.0
    assert pr.wells["W-1"].tops["Eau Claire"].name == "Eau Claire"


def test_cannot_remove_an_interval_with_data_on_it(tmp_path):
    pr = _typewell(tmp_path)
    uid = pr.chart.match_name("Res")
    with pytest.raises(ProjectError, match="data is attached.*pick in W-1"):
        pr.remove_interval(uid)
    pr.chart.add_interval("Empty", 1)
    pr.remove_interval(pr.chart.match_name("Empty"))
    assert pr.chart.names() == ["Seal", "Res", "Mt Simon", "Basement"]


def test_reordering_the_chart_against_the_wells_is_reported(tmp_path):
    pr = _typewell(tmp_path)
    assert pr.order_issues() == []
    pr.chart.move(pr.chart.match_name("Res"), 0)
    issues = pr.order_issues()
    assert any("W-1: Seal (1000) is not below Res (1200)" in m for m in issues)


# -- tops with a chart --------------------------------------------------------------------------------------------
def test_tops_of_another_well_are_mapped_to_the_chart(tmp_path):
    pr = _typewell(tmp_path)
    f = _w(tmp_path, "w2.csv", "well,top,depth\nW-2,SEAL,1020\nW-2,TOP_RES,1230\nW-2,Odd,1300\n")
    prev = inspect("tops", f, pr)
    assert prev.top_names == ["SEAL", "TOP_RES", "Odd"]
    assert prev.top_suggestions["SEAL"][0] == pr.chart.match_name("Seal")        # case-insensitive match
    assert prev.top_suggestions["TOP_RES"][0] is None and prev.top_suggestions["Odd"][0] is None
    with pytest.raises(ImportRejected, match="top 'TOP_RES': map it to a horizon.*top 'Odd'"):
        import_file("tops", f, scripted(MD), project=pr)
    res_uid = pr.chart.match_name("Res")
    m = {"TOP_RES": res_uid, "Odd": "ignore"}
    with pytest.raises(ImportRejected, match="not a horizon of the chart"):
        import_file("tops", f, scripted({**MD, "top_map": {**m, "Odd": "zzz"}}), project=pr)
    import_file("tops", f, scripted({**MD, "top_map": m}), project=pr)
    assert {k: t.depth for k, t in pr.wells["W-2"].tops.items()} == {"Seal": 1020.0, "Res": 1230.0}
    assert pr.chart.match_name("TOP_RES") == res_uid               # remembered for the next well
    nxt = inspect("tops", _w(tmp_path, "w3.csv", "well,top,depth\nW-2,top_res,1,\n".replace(",1,\n", ",1\n")), pr)
    assert nxt.top_suggestions["top_res"][0] == res_uid


def test_two_picks_cannot_land_on_one_horizon(tmp_path):
    pr = _typewell(tmp_path)
    f = _w(tmp_path, "w2.csv", "well,top,depth\nW-2,Seal,1020\nW-2,TopSeal,1030\n")
    uid = pr.chart.match_name("Seal")
    with pytest.raises(ImportRejected, match="both map to 'Seal'"):
        import_file("tops", f, scripted({**MD, "top_map": {"Seal": uid, "TopSeal": uid}}), project=pr)


def test_picks_out_of_chart_order_are_advised(tmp_path):
    pr = _typewell(tmp_path)
    f = _w(tmp_path, "w2.csv", "well,top,depth\nW-2,Seal,1300\nW-2,Res,1200\n")
    prev = inspect("tops", f, pr)
    adv = advice_for(prev, {**MD})
    assert any("W-2: Res (1200) is not below Seal (1300)" in m for m in adv)
    res = import_file("tops", f, scripted(MD), project=pr)                       # warned, not blocked (faults happen)
    assert any(n.startswith("order: W-2: Res") for n in res.notes)


def test_first_tops_import_without_a_chart_is_untouched(tmp_path):
    pr = Project()
    import_file("well_headers", _w(tmp_path, "h.csv", "well,x,y,kb\nW-1,0,0,25\n"), scripted(UM), project=pr)
    prev = inspect("tops", _w(tmp_path, "t.csv", TOPS1), pr)
    assert prev.top_names == [] and prev.advisor is None


# -- surfaces on horizons ---------------------------------------------------------------------------------------------
def test_name_hints():
    c = StratChart(["Seal", "Res", "Mt Simon"], "Basement")
    h = {i.name: i.uid for i in c.intervals}
    assert parse_hint("Top_Seal") == ("top", "Seal") and parse_hint("BASE-Res") == ("base", "Res")
    assert parse_hint("Basement") == ("top", "Basement")                 # 'Base' must be followed by a name
    assert name_match_uid(c, "Top_Seal") == h["Seal"]
    assert name_match_uid(c, "Base_Seal") == h["Res"]                    # base of Seal IS top of Res
    assert name_match_uid(c, "base_mt_simon") == BASE_UID
    assert name_match_uid(c, "Basement") == BASE_UID
    assert name_match_uid(c, "Top_Mystery") is None


def test_surface_needs_a_chart_but_works_without_a_project(tmp_path):
    p = _plane(tmp_path, "top_seal.irap", 975.0)
    s = import_file("surface", p, scripted(UM))                          # scripting without project: unchanged
    assert s.values[0, 0] == 975.0
    with pytest.raises(ImportRejected, match="define the stratigraphic chart first"):
        import_file("surface", p, scripted(UM), project=Project())


def test_surface_suggestion_by_depth_and_name_and_confirmation(tmp_path):
    pr = _typewell(tmp_path)
    import_file("tops", _w(tmp_path, "w2.csv", "well,top,depth\nW-2,Seal,1010\nW-2,Res,1215\nW-2,Mt Simon,1510\nW-2,Basement,2010\n"),
                scripted(MD), project=pr)
    seal = pr.chart.match_name("Seal")
    # file named for the Seal top and sitting on the Seal picks (975 / 985 TVDSS)
    p = _plane(tmp_path, "Top_Seal.irap", 980.0, x0=0.0)
    prev = inspect("surface", p, pr)
    assert placement_for(prev, {}) is None                                # units not chosen yet
    pl = placement_for(prev, UM)
    assert pl.suggested.uid == seal and pl.suggested.name_match and pl.suggested.n_wells == 2
    assert pl.suggested.misfit == pytest.approx(5.0) and pl.suggested.bias == pytest.approx(0.0)   # +5 and -5 -> median 0
    with pytest.raises(ImportRejected, match="Attach to horizon must be chosen"):
        import_file("surface", p, scripted(UM), project=pr)               # the suggestion is never applied silently
    res = import_file("surface", p, scripted({**UM, "horizon": seal}), project=pr)
    assert pr.surfaces[seal]["Top_Seal"].values[0, 0] == 980.0 and pr.primary[seal] == "Top_Seal"
    assert "Top of Seal" in res.applied[0]


def test_base_of_a_and_top_of_b_are_the_same_horizon(tmp_path):
    pr = _typewell(tmp_path)
    p1 = _plane(tmp_path, "Base_Seal.irap", 1175.0)
    p2 = _plane(tmp_path, "Top_Res.irap", 1176.0)
    res_uid = pr.chart.match_name("Res")
    assert placement_for(inspect("surface", p1, pr), UM).suggested.uid == res_uid
    assert placement_for(inspect("surface", p2, pr), UM).suggested.uid == res_uid
    import_file("surface", p1, scripted({**UM, "horizon": res_uid}), project=pr)
    # attaching a second surface to the same boundary is a conflict whichever name it came with
    with pytest.raises(ImportRejected, match="chart\\|surface\\|%s|choose replace" % res_uid):
        import_file("surface", p2, scripted({**UM, "horizon": res_uid}), project=pr)
    key = f"chart|surface|{res_uid}"
    import_file("surface", p2, scripted({**UM, "horizon": res_uid, "conflict_actions": {key: SKIP}}), project=pr)
    assert list(pr.surfaces[res_uid]) == ["Base_Seal"]
    res = import_file("surface", p2, scripted({**UM, "horizon": res_uid, "conflict_actions": {key: Action("suffix", "_alt")}}), project=pr)
    assert sorted(pr.surfaces[res_uid]) == ["Base_Seal", "Top_Res_alt"] and pr.primary[res_uid] == "Base_Seal"
    assert any("remains the surface the model uses" in n for n in res.notes)
    import_file("surface", p2, scripted({**UM, "horizon": res_uid, "conflict_actions": {key: REPL}}), project=pr)
    assert pr.primary[res_uid] == "Top_Res" and "Base_Seal" not in pr.surfaces[res_uid]


def test_wrong_depth_for_the_named_horizon_is_warned_and_not_suggested(tmp_path):
    pr = _typewell(tmp_path)
    p = _plane(tmp_path, "Top_Seal.irap", 1700.0)                          # far from the Seal pick (975 TVDSS)
    pl = placement_for(inspect("surface", p, pr), UM)
    assert pl.suggested is None
    assert any("points at Top of Seal" in w and "wrong horizon, datum or units" in w for w in pl.warnings)


def test_depth_only_suggestion_needs_a_clear_winner(tmp_path):
    pr = _typewell(tmp_path)
    clear = placement_for(inspect("surface", _plane(tmp_path, "mystery.irap", 1175.0), pr), UM)
    assert clear.suggested.label.startswith("Base of Seal") and not clear.suggested.name_match
    # picks 25 m apart would be needed for a tie; instead make the surface lie between two picks
    amb = placement_for(inspect("surface", _plane(tmp_path, "mystery2.irap", 1087.0), pr), UM)
    assert amb.suggested is None                                          # 1087 is ~88 m from Res(1175), ~413 from the next


def test_surface_in_feet_misplaced_shows_large_bias(tmp_path):
    pr = _typewell(tmp_path)
    p = _plane(tmp_path, "Top_Seal.irap", 975.0 / 0.3048)               # metres value read as feet by mistake
    pl = suggest_horizons(pr, __import__("cosma.surfaces", fromlist=["read_surface"]).read_surface(p, xy_unit="m", z_unit="m"), "Top_Seal")
    assert pl.suggested is None and pl.warnings
    pl2 = placement_for(inspect("surface", p, pr), {"xy_unit": "m", "z_unit": "ft_intl"})
    assert pl2.suggested.uid == pr.chart.match_name("Seal") and pl2.suggested.misfit == pytest.approx(0.0, abs=1e-6)


# -- placement details (found by mutation checks) --------------------------------------------------------------------------
def _tilted(tmp_path, name, fn):
    p = tmp_path / name
    write_irap(p, 5, 5, 0.0, 0.0, 1000.0, 1000.0, 0.0, fn)
    return p


def test_surface_is_sampled_at_each_well_in_x_and_y_and_signs_are_right(tmp_path):
    pr = _typewell(tmp_path)
    import_file("tops", _w(tmp_path, "w2.csv", "well,top,depth\nW-2,Seal,1010\n"), scripted(MD), project=pr)   # tvdss 985 at x=3000
    p = _tilted(tmp_path, "mystery.irap", lambda x, y: 975.0 + 0.005 * (x - 1000.0))                           # 975 at W-1, 985 at W-2
    pl = placement_for(inspect("surface", p, pr), UM)
    seal = next(s for s in pl.candidates if s.label == "Top of Seal")
    assert seal.n_wells == 2 and seal.misfit == pytest.approx(0.0, abs=1e-9)        # x/y swapped would give 5 m
    q = _plane(tmp_path, "mystery2.irap", 990.0)
    seal = next(s for s in placement_for(inspect("surface", q, pr), UM).candidates if s.label == "Top of Seal")
    assert seal.bias == pytest.approx(+10.0) and seal.misfit == pytest.approx(10.0)  # diffs +15 and +5 -> median +10; deeper surface = positive bias


def test_named_horizon_that_is_nearest_but_outside_tolerance_is_not_suggested(tmp_path):
    pr = _typewell(tmp_path)
    pl = placement_for(inspect("surface", _plane(tmp_path, "Top_Seal.irap", 1040.0), pr), UM)   # Seal 65 m away, tolerance 50
    assert pl.candidates[0].label == "Top of Seal" and not pl.candidates[0].fits
    assert pl.suggested is None and pl.warnings


def test_tolerance_scales_with_interval_thickness(tmp_path):
    pr = _typewell(tmp_path)
    pl = placement_for(inspect("surface", _plane(tmp_path, "Top_Seal.irap", 995.0), pr), UM)   # 20 m off; Seal is 200 m thick -> 50 m
    assert pl.suggested.label == "Top of Seal" and pl.suggested.tol == pytest.approx(50.0)


def test_close_horizons_make_a_depth_only_suggestion_ambiguous(tmp_path):
    pr = _typewell(tmp_path, tops="well,top,depth\nW-1,A,1000\nW-1,B,1006\nW-1,C,1500\nW-1,Base,2000\n")   # A 975, B 981 TVDSS
    clear = placement_for(inspect("surface", _plane(tmp_path, "m1.irap", 976.0), pr), UM)       # A 1 m, B 5 m: both within 5 m
    assert clear.suggested.label == "Top of A"                                                    # but A is 5x closer
    amb = placement_for(inspect("surface", _plane(tmp_path, "m2.irap", 978.0), pr), UM)         # 3 m from each
    assert amb.suggested is None


def test_mapping_for_picks_not_in_the_file_is_rejected(tmp_path):
    pr = _typewell(tmp_path)
    f = _w(tmp_path, "w2.csv", "well,top,depth\nW-2,Seal,1020\n")
    with pytest.raises(ImportRejected, match="mapping given for picks not in the file"):
        import_file("tops", f, scripted({**MD, "top_map": {"Ghost": "ignore"}}), project=pr)


def test_surface_suffix_cannot_overwrite_an_earlier_alternative(tmp_path):
    pr = _typewell(tmp_path)
    u = pr.chart.match_name("Res")
    p = _plane(tmp_path, "s.irap", 1200)
    vals = {"xy_unit": "m", "z_unit": "m", "positive": "down", "horizon": u}
    import_file("surface", p, scripted(vals), project=pr)
    key = f"chart|surface|{u}"
    import_file("surface", p, scripted({**vals, "conflict_actions": {key: Action("suffix", "_b")}}), project=pr)
    with pytest.raises(ImportRejected, match="already exists"):
        import_file("surface", p, scripted({**vals, "conflict_actions": {key: Action("suffix", "_b")}}), project=pr)

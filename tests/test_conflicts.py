"""Importing data a well already has: flagged, and replace / skip / suffix must be chosen."""

from __future__ import annotations

import numpy as np
import pytest

from cosma.imports import ImportRejected, import_file, inspect, scripted
from cosma.project import Action, WellLink
from test_imports import UM, _project, _w

REPL, SKIP = Action("replace"), Action("skip")
MD = {"z_unit": "m", "depth_type": "md"}


def SUF(s="_2"):
    return Action("suffix", s)


def _las(tmp_path, name, depths, curves, well="W-1"):
    names = list(curves)
    head = (f"~VERSION INFORMATION\n VERS. 2.0 : CWLS\n WRAP. NO :\n~WELL INFORMATION\n STRT.M {depths[0]} :\n"
            f" STOP.M {depths[-1]} :\n STEP.M 0.5 :\n NULL. -999.25 :\n WELL. {well} :\n~CURVE INFORMATION\n DEPT.M : Depth\n")
    head += "".join(f" {n}.UNIT : {n}\n" for n in names) + "~ASCII\n"
    rows = "\n".join(f"{d} " + " ".join(str(curves[n][i]) for n in names) for i, d in enumerate(depths))
    return _w(tmp_path, name, head + rows + "\n")


def _conf(kind, path, project, values, **extra):
    from cosma.imports import conflicts_for
    prev = inspect(kind, path, project)
    return conflicts_for(prev, {**values, **extra})


# -- tops ---------------------------------------------------------------------------------------------
def _seal(tmp_path, depth, name="t.csv"):
    return _w(tmp_path, name, f"well,top,depth\nW-1,Seal,{depth}\n")


def test_top_at_a_different_depth_is_flagged_with_both_depths(tmp_path):
    pr = _project(tmp_path)
    import_file("tops", _seal(tmp_path, 1000), scripted(MD), project=pr)
    (c,) = _conf("tops", _seal(tmp_path, 1010, "t2.csv"), pr, MD)
    assert (c.well, c.what, c.item, c.identical) == ("W-1", "top", "Seal", False)
    assert c.existing == "1000.00 m MD" and c.new == "1010.00 m MD"
    (same,) = _conf("tops", _seal(tmp_path, 1000, "t3.csv"), pr, MD)
    assert same.identical


def test_top_same_point_in_other_depth_type_is_recognised_as_identical(tmp_path):
    pr = _project(tmp_path)
    import_file("tops", _seal(tmp_path, 1000), scripted(MD), project=pr)
    tvd = float(pr.wells["W-1"].resolved_tops()[0].tvdss + 25.0)           # TVD below KB (KB = 25)
    (c,) = _conf("tops", _seal(tmp_path, repr(tvd), "t2.csv"), pr, {"z_unit": "m", "depth_type": "tvd"})
    assert c.identical and c.new.endswith("TVD")


def test_no_conflict_for_a_new_name_or_a_new_well(tmp_path):
    pr = _project(tmp_path)
    import_file("tops", _seal(tmp_path, 1000), scripted(MD), project=pr)
    assert _conf("tops", _w(tmp_path, "a.csv", "well,top,depth\nW-1,Base,1500\n"), pr, MD) == []
    assert _conf("tops", _w(tmp_path, "b.csv", "well,top,depth\nZZ,Seal,1500\n"), pr, MD) == []


def test_top_conflict_must_be_answered_and_each_answer_works(tmp_path):
    pr = _project(tmp_path)
    import_file("tops", _seal(tmp_path, 1000), scripted(MD), project=pr)
    new = _seal(tmp_path, 1010, "n.csv")
    with pytest.raises(ImportRejected, match="W-1 top Seal: choose"):
        import_file("tops", new, scripted(MD), project=pr)
    assert pr.wells["W-1"].tops["Seal"].depth == 1000.0

    import_file("tops", new, scripted({**MD, "conflict_actions": {"W-1|top|Seal": SKIP}}), project=pr)
    assert pr.wells["W-1"].tops["Seal"].depth == 1000.0 and list(pr.wells["W-1"].tops) == ["Seal"]

    res = import_file("tops", new, scripted({**MD, "conflict_actions": {"W-1|top|Seal": SUF("_v2")}}), project=pr)
    assert {k: t.depth for k, t in pr.wells["W-1"].tops.items()} == {"Seal": 1000.0, "Seal_v2": 1010.0}
    assert any("imported as Seal_v2" in n for n in res.notes)

    import_file("tops", new, scripted({**MD, "conflict_actions": {"W-1|top|Seal": REPL}}), project=pr)
    assert pr.wells["W-1"].tops["Seal"].depth == 1010.0 and pr.wells["W-1"].tops["Seal_v2"].depth == 1010.0


def test_suffix_rules(tmp_path):
    pr = _project(tmp_path)
    import_file("tops", _w(tmp_path, "t.csv", "well,top,depth\nW-1,Seal,1000\nW-1,Seal_2,1100\n"), scripted(MD), project=pr)
    new = _seal(tmp_path, 1010, "n.csv")
    with pytest.raises(ImportRejected, match="already exists - use another suffix"):
        import_file("tops", new, scripted({**MD, "conflict_actions": {"W-1|top|Seal": SUF("_2")}}), project=pr)
    with pytest.raises(ImportRejected, match="give a suffix"):
        import_file("tops", new, scripted({**MD, "conflict_actions": {"W-1|top|Seal": SUF("  ")}}), project=pr)
    with pytest.raises(ImportRejected, match="not in conflict"):
        import_file("tops", _w(tmp_path, "o.csv", "well,top,depth\nW-1,Other,5\n"), scripted(
            {**MD, "conflict_actions": {"W-1|top|Seal": SKIP}}), project=pr)
    with pytest.raises(ImportRejected, match="choose replace, skip or import with suffix"):
        import_file("tops", new, scripted({**MD, "conflict_actions": {"W-1|top|Seal": Action("merge")}}), project=pr)


def test_two_picks_in_one_file_cannot_collide_through_suffix(tmp_path):
    pr = _project(tmp_path)
    import_file("tops", _seal(tmp_path, 1000), scripted(MD), project=pr)
    f = _w(tmp_path, "n.csv", "well,top,depth\nW-1,Seal,1010\nW-1,Seal_b,1020\n")
    with pytest.raises(ImportRejected, match="already exists"):
        import_file("tops", f, scripted({**MD, "conflict_actions": {"W-1|top|Seal": SUF("_b")}}), project=pr)


# -- header / survey: no suffix possible --------------------------------------------------------------------------
def test_header_and_survey_conflicts_allow_replace_or_skip_only(tmp_path):
    pr = _project(tmp_path)
    h2 = _w(tmp_path, "h2.csv", "well,x,y,kb\nW-1,1100,2000,40\n")
    (c,) = _conf("well_headers", h2, pr, UM)
    assert c.allowed == ("replace", "skip") and "x 1000.00" in c.existing and "x 1100.00" in c.new
    with pytest.raises(ImportRejected, match="'suffix' is not possible here"):
        import_file("well_headers", h2, scripted({**UM, "conflict_actions": {"W-1|header|header": SUF()}}), project=pr)
    res = import_file("well_headers", h2, scripted({**UM, "conflict_actions": {"W-1|header|header": SKIP}}), project=pr)
    assert pr.wells["W-1"].header["x"] == 1000.0 and "kept" in res.applied[0]

    same = _w(tmp_path, "h3.csv", "well,x,y,kb\nW-1,1000,2000,25\n")
    assert _conf("well_headers", same, pr, UM)[0].identical

    dev = _w(tmp_path, "d.csv", "well,md,incl,azim\nW-1,0,0,0\nW-1,2000,0,0\n")
    (sv,) = _conf("deviation", dev, pr, {"z_unit": "m", "azimuth_reference": "grid"})
    assert sv.what == "survey" and "MD 0-1500" in sv.existing and "MD 0-2000" in sv.new
    n_before = pr.wells["W-1"].survey.md.size
    import_file("deviation", dev, scripted({"z_unit": "m", "azimuth_reference": "grid",
                                            "conflict_actions": {"W-1|survey|survey": SKIP}}), project=pr)
    assert pr.wells["W-1"].survey.md.size == n_before


# -- logs -----------------------------------------------------------------------------------------------------
D1 = [1000.0, 1000.5, 1001.0, 1001.5]
V = [10.0, 20.0, 30.0, 40.0]


def _logs_project(tmp_path):
    pr = _project(tmp_path)
    import_file("las", _las(tmp_path, "a.las", D1, {"GR": V, "RHOB": [2.3, 2.4, 2.5, 2.6]}), scripted({}), project=pr)
    return pr


def test_overlapping_curve_is_flagged_with_ranges(tmp_path):
    pr = _logs_project(tmp_path)
    f = _las(tmp_path, "b.las", [1001.0, 1001.5, 1002.0], {"GR": [1, 2, 3], "NPHI": [0.1, 0.2, 0.3]})
    (c,) = _conf("las", f, pr, {})
    assert (c.what, c.item, c.identical) == ("curve", "GR", False)
    assert "GR in a: 1000.0-1001.5 m MD" in c.existing and "overlap 0.5 m" in c.new


def test_curve_match_ignores_case_and_identical_data_is_marked(tmp_path):
    pr = _logs_project(tmp_path)
    from cosma.wells.model import LogSet
    lower = LogSet(np.array(D1), {"gr": np.array(V)}, {"gr": ""})          # lasio upper-cases mnemonics; test the project directly
    (c,) = pr.conflicts_logs(lower, "W-1", WellLink("existing", "W-1"))
    assert c.item == "gr" and c.identical
    (d,) = _conf("las", _las(tmp_path, "d.las", D1, {"GR": [10.0, 20.0, 30.0, 41.0]}), pr, {})
    assert not d.identical


def test_same_curve_at_other_depths_is_not_a_conflict(tmp_path):
    pr = _logs_project(tmp_path)
    f = _las(tmp_path, "far.las", [2000.0, 2000.5], {"GR": [1, 2]})
    assert _conf("las", f, pr, {}) == []
    import_file("las", f, scripted({}), project=pr)
    assert sorted(pr.wells["W-1"].logs) == ["a", "far"]


def test_curve_with_only_nulls_in_overlap_range_is_not_a_conflict(tmp_path):
    pr = _logs_project(tmp_path)
    f = _las(tmp_path, "nul.las", [1000.0, 1000.5, 1001.0, 3000.0], {"GR": [-999.25, -999.25, -999.25, 5.0]})
    assert _conf("las", f, pr, {}) == []                       # its defined samples are at 3000 m only


def test_curve_actions_skip_suffix_replace(tmp_path):
    f = lambda: _las(tmp_path, "b.las", [1001.0, 1001.5, 1002.0], {"GR": [1, 2, 3], "NPHI": [0.1, 0.2, 0.3]})
    key = "W-1|curve|GR"
    pr = _logs_project(tmp_path)
    import_file("las", f(), scripted({"conflict_actions": {key: SKIP}}), project=pr)
    assert sorted(pr.wells["W-1"].logs["b"].curves) == ["NPHI"]                      # GR not imported
    np.testing.assert_allclose(pr.wells["W-1"].logs["a"].curves["GR"], V)

    pr = _logs_project(tmp_path)
    import_file("las", f(), scripted({"conflict_actions": {key: SUF("_2")}}), project=pr)
    assert sorted(pr.wells["W-1"].logs["b"].curves) == ["GR_2", "NPHI"]
    assert pr.wells["W-1"].logs["b"].units["GR_2"] == "UNIT"
    np.testing.assert_allclose(pr.wells["W-1"].logs["a"].curves["GR"], V)

    pr = _logs_project(tmp_path)
    import_file("las", f(), scripted({"conflict_actions": {key: REPL}}), project=pr)
    assert sorted(pr.wells["W-1"].logs["a"].curves) == ["RHOB"]                       # old GR removed, RHOB kept
    np.testing.assert_allclose(pr.wells["W-1"].logs["b"].curves["GR"], [1, 2, 3])


def test_replace_that_empties_an_old_log_set_removes_it(tmp_path):
    pr = _project(tmp_path)
    import_file("las", _las(tmp_path, "a.las", D1, {"GR": V}), scripted({}), project=pr)
    import_file("las", _las(tmp_path, "b.las", D1, {"GR": V[::-1]}), scripted({"conflict_actions": {"W-1|curve|GR": REPL}}), project=pr)
    assert list(pr.wells["W-1"].logs) == ["b"]


def test_all_curves_skipped_adds_no_log_set(tmp_path):
    pr = _logs_project(tmp_path)
    import_file("las", _las(tmp_path, "b.las", D1, {"GR": V}), scripted({"conflict_actions": {"W-1|curve|GR": SKIP}}), project=pr)
    assert list(pr.wells["W-1"].logs) == ["a"]


def test_same_file_name_twice_never_overwrites_a_log_set_silently(tmp_path):
    pr = _logs_project(tmp_path)
    f = _las(tmp_path, "a.las", [5000.0, 5000.5], {"GR": [1, 2]})
    import_file("las", f, scripted({}), project=pr)
    import_file("las", _las(tmp_path, "a.las", [6000.0, 6000.5], {"GR": [1, 2]}), scripted({}), project=pr)
    assert sorted(pr.wells["W-1"].logs) == ["a", "a#2", "a#3"]


# -- core ---------------------------------------------------------------------------------------------------------
CORE = "well,md,porosity,perm_mD\nW-1,1000,12.5,50\nW-1,1100,20,300\n"
CP = {"z_unit": "m", "depth_type": "md", "porosity_unit": "percent"}


def _core_project(tmp_path):
    pr = _project(tmp_path)
    import_file("core", _w(tmp_path, "c1.csv", CORE), scripted(CP), project=pr)
    return pr


def test_core_overlap_flagged_per_column(tmp_path):
    pr = _core_project(tmp_path)
    f = _w(tmp_path, "c2.csv", "well,md,porosity,sw\nW-1,1000,13,0.3\nW-1,1050,15,0.4\n")
    (c,) = _conf("core", f, pr, CP)
    assert (c.what, c.item, c.identical) == ("core", "porosity", False)
    assert "2 samples, 1000.0-1100.0 m" in c.existing and "1 at the same depth" in c.new
    same = _conf("core", _w(tmp_path, "c3.csv", CORE), pr, CP)
    assert sorted(c.item for c in same) == ["perm_md", "porosity"] and all(c.identical for c in same)


def test_core_non_overlapping_depths_do_not_conflict(tmp_path):
    pr = _core_project(tmp_path)
    f = _w(tmp_path, "c4.csv", "well,md,porosity\nW-1,1300,13\nW-1,1350,15\n")
    assert _conf("core", f, pr, CP) == []


def test_core_actions(tmp_path):
    f = lambda: _w(tmp_path, "c2.csv", "well,md,porosity,sw\nW-1,1000,13,0.3\nW-1,1050,15,0.4\n")
    key = "W-1|core|porosity"
    pr = _core_project(tmp_path)
    import_file("core", f(), scripted({**CP, "conflict_actions": {key: SKIP}}), project=pr)
    assert sorted(pr.wells["W-1"].core["c2"].columns) == ["sw"]
    pr = _core_project(tmp_path)
    import_file("core", f(), scripted({**CP, "conflict_actions": {key: SUF("_b")}}), project=pr)
    assert sorted(pr.wells["W-1"].core["c2"].columns) == ["porosity_b", "sw"]
    assert pr.wells["W-1"].core["c1"].columns["porosity"][0] == pytest.approx(0.125)
    pr = _core_project(tmp_path)
    import_file("core", f(), scripted({**CP, "conflict_actions": {key: REPL}}), project=pr)
    assert sorted(pr.wells["W-1"].core["c1"].columns) == ["perm_md"]
    assert pr.wells["W-1"].core["c2"].columns["porosity"][0] == pytest.approx(0.13)


def test_core_with_unlike_depth_types_and_no_trajectory_is_flagged_not_compared(tmp_path):
    pr = _project(tmp_path, dev=False)                        # headers only: no survey, so tvd cannot become md
    import_file("core", _w(tmp_path, "c1.csv", CORE), scripted(CP), project=pr)
    f = _w(tmp_path, "c5.csv", "well,depth,porosity\nW-1,990,0.1\n")
    (c,) = _conf("core", f, pr, {"z_unit": "m", "depth_type": "tvd", "porosity_unit": "frac"})
    assert "not comparable" in c.new


# -- atomicity ---------------------------------------------------------------------------------------------------
def test_conflict_actions_do_not_half_apply(tmp_path):
    pr = _project(tmp_path)
    import_file("tops", _w(tmp_path, "t.csv", "well,top,depth\nW-1,Seal,1000\nW-1,Res,1200\n"), scripted(MD), project=pr)
    f = _w(tmp_path, "n.csv", "well,top,depth\nW-1,Seal,1010\nW-1,Res,1210\nW-1,Deep,99999\n")
    with pytest.raises(Exception, match="outside surveyed"):
        import_file("tops", f, scripted({**MD, "conflict_actions": {"W-1|top|Seal": REPL, "W-1|top|Res": REPL}}), project=pr)
    assert {k: t.depth for k, t in pr.wells["W-1"].tops.items()} == {"Seal": 1000.0, "Res": 1200.0}


# -- dialog ---------------------------------------------------------------------------------------------------------
qt = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    return qt.QApplication.instance() or qt.QApplication([])


def test_dialog_lists_conflicts_and_blocks_until_each_is_answered(app, tmp_path):
    from cosma.gui.import_dialog import ImportDialog

    pr = _project(tmp_path)
    import_file("tops", _w(tmp_path, "t.csv", "well,top,depth\nW-1,Seal,1000\nW-1,Res,1200\n"), scripted(MD), project=pr)
    f = _w(tmp_path, "n.csv", "well,top,depth\nW-1,Seal,1010\nW-1,Res,1200\nW-1,New,1300\n")
    d = ImportDialog(inspect("tops", f, pr))
    ok = d.buttons.button(qt.QDialogButtonBox.Ok)
    assert d.conflict_box.isHidden()                               # units not chosen yet: nothing to compare
    d.widgets["z_unit"].setCurrentText("m")
    d.widgets["depth_type"].setCurrentText("md")
    assert not d.conflict_box.isHidden() and d.conflict_table.rowCount() == 2
    assert "2 item(s)" in d.conflict_label.text() and "1 are identical" in d.conflict_label.text()
    texts = [d.conflict_table.item(r, 1).text() for r in range(2)]
    assert texts == ["Seal", "Res   [same data]"]
    assert d.conflict_table.item(0, 2).text() == "1000.00 m MD" and d.conflict_table.item(0, 3).text() == "1010.00 m MD"
    assert not ok.isEnabled() and "choose replace, skip" in d.problem.text()
    (c0, s0), (c1, s1) = d.conflict_rows["W-1|top|Seal"], d.conflict_rows["W-1|top|Res"]
    c0.setCurrentIndex(c0.findData("skip"))
    assert not ok.isEnabled()
    c1.setCurrentIndex(c1.findData("suffix"))
    assert s1.isEnabled() and s1.text() == "_new"                  # editable suggestion, shown
    assert ok.isEnabled()
    s1.setText("Seal")                                             # not a collision for Res...
    assert ok.isEnabled()
    s1.setText("")
    assert not ok.isEnabled() and "give a suffix" in d.problem.text()
    s1.setText("_b")
    assert d.values()["conflict_actions"] == {"W-1|top|Seal": Action("skip"), "W-1|top|Res": Action("suffix", "_b")}
    # changing a parameter that changes nothing keeps the choices; the apply-all box overrides them
    d.all_combo.setCurrentIndex(1)
    assert d.values()["conflict_actions"] == {"W-1|top|Seal": Action("replace"), "W-1|top|Res": Action("replace")}
    d.widgets["z_unit"].setCurrentIndex(0)                          # parameters incomplete again: table goes away
    assert d.conflict_box.isHidden() and d.conflict_table.rowCount() == 0 or d.conflict_box.isHidden()
    d.widgets["z_unit"].setCurrentText("m")
    assert not d.conflict_box.isHidden()
    assert d.values()["conflict_actions"] == {"W-1|top|Seal": Action("replace"), "W-1|top|Res": Action("replace")}   # choices remembered


def test_dialog_header_rows_offer_no_suffix(app, tmp_path):
    from cosma.gui.import_dialog import ImportDialog

    pr = _project(tmp_path)
    d = ImportDialog(inspect("well_headers", _w(tmp_path, "h.csv", "well,x,y,kb\nW-1,1100,2000,40\n"), pr))
    d.widgets["xy_unit"].setCurrentText("m")
    d.widgets["z_unit"].setCurrentText("m")
    combo, _ = d.conflict_rows["W-1|header|header"]
    assert [combo.itemData(i) for i in range(combo.count())] == [None, "replace", "skip"]


def test_dialog_end_to_end_with_conflict_choice(app, tmp_path, monkeypatch):
    from cosma.gui import import_dialog as idlg

    pr = _project(tmp_path)
    import_file("tops", _seal(tmp_path, 1000), scripted(MD), project=pr)

    def fake_exec(self):
        self.widgets["z_unit"].setCurrentText("m")
        self.widgets["depth_type"].setCurrentText("md")
        combo, suf = self.conflict_rows["W-1|top|Seal"]
        combo.setCurrentIndex(combo.findData("suffix"))
        suf.setText("_alt")
        self.accept()
        return qt.QDialog.Accepted

    monkeypatch.setattr(idlg.ImportDialog, "exec", fake_exec)
    import_file("tops", _seal(tmp_path, 1010, "n.csv"), idlg.confirm_with_dialog, project=pr)
    assert {k: t.depth for k, t in pr.wells["W-1"].tops.items()} == {"Seal": 1000.0, "Seal_alt": 1010.0}

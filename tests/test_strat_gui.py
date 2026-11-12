"""Chart editing (headless + dialog), the horizon picker in the import dialog, the new-project flow."""

from __future__ import annotations

import os

import pytest

from cosma.chart_edit import Row, apply_chart_edit, rows_from_chart, validate
from cosma.imports import ImportCancelled, ImportRejected, import_file, inspect, scripted
from cosma.project import Action, Project, ProjectError
from cosma.strat import BASE_UID
from test_imports import UM, _w
from test_strat import MD, TOPS1, _plane, _typewell

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qt = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    return qt.QApplication.instance() or qt.QApplication([])


# -- headless chart editing -------------------------------------------------------------------------------------------
def test_validate_messages():
    assert validate([Row("a", "A"), Row("b", "a")], "Base") == ["name 'a' is used twice"]
    assert "an interval has no name" in validate([Row("a", " ")], "Base")
    assert "the base horizon has no name" in validate([Row("a", "A")], " ")
    assert validate([Row("a", "A")], "a") == ["name 'a' is used twice"]
    assert "a chart needs at least one interval" in validate([], "B")
    assert "contact must be one of" in validate([Row("a", "A", "faulted")], "B")[0]


def test_apply_rename_reorder_add_remove(tmp_path):
    pr = _typewell(tmp_path)
    ch = pr.chart
    seal, res, mts = (ch.match_name(n) for n in ("Seal", "Res", "Mt Simon"))
    rows = [Row(res, "Reservoir"), Row(seal, "Seal", "unconformity"), Row(None, "Interlayer"), Row(mts, "Mt Simon")]
    apply_chart_edit(pr, rows, "Crystalline basement")
    assert [i.name for i in ch.intervals] == ["Reservoir", "Seal", "Interlayer", "Mt Simon"]
    assert ch.base_name == "Crystalline basement"
    assert ch.horizon(seal).contact == "unconformity"
    assert ch.horizon(seal).label == "Base of Reservoir / Top of Seal"
    assert set(pr.wells["W-1"].tops) == {"Reservoir", "Seal", "Mt Simon", "Crystalline basement"}   # picks followed
    assert pr.wells["W-1"].tops["Reservoir"].depth == 1200.0
    assert ch.match_name("Res") == res and ch.match_name("Basement") == BASE_UID                 # old spellings still work


def test_swapping_two_names_works_and_does_not_leave_junk(tmp_path):
    pr = _typewell(tmp_path)
    seal, res = pr.chart.match_name("Seal"), pr.chart.match_name("Res")
    rows = rows_from_chart(pr.chart)
    rows[0].name, rows[1].name = "Res", "Seal"
    apply_chart_edit(pr, rows, pr.chart.base_name)
    assert pr.wells["W-1"].tops["Res"].depth == 1000.0 and pr.wells["W-1"].tops["Seal"].depth == 1200.0
    assert all("\x00" not in a for iv in pr.chart.intervals for a in iv.aliases)


def test_removal_blocked_when_data_is_attached_and_rolled_back(tmp_path):
    pr = _typewell(tmp_path)
    seal, res = pr.chart.match_name("Seal"), pr.chart.match_name("Res")
    rows = rows_from_chart(pr.chart)
    rows = [r for r in rows if r.uid != res]
    rows[0].name = "Changed"
    with pytest.raises(ProjectError, match="cannot remove 'Res': pick in W-1"):
        apply_chart_edit(pr, rows, pr.chart.base_name)
    assert pr.chart.names() == ["Seal", "Res", "Mt Simon", "Basement"]                       # nothing half-applied
    assert "Seal" in pr.wells["W-1"].tops
    with pytest.raises(ProjectError, match="used twice"):
        apply_chart_edit(pr, [Row(seal, "X"), Row(res, "x")], "B")


def test_surface_stays_attached_through_reorder_and_rename(tmp_path):
    pr = _typewell(tmp_path)
    res = pr.chart.match_name("Res")
    import_file("surface", _plane(tmp_path, "Top_Res.irap", 1175.0), scripted({**UM, "horizon": res}), project=pr)
    rows = rows_from_chart(pr.chart)
    rows[1].name = "Reservoir"
    rows.insert(0, rows.pop(1))
    apply_chart_edit(pr, rows, pr.chart.base_name)
    assert pr.chart.horizon(res).name == "Reservoir" and "Top_Res" in pr.surfaces[res]
    assert pr.chart.horizon(res).label == "Top of Reservoir"


def test_manual_chart_without_type_well():
    pr = Project("m")
    ch = pr.create_chart(["Seal", "Res"], "Basement")
    assert ch.names() == ["Seal", "Res", "Basement"]
    with pytest.raises(ProjectError, match="already has"):
        pr.create_chart(["X"], "Y")
    with pytest.raises(ProjectError, match="used twice"):
        Project().create_chart(["A", "a"], "B")


# -- the editor dialog ----------------------------------------------------------------------------------------------------
def test_editor_dialog_edits_and_applies(app, tmp_path):
    from cosma.gui.chart_editor import ChartEditorDialog

    pr = _typewell(tmp_path)
    d = ChartEditorDialog(pr)
    t = d.table
    assert t.rowCount() == 4                                              # 3 intervals + base row
    assert [t.item(r, 1).text() for r in range(4)] == ["Seal", "Res", "Mt Simon", "Basement"]
    assert t.item(1, 3).text() == "Base of Seal / Top of Res"            # shared boundary shown as such
    assert t.item(1, 4).text() == "pick in W-1"
    ok = d.buttons.button(qt.QDialogButtonBox.Ok)
    t.item(1, 1).setText("Reservoir")
    assert t.item(1, 3).text() == "Base of Seal / Top of Reservoir" and t.item(2, 3).text().startswith("Base of Reservoir")
    assert ok.isEnabled()
    t.item(1, 1).setText("seal")
    assert not ok.isEnabled() and "used twice" in d.problem.text()
    t.item(1, 1).setText("Reservoir")
    d.move(1, -1)
    assert [r.name for r in d.rows] == ["Reservoir", "Seal", "Mt Simon"]
    d.add(2)
    assert d.rows[3].name == "New interval 1" and t.item(3, 4).text() == "new"
    d.remove(1)                                                           # Seal has a pick: refused
    assert "Cannot remove 'Seal'" in d.problem.text() and len(d.rows) == 4
    d.remove(3)
    assert [r.name for r in d.rows] == ["Reservoir", "Seal", "Mt Simon"]
    t.item(3, 1).setText("Crystalline")                                  # base row
    d.accept()
    assert pr.chart.names() == ["Reservoir", "Seal", "Mt Simon", "Crystalline"]
    assert pr.wells["W-1"].tops["Crystalline"].depth == 2000.0


def test_editor_dialog_keeps_open_when_the_project_refuses(app, tmp_path):
    from cosma.gui.chart_editor import ChartEditorDialog

    pr = _typewell(tmp_path)
    d = ChartEditorDialog(pr)
    d.rows = [r for r in d.rows if r.name != "Res"]                       # bypass the button to reach the project check
    d.accept()
    assert "cannot remove 'Res'" in d.problem.text() and d.result() != qt.QDialog.Accepted


# -- import dialog: horizon picker ---------------------------------------------------------------------------------------------
def test_dialog_suggests_horizon_after_units_and_user_can_override(app, tmp_path):
    from cosma.gui.import_dialog import ImportDialog

    pr = _typewell(tmp_path)
    seal, res = pr.chart.match_name("Seal"), pr.chart.match_name("Res")
    p = _plane(tmp_path, "Top_Seal.irap", 975.0)
    d = ImportDialog(inspect("surface", p, pr))
    ok = d.buttons.button(qt.QDialogButtonBox.Ok)
    hz = d.widgets["horizon"]
    assert hz.currentData() is None and "once its units are chosen" in d.suggest_label.text()
    d.widgets["xy_unit"].setCurrentText("m")
    d.widgets["z_unit"].setCurrentText("m")
    assert hz.currentData() == seal and ok.isEnabled()                     # suggested, shown, tagged
    assert "Suggested: Top of Seal" in d.suggest_label.text() and "suggested - check" in d.rows["horizon"].text()
    assert "name matches" in hz.itemText(hz.findData(seal)) and "no well picks" not in hz.itemText(hz.findData(seal))
    assert hz.itemText(hz.findData(res)).startswith("Base of Seal / Top of Res")
    hz.activated.emit(hz.findData(res))                                    # the user takes over...
    hz.setCurrentIndex(hz.findData(res))
    d.widgets["positive"].setCurrentText("down")                           # ...and later refreshes do not undo it
    assert d.values()["horizon"] == res
    assert [hz.itemData(i) for i in range(hz.count())] == [None] + [pr.chart.match_name(n) for n in ("Seal", "Res", "Mt Simon")] + [BASE_UID]


def test_dialog_without_clear_fit_leaves_the_choice_to_the_user(app, tmp_path):
    from cosma.gui.import_dialog import ImportDialog

    pr = _typewell(tmp_path)
    d = ImportDialog(inspect("surface", _plane(tmp_path, "Top_Seal.irap", 1700.0), pr))
    d.widgets["xy_unit"].setCurrentText("m")
    d.widgets["z_unit"].setCurrentText("m")
    assert d.widgets["horizon"].currentData() is None and not d.buttons.button(qt.QDialogButtonBox.Ok).isEnabled()
    assert "No horizon fits clearly" in d.suggest_label.text() and "wrong horizon, datum or units" in d.suggest_label.text()


def test_dialog_second_surface_on_same_horizon_shows_a_conflict(app, tmp_path):
    from cosma.gui.import_dialog import ImportDialog

    pr = _typewell(tmp_path)
    res = pr.chart.match_name("Res")
    import_file("surface", _plane(tmp_path, "Base_Seal.irap", 1175.0), scripted({**UM, "horizon": res}), project=pr)
    d = ImportDialog(inspect("surface", _plane(tmp_path, "Top_Res.irap", 1176.0), pr))
    d.widgets["xy_unit"].setCurrentText("m")
    d.widgets["z_unit"].setCurrentText("m")
    assert d.widgets["horizon"].currentData() == res
    assert not d.conflict_box.isHidden() and d.conflict_table.item(0, 1).text() == "Base of Seal / Top of Res"
    assert d.conflict_table.item(0, 2).text() == "Base_Seal" and d.conflict_table.item(0, 3).text() == "Top_Res"
    assert not d.buttons.button(qt.QDialogButtonBox.Ok).isEnabled()


def test_dialog_top_mapping_table_and_chart_edit_button(app, tmp_path, monkeypatch):
    from cosma.gui import import_dialog as idlg
    from cosma.gui.chart_editor import ChartEditorDialog

    pr = _typewell(tmp_path)
    f = _w(tmp_path, "w2.csv", "well,top,depth\nW-2,Seal,1020\nW-2,TOP_RES,1230\nW-2,Odd,1300\n")
    d = idlg.ImportDialog(inspect("tops", f, pr))
    ok = d.buttons.button(qt.QDialogButtonBox.Ok)
    assert d.top_table.rowCount() == 3 and not d.chart_btn.isHidden()
    seal = pr.chart.match_name("Seal")
    assert d.top_rows["Seal"].currentData() == seal and d.top_rows["TOP_RES"].currentData() is None
    d.widgets["z_unit"].setCurrentText("m")
    d.widgets["depth_type"].setCurrentText("md")
    assert not ok.isEnabled() and "map it to a horizon" in d.problem.text()
    cb = d.top_rows["TOP_RES"]
    cb.setCurrentIndex(cb.findData(pr.chart.match_name("Res")))
    d.top_rows["Odd"].setCurrentIndex(d.top_rows["Odd"].findData("ignore"))
    assert ok.isEnabled()
    # the chart is edited from inside the dialog (rename Mt Simon), and the table picks it up
    def fake_exec(self):
        self.rows[2].name = "Mount Simon"
        self.accept()
        return qt.QDialog.Accepted
    monkeypatch.setattr(ChartEditorDialog, "exec", fake_exec)
    d.chart_btn.click()
    labels = [d.top_rows["Odd"].itemText(i) for i in range(d.top_rows["Odd"].count())]
    assert any("Mount Simon" in l for l in labels) and not any(l.endswith("Top of Mt Simon") for l in labels)
    assert d.top_rows["TOP_RES"].currentData() == pr.chart.match_name("Res") and ok.isEnabled()      # selections survive
    assert "Mount Simon" in pr.wells["W-1"].tops


def test_dialog_warns_when_picks_break_chart_order(app, tmp_path):
    from cosma.gui.import_dialog import ImportDialog

    pr = _typewell(tmp_path)
    d = ImportDialog(inspect("tops", _w(tmp_path, "w2.csv", "well,top,depth\nW-2,Seal,1300\nW-2,Res,1200\n"), pr))
    d.widgets["z_unit"].setCurrentText("m")
    d.widgets["depth_type"].setCurrentText("md")
    assert "W-2: Res (1200) is not below Seal (1300)" in d.advice.text() and d.buttons.button(qt.QDialogButtonBox.Ok).isEnabled()


# -- new project -----------------------------------------------------------------------------------------------------------------
def test_new_project_dialog_requires_name_and_an_answer(app):
    from cosma.gui.new_project import NewProjectDialog

    d = NewProjectDialog()
    ok = d.buttons.button(qt.QDialogButtonBox.Ok)
    assert not ok.isEnabled()
    d.name.setText("Michigan")
    assert not ok.isEnabled()                                # the type-well question has no default answer
    d.yes.setChecked(True)
    assert ok.isEnabled() and d.result_tuple() == ("Michigan", True)
    d.no.setChecked(True)
    assert d.result_tuple() == ("Michigan", False)
    d.name.setText("  ")
    assert not ok.isEnabled()


def _files(tmp_path):
    return {
        "well_headers": _w(tmp_path, "h.csv", "well,x,y,kb\nW-1,1000,2000,25\n"),
        "deviation": _w(tmp_path, "d.csv", "well,md,incl,azim\nW-1,0,0,0\nW-1,3000,0,0\n"),
        "tops": _w(tmp_path, "t.csv", TOPS1),
    }


def _confirm(preview):
    vals = {"well_headers": UM, "deviation": {"z_unit": "m", "azimuth_reference": "grid"}, "tops": MD}[preview.kind]
    return dict(vals)


def test_new_project_with_type_well_builds_and_shows_the_chart(app, tmp_path):
    from cosma.gui.new_project import new_project_workflow

    files = _files(tmp_path)
    asked, shown = [], []
    pr = new_project_workflow(
        pick_file=lambda kind, title: files[kind],
        ask=lambda: ("Mount Simon", True),
        confirm=_confirm,
        choose_well=lambda names: asked.append(names) or names[0],
        edit_chart=lambda p: shown.append(p.chart.names()) or True)
    assert pr.name == "Mount Simon" and pr.type_well == "W-1" and asked == []        # one candidate: not asked
    assert pr.chart.names() == ["Seal", "Res", "Mt Simon", "Basement"] and shown == [["Seal", "Res", "Mt Simon", "Basement"]]


def test_new_project_order_of_steps_and_skipping_the_survey(app, tmp_path):
    from cosma.gui.new_project import new_project_workflow

    files = _files(tmp_path)
    seen = []

    def pick(kind, title):
        seen.append(kind)
        return None if kind == "deviation" else files[kind]

    pr = new_project_workflow(pick, lambda: ("p", True), _confirm, lambda n: n[0], lambda p: True)
    assert seen == ["well_headers", "deviation", "tops"]                              # well first, tops before the chart
    assert pr.chart is not None and pr.wells["W-1"].survey is None


def test_new_project_backing_out_or_no_type_well(app, tmp_path):
    from cosma.gui.new_project import new_project_workflow

    files = _files(tmp_path)
    assert new_project_workflow(lambda k, t: files[k], lambda: None, _confirm, None, None) is None
    pr = new_project_workflow(lambda k, t: pytest.fail("no file should be asked for"), lambda: ("p", False), _confirm, None, None)
    assert pr.chart is None and pr.type_well is None
    # cancelling the tops import abandons the project; cancelling the editor keeps the chart as built
    assert new_project_workflow(lambda k, t: files[k], lambda: ("p", True), lambda prev: None, None, None) is None
    assert new_project_workflow(lambda k, t: None if k == "tops" else files[k], lambda: ("p", True), _confirm, None, None) is None
    pr = new_project_workflow(lambda k, t: files[k], lambda: ("p", True), _confirm, lambda n: n[0], lambda p: False)
    assert pr.chart.names()[0] == "Seal"


def test_new_project_needs_two_tops(app, tmp_path):
    from cosma.gui.new_project import new_project_workflow

    files = _files(tmp_path)
    files["tops"] = _w(tmp_path, "one.csv", "well,top,depth\nW-1,Seal,1000\n")
    with pytest.raises(ProjectError, match="at least two tops"):
        new_project_workflow(lambda k, t: files[k], lambda: ("p", True), _confirm, lambda n: n[0], lambda p: True)


def test_surfaces_cannot_be_imported_before_the_chart_exists(app, tmp_path):
    pr = Project("p")
    with pytest.raises(ImportRejected, match="define the stratigraphic chart first"):
        import_file("surface", _plane(tmp_path, "x.irap", 1000.0), scripted(UM), project=pr)

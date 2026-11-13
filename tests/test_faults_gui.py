import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qt = pytest.importorskip("PySide6.QtWidgets")

from cosma.faults import Fault, read_fault_sticks
from cosma.gui.fault_dialog import FaultViewDialog, fit_text
from cosma.gui.import_dialog import ImportDialog
from cosma.gui.main_window import MainWindow
from cosma.imports import inspect
from cosma.project import Project
from test_faults import plane_sticks
from test_faults_project import OK, sticks_csv
from test_main_window import Hooks, items, select


@pytest.fixture(scope="module")
def app():
    return qt.QApplication.instance() or qt.QApplication([])


class FH(Hooks):
    def __init__(self, path=None, values=None):
        super().__init__()
        self.path, self.values = path, values
        self.shown, self.removed_ask, self.remove_ok, self.int_calls, self.tm_calls = [], [], True, [], []

    def dict(self):
        d = super().dict()
        d.update(pick_file=lambda k, t: self.path, pick_files=lambda k, t: [self.path], confirm=lambda prev: dict(self.values) if self.values is not None else None,
                 show_fault=self.shown.append, fault_interactions=self.int_calls.append, fault_transmissibility=self.tm_calls.append, confirm_remove=lambda t: (self.removed_ask.append(t), self.remove_ok)[1])
        return d


def test_import_works_without_a_chart_and_lists_faults(app, tmp_path):
    h = FH(sticks_csv(tmp_path), OK)
    w = MainWindow(Project("p"), hooks=h.dict())
    assert w.import_actions["fault_sticks"].isEnabled()                    # no chart needed
    assert not w.act_export_faults.isEnabled() and not w.act_view_fault.isEnabled()
    w.do_import("fault_sticks")
    assert list(w.project.faults) == ["A", "B"] and w.is_dirty() and all("VSHALE" in r for r in h.reports)
    it = select(w, "A")
    assert "3 stick(s), 18 points, dip 60 deg" in it.text(1) and it.parent().text(0) == "Faults" and it.parent().text(1) == "2"
    assert w.act_view_fault.isEnabled() and w.act_remove_fault.isEnabled() and w.act_export_faults.isEnabled()


def test_bad_file_is_reported_not_raised(app, tmp_path):
    bad = tmp_path / "b.csv"
    bad.write_text("x,y,fault\n1,2,A\n")
    h = FH(bad, OK)
    w = MainWindow(Project("p"), hooks=h.dict())
    w.do_import("fault_sticks")
    assert w.project.faults == {} and any("no column for z" in r for r in h.reports)
    h2 = FH(sticks_csv(tmp_path), None)                                      # cancelled
    w2 = MainWindow(Project("p"), hooks=h2.dict())
    w2.do_import("fault_sticks")
    assert w2.project.faults == {} and not h2.reports


def test_view_remove_and_double_click(app, tmp_path):
    h = FH(sticks_csv(tmp_path), OK)
    w = MainWindow(Project("p"), hooks=h.dict())
    w.do_import("fault_sticks")
    w.view_fault()
    assert h.shown == [] and h.reports                                      # nothing selected
    it = select(w, "B")
    w.tree.itemDoubleClicked.emit(it, 0)
    assert [f.name for f in h.shown] == ["B"] and h.shown[0] is w.project.faults["B"]
    h.remove_ok = False
    w.remove_fault()
    assert "B" in w.project.faults and "'B'" in h.removed_ask[0]
    h.remove_ok = True
    w.remove_fault()
    assert list(w.project.faults) == ["A"]
    assert not any(i.text(0) == "B" for i in items(w))


def test_export_writes_csv_that_reads_back(app, tmp_path):
    h = FH(sticks_csv(tmp_path), OK)
    w = MainWindow(Project("p"), hooks=h.dict())
    w.export_faults()
    assert h.reports and h.reports[-1] == "The project has no faults."
    w.do_import("fault_sticks")
    h.save_to = str(tmp_path / "out.csv")
    w.export_faults()
    back = read_fault_sticks(h.save_to, xy_unit="m", z_unit="m", z_is="depth")
    assert [f.name for f in back] == ["A", "B"] and all(b.same_geometry(w.project.faults[b.name]) for b in back)
    h.save_to = None
    w.export_faults()                                                      # cancelled dialog: nothing happens


def test_saved_project_keeps_faults(app, tmp_path):
    h = FH(sticks_csv(tmp_path), OK)
    w = MainWindow(Project("p"), hooks=h.dict())
    w.do_import("fault_sticks")
    from cosma.persist import load_project, save_project
    p = save_project(w.project, tmp_path / "x.cosma")
    assert all(load_project(p).faults[n].same_geometry(f) for n, f in w.project.faults.items())


def test_import_dialog_demands_every_choice_and_shows_conflicts(app, tmp_path):
    pr = Project("p")
    f = sticks_csv(tmp_path)
    d = ImportDialog(inspect("fault_sticks", f, pr))
    ok = d.buttons.button(qt.QDialogButtonBox.Ok)
    assert not ok.isEnabled()
    d.widgets["xy_unit"].setCurrentText("m")
    d.widgets["z_unit"].setCurrentText("m")
    assert not ok.isEnabled()                                              # depth / elevation still unanswered
    d.widgets["z_is"].setCurrentIndex(d.widgets["z_is"].findData("depth"))
    assert ok.isEnabled() and d.values()["z_is"] == "depth"
    pr.attach_faults(read_fault_sticks(f, xy_unit="m", z_unit="m", z_is="depth"))
    d2 = ImportDialog(inspect("fault_sticks", f, pr))
    for k, v in (("xy_unit", "m"), ("z_unit", "m")):
        d2.widgets[k].setCurrentText(v)
    d2.widgets["z_is"].setCurrentIndex(d2.widgets["z_is"].findData("depth"))
    assert d2.conflict_table.rowCount() == 2 and not d2.buttons.button(qt.QDialogButtonBox.Ok).isEnabled()


# -- the view dialog ---------------------------------------------------------------------------------
def test_view_dialog_trace_and_messages(app):
    f = Fault("F", plane_sticks(50, 90, [0, 300, 700]))
    d = FaultViewDialog(f)
    assert "Dip 50.0 deg towards 90 deg" in d.summary.text() and "treats faults as vertical" in d.summary.text()
    z1 = f.bounds()[5]
    assert d.depth.text() == f"{z1 / 2:.1f}" and d.trace.depth == pytest.approx(round(z1 / 2, 1)) and len(d.trace.xy) == 3
    assert "3 of 3 stick(s) give a point" in d.notes.toPlainText()
    d.depth.setText("1e9")
    d.show_trace()
    assert len(d.trace.xy) == 0 and "no stick reaches this depth" in d.notes.toPlainText() and d.problem.text() == ""
    d.depth.setText("abc")
    d.show_trace()
    assert d.problem.text() == "The depth must be a number."
    d.depth.setText("nan")
    d.show_trace()
    assert "number" in d.problem.text()


def test_view_dialog_lists_why_sticks_are_skipped_and_single_stick_text(app):
    good = plane_sticks(89.9, 90, [0, 300], length=10.0)
    short = np.array([[0, 150.0, 0.0], [0, 150.0, 5.0]])
    f = Fault("F", (good[0], short, good[1]))
    d = FaultViewDialog(f)
    d.depth.setText("5.0")
    d.show_trace()
    assert "stick 2:" not in d.notes.toPlainText() and "3 of 3" in d.notes.toPlainText()          # its end vertex is on 5 m: it counts
    d.depth.setText("20")
    d.show_trace()
    assert "stick 2: does not reach this depth" in d.notes.toPlainText()
    t = fit_text(Fault("S", plane_sticks(60, 90, [0])))
    assert "NOT DETERMINED" in t and "seismic section" in t
    v = fit_text(Fault("V", tuple(np.array([[k * 10.0, 0, 0], [k * 10.0, 0, 100.0]]) for k in range(3))))
    assert "vertical: no dip direction" in v and "treats faults" not in v


# -- fault ages and interactions ---------------------------------------------------------------------------

def crossing_project():
    from cosma.faults import Fault

    def vf(name, p0, p1):
        return Fault(name, tuple(np.array([[x, y, z] for z in (0.0, 3000.0)]) for x, y in (p0, p1)))
    pr = Project("p")
    pr.attach_faults([vf("A", (0, 0), (100, 100)), vf("B", (0, 100), (100, 0)), vf("T", (300, 0), (300, 100))])
    return pr


def test_import_of_two_faults_forces_the_interaction_tool_open(app, tmp_path):
    h = FH(sticks_csv(tmp_path), OK)
    w = MainWindow(Project("p"), hooks=h.dict())
    w.do_import("fault_sticks")
    assert h.int_calls == [w.project]                       # opened by force, nothing settled yet


def test_not_forced_for_one_fault_or_when_settled(app, tmp_path):
    h = FH(None, None)
    pr = crossing_project()
    w = MainWindow(pr, hooks=h.dict())
    w.settle_fault_interactions(force=True)
    assert len(h.int_calls) == 1
    for n in pr.faults:
        pr.set_fault_depth(n, 1500.0)
    pr.refresh_fault_interactions(5.0)
    pr.confirm_fault_interaction(("A", "B"), "A")
    w.settle_fault_interactions(force=True)
    assert len(h.int_calls) == 1                            # settled: no forced opening
    w.fault_interactions()                                  # the menu entry always opens it
    assert len(h.int_calls) == 2
    pr.remove_fault("B")
    pr.remove_fault("T")
    w.settle_fault_interactions(force=False)
    assert len(h.int_calls) == 2                            # one fault: nothing to settle


def test_menu_entry_needs_two_faults(app):
    w = MainWindow(Project("p"), hooks=FH().dict())
    assert not w.act_fault_ints.isEnabled()
    w.set_project(crossing_project())
    w.refresh()
    assert w.act_fault_ints.isEnabled()


def test_interaction_dialog_flow(app):
    from cosma.gui.fault_interaction_dialog import CHOOSE, FaultInteractionDialog
    pr = crossing_project()
    d = FaultInteractionDialog(pr)
    assert "not checked" in d.status.text() and d.table.rowCount() == 0
    d.reach.setText("abc")
    d.find()
    assert "numbers" in d.problem.text()
    d.reach.setText("0")
    d.find()
    assert "positive" in d.problem.text() and pr.fault_reach is None
    d.reach.setText("5")
    d.find()
    assert d.table.rowCount() == 1 and pr.fault_depths == {"A": 1500.0, "B": 1500.0, "T": 1500.0}
    assert d.table.item(0, 1).text() == "geometry cannot say" and d.table.item(0, 3).text() == "NOT CONFIRMED"
    assert "need your answer" in d.status.text() and not d.use_btn.isEnabled()
    d.combos[0].setCurrentText("B is older")
    assert pr.fault_interactions[0].answer == "B" and d.table.item(0, 3).text() == "confirmed"
    assert "Oldest to youngest" in d.status.text() and d.status.text().index("B") < d.status.text().index("A")
    d.combos[0].setCurrentText("same age")
    assert pr.fault_interactions[0].answer == "same_age"
    d.combos[0].setCurrentText(CHOOSE)                      # no effect: an answer is never un-set by accident
    assert pr.fault_interactions[0].answer == "same_age"


def test_dialog_suggestions_are_applied_only_on_request_and_depth_edits_discard_answers(app):
    from cosma.faults import Fault
    from cosma.gui.fault_interaction_dialog import FaultInteractionDialog

    def vf(name, p0, p1):
        return Fault(name, tuple(np.array([[x, y, z] for z in (0.0, 3000.0)]) for x, y in (p0, p1)))
    pr = Project("p")
    pr.attach_faults([vf("A", (0, 50), (100, 50)), vf("B", (50, 50), (50, 150))])     # B ends on A
    d = FaultInteractionDialog(pr)
    d.reach.setText("5")
    d.find()
    assert d.table.item(0, 1).text() == "A is older" and not pr.fault_interactions[0].confirmed
    assert d.use_btn.isEnabled()
    d.use_suggestions()
    assert pr.fault_interactions[0].answer == "A" and not d.use_btn.isEnabled()
    d.depth_edits["B"].setText("1600")
    d.find()
    assert not pr.fault_interactions[0].confirmed           # B's trace was re-taken, so the answer is asked again


def test_dialog_reports_faults_without_a_trace(app):
    from cosma.gui.fault_interaction_dialog import FaultInteractionDialog
    pr = crossing_project()
    d = FaultInteractionDialog(pr)
    d.reach.setText("5")
    d.depth_edits["T"].setText("9000")                      # no stick reaches it
    d.find()
    assert "T: no trace at depth 9000" in d.problem.text()


def test_import_forces_the_transmissibility_tool_until_every_fault_has_a_choice(app, tmp_path):
    from cosma.fault_transmissibility import FaultTransmissibility
    h = FH(sticks_csv(tmp_path), OK)
    w = MainWindow(Project("p"), hooks=h.dict())
    w.do_import("fault_sticks")
    assert h.tm_calls == [w.project]                        # two faults, no choice yet
    w.project.set_fault_transmissibility("A", FaultTransmissibility("open"))
    w.settle_fault_transmissibility()
    assert len(h.tm_calls) == 2                             # B still has none
    w.project.set_fault_transmissibility("B", FaultTransmissibility("fixed", 0.0))
    w.settle_fault_transmissibility()
    assert len(h.tm_calls) == 2                             # all chosen: not forced
    h2 = FH(None, None)
    w2 = MainWindow(Project("p"), hooks=h2.dict())
    w2.settle_fault_transmissibility()
    assert h2.tm_calls == []                                # no faults: nothing to ask


def test_cancelled_import_does_not_open_the_transmissibility_tool(app, tmp_path):
    h = FH(sticks_csv(tmp_path), None)                      # confirmation popup cancelled
    w = MainWindow(Project("p"), hooks=h.dict())
    w.do_import("fault_sticks")
    assert h.tm_calls == []

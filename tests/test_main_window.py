import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qt = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import Qt

from cosma.gui.main_window import MainWindow
from cosma.gui.variogram_setup import VariogramSetupDialog, make_session
from cosma.project import Action, Project
from cosma.variogram import Structure, VariogramModel, from_gslib
from synthetic_field import facies_project, synthetic


@pytest.fixture(scope="module")
def app():
    return qt.QApplication.instance() or qt.QApplication([])


class Hooks:
    def __init__(self):
        self.reports, self.editor_calls, self.setups = [], [], []
        self.setup_answer = None
        self.editor_fn = None
        self.save_to = None

    def dict(self):
        return dict(report=lambda t, error=True: self.reports.append(t), ask_setup=self.ask_setup, run_editor=self.run_editor,
                    save_path=lambda title, default: self.save_to, pick_file=lambda k, t: None)

    def ask_setup(self, pr, preset):
        self.setups.append(preset)
        return self.setup_answer

    def run_editor(self, session, label):
        self.editor_calls.append((session, label))
        return self.editor_fn(session) if self.editor_fn else None


def items(win):
    out = []

    def walk(it):
        out.append(it)
        for i in range(it.childCount()):
            walk(it.child(i))

    for i in range(win.tree.topLevelItemCount()):
        walk(win.tree.topLevelItem(i))
    return out


def select(win, text):
    it = next(i for i in items(win) if i.text(0).startswith(text))
    win.tree.setCurrentItem(it)
    return it


def vals(pr, **kw):
    return dict(uid=pr.chart.match_name("Res"), kind="log", variable="PHI", source=None, frame="top", indicator=None, **kw)


def test_no_project_disables_everything(app):
    w = MainWindow(None)
    assert not any(a.isEnabled() for a in w.import_actions.values()) and not w.act_new_vario.isEnabled()
    assert w.act_new.isEnabled()


def test_menus_follow_project_state(app):
    pr = Project("p")
    w = MainWindow(pr)
    assert w.import_actions["well_headers"].isEnabled() and not w.import_actions["surface"].isEnabled()
    assert not w.import_actions["variogram"].isEnabled() and not w.act_chart.isEnabled()
    pr = synthetic()
    w = MainWindow(pr)
    assert all(a.isEnabled() for a in w.import_actions.values()) and w.act_new_vario.isEnabled()
    assert not w.act_edit_vario.isEnabled()                                  # nothing selected


def test_tree_shows_wells_and_intervals(app):
    pr = synthetic(n=2)
    w = MainWindow(pr)
    texts = [i.text(0) for i in items(w)]
    assert "Wells" in texts and "Res" in texts and "Seal" in texts and any(t.startswith("Base: Basement") for t in texts)


def test_new_variogram_opens_editor_with_session_and_saved_result_appears(app):
    pr = synthetic(n=3)
    h = Hooks()
    h.setup_answer = vals(pr)

    def edit(session):
        session.set_nugget(0.1)
        session.set_structure(0, contribution=0.9, major=400, minor=200, vertical=20)
        return session.save("v1")

    h.editor_fn = edit
    w = MainWindow(pr, hooks=h.dict())
    w.act_new_vario.trigger()
    s, label = h.editor_calls[0]
    assert s.variable == "PHI" and label == "" and h.setups[0] is None
    texts = [i.text(0) for i in items(w)]
    assert "variogram: PHI" in texts
    row = next(i for i in items(w) if i.text(0) == "variogram: PHI")
    assert "v1" in row.text(1) and "(used)" in row.text(1) and "400/200/20" in row.text(1)


def test_cancelled_setup_does_nothing(app):
    pr = synthetic(n=2)
    h = Hooks()
    w = MainWindow(pr, hooks=h.dict())
    w.act_new_vario.trigger()
    assert h.editor_calls == [] and h.reports == []


def test_edit_existing_reopens_with_meta_and_model(app):
    pr = facies_project(n=3)
    s0 = make_session(pr, dict(uid=pr.chart.match_name("Res"), kind="log", variable="FACIES", source=None, frame="proportional", indicator=2))
    s0.set_nugget(0.05); s0.set_structure(0, contribution=0.2, major=300, minor=150, vertical=10)
    s0.save("fac")
    h = Hooks()
    w = MainWindow(pr, hooks=h.dict())
    select(w, "variogram: FACIES[2]")
    assert w.act_edit_vario.isEnabled()
    h.setup_answer = dict(uid=pr.chart.match_name("Res"), kind="log", variable="FACIES", source=None, frame="proportional", indicator=2)
    w.act_edit_vario.trigger()
    preset = h.setups[0]
    assert preset["curve"] == "FACIES" and preset["indicator"] == 2 and preset["frame"] == "proportional"
    s, label = h.editor_calls[0]
    assert label == "fac" and s.model() == pr.variogram_for(pr.chart.match_name("Res"), "FACIES[2]")


def test_edit_with_mismatched_data_is_reported_not_forced(app):
    pr = facies_project(n=3)
    s0 = make_session(pr, dict(uid=pr.chart.match_name("Res"), kind="log", variable="FACIES", source=None, frame="top", indicator=2))
    s0.set_nugget(0.05); s0.set_structure(0, contribution=0.2, major=300, minor=150, vertical=10)
    s0.save("fac")
    h = Hooks()
    h.setup_answer = dict(uid=pr.chart.match_name("Res"), kind="log", variable="PHI", source=None, frame="top", indicator=None)
    w = MainWindow(pr, hooks=h.dict())
    select(w, "variogram: FACIES[2]")
    w.act_edit_vario.trigger()
    assert h.editor_calls == [] and "computed on" in h.reports[0]


def test_edit_imported_variogram_without_meta(app, tmp_path):
    pr = synthetic(n=3)
    u = pr.chart.match_name("Res")
    pr.attach_variogram(u, "PHI", from_gslib("1 0.1\n1 0.9 0 0 0\n300 150 10\n"), "imp")
    h = Hooks()
    h.setup_answer = vals(pr)
    w = MainWindow(pr, hooks=h.dict())
    select(w, "variogram: PHI")
    w.edit_variogram()
    assert h.setups[0] == {"uid": u, "curve": "PHI"} and h.editor_calls[0][1] == "imp"


def test_export_variogram(app, tmp_path):
    pr = synthetic(n=2)
    u = pr.chart.match_name("Res")
    m = from_gslib("1 0.1\n1 0.9 30 0 0\n300 150 10\n", variable="PHI")
    pr.attach_variogram(u, "PHI", m, "exp1")
    h = Hooks()
    w = MainWindow(pr, hooks=h.dict())
    select(w, "variogram: PHI")
    h.save_to = str(tmp_path / "o.var")
    w.act_export_vario.trigger()
    assert from_gslib((tmp_path / "o.var").read_text()).azimuth == 30
    h.save_to = str(tmp_path / "o.json")
    w.export_variogram()
    import json
    assert json.loads((tmp_path / "o.json").read_text())["nugget"] == 0.1
    h.save_to = None
    w.export_variogram()                                                     # cancelled: nothing happens
    assert h.reports == []


def test_nothing_selected_gives_message(app):
    pr = synthetic(n=2)
    h = Hooks()
    w = MainWindow(pr, hooks=h.dict())
    w.edit_variogram()
    assert "Select a variogram" in h.reports[0]


def test_import_through_window_refreshes_tree_and_reports_errors(app, tmp_path):
    pr = synthetic(n=2)
    u = pr.chart.match_name("Res")
    f = tmp_path / "p.var"
    f.write_text("1 0.1\n1 0.9 0 0 0\n300 150 10\n")
    h = Hooks()
    d = h.dict()
    d["pick_file"] = lambda k, t: str(f)
    d["confirm"] = lambda prev: dict(variable="PHI", transform="none", range_unit="m", interval=u)
    w = MainWindow(pr, hooks=d)
    w.import_actions["variogram"].trigger()
    assert any(i.text(0) == "variogram: PHI" for i in items(w)) and "p -> Res / PHI" in w.statusBar().currentMessage()
    w.import_actions["variogram"].trigger()                                  # same again: conflict not answered -> reported
    assert h.reports and "p.var" in h.reports[0]
    w.confirm = lambda prev: None
    w.import_actions["variogram"].trigger()                                  # cancelled: silent
    assert len(h.reports) == 1


# -- setup dialog -------------------------------------------------------------------------------------------------------------
def test_setup_dialog_lists_properties_and_facies_codes(app):
    pr = facies_project(n=3)
    dlg = VariogramSetupDialog(pr)
    assert [dlg.prop.itemText(i) for i in range(dlg.prop.count())] == ["FACIES", "PHI"]
    dlg.prop.setCurrentText("PHI")
    dlg._changed()
    assert not dlg.indicator.isEnabled() and dlg.buttons.button(qt.QDialogButtonBox.Ok).isEnabled()
    dlg.prop.setCurrentText("FACIES")
    assert dlg.indicator.isEnabled() and dlg.code.count() == 3
    dlg.indicator.setChecked(True)
    dlg.code.setCurrentIndex(1)
    v = dlg.values()
    assert v["variable"] == "FACIES" and v["indicator"] == 2 and v["kind"] == "log" and v["source"] is None
    dlg.indicator.setChecked(False)
    assert dlg.values()["indicator"] is None


def test_setup_dialog_preset_and_sources_and_problems(app):
    pr = facies_project(n=3)
    import numpy as np
    from cosma.wells.model import LogSet
    rec = pr.wells["W0"]
    rec.logs["second"] = LogSet(np.arange(0.0, 400.0), {"PHI": np.zeros(400)})
    u = pr.chart.match_name("Seal")
    dlg = VariogramSetupDialog(pr, preset=dict(uid=u, curve="PHI", frame="proportional"))
    assert dlg.interval.currentData() == u and dlg.frame.currentText() == "proportional" and dlg.prop.currentText() == "PHI"
    assert not dlg.buttons.button(qt.QDialogButtonBox.Ok).isEnabled() and "several sources" in dlg.info.text()    # ambiguous: must name one
    dlg.source.setCurrentText("lg")
    assert dlg.buttons.button(qt.QDialogButtonBox.Ok).isEnabled() and dlg.values()["source"] == "lg"
    dlg.kind.setCurrentText("core")
    assert dlg.prop.count() == 0 and not dlg.buttons.button(qt.QDialogButtonBox.Ok).isEnabled() and "no core data" in dlg.info.text()
    dlg2 = VariogramSetupDialog(pr, preset=dict(uid=u, curve="FACIES", indicator=3))
    assert dlg2.indicator.isChecked() and dlg2.values()["indicator"] == 3


def test_real_dialogs_open_from_the_window(app):
    from PySide6.QtCore import QTimer
    pr = facies_project(n=3)
    seen = []

    def accept_setup():
        m = app.activeModalWidget()
        seen.append(type(m).__name__)
        m.prop.setCurrentText("FACIES"); m._changed()
        m.indicator.setChecked(True)
        m.accept()

    def close_editor():
        m = app.activeModalWidget()
        seen.append(type(m).__name__)
        assert m.s.variable == "FACIES[1]" and m.windowTitle().endswith("FACIES[1]")
        m.reject()

    w = MainWindow(pr, hooks=dict(report=lambda t, error=True: seen.append("REPORT " + t)))
    QTimer.singleShot(0, accept_setup)
    QTimer.singleShot(300, close_editor)
    w.act_new_vario.trigger()
    assert seen == ["VariogramSetupDialog", "VariogramEditor"], seen


def test_new_variogram_needs_wells_even_with_a_chart(app):
    pr = Project("c")
    pr.create_chart(["A", "B"], "Base")
    w = MainWindow(pr)
    assert w.act_chart.isEnabled() and not w.act_new_vario.isEnabled()


def test_edit_imported_indicator_variogram_offers_the_base_curve(app):
    pr = facies_project(n=3)
    u = pr.chart.match_name("Res")
    pr.attach_variogram(u, "FACIES[2]", from_gslib("1 0.1\n1 0.1 0 0 0\n300 150 10\n", variable="FACIES[2]", transform="indicator"), "imp")
    h = Hooks()
    w = MainWindow(pr, hooks=h.dict())
    select(w, "variogram: FACIES[2]")
    w.edit_variogram()
    assert h.setups[0]["curve"] == "FACIES"


def test_editor_locks_transform_for_indicator(app):
    from cosma.gui.variogram_editor import VariogramEditor
    pr = facies_project(n=3)
    e = VariogramEditor(make_session(pr, dict(uid=pr.chart.match_name("Res"), kind="log", variable="FACIES", source=None, frame="top", indicator=2)))
    assert e.transform.currentText() == "indicator" and not e.transform.isEnabled() and "Indicator of facies 2" in e.findChildren(qt.QLabel)[0].text()
    e2 = VariogramEditor(make_session(pr, dict(uid=pr.chart.match_name("Res"), kind="log", variable="PHI", source=None, frame="top", indicator=None)))
    assert e2.transform.isEnabled() and e2.transform.count() == 2


# -- save / open ------------------------------------------------------------------------------------------------------------
class FileHooks(Hooks):
    def __init__(self):
        super().__init__()
        self.open_to = None
        self.save_as_to = None
        self.unsaved_answer = "cancel"
        self.asked = []

    def dict(self):
        d = super().dict()
        d.update(project_open_path=lambda: self.open_to, project_save_path=lambda default: (self.asked.append(default), self.save_as_to)[1],
                 ask_unsaved=lambda text: (self.asked.append(text), self.unsaved_answer)[1],
                 new_project=lambda: Project("fresh"))
        return d


def test_save_as_then_save_then_dirty_tracking(app, tmp_path):
    pr = synthetic(n=2)
    h = FileHooks()
    w = MainWindow(pr, hooks=h.dict())
    assert w.is_dirty() and w.windowTitle().endswith(" *")
    h.save_as_to = None
    assert w.save() is False and w.path is None                                    # cancelled file dialog: nothing saved
    h.save_as_to = str(tmp_path / "a")
    assert w.save() is True and w.path == tmp_path / "a.cosma" and (tmp_path / "a.cosma").exists()
    assert not w.is_dirty() and "[a.cosma]" in w.windowTitle() and not w.windowTitle().endswith("*")
    assert "Saved a.cosma" in w.statusBar().currentMessage()
    pr.name = "renamed"
    assert w.is_dirty() and not w.windowTitle().endswith("*")                      # title refreshes on the next action, not on every edit
    w.refresh()
    assert w.windowTitle().endswith("*")
    asked = len(h.asked)
    assert w.save() and not w.is_dirty() and len(h.asked) == asked                 # a known path is reused: no dialog
    from cosma.persist import load_project
    assert load_project(tmp_path / "a.cosma").name == "renamed"


def test_open_project_and_failures(app, tmp_path):
    from cosma.persist import save_project
    src = synthetic(n=2)
    save_project(src, tmp_path / "s.cosma")
    h = FileHooks()
    w = MainWindow(None, hooks=h.dict())
    h.open_to = str(tmp_path / "s.cosma")
    w.act_open.trigger()
    assert w.project is not None and w.path == tmp_path / "s.cosma" and not w.is_dirty()
    assert any(i.text(0) == "Res" for i in items(w))
    (tmp_path / "bad.cosma").write_text("nope")
    h.open_to = str(tmp_path / "bad.cosma")
    w.project.name = "edited"
    h.unsaved_answer = "discard"
    w.open_project()
    assert "not a COSMA project" in h.reports[-1] and w.project.name == "edited"      # failed open keeps what was open
    h.open_to = None
    w.open_project()                                                                   # cancelled picker: nothing changes
    assert w.project.name == "edited"


def test_unsaved_changes_are_never_dropped_silently(app, tmp_path):
    h = FileHooks()
    pr = synthetic(n=2)
    w = MainWindow(pr, hooks=h.dict())
    for answer, replaced in (("cancel", False), ("discard", True)):
        w.set_project(pr)
        h.unsaved_answer = answer
        w.new_project()
        assert (w.project is not pr) == replaced and "unsaved changes" in h.asked[-1]
    w.set_project(pr)
    h.unsaved_answer = "save"
    h.save_as_to = None                                                                # chose Save but cancelled the file dialog
    w.new_project()
    assert w.project is pr                                                             # still the same project: nothing lost
    h.save_as_to = str(tmp_path / "keep")
    w.new_project()
    assert w.project is not pr and (tmp_path / "keep.cosma").exists()


def test_close_asks_and_clean_or_empty_projects_do_not(app, tmp_path):
    from PySide6.QtGui import QCloseEvent
    h = FileHooks()
    w = MainWindow(Project("empty"), hooks=h.dict())
    assert not w.is_dirty()
    ev = QCloseEvent(); w.closeEvent(ev)
    assert ev.isAccepted() and h.asked == []
    w2 = MainWindow(synthetic(n=2), hooks=h.dict())
    ev = QCloseEvent(); w2.closeEvent(ev)
    assert not ev.isAccepted() and h.asked                                             # default answer cancel keeps it open
    h.unsaved_answer = "discard"
    ev = QCloseEvent(); w2.closeEvent(ev)
    assert ev.isAccepted()
    assert not MainWindow(None, hooks=h.dict()).is_dirty()


def test_save_failure_is_reported_and_keeps_dirty(app, tmp_path):
    h = FileHooks()
    w = MainWindow(synthetic(n=2), hooks=h.dict())
    h.save_as_to = str(tmp_path / "no_such_dir" / "x")
    assert w.save() is False and "Could not save" in h.reports[-1] and w.is_dirty() and w.path is None


def test_save_menu_state(app):
    w = MainWindow(None)
    assert not w.act_save.isEnabled() and not w.act_save_as.isEnabled() and w.act_open.isEnabled()
    w = MainWindow(Project("x"))
    assert w.act_save.isEnabled() and w.act_save_as.isEnabled()


def test_open_asks_before_replacing_unsaved_work(app, tmp_path):
    from cosma.persist import save_project
    save_project(synthetic(n=2), tmp_path / "o.cosma")
    h = FileHooks()
    pr = synthetic(n=2)
    w = MainWindow(pr, hooks=h.dict())
    picked = []
    w.project_open_path = lambda: picked.append(1) or str(tmp_path / "o.cosma")
    h.unsaved_answer = "cancel"
    w.open_project()
    assert w.project is pr and picked == []                          # asked first; the picker was never even shown
    h.unsaved_answer = "discard"
    w.open_project()
    assert w.project is not pr and picked == [1]

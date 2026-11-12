import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qt = pytest.importorskip("PySide6.QtWidgets")

from cosma.gui.main_window import MainWindow
from cosma.gui.realizations_dialog import CHOOSE, RealizationsDialog
from cosma.persist import load_project, save_project
from test_datacheck import make_model
from test_facies_sim import SIS_S
from test_main_window import Hooks
from test_realizations import world


@pytest.fixture(scope="module")
def app():
    return qt.QApplication.instance() or qt.QApplication([])


def dlg(pr, model, tmp_path=None, saved=None):
    return RealizationsDialog(pr, build=lambda p: model, pick_dir=lambda: str(tmp_path), pick_save=lambda: str(saved))


def fill(d, perm=True, keyword="PORO"):
    d.add_job(["PHI", keyword, True, "PHI {facies}", "", "", "", 1, 8, 1, 1, 0.001])
    d._job_selected()
    for i in range(d.dom.rowCount()):
        d.dom.cellWidget(i, 2).setCurrentText("typed")
        d.dom.cellWidget(i, 3).setCurrentText("normal")
        d.dom.cellWidget(i, 4).setText("0.2")
        d.dom.cellWidget(i, 5).setText("0.03")
    if perm:
        d.use_perm.setChecked(True)
        d.phi_box.setCurrentText("PHI")
        d.kykx.setText("1")
        d.shift.setText("0")
        d.extrap.setCurrentText("allow")
        d.scat.setCurrentText("independent")
        for i in range(d.perm.rowCount()):
            d.perm.cellWidget(i, 2).setCurrentText("typed")
            for c, v in ((3, "1"), (4, "10"), (5, "0.2"), (7, "0.1")):
                d.perm.cellWidget(i, c).setText(v)
    d.count.setText("3")
    d.seed.setText("5")
    d.conf.setText("0.95")
    d.on_fail.setCurrentText("keep")
    d.workers.setText("1")


def test_nothing_is_preset_and_every_gap_is_named(app):
    pr, model = world()
    d = dlg(pr, model)
    assert d.count.text() == "" and d.seed.text() == "" and d.on_fail.currentText() == CHOOSE and not d.save_btn.isEnabled()
    d.add_job(["PHI", "PORO", True, "PHI {facies}", "", "", "", 1, 8, 1, 1, 0.001])
    with pytest.raises(Exception, match="choose where the distribution comes from"):
        d.read_plan()
    d.check_setup()
    d.panel.wait()
    assert "choose where" in d.problem.text() and not d.save_btn.isEnabled()
    d.jobs.selectRow(0)
    d.remove_job()
    fill(d, perm=False)
    d.on_fail.setCurrentText(CHOOSE)
    with pytest.raises(Exception, match="failing realization"):
        d.read_plan()
    d.on_fail.setCurrentText("redraw")
    with pytest.raises(Exception, match="what a redraw changes"):
        d.read_plan()
    d.on_fail.setCurrentText("keep")
    d.use_perm.setChecked(True)
    with pytest.raises(Exception, match="porosity property"):
        d.read_plan()


def test_check_save_run_export_and_reload(app, tmp_path):
    pr, model = world()
    out = tmp_path / "r.grdecl"
    from test_realizations import full_model
    d = dlg(pr, full_model(model), tmp_path, out)
    fill(d)
    assert not d.save()                                   # not checked yet
    assert d.check_setup()
    d.panel.wait()
    assert d.save_btn.isEnabled()
    d.kykx.setText("0.9")                                 # an edit invalidates the check
    assert not d.save_btn.isEnabled() and not d.save()
    assert "Check the setup" in d.problem.text()
    d.kykx.setText("1")
    assert d.check_setup()
    d.panel.wait()
    assert d.save() and pr.realization_plan.count == 3
    d.cache.setText(str(tmp_path / "cache"))
    d.run_all()
    d.panel.wait()
    assert d.list.count() == 3 and "3 realization(s)" in d.report.toPlainText()
    d.list.setCurrentRow(1)
    d.export_selected()
    d.panel.wait()
    assert out.exists() and "Written" in d.report.toPlainText() and "PERMZ" in out.read_text()
    assert (tmp_path / "cache" / "realization_001.npz").exists()
    # reload: the dialog shows the saved plan and round-trips it
    back = load_project(save_project(pr, tmp_path / "p.cosma"))
    d2 = dlg(back, model)
    assert d2.read_plan() == pr.realization_plan


def test_flagged_export_needs_the_tick(app, tmp_path, monkeypatch):
    import cosma.realizations as R
    pr, model = world()
    from test_realizations import full_model
    d = dlg(pr, full_model(model), tmp_path, tmp_path / "x.grdecl")
    fill(d)
    monkeypatch.setattr(R, "check_realization", lambda *a, **k: dict(rows=[], ntests=1, alpha=.05, failed=[("PORO", "A", 1, "mean")], text="bad"))
    d.run_all()
    d.panel.wait()
    assert "flagged" in d.list.item(0).text()
    d.list.setCurrentRow(0)
    d.export_selected()
    d.panel.wait()
    assert "flagged" in d.problem.text() and not (tmp_path / "x.grdecl").exists()
    d.allow.setChecked(True)
    d.export_selected()
    d.panel.wait()
    assert (tmp_path / "x.grdecl").exists()


def test_menu_entry(app):
    pr, _ = world()
    seen = []
    h = Hooks()
    w = MainWindow(hooks={**h.dict(), "realizations": lambda p: seen.append(p)})
    assert not w.act_real.isEnabled()
    w.project = pr
    from cosma.grid_definition import GridDefinition
    w.project.grid_def = GridDefinition(0, 0, 0, 6, 4, 100, 100)
    w.refresh()
    assert w.act_real.isEnabled()
    w.act_real.trigger()
    assert seen == [pr]


def test_run_is_threaded_shows_progress_and_disables_editing(app, tmp_path):
    import time
    from cosma.gui.progress_widget import ProgressPanel
    p = ProgressPanel(interval_ms=20)
    gate = {"go": False}

    def job(ctx):
        ctx.progress("stage one", 0.0)
        while not gate["go"]:
            time.sleep(0.005)
        ctx.progress("stage two", 0.5)
        return 42
    p.start(job, items=1)
    assert p.running and p.stop_btn.isEnabled()
    with pytest.raises(RuntimeError):
        p.start(job)
    app.processEvents()
    assert p.stage.text() == "stage one" and "estimating" in p.time.text() or "done" in p.time.text()
    gate["go"] = True
    p.wait()
    assert p.outcome == ("ok", 42) and p.bar.value() == 1000 and "finished in" in p.time.text() and not p.stop_btn.isEnabled()


def test_panel_reports_failure_and_stop(app):
    import time
    from cosma.gui.progress_widget import ProgressPanel
    p = ProgressPanel(interval_ms=20)

    def bad(ctx):
        raise ValueError("boom")
    p.start(bad)
    p.wait()
    assert p.outcome == ("failed", "boom")

    def slow(ctx):
        for i in range(2000):
            ctx.progress("working", i / 2000)
            time.sleep(0.002)
    p.start(slow)
    time.sleep(0.05)
    p.stop()
    assert "Stopping" in p.stage.text()
    p.wait()
    assert p.outcome[0] == "cancelled" and p.bar.value() < 1000 and "stopped after" in p.time.text()


def test_dialog_run_stop_cache_and_busy_state(app, tmp_path):
    import cosma.facies_sim as FS
    pr, model = world()
    from test_realizations import full_model
    d = dlg(pr, full_model(model), tmp_path, tmp_path / "x.grdecl")
    fill(d)
    d.count.setText("4")
    d.cache.setText(str(tmp_path / "c"))
    # stop pressed while realization 1 is being made
    real = FS.run
    calls = []

    def run_and_stop(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            d.panel.ctx.cancel_requested = True
        return real(*a, **k)
    FS.run = run_and_stop
    try:
        d.run_all()
        assert d.panel.running and not d.tabs.isEnabled() and not d.check_btn.isEnabled()      # editing is blocked while it runs
        d.panel.wait()
    finally:
        FS.run = real
    assert d.panel.outcome[0] == "cancelled" and d.tabs.isEnabled() and d.check_btn.isEnabled()
    assert "STOPPED after 2 of 4" in d.report.toPlainText() or "STOPPED after 1 of 4" in d.report.toPlainText()
    done_before = d.list.count()
    assert 1 <= done_before <= 2 and (tmp_path / "c" / "realization_000.npz").exists()
    # a second run takes the finished ones from the cache and says so
    d.run_all()
    d.panel.wait()
    assert d.panel.outcome[0] == "ok" and d.list.count() == 4 and "[cache]" in d.list.item(0).text()
    assert "Total time" in d.report.toPlainText() and d.panel.bar.value() == 1000
    # everything cached: nothing to compute, the run is immediate and complete
    d.run_all()
    d.panel.wait()
    assert d.panel.outcome[0] == "ok" and d.panel.tracker.fraction == 1.0
    # closing while running stops the worker first
    d.cache.setText("")
    d.run_all()
    d.accept()
    assert not d.panel.running


def test_workers_must_be_chosen_and_memory_is_checked(app, tmp_path, monkeypatch):
    import cosma.realization_runner as RR
    import cosma.gui.realizations_dialog as RD
    pr, model = world()
    d = dlg(pr, model, tmp_path)
    fill(d)
    d.workers.setText("")
    d.run_all()
    assert "worker processes: enter a whole number" in d.problem.text() and not d.panel.running
    d.workers.setText("0")
    d.run_all()
    assert "1 or more" in d.problem.text()
    d.workers.setText("2")
    monkeypatch.setattr(RR, "physical_memory", lambda: 1e6)
    d.run_all()
    d.panel.wait()
    assert "does not fit" in d.problem.text() and d.list.count() == 0
    monkeypatch.setattr(RR, "physical_memory", lambda: None)
    msg, ok = RR.check_memory(1000, 2)
    assert ok and "could not be read" in msg
    monkeypatch.setattr(RR, "physical_memory", lambda: 1e10)
    assert "above 70%" in RR.check_memory(int(1.8e7), 1)[0] and RR.check_memory(1000, 2)[1]
    assert not RR.check_memory(int(1e9), 2)[1]


def test_parallel_run_equals_serial_and_export_regenerates_without_a_cache(app, tmp_path):
    from test_realizations import full_model
    pr, model = world()
    outs = {}
    for w in ("1", "2"):
        d = dlg(pr, full_model(model), tmp_path, tmp_path / f"w{w}.grdecl")
        fill(d)
        d.count.setText("4")
        d.workers.setText(w)
        d.run_all()
        d.panel.wait()
        assert d.panel.outcome[0] == "ok" and d.list.count() == 4
        assert [d.list.item(i).text().split(":")[0] for i in range(4)] == ["0", "1", "2", "3"]
        assert all(r.facies is None for r in d.results)                                  # no arrays kept in memory
        if w == "2":
            assert "worker(s) on" in d.report.toPlainText()
        d.list.setCurrentRow(2)
        d.export_selected()                                                              # no cache: regenerated first
        d.panel.wait()
        assert "regenerating it from its seeds" in d.report.toPlainText() and "Written" in d.report.toPlainText()
        outs[w] = (tmp_path / f"w{w}.grdecl").read_text()
    assert outs["1"] == outs["2"]


def test_parallel_stop_keeps_what_finished(app, tmp_path, monkeypatch):
    import cosma.gui.realizations_dialog as RD
    pr, model = world()
    d = dlg(pr, model, tmp_path)
    fill(d)
    d.count.setText("8")
    d.workers.setText("2")
    d.cache.setText(str(tmp_path / "c"))
    real = RD.run_parallel

    def stop_after_first(mgr, todo, w, tracker, cancelled, got):
        return real(mgr, todo, w, tracker, lambda: len(d.results) >= 1, got, poll=0.05)
    monkeypatch.setattr(RD, "run_parallel", stop_after_first)
    d.run_all()
    d.panel.wait()
    assert d.panel.outcome[0] == "cancelled" and "STOPPED after" in d.report.toPlainText()
    got = len(list((tmp_path / "c").glob("*.npz")))
    assert 1 <= got < 8 and d.list.count() == got and d.tabs.isEnabled()


def test_correlation_and_correlated_scatter_fields(app, tmp_path):
    pr, model = world()
    d = dlg(pr, model, tmp_path)
    fill(d)
    assert d.read_plan().perm.scatter_mode == "independent"
    d.scat.setCurrentText(CHOOSE)
    with pytest.raises(Exception, match="independent or spatially correlated"):
        d.read_plan()
    d.scat.setCurrentText("independent")
    d.scat_vg.setText("PHI {facies}")
    with pytest.raises(Exception, match="clear those fields"):
        d.read_plan()
    d.scat.setCurrentText("correlated")
    with pytest.raises(Exception, match="neighbours"):
        d.read_plan()
    d.scat_nn.setText("8")
    d.scat_sf.setText("1")
    d.scat_lv.setText("1")
    assert d.read_plan().perm.scatter_variogram == "PHI {facies}"
    # a second property correlated with the first
    d.add_job(["VSH", "VSHALE", False, "PHI {facies}", "", "", "", 1, 8, 1, 1, 0.001, "PHI"])
    d._job_selected()
    for i in range(d.dom.rowCount()):
        d.dom.cellWidget(i, 2).setCurrentText("typed")
        d.dom.cellWidget(i, 3).setCurrentText("normal")
        d.dom.cellWidget(i, 4).setText("0.3")
        d.dom.cellWidget(i, 5).setText("0.1")
    with pytest.raises(Exception, match="enter the correlation with PHI"):
        d.read_plan()
    for i in range(d.dom.rowCount()):
        d.dom.cellWidget(i, 6).setText("-0.7")
    p = d.read_plan()
    assert p.jobs[1].setup.correlate_with == "PHI" and dict(p.jobs[1].setup.correlation)[("A", 1)] == -0.7
    d.jobs.item(1, 12).setText("")
    with pytest.raises(Exception, match="no property to correlate with"):
        d.read_plan()
    d.jobs.item(1, 12).setText("NOPE")
    with pytest.raises(Exception, match="listed before"):
        d.read_plan()
    d.jobs.item(1, 12).setText("PHI")
    assert d.check_setup()
    d.panel.wait()
    assert d.panel.outcome[0] == "ok" and d.save()
    d2 = dlg(pr, model, tmp_path)                       # the saved plan comes back into the fields
    assert d2.read_plan() == pr.realization_plan and d2.scat.currentText() == "correlated"

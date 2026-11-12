import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qt = pytest.importorskip("PySide6.QtWidgets")

from cosma.facies_sim import FaciesSimError, FaciesSimSetup, from_state, run, to_state
from cosma.gui.facies_sim_dialog import CHOOSE, KEEP, SIS, TGS, FaciesSimDialog
from cosma.gui.main_window import MainWindow
from cosma.persist import load_project, save_project
from cosma.proportion_maps import ProportionMap
from cosma.proportions import ProportionSetup, Settings
from cosma.sis import SisSettings
from cosma.strat import StratChart
from cosma.tgs import TgsSettings
from cosma.variogram import Structure, VariogramModel
from test_datacheck import make_model, make_project
from test_main_window import Hooks


@pytest.fixture(scope="module")
def app():
    return qt.QApplication.instance() or qt.QApplication([])


def project(maps=False):
    pr = make_project()
    pr.chart = StratChart(["A", "B"], "Base")
    pr.set_proportions(ProportionSetup(Settings("relative", nbins=5), (("A", "wells"), ("B", "wells"))))
    for iv in pr.chart.intervals:
        for f in pr.facies_setup.scheme:
            pr.attach_variogram(iv.uid, f.name, VariogramModel(0.0, (Structure("spherical", 1.0, 300.0, 300.0, 10.0),), 0.0, "indicator"), "v")
        pr.attach_variogram(iv.uid, "GAUSS", VariogramModel(0.0, (Structure("spherical", 1.0, 300.0, 300.0, 10.0),), 0.0, "normal_score"), "g")
    if maps:
        fr = np.zeros((6, 3, 2))
        fr[..., 0], fr[..., 1] = 0.5, 0.5
        pr.set_proportion_map(ProportionMap("A", (1, 2), fr, {}, ""))
    return pr


SIS_S = FaciesSimSetup("sis", SisSettings(8, 1.0, 1), None, 3)
TGS_S = FaciesSimSetup("tgs", None, TgsSettings((1, 2), "GAUSS", 8, 1.0, 1, 3), 3)


def test_setup_validation_and_state_roundtrip():
    for s in (SIS_S, TGS_S, FaciesSimSetup("sis", SisSettings(8, 1.0, 1), None, 3, "curve", 0.2)):
        assert from_state(to_state(s)) == s
    assert from_state(None) is None and to_state(None) is None
    for bad in (lambda: FaciesSimSetup("x", None, None, 1), lambda: FaciesSimSetup("sis", None, None, 1),
                lambda: FaciesSimSetup("tgs", SisSettings(8, 1.0, 1), None, 1), lambda: FaciesSimSetup("sis", SisSettings(8, 1.0, 1), None, -1),
                lambda: FaciesSimSetup("sis", SisSettings(8, 1.0, 1), None, 1, "guess", 0.1),
                lambda: FaciesSimSetup("sis", SisSettings(8, 1.0, 1), None, 1, "curve", 2.0)):
        with pytest.raises(FaciesSimError):
            bad()
    for mut in (lambda d: d.pop("method"), lambda d: d["sis"].update(levels=0), lambda d: d.update(seed="x")):
        d = to_state(SIS_S)
        mut(d)
        with pytest.raises(FaciesSimError):
            from_state(d)


def test_run_end_to_end_for_both_methods_honours_the_wells():
    pr, model = project(), make_model()
    for s in (SIS_S, TGS_S):
        res, notes = run(pr, model, s)
        f = res.facies
        assert f.shape == (6, 3, 10) and set(np.unique(f)) <= {1, 2} and (f > 0).all()
        assert res.per_interval["A"]["hard"] > 0
        again, _ = run(pr, model, s)
        np.testing.assert_array_equal(f, again.facies)


def test_run_refuses_what_is_missing():
    model = make_model()
    pr = project()
    pr.proportions = None
    with pytest.raises(FaciesSimError, match="proportion curves first"):
        run(pr, model, SIS_S)
    pr = project()
    pr.facies_setup = None
    with pytest.raises(FaciesSimError, match="define the facies"):
        run(pr, model, SIS_S)
    pr = project()
    uid = pr.chart.intervals[1].uid
    name = pr.facies_setup.scheme[0].name
    del pr.variograms[uid][name]
    del pr.var_primary[(uid, name)]
    with pytest.raises(FaciesSimError, match="interval 'B': no indicator variogram"):
        run(pr, model, SIS_S)
    pr = project(maps=True)
    with pytest.raises(FaciesSimError, match="proportion maps"):
        run(pr, model, SIS_S)
    res, notes = run(pr, model, FaciesSimSetup("sis", SisSettings(8, 1.0, 1), None, 3, "curve", 0.5))
    assert any("A: map applied" in n for n in notes) and res.facies.shape == (6, 3, 10)


def dialog(pr=None, **kw):
    return FaciesSimDialog(pr or project(), build=lambda p: make_model(), **kw)


def fill_sis(d):
    d.method.setCurrentText(SIS)
    for e, v in ((d.nn, "8"), (d.search, "1"), (d.levels, "1"), (d.seed, "3")):
        e.setText(v)


def test_nothing_is_preset_except_the_sis_method_and_gaps_are_named(app):
    d = dialog()
    assert d.method.currentText() == SIS and all(e.text() == "" for e in (d.nn, d.search, d.levels, d.seed, d.gname, d.gibbs))
    assert d.tgs_box.isHidden() and d.map_box.isHidden() and not d.save_btn.isEnabled()
    d.simulate()
    d.panel.wait()
    assert "neighbours: enter a whole number" in d.problem.text()
    d.nn.setText("8")
    d.simulate()
    d.panel.wait()
    assert "search radius" in d.problem.text()
    d.search.setText("0")
    d.levels.setText("1")
    d.seed.setText("3")
    d.simulate()
    d.panel.wait()
    assert "search_factor" in d.problem.text()


def test_simulate_then_save_stores_exactly_what_was_run(app):
    pr = project()
    d = dialog(pr)
    fill_sis(d)
    assert not d.save() and "Simulate with these entries" in d.problem.text() and pr.facies_sim is None
    d.simulate()
    d.panel.wait()
    assert d.problem.text() == "" and "cells simulated" in d.report.toPlainText() and d.save_btn.isEnabled()
    assert d.layer.maximum() == 9 and d.row.maximum() == 2
    d.layer.setValue(3)
    d.seed.setText("4")
    assert not d.save_btn.isEnabled() and not d.save() and pr.facies_sim is None
    d.simulate()
    d.panel.wait()
    assert d.save() and pr.facies_sim.seed == 4 and pr.facies_sim.method == "sis"


def test_truncated_gaussian_needs_a_confirmed_order_and_a_variogram_name(app):
    pr = project()
    d = dialog(pr)
    fill_sis(d)
    d.method.setCurrentText(TGS)
    assert not d.tgs_box.isHidden()
    d.simulate()
    d.panel.wait()
    assert "confirm the facies order" in d.problem.text()
    d.order.setCurrentRow(0)
    d._move(1)
    assert [d.order.item(i).text() for i in range(2)] == [pr.facies_setup.scheme[1].name, pr.facies_setup.scheme[0].name]
    d.order_ok.setChecked(True)
    d.gname.setText("GAUSS")
    d.gibbs.setText("3")
    d.simulate()
    d.panel.wait()
    assert d.problem.text() == "", d.problem.text()
    d.order_ok.setChecked(False)
    assert not d.save_btn.isEnabled()
    d.order_ok.setChecked(True)
    d.simulate()
    d.panel.wait()
    assert d.save() and pr.facies_sim.tgs.order == (2, 1)
    d2 = dialog(pr)
    assert d2.method.currentText() == TGS and d2.order_ok.isChecked() and d2.gname.text() == "GAUSS"
    assert [d2.order.item(i).text() for i in range(2)] == [pr.facies_setup.scheme[1].name, pr.facies_setup.scheme[0].name]


def test_map_options_appear_only_when_maps_exist_and_must_be_chosen(app):
    d = dialog(project(maps=True))
    fill_sis(d)
    assert not d.map_box.isHidden()
    d.simulate()
    d.panel.wait()
    assert "choose what to do where a proportion map has no data" in d.problem.text()
    d.outside.setCurrentText(KEEP)
    d.simulate()
    d.panel.wait()
    assert "largest share" in d.problem.text()
    d.incompat.setText("0.5")
    d.simulate()
    d.panel.wait()
    assert d.problem.text() == "" and d.save_btn.isEnabled()


def test_a_failing_run_is_reported_and_blocks_saving(app):
    d = dialog(runner=lambda p, m, s: (_ for _ in ()).throw(FaciesSimError("no variogram")))
    fill_sis(d)
    d.simulate()
    d.panel.wait()
    assert d.problem.text() == "no variogram" and not d.save_btn.isEnabled()


def test_setup_survives_project_save_and_load_and_menu_entry(app, tmp_path):
    pr = project()
    pr.set_facies_sim(TGS_S)
    back = load_project(save_project(pr, tmp_path / "p.cosma"))
    assert back.facies_sim == TGS_S
    seen = []
    h = Hooks()
    w = MainWindow(hooks={**h.dict(), "facies_sim": lambda p: seen.append(p)})
    assert not w.act_fsim.isEnabled()
    w.project = project()
    from cosma.grid_definition import GridDefinition
    w.project.grid_def = GridDefinition(0, 0, 0, 6, 4, 100, 100)
    w.refresh()
    assert w.act_fsim.isEnabled()
    w.facies_sim_dialog()
    assert seen == [w.project]


def test_simulation_runs_in_a_thread_with_progress_and_can_be_stopped(app):
    import cosma.gui.facies_sim_dialog as FD
    d = dialog()
    fill_sis(d)
    labels = []
    real = FD.run

    def watch(project, model, setup, prepared=None, seed=None, progress=None):
        def p(l, dn, t):
            labels.append((l, dn, t))
            d.panel.ctx.cancel_requested = len(labels) >= 2
            progress(l, dn, t)
        return real(project, model, setup, prepared, seed, p)
    d.runner = watch
    d.simulate()
    assert d.panel.running and not d.sim_btn.isEnabled()
    d.panel.wait()
    assert labels and labels[0][0].startswith("facies, interval") and labels[0][1] == 0
    assert d.panel.outcome[0] == "cancelled" and "Stopped" in d.problem.text() and not d.save_btn.isEnabled() and d.sim_btn.isEnabled()
    d.runner = real
    d.simulate()
    d.panel.wait()
    assert d.panel.outcome[0] == "ok" and d.save_btn.isEnabled() and d.panel.bar.value() == 1000

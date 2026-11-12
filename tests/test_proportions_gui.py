import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qt = pytest.importorskip("PySide6.QtWidgets")

from cosma.gui.main_window import MainWindow
from cosma.gui.proportions_dialog import CHOOSE, M_DEPTH, M_REL, S_USER, S_WELLS, ProportionsDialog
from cosma.proportions import ProportionError
from cosma.strat import StratChart
from test_datacheck import make_model, make_project
from test_main_window import Hooks


@pytest.fixture(scope="module")
def app():
    return qt.QApplication.instance() or qt.QApplication([])


def project():
    pr = make_project()
    pr.chart = StratChart(["A", "B"], "Base")
    return pr


def dialog(pr=None, fill=True, sources=(S_WELLS, S_WELLS)):
    pr = pr or project()
    d = ProportionsDialog(pr, build=lambda p: make_model())
    if fill:
        d.mode.setCurrentText(M_REL)
        for e, v in ((d.nbins, "5"), (d.window, "1"), (d.min_wells, "1"), (d.min_cells, "1")):
            e.setText(v)
        for n, s in zip(d.names, sources):
            d.sources[n].setCurrentText(s)
    return d


def test_nothing_preselected_and_each_gap_is_named(app):
    d = dialog(fill=False)
    assert d.mode.currentText() == CHOOSE and all(c.currentText() == CHOOSE for c in d.sources.values())
    assert not d.save_btn.isEnabled()
    d.compute()
    assert "choose how positions" in d.problem.text()
    d.mode.setCurrentText(M_REL)
    d.compute()
    assert "number of bins" in d.problem.text()
    d.nbins.setText("2.5")
    d.compute()
    assert "whole number" in d.problem.text()
    d.nbins.setText("5")
    for e in (d.window, d.min_wells, d.min_cells):
        e.setText("1")
    d.compute()
    assert "choose a source" in d.problem.text() and not d.save_btn.isEnabled()


def test_compute_then_save_stores_exactly_what_was_computed(app):
    pr = project()
    d = dialog(pr)
    assert not d.save() and "Compute" in d.problem.text() and pr.proportions is None
    d.compute()
    assert d.problem.text() == "" and "Vertical proportion curves" in d.report.toPlainText() and d.save_btn.isEnabled()
    d.nbins.setText("4")
    assert not d.save_btn.isEnabled()
    assert not d.save() and pr.proportions is None
    d.compute()
    assert d.save() and pr.proportions.settings.nbins == 4
    assert dict(pr.proportions.choices) == {n: "wells" for n in d.names}


def test_depth_mode_needs_a_bin_size(app):
    d = dialog(fill=False, sources=(S_WELLS, S_WELLS))
    d.mode.setCurrentText(M_DEPTH)
    for e in (d.window, d.min_wells, d.min_cells):
        e.setText("1")
    for c in d.sources.values():
        c.setCurrentText(S_WELLS)
    d.compute()
    assert "bin size" in d.problem.text()


def test_typed_curve_must_sum_to_one_and_roundtrips(app):
    pr = project()
    d = dialog(pr, sources=(S_USER, S_WELLS))
    d.compute()
    assert "type at least one bin" in d.problem.text()
    first = d.names[0]
    assert d.curve_for.currentText() == first
    d._add_bin()
    d.curve_table.setItem(0, 0, qt.QTableWidgetItem("0.5"))
    d.compute()
    assert "fill in a proportion" in d.problem.text()
    d.curve_table.setItem(0, 1, qt.QTableWidgetItem("0.4"))
    d.compute()
    assert "sum to 0.9" in d.problem.text() and not d.save_btn.isEnabled()
    d.curve_table.setItem(0, 1, qt.QTableWidgetItem("0.5"))
    d.compute()
    assert d.problem.text() == "" and d.save_btn.isEnabled()
    assert f"{first}: source user" in d.report.toPlainText()
    assert d.save()
    d2 = ProportionsDialog(pr, build=lambda p: make_model())
    assert d2.sources[first].currentText() == S_USER and d2.typed[first] == [["0.5", "0.5"]]
    assert d2.mode.currentText() == M_REL and d2.nbins.text() == "5"


def test_unbuildable_model_is_reported(app):
    def bad(p):
        raise ProportionError("the grid could not be built: x")
    d = dialog()
    d.build = bad
    d.compute()
    assert "grid could not be built" in d.problem.text() and not d.save_btn.isEnabled()


def test_menu_entry_enabled_with_wells_facies_and_chart(app, tmp_path):
    seen = []
    h = Hooks()
    w = MainWindow(hooks={**h.dict(), 'proportions': lambda p: seen.append(p)})
    assert not w.act_proportions.isEnabled()
    w.project = project()
    w.refresh()
    assert w.act_proportions.isEnabled()
    w.proportions_dialog()
    assert seen == [w.project]

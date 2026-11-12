import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qt = pytest.importorskip("PySide6.QtWidgets")

from cosma.datacheck import CoreSpec, DataCheckError, DataCheckSetup, PropertySpec
from cosma.gui.datacheck_dialog import FIELDS, DataCheckDialog
from cosma.gui.main_window import MainWindow
from test_datacheck import add_core, make_model, make_project, phi_curve, th
from test_main_window import Hooks


@pytest.fixture(scope="module")
def app():
    return qt.QApplication.instance() or qt.QApplication([])


VALUES = dict(min_hard_cells=1, min_wells=1, lag_v=0.7, nlags_v=4, min_pairs_v=1, min_usable_lag_v=0.7, lag_h=200, nlags_h=3,
              min_pairs_h=1, min_usable_lag_h=200, same_layer_tol=2, min_wells_h=2, min_core_pairs=5)


def dialog(pr=None, vals=None, props=True, core=False):
    pr = pr or make_project(extra={"PHIE": phi_curve})
    d = DataCheckDialog(pr, build=lambda p: make_model())
    for k, v in (vals or VALUES).items():
        d.edits[k].setText(str(v))
    if props:
        for c, v in enumerate(("PHI", "PHIE", "arithmetic")):
            d.props.setItem(0, c, qt.QTableWidgetItem(v))
    if core:
        for k, v in dict(porosity="CPOR", permeability="CKH", depth_tol="0.3").items():
            d.core[k].setText(v)
    return d


def test_nothing_has_a_default_and_save_needs_a_run(app):
    pr = make_project()
    d = DataCheckDialog(pr, build=lambda p: make_model())
    assert all(e.text() == "" for e in d.edits.values()) and not d.save_btn.isEnabled()
    d.run()
    assert "Minimum hard cells per interval and facies: enter a value" in d.problem.text() and not d.save_btn.isEnabled()
    d = dialog(pr)
    assert not d.save() and "Run the checks" in d.problem.text() and pr.data_checks is None


def test_run_shows_the_report_and_save_stores_what_was_run(app):
    pr = make_project(extra={"PHIE": phi_curve})
    d = dialog(pr)
    d.run()
    t = d.report.toPlainText()
    assert d.problem.text() == "" and d.save_btn.isEnabled()
    assert "PHI: A / Sand: 6 cells (3 well(s))" in t and "variance ratio cell/log" in t
    assert d.save() and d.saved
    s = pr.data_checks
    assert s.thresholds.min_core_pairs == 5 and s.thresholds.min_r2 is None and s.properties == (PropertySpec("PHI", "PHIE", "arithmetic"),)
    assert s.core is None
    d2 = DataCheckDialog(pr, build=lambda p: make_model())             # reopened: the saved entries come back
    assert d2.edits["min_hard_cells"].text() == "1" and d2.props.item(0, 1).text() == "PHIE" and d2.edits["min_r2"].text() == ""


def test_any_edit_after_a_run_switches_save_off_and_a_changed_entry_cannot_be_saved(app):
    pr = make_project(extra={"PHIE": phi_curve})
    d = dialog(pr)
    d.run()
    d.edits["lag_v"].setText("0.8")
    assert not d.save_btn.isEnabled() and not d.save() and pr.data_checks is None
    d.run()
    d.props.item(0, 2).setText("harmonic")
    assert not d.save_btn.isEnabled()
    d.run()
    qt.QPushButton.click(d.findChildren(qt.QPushButton)[0])           # Add row
    assert not d.save_btn.isEnabled()
    d.run()
    d.core["source"].setText("x")
    assert not d.save_btn.isEnabled()
    d.core["source"].setText("")
    d.run()
    d._ran = None
    d.save_btn.setEnabled(True)                                        # forced on: the save itself re-checks
    assert not d.save() and not d.save_btn.isEnabled()
    d.run()
    d.edits["nlags_v"].blockSignals(True)
    d.edits["nlags_v"].setText("5")                                    # edit without the signal reaching Save
    d.save_btn.setEnabled(True)
    assert not d.save() and pr.data_checks is None


def test_bad_entries_are_named(app):
    for key, val, msg in (("nlags_v", "2.5", "number of lags: enter a whole number"), ("lag_v", "abc", "enter a number"),
                          ("lag_v", "0", "must be a number > 0"), ("min_r2", "2", "between 0 and 1")):
        d = dialog(vals={**VALUES, key: val})
        d.run()
        assert msg in d.problem.text() and not d.save_btn.isEnabled(), key
    d = dialog()
    d.props.item(0, 2).setText("median")
    d.run()
    assert "averaging must be one of" in d.problem.text()
    d = dialog()
    d.props.item(0, 1).setText("")
    d.run()
    assert "property row 1" in d.problem.text()
    d = dialog(core=True)
    d.core["depth_tol"].setText("")
    d.run()
    assert "give the porosity column, the permeability column and the depth tolerance" in d.problem.text()
    d = dialog(core=True)
    d.core["depth_tol"].setText("x")
    d.run()
    assert "depth tolerance: enter a number" in d.problem.text()
    d = dialog()
    d.core["source"].setText("c1")
    d.run()
    assert "no core columns" in d.problem.text()


def test_core_entries_run_the_regression_and_roundtrip(app):
    pr = make_project(extra={"PHIE": phi_curve})
    add_core(pr)
    d = dialog(pr, core=True)
    d.run()
    t = d.report.toPlainText()
    assert d.problem.text() == "" and "facies, all intervals" in t
    assert d.save() and pr.data_checks.core == CoreSpec("CPOR", "CKH", 0.3, None)
    d2 = DataCheckDialog(pr, build=lambda p: make_model())
    assert d2.core["porosity"].text() == "CPOR" and d2.core["depth_tol"].text() == "0.3" and d2.core["source"].text() == ""


def test_run_refuses_without_facies_and_reports_build_failures(app):
    pr = make_project()
    pr.facies_setup = None
    d = dialog(pr)
    d.run()
    assert "define the facies first" in d.problem.text() and d.report.toPlainText() == ""

    def broken(p):
        raise DataCheckError("the grid cannot be built yet (x)")
    d = DataCheckDialog(make_project(), build=broken)
    for k, v in VALUES.items():
        d.edits[k].setText(str(v))
    d.run()
    assert "the grid cannot be built yet" in d.problem.text() and not d.save_btn.isEnabled()


def test_menu_entry_needs_wells_and_a_facies_setup_and_calls_the_hook(app):
    calls = []
    pr = make_project()
    fs, pr.facies_setup = pr.facies_setup, None
    win = MainWindow(pr, hooks={**Hooks().dict(), "datacheck": lambda p: calls.append(p)})
    assert not win.act_datacheck.isEnabled()
    pr.facies_setup = fs
    win.refresh()
    assert win.act_datacheck.isEnabled()
    win.datacheck_dialog()
    assert calls == [pr]
    pr.facies_setup = None
    win.refresh()
    win.datacheck_dialog()
    assert calls == [pr]

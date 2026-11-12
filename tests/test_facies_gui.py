import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qt = pytest.importorskip("PySide6.QtWidgets")

from cosma.facies import CodeMap, Condition, Facies, FaciesSetup, Flag, Rule
from cosma.gui.facies_dialog import CHOOSE, M_CODES, M_FLAGS, M_RULES, FaciesDialog
from cosma.gui.main_window import MainWindow
from cosma.project import Project
from test_facies import gr_log, make_grid, project_with, well
from test_main_window import Hooks

import numpy as np


@pytest.fixture(scope="module")
def app():
    return qt.QApplication.instance() or qt.QApplication([])


def project():
    md, gr = gr_log()
    return project_with(well("A", 50.0, 50.0, {"GR": gr}, md), well("B", 150.0, 50.0))


def fill(t, rows):
    t.setRowCount(max(t.rowCount(), len(rows)))
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            t.setItem(r, c, qt.QTableWidgetItem(str(v)))


def rules_dialog(pr=None, build=None):
    d = FaciesDialog(pr or project(), build=build or (lambda p: (None, "no grid in this test")))
    fill(d.scheme, [[1, "Sand"], [2, "Silt"]])
    d.method.setCurrentText(M_RULES)
    fill(d.rules, [["Sand", "GR", "<", 50, "", "", ""], ["Silt", "GR", ">=", 50, "GR", "<", 100]])
    return d


def test_nothing_is_chosen_at_first_and_save_needs_a_preview(app):
    pr = project()
    d = FaciesDialog(pr, build=lambda p: (None, ""))
    assert d.method.currentText() == CHOOSE and not d.save_btn.isEnabled()
    d.preview()
    assert "choose how facies are found" in d.problem.text() and not d.save_btn.isEnabled()
    d = rules_dialog(pr)
    assert not d.save() and "Preview these entries first" in d.problem.text() and pr.facies_setup is None


def test_preview_shows_the_classification_and_save_stores_what_was_previewed(app):
    pr = project()
    d = rules_dialog(pr)
    d.preview()
    t = d.report.toPlainText()
    assert d.problem.text() == "" and d.save_btn.isEnabled()
    assert "A [lg]" in t and "Sand 24.9%" in t and "Silt 29.0%" in t and "undefined 46.1%" in t      # by length: 30, 35 and 55.5 of 120.5 m
    assert "skipped B" in t and "Choose the minimum classified share" in t
    assert d.save() and d.saved
    want = FaciesSetup((Facies(1, "Sand"), Facies(2, "Silt")),
                       (Rule(1, (Condition("GR", "<", 50.0),)), Rule(2, (Condition("GR", ">=", 50.0), Condition("GR", "<", 100.0)))))
    assert pr.facies_setup == want


def test_any_edit_after_the_preview_switches_save_off(app):
    for edit in (lambda d: d.rules.item(0, 3).setText("60"), lambda d: d.scheme.item(1, 1).setText("Mud"),
                 lambda d: d.source.setText("lg"), lambda d: d.coverage.setText("0.5"), lambda d: d.conflict.setCurrentIndex(1),
                 lambda d: [b for b in d.findChildren(qt.QPushButton) if b.text() == "Remove last"][1].click(),
                 lambda d: [b for b in d.findChildren(qt.QPushButton) if b.text() == "Add row"][3].click(),
                 lambda d: d.step.setText("1")):
        pr = project()
        d = rules_dialog(pr)
        d.preview()
        assert d.save_btn.isEnabled()
        edit(d)
        assert not d.save_btn.isEnabled() and not d.save() and pr.facies_setup is None


def test_bad_entries_are_named(app):
    cases = [
        (lambda d: d.scheme.item(0, 0).setText("x"), "facies row 1: the code must be a whole number"),
        (lambda d: d.scheme.item(0, 0).setText("0"), "facies row 1"),
        (lambda d: d.rules.item(0, 0).setText("Gravel"), "rule 1: 'Gravel' is not a facies of the scheme"),
        (lambda d: d.rules.item(0, 3).setText("abc"), "rule 1: enter a number"),
        (lambda d: d.rules.item(0, 2).setText(""), "rule 1: a condition needs a curve, an operator and a value"),
        (lambda d: d.rules.item(0, 2).setText("=="), "operator"),
        (lambda d: [d.rules.item(0, c).setText("") for c in range(1, 4)], "rule 1 has no condition"),
        (lambda d: d.step.setText("abc"), "flag step: enter a number"),
        (lambda d: d.coverage.setText("1.5"), "min_coverage"),
        (lambda d: d.coverage.setText("abc"), "minimum classified share: enter a number"),
    ]
    for edit, msg in cases:
        d = rules_dialog()
        edit(d)
        d.preview()
        assert msg in d.problem.text(), (msg, d.problem.text())
        assert not d.save_btn.isEnabled()


def test_code_map_and_flags_flows(app):
    pr = project()
    md = np.arange(1000.0, 1010.0)
    pr.wells["C"] = well("C", 250.0, 50.0, {"FAC": np.array([10, 10, 20, 20, 30, 30, 10, 10, 20, 20.0])}, md)
    d = FaciesDialog(pr, build=lambda p: (None, "n/a"))
    fill(d.scheme, [[1, "Sand"], [2, "Silt"]])
    d.method.setCurrentText(M_CODES)
    d.map_curve.setText("FAC")
    fill(d.codes, [[10, "Sand"], [20, "Silt"], [10, "Silt"]])
    d.preview()
    assert "mapped twice" in d.problem.text()
    d.codes.setRowCount(2)
    d.preview()
    t = d.report.toPlainText()
    assert d.problem.text() == "" and "C [lg]" in t and "curve values with no mapping: 30 (2)" in t and "skipped A" in t
    # flags only: well B has no log, so a flag step is needed
    d.method.setCurrentText(M_FLAGS)
    fill(d.flags, [["B", 1000, 1010, "md", "Sand"], ["B", 1010, 1020, "md", "Silt"]])
    d.preview()
    assert "flag step" in d.problem.text()
    d.step.setText("0.5")
    d.preview()
    assert d.problem.text() == "" and "B [flags only]" in d.report.toPlainText() and "Sand 50.0%" in d.report.toPlainText()
    assert d.save()
    s = pr.facies_setup
    assert s.flags == (Flag("B", 1000.0, 1010.0, "md", 1), Flag("B", 1010.0, 1020.0, "md", 2)) and s.flag_step == 0.5
    d.preview()
    assert d.save_btn.isEnabled()
    d.flags.item(0, 4).setText("Mud")                       # an edit of the flag table switches Save off ...
    assert not d.save_btn.isEnabled()
    d.preview()
    assert "'Mud' is not a facies" in d.problem.text()
    d.flags.item(0, 4).setText("Sand")
    d.method.setCurrentText(M_CODES)
    d.preview()
    assert d.save_btn.isEnabled()
    d.codes.item(0, 1).setText("Silt")                      # ... and so does an edit of the code map
    assert not d.save_btn.isEnabled()


def test_hard_cells_appear_when_a_grid_and_the_upscaling_choices_exist(app):
    pr = project()
    g = make_grid()
    d = rules_dialog(pr, build=lambda p: (g, ""))
    d.coverage.setText("0.5")
    d.conflict.setCurrentIndex(2)
    d.preview()
    t = d.report.toPlainText()
    assert "Hard cells in the grid" in t and "Sand (code 1): 3 hard cells" in t and "Silt (code 2): 4 hard cells" in t
    assert d.save() and pr.facies_setup.min_coverage == 0.5 and pr.facies_setup.conflict == "drop"
    d2 = rules_dialog(pr, build=lambda p: (None, "the grid could not be built: boom"))
    d2.coverage.setText("0.5")
    d2.conflict.setCurrentIndex(1)
    d2.preview()
    assert "Hard cells not shown: the grid could not be built: boom" in d2.report.toPlainText() and d2.save_btn.isEnabled()


def test_the_dialog_shows_a_stored_setup_again(app):
    pr = project()
    pr.set_facies_setup(FaciesSetup((Facies(1, "Sand"), Facies(2, "Silt")),
                                    (Rule(1, (Condition("GR", "<", 50.0),)),
                                     Rule(2, (Condition("GR", ">=", 50.0), Condition("GR", "<", 100.0)))),
                                    source="lg", flags=(Flag("A", 1000.0, 1002.5, "tvdss", 2),), flag_step=0.25,
                                    min_coverage=0.6, conflict="longest"))
    d = FaciesDialog(pr, build=lambda p: (None, ""))
    assert d.method.currentText() == M_RULES and d.source.text() == "lg" and d.step.text() == "0.25"
    assert d.coverage.text() == "0.6" and d.conflict.currentIndex() == 1
    assert d.rules.item(1, 4).text() == "GR" and d.rules.item(1, 5).text() == "<" and d.rules.item(1, 6).text() == "100"
    assert [d.flags.item(0, c).text() for c in range(5)] == ["A", "1000", "1002.5", "tvdss", "Silt"]
    assert not d.save_btn.isEnabled()                      # nothing edited, nothing previewed
    assert d.read_setup() == pr.facies_setup
    d.preview()
    assert d.problem.text() == "" and d.save()
    pr2 = project()
    pr2.set_facies_setup(FaciesSetup((Facies(1, "Sand"),), code_map=CodeMap("FAC", {10: 1})))
    d2 = FaciesDialog(pr2, build=lambda p: (None, ""))
    assert d2.method.currentText() == M_CODES and d2.map_curve.text() == "FAC" and d2.codes.item(0, 1).text() == "Sand"


def test_menu_entry_needs_wells(app):
    calls = []
    d = Hooks().dict()
    d["facies"] = calls.append
    w = MainWindow(Project("p"), hooks=d)
    assert not w.act_facies.isEnabled()
    pr = project()
    w.set_project(pr)
    w.refresh()
    assert w.act_facies.isEnabled()
    w.facies_dialog()
    assert calls == [pr]


def test_what_is_saved_must_be_what_was_previewed_even_if_a_signal_is_missed(app):
    pr = project()
    d = rules_dialog(pr)
    d.preview()
    assert d.save_btn.isEnabled()
    d.rules.blockSignals(True)
    d.rules.item(0, 3).setText("10")                         # changed without the dialog noticing
    d.rules.blockSignals(False)
    assert d.save_btn.isEnabled()
    assert not d.save() and "Preview these entries first" in d.problem.text() and pr.facies_setup is None
    assert not d.save_btn.isEnabled()

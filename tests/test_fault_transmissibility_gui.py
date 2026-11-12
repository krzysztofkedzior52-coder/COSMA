import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qt = pytest.importorskip("PySide6.QtWidgets")

from cosma.fault_transmissibility import FaultTransmissibility as FT
from cosma.gui.fault_transmissibility_dialog import (BY_THICKNESS, BY_THROW, CALC, CHOOSE, FIXED, OPEN, SGR, builtin_label, own_label,
                                                    FaultTransmissibilityDialog)
from cosma.gui.main_window import MainWindow
from cosma.project import Project
from test_faults_gui import crossing_project
from test_main_window import Hooks


@pytest.fixture(scope="module")
def app():
    return qt.QApplication.instance() or qt.QApplication([])


def choose(d, name, mode, val="", basis=None, thick=""):
    e = d.rows[name]
    e["mode"].setCurrentText(mode)
    e["val"].setText(val)
    if basis:
        e["basis"].setCurrentText(basis)
    e["thick"].setText(thick)


def test_nothing_is_preselected_and_saving_without_choices_is_refused(app):
    pr = crossing_project()
    d = FaultTransmissibilityDialog(pr)
    assert set(d.rows) == {"A", "B", "T"} and all(e["mode"].currentText() == CHOOSE for e in d.rows.values())
    assert not d.save() and "no choice made" in d.problem.text()
    assert pr.fault_transmissibility == {}


def test_choices_are_stored_and_shown_again(app):
    pr = crossing_project()
    d = FaultTransmissibilityDialog(pr)
    choose(d, "A", OPEN)
    choose(d, "B", FIXED, "0")
    choose(d, "T", CALC, "0.02", BY_THROW, "0.015")
    assert d.save() and d.problem.text() == ""
    assert pr.fault_transmissibility == {"A": FT("open"), "B": FT("fixed", multiplier=0.0),
                                         "T": FT("calculated", method="slab", kf=0.02, thickness_per_throw=0.015)}
    d2 = FaultTransmissibilityDialog(pr)
    assert d2.rows["B"]["mode"].currentText() == FIXED and d2.rows["B"]["val"].text() == "0"
    t = d2.rows["T"]
    assert t["mode"].currentText() == CALC and t["basis"].currentText() == BY_THROW
    assert t["val"].text() == "0.02" and t["thick"].text() == "0.015"
    choose(d2, "T", CALC, "0.5", BY_THICKNESS, "0.3")
    assert d2.save() and pr.fault_transmissibility["T"] == FT("calculated", method="slab", kf=0.5, thickness=0.3)


def test_bad_entries_are_named_and_nothing_wrong_is_stored(app):
    pr = crossing_project()
    d = FaultTransmissibilityDialog(pr)
    choose(d, "A", OPEN)
    choose(d, "B", FIXED, "-2")
    choose(d, "T", FIXED, "abc")
    assert not d.save()
    assert d.problem.text().startswith("B:") and ">= 0" in d.problem.text()
    assert pr.fault_transmissibility == {"A": FT("open")}              # the valid row is kept, the bad ones are not
    choose(d, "B", CALC, "0.1", BY_THICKNESS, "")
    assert not d.save() and "thickness value" in d.problem.text()


def test_fields_follow_the_chosen_treatment(app):
    d = FaultTransmissibilityDialog(crossing_project())
    e = d.rows["A"]
    assert not e["val"].isEnabled() and not e["thick"].isEnabled() and not e["basis"].isEnabled()
    e["mode"].setCurrentText(FIXED)
    assert e["val"].isEnabled() and not e["thick"].isEnabled()
    e["mode"].setCurrentText(CALC)
    assert e["val"].isEnabled() and e["thick"].isEnabled() and e["basis"].isEnabled()
    e["mode"].setCurrentText(OPEN)
    assert not e["val"].isEnabled()


def test_calculated_is_not_offered_for_a_fault_known_to_have_no_throw(app):
    pr = crossing_project()
    pr.fault_throw_seen = {"A": True, "B": False}
    d = FaultTransmissibilityDialog(pr)
    items = lambda n: [d.rows[n]["mode"].itemText(k) for k in range(d.rows[n]["mode"].count())]   # noqa: E731
    assert CALC in items("A") and CALC not in items("B") and CALC in items("T")      # T: not built yet, unknown
    assert d.table.cellWidget(1, 1).text() == "negligible" and d.table.cellWidget(0, 1).text() == "visible"
    assert "unknown" in d.table.cellWidget(2, 1).text()


def test_menu_entry_and_forgetting(app):
    calls = []
    h = Hooks()
    d = h.dict()
    d["fault_transmissibility"] = calls.append
    w = MainWindow(Project("p"), hooks=d)
    assert not w.act_fault_tm.isEnabled()
    pr = crossing_project()
    w.set_project(pr)
    w.refresh()
    assert w.act_fault_tm.isEnabled()
    w.fault_transmissibility()
    assert calls == [pr]
    pr.set_fault_transmissibility("A", FT("open"))
    pr.fault_throw_seen["A"] = True
    pr.remove_fault("A")
    assert "A" not in pr.fault_transmissibility and "A" not in pr.fault_throw_seen


def sgr_row(d, name, method, thick="0.015", basis=BY_THROW, minthr="1"):
    e = d.rows[name]
    e["mode"].setCurrentText(SGR)
    e["rel"].setCurrentText(method)
    e["basis"].setCurrentText(basis)
    e["thick"].setText(thick)
    e["minthr"].setText(minthr)


BUILTIN = builtin_label("manzocchi1999")


def add_mine(d, name="mine", expr="0.5 - 3*SGR", cite="Own study 2026", note="basin X"):
    d.m_name.setText(name)
    d.m_expr.setText(expr)
    d.m_cite.setText(cite)
    d.m_note.setText(note)
    return d.add_method()


def test_sgr_with_the_builtin_relation_or_own_method_is_stored_and_shown_again(app):
    pr = crossing_project()
    d = FaultTransmissibilityDialog(pr)
    assert d.rows["A"]["rel"].count() == 2                                       # choose + the one built-in
    assert add_mine(d) and "mine" in pr.sgr_methods and d.rows["A"]["rel"].count() == 3
    choose(d, "A", OPEN)
    sgr_row(d, "B", BUILTIN)
    sgr_row(d, "T", own_label("mine"), "0.3", BY_THICKNESS, "2.5")
    assert d.save(), d.problem.text()
    assert pr.fault_transmissibility["B"] == FT("calculated", method="sgr", relation="manzocchi1999",
                                                thickness_per_throw=0.015, min_throw=1.0)
    assert pr.fault_transmissibility["T"] == FT("calculated", method="sgr", expression="0.5 - 3*SGR", expression_name="mine",
                                                thickness=0.3, min_throw=2.5)
    d2 = FaultTransmissibilityDialog(pr)
    b, t = d2.rows["B"], d2.rows["T"]
    assert b["mode"].currentText() == SGR and b["rel"].currentText() == BUILTIN and b["minthr"].text() == "1"
    assert t["rel"].currentText() == own_label("mine") and t["basis"].currentText() == BY_THICKNESS and t["thick"].text() == "0.3"
    assert not b["val"].isEnabled()
    t["rel"].setCurrentText(own_label("mine"))
    assert "Your method 'mine'" in d2.relation_note.text() and "Own study 2026" in d2.relation_note.text() and "basin X" in d2.relation_note.text()


def test_a_choice_keeps_its_own_copy_when_the_stored_method_changes_or_is_removed(app):
    pr = crossing_project()
    d = FaultTransmissibilityDialog(pr)
    add_mine(d)
    choose(d, "A", OPEN)
    choose(d, "B", OPEN)
    sgr_row(d, "T", own_label("mine"))
    assert d.save()
    add_mine(d, expr="-9 - SGR")                                                  # replace the stored method
    assert pr.sgr_methods["mine"].expression == "-9 - SGR"
    assert pr.fault_transmissibility["T"].expression == "0.5 - 3*SGR"            # the fault's choice did not change
    d2 = FaultTransmissibilityDialog(pr)
    label = d2.rows["T"]["rel"].currentText()
    assert "saved copy" in label and "0.5 - 3*SGR" in label
    d2.rows["T"]["rel"].setCurrentText(own_label("mine"))
    d2.rows["T"]["rel"].setCurrentText(label)
    assert "keeps its own saved copy" in d2.relation_note.text() and "0.5 - 3*SGR" in d2.relation_note.text()
    d2.rows["T"]["rel"].setCurrentText(label)
    assert d2.save() and pr.fault_transmissibility["T"].expression == "0.5 - 3*SGR"      # saving again keeps the copy
    d2.rows["T"]["rel"].setCurrentText(own_label("mine"))
    assert d2.save() and pr.fault_transmissibility["T"].expression == "-9 - SGR"         # choosing the new one is explicit
    d2.m_list.setCurrentText("mine")
    d2.remove_method()
    assert "mine" not in pr.sgr_methods and pr.fault_transmissibility["T"].expression == "-9 - SGR"
    assert d2.rows["T"]["rel"].findText(own_label("mine")) < 0


def test_methods_are_tested_added_loaded_removed_and_refused_with_reasons(app):
    pr = crossing_project()
    d = FaultTransmissibilityDialog(pr)
    d.m_expr.setText("-4*SGR - 0.25*log10(D)*(1 - SGR)**5")
    d.m_name.setText("t")
    d.test_method()
    t = d.m_report.toPlainText()
    from cosma.fault_relations import LIBRARY
    expect = f"{10 ** float(LIBRARY['manzocchi1999'].log10_kf(0.5, 20.0)):>9.3g}"
    assert "kf (mD) at SGR" in t and "D =    20 m:" in t and expect in t.splitlines()[3]
    for name, expr, cite, msg in (("", "-1", "src", "needs a name"), ("x", "q", "src", "unknown name"), ("x", "-1", "", "where this relation comes from"),
                                  ("manzocchi1999", "-1", "src", "built-in relation"), ("x", "__import__('os')", "src", "only these functions"),
                                  ("x", "1/0", "src", "not finite")):
        d.m_name.setText(name)
        d.m_expr.setText(expr)
        d.m_cite.setText(cite)
        assert not d.add_method() and msg in d.m_report.toPlainText(), (name, expr)
        assert pr.sgr_methods == {}
    assert add_mine(d, "a") and add_mine(d, "b", "-1 - log10(D)")
    d.m_list.setCurrentText("b")
    d.m_name.setText("")
    d.load_method()
    assert d.m_name.text() == "b" and d.m_expr.text() == "-1 - log10(D)" and d.m_cite.text() == "Own study 2026"
    d.m_list.setCurrentText("a")
    d.remove_method()
    assert list(pr.sgr_methods) == ["b"] and d.m_list.count() == 1


def test_methods_export_and_import_as_a_file_and_bad_files_are_reported(app, tmp_path):
    pr = crossing_project()
    path = tmp_path / "m.json"
    d = FaultTransmissibilityDialog(pr, pick_save=lambda t: str(path), pick_open=lambda t: str(path))
    d.export_methods()
    assert "no methods of your own" in d.m_report.toPlainText() and not path.exists()
    add_mine(d, "a")
    add_mine(d, "b", "-1 - log10(D)")
    d.export_methods()
    assert path.exists() and "Exported 2" in d.m_report.toPlainText()
    pr2 = crossing_project()
    d2 = FaultTransmissibilityDialog(pr2, pick_open=lambda t: str(path))
    d2.import_methods()
    assert pr2.sgr_methods == pr.sgr_methods and "Imported 2 method(s): a, b" in d2.m_report.toPlainText()
    assert d2.rows["A"]["rel"].findText(own_label("b")) >= 0
    d2.import_methods()
    assert "Replaced existing: a, b" in d2.m_report.toPlainText()
    path.write_text("{}")
    d2.import_methods()
    assert "Could not import" in d2.m_report.toPlainText() and set(pr2.sgr_methods) == {"a", "b"}
    d3 = FaultTransmissibilityDialog(pr2, pick_open=lambda t: str(tmp_path / "missing.json"))
    d3.import_methods()
    assert "Could not import" in d3.m_report.toPlainText()
    d4 = FaultTransmissibilityDialog(pr2, pick_open=lambda t: "")
    d4.import_methods()
    assert d4.m_report.toPlainText() == ""


def test_selecting_the_builtin_shows_its_citation_and_caveat(app):
    pr = crossing_project()
    d = FaultTransmissibilityDialog(pr)
    sgr_row(d, "A", BUILTIN)
    t = d.relation_note.text()
    assert "Petroleum Geoscience 5, 53-63" in t and "SGR tends to 0" in t and "median of a log-normal" in t
    assert "-4*SGR" in t


def test_bad_sgr_entries_are_named(app):
    for kw, msg in ((dict(rel=CHOOSE), "choose an SGR method"),
                    (dict(rel=BUILTIN, minthr=""), "minimum throw: enter a number"),
                    (dict(rel=BUILTIN, minthr="0"), "minimum throw must be"),
                    (dict(rel=BUILTIN, thick="abc"), "thickness value: enter a number")):
        pr = crossing_project()
        d = FaultTransmissibilityDialog(pr)
        choose(d, "A", OPEN)
        choose(d, "B", OPEN)
        sgr_row(d, "T", kw.get("rel"), thick=kw.get("thick", "0.015"), minthr=kw.get("minthr", "1"))
        assert not d.save() and msg in d.problem.text(), (kw, d.problem.text())
        assert "T" not in pr.fault_transmissibility


def test_the_dialog_says_whether_vshale_exists_in_the_wells(app):
    import numpy as np
    from cosma.project import WellRecord
    from cosma.wells.model import LogSet
    pr = crossing_project()
    d = FaultTransmissibilityDialog(pr)
    assert "no well has a VSHALE curve" in d.vshale.text()
    rec = WellRecord("W1")
    rec.logs["lg"] = LogSet(np.arange(3.0), {"vshale": np.zeros(3)})
    pr.wells["W1"] = rec
    d = FaultTransmissibilityDialog(pr)
    assert d.vshale.text() == "VSHALE curve found in: W1"
    assert "no well" not in FaultTransmissibilityDialog(Project("empty")).vshale.text()


def test_the_vshale_notice_is_shown_once_after_a_fault_or_log_import_and_not_when_vshale_exists(app, tmp_path):
    from test_faults_gui import FH, OK, sticks_csv
    h = FH(sticks_csv(tmp_path), OK)
    w = MainWindow(Project("p"), hooks=h.dict())
    w.do_import("fault_sticks")
    assert sum("no well has a VSHALE curve" in r for r in h.reports) == 1
    w.notify_vshale()
    assert sum("no well has a VSHALE curve" in r for r in h.reports) == 1           # once per session
    import numpy as np
    from cosma.project import WellRecord
    from cosma.wells.model import LogSet
    pr = Project("q")
    rec = WellRecord("W1")
    rec.logs["lg"] = LogSet(np.arange(3.0), {"VSHALE": np.zeros(3)})
    pr.wells["W1"] = rec
    h2 = FH(sticks_csv(tmp_path), OK)
    w2 = MainWindow(pr, hooks=h2.dict())
    w2.do_import("fault_sticks")
    assert not any("VSHALE" in r for r in h2.reports)
    h3 = FH(None, None)
    w3 = MainWindow(Project("r"), hooks=h3.dict())
    w3.notify_vshale()                                                              # no faults: nothing to say
    assert not h3.reports

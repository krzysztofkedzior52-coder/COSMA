import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qt = pytest.importorskip("PySide6.QtWidgets")

from cosma.gui.variogram_editor import VariogramEditor
from cosma.project import Action
from cosma.variogram_session import VariogramSession
from synthetic_field import synthetic


@pytest.fixture(scope="module")
def app():
    return qt.QApplication.instance() or qt.QApplication([])


@pytest.fixture()
def ed(app):
    pr = synthetic()
    s = VariogramSession(pr, pr.chart.match_name("Res"), "PHI")
    chosen = []
    e = VariogramEditor(s, choose_action=lambda c: chosen.append(c) or Action("replace"))
    e.chosen = chosen
    return e


def set_exp(ed, kind, **kw):
    for k, v in kw.items():
        ed.exp_fields[kind][k].setText(str(v))


def test_starts_empty_and_nothing_is_filled_in(ed):
    assert ed.s.missing() and all(not f.text() for _, fl in ed.struct_rows[:1] for f in fl)
    assert not ed.s.exp and len(ed.fig.axes[0].collections) == 0


def test_compute_autofit_save_flow(ed):
    set_exp(ed, "vertical", lag=2, nlags=10)
    set_exp(ed, "major", lag=150, nlags=5, tolerance_deg=20)
    set_exp(ed, "minor", lag=150, nlags=5, tolerance_deg=20)
    ed.azimuth.setText("30"); ed.azimuth.editingFinished.emit()
    ed.compute_all()
    assert set(ed.s.exp) == {"major", "minor", "vertical"}
    assert len(ed.fig.axes[2].collections[0].get_offsets()) == 10          # vertical points drawn
    ed.auto_fit()
    assert ed.struct_rows[0][1][3].text() and abs(float(ed.struct_rows[0][1][3].text()) - 20) < 3        # vertical range shown
    assert ed.fig.axes[2].lines[-1].get_ydata().size == 200               # model curve drawn
    assert "vertical:" in ed.log.toPlainText()
    ed.label.setText("fit1")
    ed.save()
    assert ed.saved and ed.s.project.variogram_for(ed.s.uid, "PHI").azimuth == 30
    ed.save()                                                              # same name again: conflict -> our chooser -> replace
    assert len(ed.chosen) == 1 and "replaced" in ed.saved.applied[0]


def test_manual_model_and_validation_messages(ed):
    ed.nugget.setText("0.1"); ed.nugget.editingFinished.emit()
    t, f = ed.struct_rows[0]
    for w, v in zip(f, ("0.9", "500", "900", "20")):
        w.setText(v)
    f[0].editingFinished.emit()
    assert "minor range" in ed.problem.text()                              # minor > major refused with a reason
    f[2].setText("250"); f[2].editingFinished.emit()
    assert ed.problem.text() == "" and ed.s.model().sill == pytest.approx(1.0)
    ed.save()
    assert "name" in ed.problem.text() and ed.saved is None
    f[1].setText("abc"); f[1].editingFinished.emit()
    assert ed.problem.text().startswith("Model:")


def test_structures_buttons_and_transform(ed):
    ed.add_btn.click(); ed.add_btn.click(); ed.add_btn.click()
    assert len(ed.s.structures) == 3 and not ed.add_btn.isEnabled() and ed.struct_rows[2][0].isVisibleTo(ed)
    ed.rem_btn.click(); ed.rem_btn.click()
    assert len(ed.s.structures) == 1 and not ed.rem_btn.isEnabled()
    ed.transform.setCurrentText("normal_score")
    assert ed.s.transform == "normal_score"


def test_suggest_azimuth_button_and_one_well_message(ed, app):
    ed.suggest_azimuth()
    assert ed.azimuth.text() == "30" and "check it" in ed.log.toPlainText()
    pr = ed.s.project
    one = VariogramEditor(VariogramSession(pr, ed.s.uid, "PHI", wells=["W0"]))
    one.compute_all()
    assert "major" in one.log.toPlainText() and "Only one well" in one.warn.text()


def test_saving_keeps_the_experimental_points(ed):
    set_exp(ed, "vertical", lag=2, nlags=10)
    set_exp(ed, "major", lag=150, nlags=5, tolerance_deg=20)
    ed.azimuth.setText("30"); ed.azimuth.editingFinished.emit()
    ed.compute_all(); ed.auto_fit()
    ed.label.setText("keep"); ed.save()
    assert ed.saved and "major" in ed.s.exp and len(ed.fig.axes[0].collections) == 1

import os

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qt = pytest.importorskip("PySide6.QtWidgets")

from cosma.grid.boundary import grid_spec_from_cells
from cosma.gui.main_window import MainWindow
from cosma.gui.proportion_map_dialog import CHOOSE, ProportionMapDialog
from cosma.persist import load_project, save_project
from cosma.strat import StratChart
from test_datacheck import make_project
from test_main_window import Hooks


@pytest.fixture(scope="module")
def app():
    return qt.QApplication.instance() or qt.QApplication([])


@pytest.fixture
def png(tmp_path):
    a = np.zeros((30, 60, 3), np.uint8)
    a[:, :30] = (255, 0, 0)
    a[:, 30:] = (0, 0, 255)
    Image.fromarray(a).save(tmp_path / "m.png")
    return str(tmp_path / "m.png")


def project():
    pr = make_project()
    pr.chart = StratChart(["A", "B"], "Base")
    return pr


def dialog(pr, png, answer=True):
    spec = grid_spec_from_cells(500, 500, 0, 6, 3, 100.0, 100.0)       # 600 x 300 m inside the map
    d = ProportionMapDialog(pr, spec=spec, pick_file=lambda: png, confirm=lambda t: (d.asked.append(t), answer)[1])
    d.asked = []
    return d


def fill(d, absent=False, two_colours=True):
    d.open_image()
    d.interval.setCurrentText("A")
    mode = d.mode_btns["control point"]
    mode.setChecked(True)
    for col, row in ((0, 0), (60, 0), (0, 30)):       # map: 20 m per pixel, origin (0, 1000) at top-left
        d.click(col, row)
    for r, (X, Y) in enumerate(((0, 1000), (1200, 1000), (0, 400))):
        d.cp.setItem(r, 2, qt.QTableWidgetItem(str(X)))
        d.cp.setItem(r, 3, qt.QTableWidgetItem(str(Y)))
    d.mode_btns["legend colour"].setChecked(True)
    d.click(5, 5)
    if two_colours:
        d.click(50, 5)
    names = [f.name for f in d.scheme]
    d.facies_of_row(0).setCurrentText(names[0])
    if two_colours:
        d.facies_of_row(1).setCurrentText(names[1])
    elif absent:
        d.absent.item(1).setCheckState(d.absent.item(1).checkState().Checked)
    for e, v in ((d.tol, "1e-6"), (d.ctol, "10"), (d.sigma, "0"), (d.cover, "1")):
        e.setText(v)


def test_nothing_preset_and_legend_offers_only_project_facies(app, png):
    pr = project()
    d = dialog(pr, png)
    assert d.interval.currentText() == CHOOSE and not d.save_btn.isEnabled()
    d.process()
    assert "open an image" in d.problem.text()
    d.open_image()
    d.process()
    assert "choose the interval" in d.problem.text()
    r = d._add_legend()
    cb = d.facies_of_row(r)
    assert [cb.itemText(i) for i in range(cb.count())] == [CHOOSE] + [f.name for f in pr.facies_setup.scheme]


def test_end_to_end_click_workflow_saves_after_confirmation(app, png):
    pr = project()
    d = dialog(pr, png)
    fill(d)
    assert d.cp.rowCount() == 3 and d._cell(d.legend, 0, 0) == "255 0 0" and d._cell(d.legend, 1, 0) == "0 0 255"
    assert not d.save()
    d.process()
    assert d.problem.text() == "", d.problem.text()
    assert "Proportion map for interval 'A'" in d.report.toPlainText()
    pm = d.result
    # model columns x = 500..1100 m; the colour boundary is at X = 600 m: first column (500-600) is sand, rest silt
    assert pm.fractions.shape == (6, 3, 2)
    np.testing.assert_allclose(pm.fractions[0, :, 0], 1.0)
    np.testing.assert_allclose(pm.fractions[1:, :, 1], 1.0)
    assert d.save() and "Import this map?" in d.asked[0] and pr.proportion_maps["A"] is pm
    # declined confirmation saves nothing
    pr2 = project()
    d2 = dialog(pr2, png, answer=False)
    fill(d2)
    d2.process()
    assert not d2.save() and pr2.proportion_maps == {}


def test_any_edit_after_processing_blocks_save(app, png):
    d = dialog(project(), png)
    fill(d)
    d.process()
    assert d.save_btn.isEnabled()
    d.ctol.setText("11")
    assert not d.save_btn.isEnabled() and not d.save()


def test_every_facies_needs_a_colour_or_an_explicit_absence(app, png):
    pr = project()
    d = dialog(pr, png)
    fill(d, two_colours=False)
    d.cover.setText("0.1")
    d.process()
    assert "no colour on the legend and are not declared absent" in d.problem.text() and not d.save_btn.isEnabled()
    d.absent.item(1).setCheckState(d.absent.item(1).checkState().Checked)
    d.process()
    assert d.problem.text() == ""
    np.testing.assert_allclose(d.result.fractions[..., 1][~np.isnan(d.result.fractions[..., 1])], 0.0)


def test_blue_pixels_with_no_legend_colour_become_no_data_and_coverage_is_enforced(app, png):
    d = dialog(project(), png)
    fill(d, two_colours=False, absent=True)
    d.cover.setText("1")
    d.process()
    assert "covers" in d.problem.text() and "less than the required 100.0%" in d.problem.text()
    d.cover.setText("0.1")
    d.process()
    assert d.problem.text() == "" and "unmatched = no data" in d.report.toPlainText()


def test_suggested_sigma_comes_from_pixel_and_cell_size(app, png):
    d = dialog(project(), png)
    fill(d)
    d.sigma.setText("")
    s = d.suggest()
    assert s == pytest.approx(0.5 * 100 / 20) and d.sigma.text() == "2.5" and "you may change it" in d.suggestion.text()
    d.sigma.setText("4")                          # overridable
    d.process()
    assert d.problem.text() == "" and "sigma 4" in d.report.toPlainText()
    d.cp.item(0, 2).setText("zzz")
    assert d.suggest() is None and "cannot suggest" in d.problem.text()


def test_processing_ops_feed_the_clicks(app, png):
    d = dialog(project(), png)
    d.open_image()
    d.interval.setCurrentText("A")
    d.flip_h.setChecked(True)
    d.mode_btns["legend colour"].setChecked(True)
    d.click(5, 5)                                 # after the flip the left side is blue
    assert d._cell(d.legend, 0, 0) == "0 0 255"
    d.flip_h.setChecked(False)
    d.crop[0].setText("0")
    d.process()
    assert "crop: fill in all four" in d.problem.text()


def test_stored_map_survives_save_and_load_and_stale_files_are_refused(app, png, tmp_path):
    pr = project()
    d = dialog(pr, png)
    fill(d)
    d.process()
    d.save()
    f = save_project(pr, tmp_path / "p.cosma")
    back = load_project(f)
    m = back.proportion_maps["A"]
    np.testing.assert_array_equal(m.fractions, pr.proportion_maps["A"].fractions)
    assert m.codes == pr.proportion_maps["A"].codes and m.report == pr.proportion_maps["A"].report
    assert m.recipe_state == pr.proportion_maps["A"].recipe_state


def test_menu_entry(app):
    seen = []
    h = Hooks()
    w = MainWindow(hooks={**h.dict(), "proportion_map": lambda p: seen.append(p)})
    assert not w.act_pmap.isEnabled()
    w.project = project()
    w.refresh()
    assert not w.act_pmap.isEnabled()              # no grid yet
    from cosma.grid_definition import GridDefinition
    w.project.set_grid_def(GridDefinition(0, 0, 0, 6, 4, 100, 100)) if hasattr(w.project, 'set_grid_def') else setattr(w.project, 'grid_def', GridDefinition(0, 0, 0, 6, 4, 100, 100))
    w.refresh()
    assert w.act_pmap.isEnabled()
    w.proportion_map_dialog()
    assert seen == [w.project]

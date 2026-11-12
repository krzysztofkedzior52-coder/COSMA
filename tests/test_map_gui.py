import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qt = pytest.importorskip("PySide6.QtWidgets")

from cosma.gui.main_window import MainWindow
from cosma.gui.map_dialog import MapComputeDialog, MapViewDialog
from cosma.maps import MapGrid, PropertyMap
from cosma.project import Action, Project
from cosma.surfaces import Surface
from cosma.variogram_session import VariogramSession
from synthetic_field import facies_project, synthetic
from test_main_window import Hooks, items, select
from test_map_session import ready


@pytest.fixture(scope="module")
def app():
    return qt.QApplication.instance() or qt.QApplication([])


def fill_wells_grid(d):
    d.grid_src.setCurrentIndex(2)
    d.inc.setText("100"); d.margin.setText("100")


def test_dialog_lists_properties_and_variograms_in_order(app):
    pr, u = ready()
    d = MapComputeDialog(pr, uid=u)
    assert d.interval.currentData() == u and [d.prop.itemText(i) for i in range(d.prop.count())] == ["PHI"]
    assert [d.vario.itemText(i) for i in range(d.vario.count())][0].startswith("PHI (layer average) / avg")
    assert d.vwarn.text() == ""
    d.vario.setCurrentIndex(1)
    assert "smoother" in d.vwarn.text()
    assert not d.save_btn.isEnabled()


def test_dialog_messages_for_missing_choices(app):
    pr, u = ready()
    d = MapComputeDialog(pr, uid=u)
    d.compute()
    assert "unit" in d.problem.text()                                         # nothing assumed
    d.unit.setCurrentText("fraction")
    d.compute()
    assert "increment" in d.problem.text() or "no surface" in d.problem.text()
    d.grid_src.setCurrentIndex(2)
    d.compute()
    assert "increment is required" in d.problem.text()
    d.inc.setText("abc"); d.margin.setText("100")
    d.compute()
    assert "not a number" in d.problem.text()
    d.inc.setText("100"); d.lo.setText("0")
    d.compute()
    assert "both ends" in d.problem.text()
    d.thickness()
    assert "top, base" in d.problem.text() or "needs a surface" in d.problem.text() or "no surface" in d.problem.text()
    d.interval.setCurrentIndex(0)                                              # Seal: no variogram stored
    assert d.vario.count() == 0 and "compute or import one first" in d.vwarn.text()
    d.unit.setCurrentText("fraction")
    d.compute()
    assert "choose a variogram" in d.problem.text()


def test_dialog_computes_previews_and_saves_with_conflict_choice(app):
    pr, u = ready()
    chosen = []
    d = MapComputeDialog(pr, uid=u, choose_action=lambda c: chosen.append(c) or Action("replace"))
    d.unit.setCurrentText("fraction")
    fill_wells_grid(d)
    d.coverage.setText("0.5")
    d.compute()
    assert d.problem.text() == "" and d.ms.result.method == "kriged" and d.save_btn.isEnabled()
    assert len(d.cv.fig.axes) >= 2 and d.cv.fig.axes[0].collections                       # estimate drawn
    assert any(l.get_marker() == "^" for ax in d.cv.fig.axes[:2] for l in ax.lines)        # wells marked
    d.label.setText("m1")
    d.save()
    assert d.saved and pr.map_for(u, "PHI").info["variogram_label"] == "avg"
    d.save()
    assert len(chosen) == 1 and "replaced" in d.saved.applied[0]
    d.label.setText("")
    d.save()
    assert "name" in d.problem.text()


def test_dialog_clip_mask_and_coverage_inputs(app):
    pr, u = ready()
    d = MapComputeDialog(pr, uid=u)
    d.unit.setCurrentText("fraction")
    fill_wells_grid(d)
    d.lo.setText("0.3"); d.hi.setText("0.31")
    d.compute()
    pm = d.ms.result
    assert pm.info["clipped"] > 0 and np.nanmax(pm.values) <= 0.31 and "clipped" in d.notes.toPlainText()
    d.lo.setText(""); d.hi.setText("")
    d.coverage.setText("1.0")
    d.compute()
    assert d.problem.text() != "" or all(s for s in d.ms.result.info["skipped"])          # either refused or skipped wells shown
    d.coverage.setText("2")
    d.compute()
    assert "between 0 and 1" in d.problem.text()


def test_dialog_thickness_and_surface_grid(app):
    pr, u = ready()
    pr.attach_surface(u, Surface(-100, -100, 100.0, 100.0, 0.0, 1, np.full((10, 10), 100.0)), "t")
    pr.attach_surface("base", Surface(-100, -100, 100.0, 100.0, 0.0, 1, np.full((10, 10), 150.0)), "b")
    d = MapComputeDialog(pr, uid=u)
    d.thickness()
    assert d.problem.text() == "" and np.all(d.ms.result.values == 50.0) and d.save_btn.isEnabled()
    d.label.setText("thk"); d.save()
    assert pr.map_for(u, "thickness").unit == "m"
    d.unit.setCurrentText("fraction")
    d.compute()                                                                  # grid from the top surface, masked by thickness
    assert d.problem.text() == "" and d.ms.result.grid.ncol == 10 and np.isfinite(d.ms.result.values).all()


def test_dialog_facies_proportion_choice(app):
    fp = facies_project(n=3, spacing=100.0)
    u = fp.chart.match_name("Res")
    a = VariogramSession(fp, u, "FACIES", indicator=2, average=True)
    a.set_nugget(0.01); a.set_structure(0, contribution=0.05, major=300, minor=150, vertical=50)
    a.save("avg")
    d = MapComputeDialog(fp, uid=u)
    d.prop.setCurrentText("FACIES")
    assert d.indicator.isEnabled() and d.code.count() == 3 and d.vario.count() == 0
    d.indicator.setChecked(True)
    d.code.setCurrentIndex(1)
    assert d.vario.count() == 1 and d.vario.itemText(0).startswith("FACIES[2] (layer average)")
    d.unit.setCurrentText("fraction"); fill_wells_grid(d)
    d.compute()
    assert d.ms.result.variable == "FACIES[2]" and d.problem.text() == ""


def test_view_dialog(app):
    g = MapGrid(0, 0, 10, 10, 3, 3)
    v = MapViewDialog(PropertyMap(g, np.arange(9.0).reshape(3, 3), "PHI", "fraction", "imported"))
    assert "values 0 to 8" in v.findChildren(qt.QLabel)[0].text() and v.cv.fig.axes[0].collections
    e = MapViewDialog(PropertyMap(g, np.full((3, 3), np.nan), "PHI", "fraction", "imported"))
    assert "no defined nodes" in e.findChildren(qt.QLabel)[0].text()


# -- main window ----------------------------------------------------------------------------------------------------------------------------------
def test_window_map_menu_tree_view_export_and_new(app, tmp_path):
    pr, u = ready()
    pr.attach_map(u, PropertyMap(MapGrid(0, 0, 10, 10, 3, 3), np.arange(9.0).reshape(3, 3), "PHI", "fraction", "imported"), "imp")
    h = Hooks()
    d = h.dict()
    shown, dialogs = [], []
    d["show_map"] = lambda pm: shown.append(pm)
    d["run_map_dialog"] = lambda p, uid: dialogs.append(uid)
    w = MainWindow(pr, hooks=d)
    assert w.act_new_map.isEnabled() and not w.act_view_map.isEnabled() and not w.act_export_map.isEnabled()
    row = next(i for i in items(w) if i.text(0) == "map: PHI")
    assert "imp (imported, fraction, 0 to 8)" in row.text(1) and "(used)" in row.text(1)
    select(w, "map: PHI")
    assert w.act_view_map.isEnabled() and w.act_export_map.isEnabled() and not w.act_edit_vario.isEnabled()
    w.act_view_map.trigger()
    assert shown and shown[0].variable == "PHI"
    h.save_to = str(tmp_path / "x.irap")
    w.act_export_map.trigger()
    from cosma.surfaces import read_surface
    assert read_surface(tmp_path / "x.irap", xy_unit="m", z_unit="m").values[2, 2] == 8.0
    w.act_new_map.trigger()
    assert dialogs == [u]                                                         # dialog opened on the selected map's interval
    select(w, "variogram: PHI")
    w.act_new_map.trigger()
    assert dialogs[-1] == u
    w.tree.setCurrentItem(w.tree.topLevelItem(0))
    w.act_new_map.trigger()
    assert dialogs[-1] is None


def test_window_property_map_import_and_empty_project_states(app, tmp_path):
    from test_surfaces import write_irap
    pr, u = ready()
    f = tmp_path / "phi.irap"
    write_irap(f, 4, 4, 0.0, 0.0, 100.0, 100.0, 0.0, lambda x, y: 18.0)
    h = Hooks()
    d = h.dict()
    d["pick_file"] = lambda k, t: str(f)
    d["confirm"] = lambda prev: dict(xy_unit="m", variable="PHI", value_unit="percent", interval=u)
    w = MainWindow(pr, hooks=d)
    w.import_actions["property_map"].trigger()
    assert pr.map_for(u, "PHI").values[0, 0] == pytest.approx(0.18) and any(i.text(0) == "map: PHI" for i in items(w))
    w2 = MainWindow(Project("x"))
    assert not w2.act_new_map.isEnabled() and not w2.import_actions["property_map"].isEnabled()
    h3 = Hooks()
    w3 = MainWindow(pr, hooks=h3.dict())
    w3.view_map()
    w3.export_map()
    assert len(h3.reports) == 2 and all("Select a map" in r for r in h3.reports)


def test_dialog_mask_checkbox_controls_masking(app):
    pr, u = ready()
    pr.attach_surface(u, Surface(-100, -100, 100.0, 100.0, 0.0, 1, np.full((10, 10), 100.0)), "t")
    pr.attach_surface("base", Surface(-100, -100, 100.0, 100.0, 0.0, 1, np.full((10, 10), 90.0)), "b")   # base above top: nowhere is inside
    d = MapComputeDialog(pr, uid=u)
    d.unit.setCurrentText("fraction"); fill_wells_grid(d)
    d.mask.setChecked(True); d.compute()
    assert d.ms.result.info["masked"] > 0 and np.all(np.isnan(d.ms.result.values))
    d.mask.setChecked(False); d.compute()
    assert d.ms.result.info["masked"] == 0 and np.isfinite(d.ms.result.values).all()

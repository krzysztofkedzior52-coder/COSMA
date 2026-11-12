import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qt = pytest.importorskip("PySide6.QtWidgets")

from cosma.gui.grid_dialog import CHOOSE, GridDialog
from cosma.gui.main_window import MainWindow
from cosma.project import Project
from test_grid_definition import flat_surface, model, vf
from test_main_window import Hooks


@pytest.fixture(scope="module")
def app():
    return qt.QApplication.instance() or qt.QApplication([])


def fill(d, **kw):
    vals = dict(origin_x="0", origin_y="0", azimuth_deg="0", nx="6", ny="4", dx="100", dy="100", min_thickness="0.1")
    vals.update(kw)
    for k, v in vals.items():
        d.fields[k].setText(v)


def choose_layering(d, rows=((("fixed number of layers", "2")), ("follow top (layers of fixed thickness)", "10"))):
    for mb, ve, (mode, v) in zip(d.mode_boxes, d.value_edits, rows):
        mb.setCurrentText(mode)
        ve.setText(v)


def test_menu_needs_a_chart_and_hook_is_called(app):
    calls = []
    h = Hooks().dict()
    h["grid_dialog"] = calls.append
    w = MainWindow(Project("p"), hooks=h)
    assert not w.act_grid.isEnabled()
    w.set_project(model())
    w.refresh()
    assert w.act_grid.isEnabled()
    w.grid_dialog()
    assert len(calls) == 1


def test_nothing_is_stored_until_the_form_is_valid(app):
    pr = model()
    d = GridDialog(pr)
    d.save()
    assert "enter origin x" in d.problem.text() and pr.grid_def is None
    fill(d, nx="2.5")
    d.save()
    assert "whole numbers" in d.problem.text() and pr.grid_def is None
    fill(d, dx="abc")
    d.save()
    assert "must be a number" in d.problem.text()
    fill(d, dx="-1")
    d.save()
    assert "positive cell size" in d.problem.text() and pr.grid_def is None


def test_layering_must_be_chosen_per_interval_and_values_checked(app):
    pr = model()
    d = GridDialog(pr)
    fill(d)
    assert d.save()                                    # grid saved, layering not chosen yet: build says what is missing
    d.show_missing()
    assert d.report.toPlainText().count("choose the layering") == 2
    d.mode_boxes[0].setCurrentText("fixed number of layers")
    d.value_edits[0].setText("2.5")
    d.save()
    assert "whole number" in d.problem.text()
    d.value_edits[0].setText("0")
    d.save()
    assert "n >= 1" in d.problem.text()
    d.mode_boxes[0].setCurrentText("follow top (layers of fixed thickness)")
    d.value_edits[0].setText("")
    d.save()
    assert "layer thickness" in d.problem.text()


def test_build_and_export_end_to_end(app, tmp_path):
    pr = model()
    out = tmp_path / "m.grdecl"
    d = GridDialog(pr, ask_export_path=lambda default: str(out))
    fill(d)
    choose_layering(d)
    d.build()
    assert not d.problem.text() and "6 x 4 x 7" in d.report.toPlainText() and d.export_btn.isEnabled()
    d.export()
    text = out.read_text()
    assert "COORD" in text and "ZCORN" in text and "ACTNUM" in text and "Exported" in d.report.toPlainText()
    assert pr.grid_def.nx == 6 and pr.grid_def.layerings                 # the definition is kept in the project


def test_build_refuses_with_reasons_and_does_not_allow_export(app):
    pr = model()
    pr.attach_faults([vf("F", (312.0, -100.0), (312.0, 500.0))])
    d = GridDialog(pr)
    fill(d)
    choose_layering(d)
    d.build()
    assert "trace depth" in d.problem.text() and not d.export_btn.isEnabled() and d.model is None
    d.export()
    assert "Build the grid first" in d.problem.text()


def test_unconformity_type_is_asked_only_where_there_is_an_unconformity(app):
    pr = model()
    pr.chart.intervals[1].contact_above = "unconformity"
    d = GridDialog(pr)
    assert not d.unc_boxes[0].isEnabled() and d.unc_boxes[1].isEnabled()
    fill(d)
    choose_layering(d)
    d.build()
    assert "erosional or onlap" in d.problem.text()
    d.unc_boxes[1].setCurrentText("erosional")
    d.build()
    assert not d.problem.text() and d.model is not None
    assert pr.grid_def.unconformity == {pr.chart.intervals[1].uid: "erosional"}


def test_fit_extent_and_orientation_suggestions(app):
    pr = model()
    pr.attach_faults([vf("N", (400.0, 0.0), (400.0, 800.0))])
    d = GridDialog(pr)
    fill(d, origin_x="", origin_y="", nx="", ny="")
    d.fit_extent()
    assert d.fields["nx"].text() == "14" and d.fields["ny"].text() == "14"          # surface spans -300..1100 m
    assert float(d.fields["origin_x"].text()) == pytest.approx(-300.0)
    fill(d, origin_x="", origin_y="", nx="", ny="", dx="60", dy="60")
    d.fit_extent()
    assert d.fields["nx"].text() == "24"                               # 1400 / 60 rounds UP so the surface is covered
    fill(d, origin_x="", origin_y="", nx="", ny="")
    d.fit_extent()
    d.suggest()
    labels = [d.sugg.item(r, 0).text() for r in range(d.sugg.rowCount())]
    assert labels[0].startswith("north-aligned") and any("fault N" in l for l in labels)
    assert d.sugg.item(0, 3).text() == "x 1.00"
    d.use_suggestion(1)
    assert float(d.fields["azimuth_deg"].text()) % 180 == pytest.approx(0.0, abs=0.1)
    fill(d, dx="0")
    d.fit_extent()
    assert "positive" in d.problem.text()
    d.suggest()
    assert "positive cell sizes" in d.problem.text()


def test_dialog_reopens_with_the_saved_definition(app):
    pr = model()
    d = GridDialog(pr)
    fill(d, azimuth_deg="12.5", min_thickness="0.25")
    choose_layering(d)
    assert d.save()
    e = GridDialog(pr)
    assert e.fields["azimuth_deg"].text() == "12.5" and e.fields["min_thickness"].text() == "0.25"
    assert e.mode_boxes[1].currentText().startswith("follow top") and e.value_edits[1].text() == "10"
    assert e.mode_boxes[0].currentText() == "fixed number of layers" and e.value_edits[0].text() == "2"


def test_grid_appears_in_the_tree_and_makes_the_project_dirty(app):
    pr = model()
    w = MainWindow(pr, hooks=Hooks().dict())
    w.refresh()
    before = w.is_dirty()
    d = GridDialog(pr)
    fill(d)
    choose_layering(d)
    d.save()
    w.refresh()
    tops = [w.tree.topLevelItem(i).text(0) for i in range(w.tree.topLevelItemCount())]
    assert "Grid" in tops and w.is_dirty()


# -- boundary section --------------------------------------------------------------------------------------------------------

def ready(pr=None):
    from shapely.geometry import box
    pr = pr or model()
    pr.set_boundary(box(100.0, 100.0, 500.0, 500.0))
    d = GridDialog(pr)
    fill(d, nx="8", ny="8", dx="75", dy="75")
    choose_layering(d)
    return pr, d


def test_default_is_closed_and_polygon_state_is_shown(app):
    pr, d = ready()
    assert d.save() and pr.grid_def.boundary.closed
    assert "Boundary polygon: 4 vertices" in d.poly_label.text()
    e = GridDialog(model())
    assert "No boundary polygon" in e.poly_label.text()


def test_buffer_pv_and_aquifer_inputs_are_checked(app):
    pr, d = ready()
    d.chk_buffer.setChecked(True)
    d.save()
    assert "enter the buffer distance" in d.problem.text()
    d.buffer.setText("-5")
    d.save()
    assert "positive" in d.problem.text()
    d.buffer.setText("75")
    d.chk_pv.setChecked(True)
    d.pv_factor.setText("20")
    d.pv_rings.setText("1.5")
    d.save()
    assert "whole number" in d.problem.text()
    d.pv_rings.setText("1")
    d.chk_aq.setChecked(True)
    d.save()
    assert "outer faces" in d.problem.text()
    d.aq_faces["I-"].setChecked(True)
    d.save()
    assert "enter datum depth" in d.problem.text()
    d.aq_fields["datum_depth"].setText("1500")
    d.aq_fields["compressibility"].setText("1e-4")
    d.aq_fields["influx_multiplier"].setText("1")
    d.save()
    assert "initial water volume" in d.problem.text()
    d.aq_fields["volume"].setText("1e9")
    d.aq_fields["productivity_index"].setText("50")
    assert d.save() and not d.problem.text()
    b = pr.grid_def.boundary
    assert b.buffer == 75.0 and b.pv.factor == 20.0 and b.pv.rings == 1
    assert b.aquifer.kind == "fetkovich" and b.aquifer.initial_pressure is None and b.aquifer.faces == ("I-",)


def test_aquifer_kind_switches_the_fields_that_apply(app):
    pr, d = ready()
    assert d.aq_fields["volume"].isEnabled() and not d.aq_fields["permeability"].isEnabled()
    d.aq_kind.setCurrentText("Carter-Tracy (radial, influence table)")
    assert not d.aq_fields["volume"].isEnabled() and d.aq_fields["permeability"].isEnabled()
    d.chk_aq.setChecked(True)
    d.aq_faces["J+"].setChecked(True)
    for k, v in dict(datum_depth="1500", compressibility="1e-4", influx_multiplier="1", permeability="200", porosity="0.2",
                     external_radius="5000", thickness="60", influx_angle="360", table="1").items():
        d.aq_fields[k].setText(v)
    assert d.save() and pr.grid_def.boundary.aquifer.kind == "carter_tracy" and pr.grid_def.boundary.aquifer.table == 1


def test_fraction_rule_needs_its_fraction_and_buffer_needs_a_polygon(app):
    pr, d = ready()
    d.rule.setCurrentText("at least this fraction of the column inside")
    d.min_fraction.setText("")
    d.save()
    assert "minimum fraction" in d.problem.text()
    d.min_fraction.setText("0.3")
    assert d.save() and pr.grid_def.boundary.rule == "fraction" and pr.grid_def.boundary.min_fraction == 0.3
    pr2 = model()
    d2 = GridDialog(pr2)
    fill(d2)
    choose_layering(d2)
    d2.chk_buffer.setChecked(True)
    d2.buffer.setText("100")
    d2.build()
    assert "boundary polygon" in d2.problem.text()


def test_build_all_four_and_export_lists_every_file(app, tmp_path):
    pr, d = ready()
    out = tmp_path / "m.grdecl"
    d.ask_export_path = lambda default: str(out)
    d.chk_buffer.setChecked(True)
    d.buffer.setText("75")
    d.chk_pv.setChecked(True)
    d.pv_factor.setText("20")
    d.pv_rings.setText("1")
    d.chk_aq.setChecked(True)
    for f in ("I-", "I+", "J-", "J+"):
        d.aq_faces[f].setChecked(True)
    for k, v in dict(datum_depth="1500", compressibility="1e-4", influx_multiplier="1", volume="1e9", productivity_index="50").items():
        d.aq_fields[k].setText(v)
    d.build()
    assert not d.problem.text() and "buffer 75 m" in d.report.toPlainText() and "double-count" in d.report.toPlainText()
    d.export()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.grdecl", "m_grid_boundary.inc", "m_runspec_boundary.inc",
                                                         "m_solution_boundary.inc"]
    assert "Include each" in d.report.toPlainText()


def test_reopen_restores_every_boundary_choice(app):
    pr, d = ready()
    d.rule.setCurrentText("at least this fraction of the column inside")
    d.min_fraction.setText("0.4")
    d.chk_buffer.setChecked(True)
    d.buffer.setText("75")
    d.chk_pv.setChecked(True)
    d.pv_factor.setText("20")
    d.pv_rings.setText("2")
    d.chk_aq.setChecked(True)
    d.aq_faces["J-"].setChecked(True)
    d.aq_fields["initial_pressure"].setText("160")
    for k, v in dict(datum_depth="1500", compressibility="1e-4", influx_multiplier="2", volume="1e9", productivity_index="50").items():
        d.aq_fields[k].setText(v)
    assert d.save()
    e = GridDialog(pr)
    assert e.rule.currentText().startswith("at least") and e.min_fraction.text() == "0.4"
    assert e.chk_buffer.isChecked() and e.buffer.text() == "75" and e.chk_pv.isChecked() and e.pv_rings.text() == "2"
    assert e.chk_aq.isChecked() and e.aq_faces["J-"].isChecked() and not e.aq_faces["I-"].isChecked()
    assert e.aq_fields["initial_pressure"].text() == "160" and e.aq_fields["influx_multiplier"].text() == "2"


def test_polygon_import_menu_sets_the_project_boundary(app, tmp_path):
    h = Hooks()
    p = tmp_path / "b.csv"
    p.write_text("x,y\n0,0\n100,0\n100,100\n0,100\n")
    d = h.dict()
    d.update(pick_file=lambda k, t: str(p), confirm=lambda prev: {})
    pr = model()
    w = MainWindow(pr, hooks=d)
    assert w.import_actions["polygon"].isEnabled()
    w.do_import("polygon")
    assert pr.boundary is not None and pr.boundary.area == pytest.approx(10000.0)
    w.do_import("polygon")
    assert "replaced the previous boundary" in w.statusBar().currentMessage()

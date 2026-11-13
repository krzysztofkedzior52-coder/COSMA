"""COSMA main window: what is in the project, and the actions on it.

Every dialog is reached through a replaceable hook, so the window can be driven (and tested)
without any modal dialog opening.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QFileDialog, QMainWindow, QMenu, QMessageBox, QTreeWidget, QTreeWidgetItem)

from ..imports import ImportCancelled, ImportRejected, import_file, import_las_batch
from ..persist import EXTENSION, PersistError, fingerprint, load_project, save_project
from ..project import Project, ProjectError
from ..maps import MapError, export_map
from ..variogram import VariogramError, export_variogram
from .variogram_setup import make_session

IMPORTS = (("surface", "Surface (depth grid)...", "Surfaces (*.irap* *.gri *.txt *.dat *.bin *.map);;All files (*)"),
           ("well_headers", "Well headers...", "Well headers (*.csv *.txt);;All files (*)"),
           ("deviation", "Deviation survey...", "Deviation surveys (*.csv *.txt);;All files (*)"),
           ("tops", "Formation tops...", "Formation tops (*.csv *.txt);;All files (*)"),
           ("las", "Well logs (LAS)...", "LAS files (*.las);;All files (*)"),
           ("core", "Core data...", "Core data (*.csv *.txt);;All files (*)"),
           ("property_map", "Property map (surface file)...", "Surfaces (*.irap* *.gri *.txt *.dat *.bin *.map);;All files (*)"),
           ("variogram", "Variogram (GSLIB / JSON)...", "Variograms (*.var *.vgm *.txt *.json);;All files (*)"),
           ("fault_sticks", "Fault sticks...", "Fault sticks (*.csv *.txt *.dat *.xyz *.tsv);;All files (*)"),
           ("polygon", "Model boundary (polygon)...", "Polygons (*.csv *.txt *.xyz *.dat *.pol *.geojson *.json *.shp *.kml *.kmz);;All files (*)"))


class MainWindow(QMainWindow):
    def __init__(self, project: Project | None = None, *, hooks: dict | None = None, path=None):
        super().__init__()
        h = hooks or {}
        self.pick_file = h.get("pick_file") or self._pick_file
        self.pick_files = h.get("pick_files") or self._pick_files
        self.confirm = h.get("confirm") or self._confirm
        self.edit_chart_dialog = h.get("edit_chart") or self._edit_chart
        self.start_chart_dialog = h.get("start_chart") or self._start_chart
        self.ask_setup = h.get("ask_setup") or self._ask_setup
        self.run_editor = h.get("run_editor") or self._run_editor
        self.run_map_dialog = h.get("run_map_dialog") or self._run_map_dialog
        self.show_map = h.get("show_map") or self._show_map
        self.show_fault = h.get("show_fault") or self._show_fault
        self.show_wellmap = h.get("well_map") or self._well_map
        self.run_fault_interactions = h.get("fault_interactions") or self._fault_interactions
        self.run_fault_tm = h.get("fault_transmissibility") or self._fault_tm
        self.run_facies = h.get("facies") or self._facies
        self.run_datacheck = h.get("datacheck") or self._datacheck
        self.run_proportions = h.get("proportions") or self._proportions
        self.run_proportion_map = h.get("proportion_map") or self._proportion_map
        self.run_facies_sim = h.get("facies_sim") or self._facies_sim
        self.run_realizations = h.get("realizations") or self._realizations
        self.run_grid_dialog = h.get("grid_dialog") or self._grid_dialog
        self.confirm_remove = h.get("confirm_remove") or self._confirm_remove
        self.report = h.get("report") or self._report
        self.new_project_flow = h.get("new_project") or self._new_project
        self.save_path = h.get("save_path") or self._save_path
        self.project_open_path = h.get("project_open_path") or self._project_open_path
        self.project_save_path = h.get("project_save_path") or self._project_save_path
        self.ask_unsaved = h.get("ask_unsaved") or self._ask_unsaved
        self.project = None
        self.path = None
        self._saved_fp = None
        self.set_project(project, path, saved=path is not None)
        self.setWindowTitle("COSMA")
        self.resize(900, 600)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Project", "Details"])
        self.tree.itemDoubleClicked.connect(lambda *_: self.view_map() if self._selected("map") else
                                            self.view_fault() if self._selected("fault") else
                                            self.well_map() if self._selected("wells") else self.edit_variogram())
        self.tree.currentItemChanged.connect(lambda *_: self.menu_state())
        self.setCentralWidget(self.tree)
        self._menus()
        self.refresh()

    # -- menus ------------------------------------------------------------------------------------------------
    def _action(self, menu, text, fn):
        a = QAction(text, self)
        a.triggered.connect(lambda *_: fn())
        menu.addAction(a)
        return a

    def _menus(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        self.act_new = self._action(f, "New project...", self.new_project)
        self.act_open = self._action(f, "Open project...", self.open_project)
        self.act_save = self._action(f, "Save", self.save)
        self.act_save.setShortcut("Ctrl+S")
        self.act_save_as = self._action(f, "Save as...", self.save_as)
        f.addSeparator()
        self._action(f, "Quit", self.close)
        vw = mb.addMenu("&View")
        self.act_wellmap = self._action(vw, "Well map...", self.well_map)
        self.import_menu = mb.addMenu("&Import")
        self.import_actions = {}
        for kind, text, _ in IMPORTS:
            self.import_actions[kind] = self._action(self.import_menu, text, lambda k=kind: self.do_import(k))
        s = mb.addMenu("&Stratigraphy")
        self.act_chart = self._action(s, "Define chart...", self.edit_chart)
        v = mb.addMenu("&Variograms")
        self.act_new_vario = self._action(v, "New variogram (compute and edit)...", self.new_variogram)
        self.act_edit_vario = self._action(v, "Edit selected variogram...", self.edit_variogram)
        self.act_export_vario = self._action(v, "Export selected variogram...", self.export_variogram)
        m = mb.addMenu("&Maps")
        self.act_new_map = self._action(m, "Compute property / thickness map...", self.new_map)
        self.act_view_map = self._action(m, "View selected map...", self.view_map)
        self.act_export_map = self._action(m, "Export selected map (IRAP)...", self.export_map)
        gm = mb.addMenu("&Grid")
        self.act_grid = self._action(gm, "Define, build and export grid...", self.grid_dialog)
        cm = mb.addMenu("F&acies")
        self.act_facies = self._action(cm, "Facies from logs...", self.facies_dialog)
        self.act_datacheck = self._action(cm, "Data checks...", self.datacheck_dialog)
        self.act_proportions = self._action(cm, "Vertical proportion curves...", self.proportions_dialog)
        self.act_pmap = self._action(cm, "Import proportion map...", self.proportion_map_dialog)
        self.act_fsim = self._action(cm, "Simulate facies...", self.facies_sim_dialog)
        self.act_real = self._action(cm, "Properties, permeability and realizations...", self.realizations_dialog)
        fm = mb.addMenu("&Faults")
        self.act_view_fault = self._action(fm, "View selected fault...", self.view_fault)
        self.act_export_faults = self._action(fm, "Export all fault sticks (CSV)...", self.export_faults)
        self.act_fault_ints = self._action(fm, "Fault ages and interactions...", self.fault_interactions)
        self.act_fault_tm = self._action(fm, "Fault transmissibility...", self.fault_transmissibility)
        self.act_remove_fault = self._action(fm, "Remove selected fault", self.remove_fault)
        for menu in mb.findChildren(QMenu):
            menu.setToolTipsVisible(True)
        self.menu_state()

    def menu_state(self):
        """Enable each action only when what it needs exists, and say what is missing in its tooltip."""
        pr = self.project
        has = pr is not None

        def gate(action, *needs):
            """needs: (condition, what is missing) pairs."""
            missing = [msg for ok, msg in needs if not ok]
            action.setEnabled(not missing)
            tip = "" if not missing else "Not available yet: " + "; ".join(missing) + "."
            action.setToolTip(tip)
            action.setStatusTip(tip)

        P = (has, "no project is open (File > New project...)")
        chart = has and pr.chart is not None
        wells = has and bool(pr.wells)
        C = (chart, "the stratigraphic chart is not defined (Stratigraphy > Define chart...)")
        W = (wells, "no wells yet (Import > Well headers...)")
        faces = has and pr.facies_setup is not None
        F = (faces, "facies are not set up (Facies > Facies from logs...)")
        G = (has and pr.grid_def is not None, "the grid is not defined (Grid > Define, build and export grid...)")
        gate(self.act_save, P)
        gate(self.act_save_as, P)
        for k, a in self.import_actions.items():
            gate(a, P, *([C] if k in ("surface", "variogram", "property_map") else []))
        self.act_chart.setText("Edit chart..." if chart else "Define chart...")
        gate(self.act_chart, P)
        gate(self.act_grid, P, C)
        gate(self.act_new_vario, P, C, W)
        sel = self._selected_variogram() is not None
        S_v = (sel, "select a variogram in the project tree")
        gate(self.act_edit_vario, P, C, S_v)
        gate(self.act_export_vario, P, C, S_v)
        gate(self.act_new_map, P, C, W)
        S_m = (self._selected("map") is not None, "select a map in the project tree")
        gate(self.act_view_map, P, C, S_m)
        gate(self.act_export_map, P, C, S_m)
        S_f = (self._selected("fault") is not None, "select a fault in the project tree")
        gate(self.act_view_fault, P, S_f)
        gate(self.act_remove_fault, P, S_f)
        gate(self.act_export_faults, P, (has and bool(pr.faults), "no faults imported (Import > Fault sticks...)"))
        gate(self.act_fault_ints, P, (has and len(pr.faults) > 1, "interactions exist only between two or more faults"))
        gate(self.act_fault_tm, P, (has and bool(pr.faults), "no faults imported (Import > Fault sticks...)"))
        gate(self.act_wellmap, P, W)
        gate(self.act_facies, P, W)
        gate(self.act_datacheck, P, W, F)
        PR = (has and pr.proportions is not None, "vertical proportion curves are not built (Facies > Vertical proportion curves...)")
        gate(self.act_proportions, P, W, F, C)
        gate(self.act_fsim, P, W, F, PR, G)
        gate(self.act_real, P, (has and pr.facies_sim is not None, "facies simulation is not set (Facies > Simulate facies...)"), C, F, G)
        gate(self.act_pmap, P, F, C, G)

    # -- default dialogs (replaced in tests) ----------------------------------------------------------------------
    def _pick_file(self, kind, title):
        filt = next((f for k, _, f in IMPORTS if k == kind), "All files (*)")
        path, _ = QFileDialog.getOpenFileName(self, title, "", filt)
        return path or None

    def _pick_files(self, kind, title):
        filt = next((f for k, _, f in IMPORTS if k == kind), "All files (*)")
        paths, _ = QFileDialog.getOpenFileNames(self, title + " (select one or more files)", "", filt)
        return list(paths)

    def _confirm(self, preview):
        from .import_dialog import confirm_with_dialog
        return confirm_with_dialog(preview, self)

    def _edit_chart(self, project):
        from .chart_editor import edit_chart_with_dialog
        return edit_chart_with_dialog(project, self)

    def _start_chart(self, project):
        from .chart_editor import start_chart_with_dialog
        return start_chart_with_dialog(project, self)

    def _ask_setup(self, project, preset):
        from .variogram_setup import ask_variogram_setup
        return ask_variogram_setup(project, self, preset)

    def _run_editor(self, session, label):
        from .variogram_editor import VariogramEditor
        dlg = VariogramEditor(session, self, label=label)
        dlg.exec()
        return dlg.saved

    def _grid_dialog(self, project):
        from .grid_dialog import GridDialog
        GridDialog(project, self).exec()

    def _fault_interactions(self, project):
        from .fault_interaction_dialog import FaultInteractionDialog
        FaultInteractionDialog(project, self).exec()

    def _facies(self, project):
        from .facies_dialog import FaciesDialog
        FaciesDialog(project, self).exec()

    def facies_dialog(self):
        if self.project is not None and self.project.wells:
            self.run_facies(self.project)
            self.refresh()

    def _facies_sim(self, project):
        from .facies_sim_dialog import FaciesSimDialog
        FaciesSimDialog(project, self).exec()

    def facies_sim_dialog(self):
        pr = self.project
        if pr is not None and pr.wells and pr.facies_setup is not None and pr.proportions is not None and pr.grid_def is not None:
            self.run_facies_sim(pr)
            self.refresh()

    def _realizations(self, project):
        from .realizations_dialog import RealizationsDialog
        RealizationsDialog(project, self).exec()

    def realizations_dialog(self):
        pr = self.project
        if pr is not None and pr.facies_sim is not None and pr.chart is not None and pr.facies_setup is not None and pr.grid_def is not None:
            self.run_realizations(pr)
            self.refresh()

    def _proportion_map(self, project):
        from .proportion_map_dialog import ProportionMapDialog
        ProportionMapDialog(project, self).exec()

    def proportion_map_dialog(self):
        pr = self.project
        if pr is not None and pr.facies_setup is not None and pr.chart is not None and pr.grid_def is not None:
            self.run_proportion_map(pr)
            self.refresh()

    def _proportions(self, project):
        from .proportions_dialog import ProportionsDialog
        ProportionsDialog(project, self).exec()

    def proportions_dialog(self):
        pr = self.project
        if pr is not None and pr.wells and pr.facies_setup is not None and pr.chart is not None:
            self.run_proportions(pr)
            self.refresh()

    def _datacheck(self, project):
        from .datacheck_dialog import DataCheckDialog
        DataCheckDialog(project, self).exec()

    def datacheck_dialog(self):
        if self.project is not None and self.project.wells and self.project.facies_setup is not None:
            self.run_datacheck(self.project)
            self.refresh()

    def _fault_tm(self, project):
        from .fault_transmissibility_dialog import FaultTransmissibilityDialog
        FaultTransmissibilityDialog(project, self).exec()

    def fault_transmissibility(self):
        if self.project is not None and self.project.faults:
            self.run_fault_tm(self.project)
            self.refresh()

    def _well_map(self, project):
        from .well_map_dialog import WellMapDialog
        WellMapDialog(project, self).exec()

    def well_map(self):
        if self.project is not None and self.project.wells:
            self.show_wellmap(self.project)

    def _show_fault(self, fault):
        from .fault_dialog import FaultViewDialog
        FaultViewDialog(fault, self).exec()

    def _confirm_remove(self, text):
        return QMessageBox.question(self, "COSMA", text) == QMessageBox.Yes

    def _run_map_dialog(self, project, uid):
        from .map_dialog import MapComputeDialog
        dlg = MapComputeDialog(project, self, uid)
        dlg.exec()
        return dlg.saved

    def _show_map(self, pm):
        from .map_dialog import MapViewDialog
        MapViewDialog(pm, self).exec()

    def _report(self, text, error=True):
        (QMessageBox.warning if error else QMessageBox.information)(self, "COSMA", text)

    def _save_path(self, title, default):
        filt = {".csv": "CSV (*.csv)", ".irap": "IRAP ASCII (*.irap)"}.get(Path(default).suffix.lower(), "GSLIB (*.var);;JSON (*.json)")
        path, _ = QFileDialog.getSaveFileName(self, title, default, filt)
        return path or None

    def _project_open_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", f"COSMA projects (*{EXTENSION})")
        return path or None

    def _project_save_path(self, default):
        path, _ = QFileDialog.getSaveFileName(self, "Save project", default, f"COSMA projects (*{EXTENSION})")
        return path or None

    def _ask_unsaved(self, text):
        box = QMessageBox(QMessageBox.Question, "COSMA", text, QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, self)
        r = box.exec()
        return {QMessageBox.Save: "save", QMessageBox.Discard: "discard"}.get(r, "cancel")

    def _new_project(self):
        from .import_dialog import confirm_with_dialog
        from .new_project import ask_new_project, new_project_workflow, qt_pick_file
        from PySide6.QtWidgets import QInputDialog
        return new_project_workflow(qt_pick_file(self), lambda: ask_new_project(self), self.confirm,
                                    lambda names: QInputDialog.getItem(self, "Type well", "Which well is the type well?", names, 0, False)[0],
                                    self.edit_chart_dialog)

    # -- project file -------------------------------------------------------------------------------------------------------------
    def set_project(self, pr, path=None, saved: bool = False):
        self.project, self.path = pr, (Path(path) if path else None)
        # a freshly made empty project has nothing to lose; anything else is unsaved until written
        self._saved_fp = fingerprint(pr) if pr is not None and (saved or (not pr.wells and pr.chart is None)) else None
        if hasattr(self, "tree"):
            self.refresh()

    def is_dirty(self) -> bool:
        return self.project is not None and fingerprint(self.project) != self._saved_fp

    def _can_discard(self) -> bool:
        """True if it is fine to replace/close the current project (saved, discarded or nothing to lose)."""
        if not self.is_dirty():
            return True
        what = self.path.name if self.path else (self.project.name or "the new project")
        a = self.ask_unsaved(f"{what} has unsaved changes.")
        if a == "discard":
            return True
        return a == "save" and self.save()

    def save(self) -> bool:
        if self.project is None:
            return False
        return self._write(self.path) if self.path else self.save_as()

    def save_as(self) -> bool:
        if self.project is None:
            return False
        default = str(self.path) if self.path else f"{self.project.name or 'project'}{EXTENSION}"
        path = self.project_save_path(default)
        return bool(path) and self._write(Path(path))

    def _write(self, path) -> bool:
        try:
            written = save_project(self.project, path)
        except (OSError, PersistError, ValueError) as exc:
            self.report(f"Could not save: {exc}")
            return False
        self.path, self._saved_fp = written, fingerprint(self.project)
        self.statusBar().showMessage(f"Saved {written.name}", 8000)
        self.refresh()
        return True

    def open_project(self, path=None):
        if not self._can_discard():
            return
        path = path or self.project_open_path()
        if not path:
            return
        try:
            pr = load_project(path)
        except (PersistError, OSError) as exc:
            self.report(f"{Path(path).name}: {exc}")
            return
        self.set_project(pr, path, saved=True)

    def closeEvent(self, ev):
        if self._can_discard():
            ev.accept()
        else:
            ev.ignore()

    # -- actions ----------------------------------------------------------------------------------------------------------------
    def new_project(self):
        if not self._can_discard():
            return
        try:
            pr = self.new_project_flow()
        except (ProjectError, ImportRejected, ValueError) as exc:
            self.report(str(exc))
            return
        if pr is not None:
            self.set_project(pr)

    def do_import(self, kind: str):
        pr = self.project
        title = next(t for k, t, _ in IMPORTS if k == kind).rstrip(".")
        if kind == "las":
            paths = self.pick_files(kind, title)
            if not paths:
                return
            path = paths[0] if len(paths) == 1 else f"{len(paths)} LAS files"
        else:
            path = self.pick_file(kind, title)
            paths = None
            if not path:
                return
        try:
            res = (import_las_batch(paths, self.confirm, project=pr) if paths
                   else import_file(kind, path, self.confirm, project=pr))
        except ImportCancelled:
            return
        except (ImportRejected, ProjectError, ValueError, OSError) as exc:
            self.report(f"{Path(path).name}: {exc}")
            return
        msg = "; ".join(getattr(res, "applied", []) + getattr(res, "notes", []))
        self.statusBar().showMessage(msg or f"{Path(path).name} imported", 15000)
        if paths and len(paths) > 1:
            n_asked = sum(1 for _, _, how in res.files if how == "asked")
            self.report(f"{len(paths)} LAS files imported ({n_asked} confirmed one by one, "
                        f"{len(paths) - n_asked} with the same choices):\n" + "\n".join(res.applied), error=False)
        self.refresh()
        if kind == "fault_sticks":
            self.settle_fault_interactions(force=True)
            self.settle_fault_transmissibility()
        if kind in ("fault_sticks", "las"):
            self.notify_vshale()

    def notify_vshale(self):
        """Tell the user, once per session, that faults are present but no well has VSHALE (needed for SGR)."""
        from ..fault_sgr import vshale_notice
        msg = vshale_notice(self.project) if self.project is not None else None
        if msg and not getattr(self, "_vshale_told", False):
            self._vshale_told = True
            self.report(msg, error=False)

    def settle_fault_transmissibility(self):
        """Open the fault transmissibility tool when any fault has no explicit choice yet (after a fault import)."""
        pr = self.project
        if pr is None or all(n in pr.fault_transmissibility for n in pr.faults):
            return
        self.run_fault_tm(pr)
        self.refresh()

    def grid_dialog(self):
        if self.project is None or self.project.chart is None:
            return
        self.run_grid_dialog(self.project)
        self.refresh()

    def fault_interactions(self):
        self.settle_fault_interactions(force=False)

    def settle_fault_interactions(self, force: bool):
        """Open the fault age/interaction tool. Forced after a fault import while two or more faults are
        not settled (no depth, interactions unchecked, or an interaction without an answer)."""
        pr = self.project
        if pr is None or len(pr.faults) < 2:
            return
        if force:
            from ..fault_interactions import unconfirmed
            settled = (all(n in pr.fault_depths for n in pr.faults) and pr.fault_reach is not None
                       and not unconfirmed(pr.fault_interactions))
            if settled:
                return
        self.run_fault_interactions(pr)
        self.refresh()

    def edit_chart(self):
        """Edit the chart; with none yet, first make one (from a well's tops or typed in), then show it in the editor."""
        pr = self.project
        if pr is None:
            return
        if pr.chart is None:
            from ..chart_edit import start_chart
            how = self.start_chart_dialog(pr)
            if not how:
                return
            try:
                start_chart(pr, how)
            except ProjectError as exc:
                self.report(str(exc))
                return
            self.refresh()
        self.edit_chart_dialog(pr)
        self.refresh()

    def _selected(self, what):
        it = self.tree.currentItem()
        d = it.data(0, Qt.UserRole) if it is not None else None
        return d if d and d[0] == what else None

    def _selected_variogram(self):
        return self._selected("variogram")

    def view_fault(self):
        sel = self._selected("fault")
        if sel is None:
            self.report("Select a fault in the project tree first.", error=False)
            return
        self.show_fault(self.project.faults[sel[1]])

    def remove_fault(self):
        sel = self._selected("fault")
        if sel is None:
            self.report("Select a fault in the project tree first.", error=False)
            return
        if not self.confirm_remove(f"Remove fault {sel[1]!r} from the project?"):
            return
        self.project.remove_fault(sel[1])
        self.refresh()

    def export_faults(self):
        from ..faults import export_sticks
        pr = self.project
        if pr is None or not pr.faults:
            self.report("The project has no faults.", error=False)
            return
        path = self.save_path("Export fault sticks", "fault_sticks.csv")
        if not path:
            return
        try:
            n = export_sticks(list(pr.faults.values()), path)
        except OSError as exc:
            self.report(str(exc))
            return
        self.statusBar().showMessage(f"{n} stick points exported (x, y, depth in metres, depth positive down)", 10000)

    def new_map(self):
        pr = self.project
        if pr is None or pr.chart is None or not pr.wells:
            self.report("Load wells and define the stratigraphic chart first.")
            return
        d = self._selected("map") or self._selected("variogram")
        uid = d[1] if d else None
        res = self.run_map_dialog(pr, uid)
        if res is not None:
            self.statusBar().showMessage("; ".join(res.applied), 15000)
        self.refresh()

    def view_map(self):
        sel = self._selected("map")
        if sel is None:
            self.report("Select a map in the project tree first.", error=False)
            return
        _, uid, var, lab = sel
        self.show_map(self.project.maps[uid][var][lab])

    def export_map(self):
        sel = self._selected("map")
        if sel is None:
            self.report("Select a map in the project tree first.", error=False)
            return
        _, uid, var, lab = sel
        path = self.save_path("Export map", f"{lab}.irap")
        if not path:
            return
        try:
            export_map(self.project.maps[uid][var][lab], path)
        except (MapError, OSError, ValueError) as exc:
            self.report(str(exc))
            return
        self.statusBar().showMessage(f"{lab} exported", 10000)

    def new_variogram(self):
        self._open_editor(None)

    def edit_variogram(self):
        sel = self._selected_variogram()
        if sel is None:
            self.report("Select a variogram in the project tree first.", error=False)
            return
        self._open_editor(sel)

    def _open_editor(self, sel):
        pr = self.project
        if pr is None or pr.chart is None or not pr.wells:
            self.report("Load wells and define the stratigraphic chart first.")
            return
        preset, model, label = None, None, ""
        if sel is not None:
            _, uid, variable, label = sel
            model = pr.variograms[uid][variable][label]
            meta = pr.var_meta.get((uid, variable, label))
            preset = dict(meta or {}, uid=uid)
            if not meta:
                preset["curve"] = variable.split("[")[0]
        values = self.ask_setup(pr, preset)
        if values is None:
            return
        try:
            session = make_session(pr, values, wells=(preset or {}).get("wells"))
            if model is not None:
                session.load(model)
        except (ProjectError, VariogramError) as exc:
            self.report(str(exc))
            return
        self.run_editor(session, label)
        self.refresh()

    def export_variogram(self):
        sel = self._selected_variogram()
        if sel is None:
            self.report("Select a variogram in the project tree first.", error=False)
            return
        _, uid, variable, label = sel
        path = self.save_path("Export variogram", f"{label}.var")
        if not path:
            return
        try:
            fmt = export_variogram(self.project.variograms[uid][variable][label], path)
        except (VariogramError, OSError) as exc:
            self.report(str(exc))
            return
        self.statusBar().showMessage(f"{label} exported ({fmt})", 10000)

    # -- tree -------------------------------------------------------------------------------------------------------------------------
    def refresh(self):
        self.tree.clear()
        pr = self.project
        if pr is None:
            self.tree.addTopLevelItem(QTreeWidgetItem(["No project", "File > New project..."]))
            self.menu_state()
            return
        self.setWindowTitle("COSMA - " + (pr.name or "untitled") + (f" [{self.path.name}]" if self.path else "")
                            + (" *" if self.is_dirty() else ""))
        wells = QTreeWidgetItem(["Wells", str(len(pr.wells))])
        for n, r in pr.wells.items():
            w = QTreeWidgetItem([n + (" (type well)" if n == pr.type_well else ""),
                                 f"{len(r.tops)} tops, {sum(len(l.curves) for l in r.logs.values())} curves, "
                                 f"{len(r.core)} core sets" + (f"; missing {', '.join(r.missing())}" if r.missing() else "")])
            wells.addChild(w)
        wells.setData(0, Qt.UserRole, ("wells",))
        self.tree.addTopLevelItem(wells)
        if pr.faults:
            fa = QTreeWidgetItem(["Faults", str(len(pr.faults))])
            for n, f in pr.faults.items():
                c = QTreeWidgetItem([n, f.label()])
                c.setData(0, Qt.UserRole, ("fault", n))
                fa.addChild(c)
            self.tree.addTopLevelItem(fa)
        if pr.grid_def is not None:
            g = pr.grid_def
            self.tree.addTopLevelItem(QTreeWidgetItem(
                ["Grid", f"{g.nx} x {g.ny} columns of {g.dx:g} x {g.dy:g} m, azimuth {g.azimuth_deg:g} deg"]))
        if pr.chart is None:
            self.tree.addTopLevelItem(QTreeWidgetItem(["Stratigraphy", "not defined - Stratigraphy > Define chart... "
                                                       "(surfaces, maps and variograms attach to it)"]))
        if pr.chart is not None:
            strat = QTreeWidgetItem(["Stratigraphy", f"{len(pr.chart.intervals)} intervals"])
            for iv in pr.chart.intervals:
                it = QTreeWidgetItem([iv.name, ""])
                for lab in pr.surfaces.get(iv.uid, {}):
                    it.addChild(QTreeWidgetItem([f"surface: {lab}", "top" + (" (used)" if pr.primary.get(iv.uid) == lab else "")]))
                for var, bucket in pr.variograms.get(iv.uid, {}).items():
                    for lab, m in bucket.items():
                        c = QTreeWidgetItem([f"variogram: {var}", f"{lab} ({m.structures[0].type}, "
                                             f"{m.range_of('major'):g}/{m.range_of('minor'):g}/{m.range_of('vertical'):g} m)"
                                             + (" (used)" if pr.var_primary.get((iv.uid, var)) == lab else "")])
                        c.setData(0, Qt.UserRole, ("variogram", iv.uid, var, lab))
                        it.addChild(c)
                for var, bucket in pr.maps.get(iv.uid, {}).items():
                    for lab, m in bucket.items():
                        st = m.stats()
                        c = QTreeWidgetItem([f"map: {var}", f"{lab} ({m.method}, {m.unit}"
                                             + (f", {st['min']:.4g} to {st['max']:.4g}" if st["n"] else ", empty") + ")"
                                             + (" (used)" if pr.map_primary.get((iv.uid, var)) == lab else "")])
                        c.setData(0, Qt.UserRole, ("map", iv.uid, var, lab))
                        it.addChild(c)
                strat.addChild(it)
            base = QTreeWidgetItem([f"Base: {pr.chart.base_name}", ""])
            for lab in pr.surfaces.get("base", {}):
                base.addChild(QTreeWidgetItem([f"surface: {lab}", ""]))
            strat.addChild(base)
            self.tree.addTopLevelItem(strat)
        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)
        self.menu_state()


def run(project: Project | None = None, path=None):
    """Start the application, optionally opening a project file (also taken from the command line)."""
    import multiprocessing
    import sys
    multiprocessing.freeze_support()           # needed in a frozen Windows build: realization workers are separate processes
    from .icon import set_app_identity
    app = QApplication.instance() or QApplication([])
    set_app_identity(app)
    path = path or (sys.argv[1] if len(sys.argv) > 1 else None)
    w = MainWindow(project)
    if path:
        w.open_project(path)
    w.show()
    code = app.exec()
    del w
    from .shutdown import release_widgets
    release_widgets(app)                       # otherwise the process can crash on quit (see shutdown.py)
    return code

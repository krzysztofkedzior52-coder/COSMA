"""Define, build and export the grid. Every choice is explicit; the dialog proposes, the user decides."""

from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..grid.boundary_options import (FACES, AquiferSpec, BoundaryOptionError, BoundaryOptions, PVMultiplier)
from ..grid.builder import BuildError, Layering
from ..grid_definition import (GridDefinition, GridDefinitionError, build_model, orientation_suggestions,
                               required_items, suggest_extent, write_model)
from ..project import ProjectError

CHOOSE = "- choose -"
MODES = {"fixed number of layers": "fixed_count", "fixed thickness": "fixed_thickness",
         "follow top (layers of fixed thickness)": "follow_top", "follow base (layers of fixed thickness)": "follow_base"}
MODE_TEXT = {v: k for k, v in MODES.items()}
RULE_TEXT = {"center": "column centre inside the polygon", "fraction": "at least this fraction of the column inside"}
AQ_KINDS = {"Fetkovich (productivity index)": "fetkovich", "Carter-Tracy (radial, influence table)": "carter_tracy"}
AQ_FIELDS = (("datum_depth", "Datum depth (m)"), ("initial_pressure", "Initial pressure (bar; blank = equilibrate with the reservoir)"),
             ("compressibility", "Total compressibility (1/bar)"), ("influx_multiplier", "Influx multiplier"),
             ("volume", "Initial water volume (m3)"), ("productivity_index", "Productivity index (sm3/day/bar)"),
             ("permeability", "Permeability (mD)"), ("porosity", "Porosity (fraction)"),
             ("external_radius", "External radius (m)"), ("thickness", "Thickness (m)"),
             ("influx_angle", "Influx angle (deg)"), ("table", "Influence table number (1 = infinite aquifer)"))
FETKOVICH_ONLY = ("volume", "productivity_index")
CT_ONLY = ("permeability", "porosity", "external_radius", "thickness", "influx_angle", "table")


class GridDialog(QDialog):
    def __init__(self, project, parent: QWidget | None = None, ask_export_path=None):
        super().__init__(parent)
        self.project = project
        self.model = None
        self.ask_export_path = ask_export_path or self._ask_export_path
        self.setWindowTitle("Grid")
        self.setMinimumSize(980, 900)
        lay = QVBoxLayout(self)
        gd = project.grid_def
        grid = QGridLayout()
        lay.addLayout(grid)
        self.fields: dict[str, QLineEdit] = {}
        spec = [("origin_x", "Origin X (m, map)"), ("origin_y", "Origin Y (m, map)"),
                ("azimuth_deg", "Azimuth of J axis (deg clockwise from north)"), ("nx", "Cells in I"),
                ("ny", "Cells in J"), ("dx", "Cell size in I (m)"), ("dy", "Cell size in J (m)"),
                ("min_thickness", "Minimum cell thickness (m)")]
        for k, (key, text) in enumerate(spec):
            grid.addWidget(QLabel(text), k // 2, (k % 2) * 2)
            e = QLineEdit("" if gd is None else f"{getattr(gd, key):g}")
            if key == "min_thickness" and gd is None:
                e.setText("0.1")
            self.fields[key] = e
            grid.addWidget(e, k // 2, (k % 2) * 2 + 1)
        row = QHBoxLayout()
        b = QPushButton("Fit origin and cell counts to the top surface")
        b.clicked.connect(self.fit_extent)
        row.addWidget(b)
        b = QPushButton("Suggest orientations")
        b.clicked.connect(self.suggest)
        row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.sugg = QTableWidget(0, 4)
        self.sugg.setHorizontalHeaderLabels(["Orientation", "Azimuth", "Cells to cover the top", "Cost vs north-aligned"])
        self.sugg.setColumnWidth(0, 420)
        self.sugg.setMaximumHeight(150)
        self.sugg.cellDoubleClicked.connect(self.use_suggestion)
        lay.addWidget(self.sugg)
        lay.addWidget(QLabel("Double-click a suggestion to use its azimuth. Cost is the number of cells needed "
                             "to cover the top surface compared with a north-aligned grid of the same cell size."))
        lay.addWidget(self._boundary_group(project, gd))
        self.ints = QTableWidget(0, 5)
        self.ints.setHorizontalHeaderLabels(["Interval", "Contact at its top", "Layering", "Value (layers or m)",
                                             "Unconformity type"])
        self.ints.setColumnWidth(0, 160)
        self.ints.setColumnWidth(1, 130)
        self.ints.setColumnWidth(2, 270)
        lay.addWidget(self.ints, 1)
        self.mode_boxes, self.value_edits, self.unc_boxes = [], [], []
        self._fill_intervals()
        row = QHBoxLayout()
        for text, fn in (("What is still missing?", self.show_missing), ("Save definition", self.save),
                         ("Build grid", self.build), ("Export GRDECL...", self.export)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
            if text.startswith("Export"):
                self.export_btn = b
                b.setEnabled(False)
        row.addStretch(1)
        c = QPushButton("Close")
        c.clicked.connect(self.accept)
        row.addWidget(c)
        lay.addLayout(row)
        self.problem = QLabel("")
        self.problem.setStyleSheet("color: #b00020")
        self.problem.setWordWrap(True)
        lay.addWidget(self.problem)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(170)
        lay.addWidget(self.report)

    # -- boundary ----------------------------------------------------------------------------------
    def _boundary_group(self, project, gd):
        ob = BoundaryOptions() if gd is None else gd.boundary
        box = QGroupBox("Model boundary: tick any treatments (none ticked = closed, no-flow edge)")
        g = QGridLayout(box)
        poly = project.boundary
        self.poly_label = QLabel("No boundary polygon imported (Import > Model boundary): all columns with defined horizons are active."
                                 if poly is None else
                                 f"Boundary polygon: {len(poly.exterior.coords) - 1} vertices, {poly.area:,.0f} m2.")
        g.addWidget(self.poly_label, 0, 0, 1, 4)
        g.addWidget(QLabel("A column is inside when its"), 1, 0)
        self.rule = QComboBox()
        self.rule.addItems(list(RULE_TEXT.values()))
        self.rule.setCurrentText(RULE_TEXT[ob.rule])
        g.addWidget(self.rule, 1, 1)
        self.min_fraction = QLineEdit(f"{ob.min_fraction:g}")
        g.addWidget(self.min_fraction, 1, 2)
        self.chk_buffer = QCheckBox("Extended buffer zone: distance beyond the polygon (m)")
        self.chk_buffer.setChecked(ob.buffer is not None)
        self.buffer = QLineEdit("" if ob.buffer is None else f"{ob.buffer:g}")
        g.addWidget(self.chk_buffer, 2, 0, 1, 2)
        g.addWidget(self.buffer, 2, 2)
        self.chk_pv = QCheckBox("Pore-volume multiplier on the outer columns: factor / rings")
        self.chk_pv.setChecked(ob.pv is not None)
        self.pv_factor = QLineEdit("" if ob.pv is None else f"{ob.pv.factor:g}")
        self.pv_rings = QLineEdit("" if ob.pv is None else f"{ob.pv.rings}")
        g.addWidget(self.chk_pv, 3, 0, 1, 2)
        g.addWidget(self.pv_factor, 3, 2)
        g.addWidget(self.pv_rings, 3, 3)
        self.chk_aq = QCheckBox("Aquifer")
        self.chk_aq.setChecked(ob.aquifer is not None)
        g.addWidget(self.chk_aq, 4, 0)
        self.aq_kind = QComboBox()
        self.aq_kind.addItems(list(AQ_KINDS))
        a = ob.aquifer
        if a is not None:
            self.aq_kind.setCurrentText(next(k for k, v in AQ_KINDS.items() if v == a.kind))
        self.aq_kind.currentTextChanged.connect(self._aq_kind_changed)
        g.addWidget(self.aq_kind, 4, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Connect to faces:"))
        self.aq_faces = {}
        for f in FACES:
            c = QCheckBox(f)
            c.setChecked(a is not None and f in a.faces)
            self.aq_faces[f] = c
            row.addWidget(c)
        w = QWidget()
        w.setLayout(row)
        g.addWidget(w, 4, 2, 1, 2)
        self.aq_fields, self.aq_labels = {}, {}
        for k, (key, text) in enumerate(AQ_FIELDS):
            lab = QLabel(text)
            e = QLineEdit("" if a is None or getattr(a, key) is None else f"{getattr(a, key):g}")
            self.aq_fields[key], self.aq_labels[key] = e, lab
            g.addWidget(lab, 5 + k // 2, (k % 2) * 2)
            g.addWidget(e, 5 + k // 2, (k % 2) * 2 + 1)
        g.addWidget(QLabel("Faces are grid index directions (I- = towards lower I). Units: METRIC deck (m, bar, mD, sm3/day/bar, 1/bar). "
                           "Use the deck's EQUIL datum depth for the aquifer datum, and leave the pressure blank to equilibrate."),
                    5 + len(AQ_FIELDS) // 2, 0, 1, 4)
        self._aq_kind_changed()
        return box

    def _aq_kind_changed(self, *_):
        kind = AQ_KINDS[self.aq_kind.currentText()]
        for key in FETKOVICH_ONLY + CT_ONLY:
            on = (key in FETKOVICH_ONLY) == (kind == "fetkovich")
            self.aq_fields[key].setEnabled(on)
            self.aq_labels[key].setEnabled(on)

    def _read_boundary(self) -> BoundaryOptions:
        def num(widget, what):
            t = widget.text().strip()
            if not t:
                raise GridDefinitionError(f"enter {what}")
            try:
                return float(t)
            except ValueError:
                raise GridDefinitionError(f"{what} must be a number") from None
        try:
            rule = next(k for k, v in RULE_TEXT.items() if v == self.rule.currentText())
            frac = num(self.min_fraction, "the minimum fraction") if rule == "fraction" else 0.5
            buf = num(self.buffer, "the buffer distance") if self.chk_buffer.isChecked() else None
            pv = None
            if self.chk_pv.isChecked():
                rings = num(self.pv_rings, "the number of edge rings")
                pv = PVMultiplier(num(self.pv_factor, "the pore-volume multiplier"), int(rings) if rings == int(rings) else rings)
            aq = None
            if self.chk_aq.isChecked():
                kind = AQ_KINDS[self.aq_kind.currentText()]
                if not any(self.aq_faces[f].isChecked() for f in FACES):
                    raise GridDefinitionError("choose which outer faces the aquifer connects to: " + ", ".join(FACES))
                vals = {}
                for key, text in AQ_FIELDS:
                    w = self.aq_fields[key]
                    if not w.isEnabled():
                        vals[key] = None
                    elif key == "initial_pressure" and not w.text().strip():
                        vals[key] = None
                    else:
                        v = num(w, text.split(" (")[0].lower())
                        vals[key] = int(v) if key == "table" and v == int(v) else v
                aq = AquiferSpec(kind, tuple(f for f in FACES if self.aq_faces[f].isChecked()), **vals)
            return BoundaryOptions(rule, frac, buf, pv, aq)
        except BoundaryOptionError as exc:
            raise GridDefinitionError(str(exc)) from None

    # -- intervals ---------------------------------------------------------------------------------
    def _fill_intervals(self):
        chart = self.project.chart
        gd = self.project.grid_def
        self.ints.setRowCount(len(chart.intervals))
        self.mode_boxes, self.value_edits, self.unc_boxes = [], [], []
        for r, iv in enumerate(chart.intervals):
            self.ints.setItem(r, 0, QTableWidgetItem(iv.name))
            first = r == 0
            self.ints.setItem(r, 1, QTableWidgetItem("top of model" if first else iv.contact_above))
            mb = QComboBox()
            mb.addItem(CHOOSE)
            mb.addItems(list(MODES))
            ve = QLineEdit()
            lay_ = None if gd is None else gd.layerings.get(iv.uid)
            if lay_ is not None:
                mb.setCurrentText(MODE_TEXT[lay_.mode])
                ve.setText(f"{lay_.n if lay_.mode == 'fixed_count' else lay_.dz:g}")
            ub = QComboBox()
            if not first and iv.contact_above == "unconformity":
                ub.addItems([CHOOSE, "erosional", "onlap"])
                if gd is not None and iv.uid in gd.unconformity:
                    ub.setCurrentText(gd.unconformity[iv.uid])
            else:
                ub.addItem("-")
                ub.setEnabled(False)
            self.ints.setCellWidget(r, 2, mb)
            self.ints.setCellWidget(r, 3, ve)
            self.ints.setCellWidget(r, 4, ub)
            self.mode_boxes.append(mb)
            self.value_edits.append(ve)
            self.unc_boxes.append(ub)

    # -- reading the form ---------------------------------------------------------------------------
    def _read(self) -> GridDefinition:
        vals = {}
        for k, e in self.fields.items():
            t = e.text().strip()
            if not t:
                raise GridDefinitionError(f"enter {k.replace('_', ' ')}")
            try:
                vals[k] = float(t)
            except ValueError:
                raise GridDefinitionError(f"{k.replace('_', ' ')} must be a number") from None
        if vals["nx"] != int(vals["nx"]) or vals["ny"] != int(vals["ny"]):
            raise GridDefinitionError("cell counts must be whole numbers")
        layerings, unc = {}, {}
        for iv, mb, ve, ub in zip(self.project.chart.intervals, self.mode_boxes, self.value_edits, self.unc_boxes):
            if mb.currentText() != CHOOSE:
                mode = MODES[mb.currentText()]
                try:
                    v = float(ve.text())
                except ValueError:
                    raise GridDefinitionError(f"interval {iv.name!r}: enter the "
                                              + ("number of layers" if mode == "fixed_count" else "layer thickness")) from None
                if mode == "fixed_count" and v != int(v):
                    raise GridDefinitionError(f"interval {iv.name!r}: the number of layers must be a whole number")
                try:
                    layerings[iv.uid] = (Layering(mode, n=int(v)) if mode == "fixed_count" else Layering(mode, dz=v))
                except BuildError as exc:
                    raise GridDefinitionError(f"interval {iv.name!r}: {exc}") from None
            if ub.isEnabled() and ub.currentText() != CHOOSE:
                unc[iv.uid] = ub.currentText()
        return GridDefinition(vals["origin_x"], vals["origin_y"], vals["azimuth_deg"], int(vals["nx"]), int(vals["ny"]),
                              vals["dx"], vals["dy"], vals["min_thickness"], layerings, unc, self._read_boundary())

    def _numbers(self, *keys):
        try:
            return [float(self.fields[k].text()) for k in keys]
        except ValueError:
            self.problem.setText("Enter numbers for the azimuth and the cell sizes first.")
            return None

    # -- actions ---------------------------------------------------------------------------------------
    def fit_extent(self):
        self.problem.setText("")
        v = self._numbers("azimuth_deg", "dx", "dy")
        if v is None:
            return
        if not (v[1] > 0 and v[2] > 0):
            self.problem.setText("The cell sizes must be positive.")
            return
        r = suggest_extent(self.project, *v)
        if r is None:
            self.problem.setText("The top horizon has no primary depth surface to fit to.")
            return
        for k, x in zip(("origin_x", "origin_y", "nx", "ny"), r):
            self.fields[k].setText(f"{x:.6g}" if k.startswith("origin") else f"{x}")

    def suggest(self):
        self.problem.setText("")
        v = self._numbers("dx", "dy")
        if v is None or not (v[0] > 0 and v[1] > 0):
            self.problem.setText("Enter positive cell sizes first: the cost depends on them.")
            return
        self._suggestions = orientation_suggestions(self.project, *v)
        self.sugg.setRowCount(len(self._suggestions))
        for r, s in enumerate(self._suggestions):
            self.sugg.setItem(r, 0, QTableWidgetItem(s.label))
            self.sugg.setItem(r, 1, QTableWidgetItem(f"{s.azimuth_deg:.1f}"))
            self.sugg.setItem(r, 2, QTableWidgetItem("-" if s.cells is None else f"{s.cells:,}"))
            self.sugg.setItem(r, 3, QTableWidgetItem("-" if s.cost_ratio is None else f"x {s.cost_ratio:.2f}"))

    def use_suggestion(self, row, _col=0):
        self.fields["azimuth_deg"].setText(f"{self._suggestions[row].azimuth_deg:.1f}")

    def save(self) -> bool:
        self.problem.setText("")
        try:
            self.project.grid_def = self._read()
        except (GridDefinitionError, BuildError) as exc:
            self.problem.setText(str(exc))
            return False
        return True

    def show_missing(self):
        if not self.save():
            return
        items = required_items(self.project, self.project.grid_def)
        self.report.setPlainText("Nothing is missing: the grid can be built." if not items else
                                 "Still missing:\n- " + "\n- ".join(items))

    def build(self):
        self.model = None
        self.export_btn.setEnabled(False)
        if not self.save():
            return
        try:
            self.model = build_model(self.project, self.project.grid_def)
        except (GridDefinitionError, ProjectError, BuildError, ValueError) as exc:
            self.problem.setText(str(exc))
            self.report.setPlainText("")
            return
        self.report.setPlainText(self.model.summary())
        self.export_btn.setEnabled(True)

    def _ask_export_path(self, default):
        p, _ = QFileDialog.getSaveFileName(self, "Export GRDECL", default, "GRDECL (*.grdecl *.GRDECL)")
        return p

    def export(self):
        if self.model is None:
            self.problem.setText("Build the grid first.")
            return
        path = self.ask_export_path(f"{self.project.name or 'model'}.grdecl")
        if not path:
            return
        try:
            files = write_model(self.model, path)
        except (OSError, ValueError, BoundaryOptionError) as exc:
            self.problem.setText(f"Could not export: {exc}")
            return
        self.report.appendPlainText("\nExported:\n" + "\n".join(f"  {f}" for f in files)
                                    + ("\nInclude each *_boundary.inc file in the deck section named in its first lines."
                                       if len(files) > 1 else ""))

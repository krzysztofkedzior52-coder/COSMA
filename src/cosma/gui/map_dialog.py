"""Compute, preview and save a per-interval property map; and a plain map viewer."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QVBoxLayout, QWidget)

from ..map_session import UNITS, MapSession
from ..maps import MapError, PropertyMap
from ..project import Action, Project, ProjectError
from ..variogram_data import SOURCES, collect


def draw_map(axes, pm: PropertyMap):
    """Estimate (and kriging variance, if any) as coloured maps with the wells that were used."""
    X, Y = pm.grid.node_xy()
    wxy = np.array(list(pm.info.get("well_xy", {}).values()), float).reshape(-1, 2)
    panels = [(pm.values, f"{pm.variable} [{pm.unit}]")] + ([(pm.variance, "kriging variance")] if pm.variance is not None else [])
    for ax in axes:
        ax.clear()
        ax.set_visible(False)
    for ax, (v, title) in zip(axes, panels):
        ax.set_visible(True)
        if np.isfinite(v).any():
            q = ax.pcolormesh(X, Y, np.ma.masked_invalid(v), shading="auto")
            ax.figure.colorbar(q, ax=ax, fraction=0.046)
        if len(wxy):
            ax.plot(wxy[:, 0], wxy[:, 1], "k^", ms=4)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=9, loc="left")
        ax.tick_params(labelsize=7)


class _Canvas:
    def __init__(self):
        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(7, 4))
        self.axes = list(self.fig.subplots(1, 2))
        self.canvas = FigureCanvasQTAgg(self.fig)

    def show(self, pm):
        self.fig.clear()
        self.axes = list(self.fig.subplots(1, 2))
        draw_map(self.axes, pm)
        self.fig.tight_layout()
        self.canvas.draw_idle()


class MapViewDialog(QDialog):
    def __init__(self, pm: PropertyMap, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Map - {pm.variable} ({pm.method})")
        self.setMinimumSize(900, 450)
        lay = QVBoxLayout(self)
        st = pm.stats()
        lay.addWidget(QLabel(f"{pm.method}; unit {pm.unit}; grid {pm.grid.ncol} x {pm.grid.nrow}, increment {pm.grid.xinc:g} x {pm.grid.yinc:g} m; "
                             + (f"values {st['min']:.4g} to {st['max']:.4g}, mean {st['mean']:.4g}" if st["n"] else "no defined nodes")))
        self.cv = _Canvas()
        lay.addWidget(self.cv.canvas)
        self.cv.show(pm)


class MapComputeDialog(QDialog):
    def __init__(self, project: Project, parent: QWidget | None = None, uid: str | None = None, choose_action=None):
        super().__init__(parent)
        self.project = project
        self.choose_action = choose_action or self._ask_action
        self.setWindowTitle("Property map")
        self.setMinimumSize(1000, 640)
        self.saved = None
        self.ms: MapSession | None = None
        self._busy = True
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        root.addLayout(left, 0)
        form = QFormLayout()
        left.addLayout(form)
        self.interval = QComboBox()
        for iv in project.chart.intervals:
            self.interval.addItem(iv.name, iv.uid)
        if uid:
            self.interval.setCurrentIndex(max(0, self.interval.findData(uid)))
        self.kind, self.prop, self.source = QComboBox(), QComboBox(), QComboBox()
        self.kind.addItems(SOURCES)
        self.indicator = QCheckBox("Facies proportion of one code")
        self.code = QComboBox()
        self.coverage = QLineEdit("0.5")
        self.vario = QComboBox()
        self.vwarn = QLabel("")
        self.vwarn.setWordWrap(True)
        self.vwarn.setStyleSheet("color: #8a5a00")
        self.grid_src = QComboBox()
        self.grid_src.addItems(["from the top surface of the interval", "from the base surface of the interval", "around the wells"])
        self.inc, self.margin = QLineEdit(""), QLineEdit("")
        self.inc.setPlaceholderText("increment (m)")
        self.margin.setPlaceholderText("margin (m)")
        self.unit = QComboBox()
        self.unit.addItem("")
        self.unit.addItems(UNITS)
        self.mask = QCheckBox("Leave nodes outside the interval undefined (needs top and base surfaces)")
        self.mask.setChecked(True)
        self.lo, self.hi = QLineEdit(""), QLineEdit("")
        self.lo.setPlaceholderText("min"); self.hi.setPlaceholderText("max")
        for label, w in (("Interval", self.interval), ("Data", self.kind), ("Property", self.prop), ("Source", self.source),
                         (None, self.indicator), ("Facies code", self.code), ("Min. well coverage (0-1)", self.coverage),
                         ("Variogram", self.vario), (None, self.vwarn), ("Grid", self.grid_src)):
            form.addRow(label, w) if label else form.addRow(w)
        gr = QHBoxLayout(); gr.addWidget(self.inc); gr.addWidget(self.margin)
        form.addRow("  around wells:", gr)
        form.addRow("Unit of the values", self.unit)
        form.addRow(self.mask)
        br = QHBoxLayout(); br.addWidget(self.lo); br.addWidget(self.hi)
        form.addRow("Clip to range (optional)", br)
        row = QHBoxLayout()
        self.compute_btn, self.thick_btn = QPushButton("Compute map"), QPushButton("Thickness map")
        row.addWidget(self.compute_btn); row.addWidget(self.thick_btn)
        left.addLayout(row)
        self.problem = QLabel("")
        self.problem.setWordWrap(True)
        self.problem.setStyleSheet("color: #b00020")
        left.addWidget(self.problem)
        self.notes = QPlainTextEdit()
        self.notes.setReadOnly(True)
        left.addWidget(self.notes)
        srow = QHBoxLayout()
        self.label = QLineEdit("")
        self.label.setPlaceholderText("name of this map")
        self.save_btn = QPushButton("Save to project")
        self.save_btn.setEnabled(False)
        srow.addWidget(self.label); srow.addWidget(self.save_btn)
        left.addLayout(srow)
        self.cv = _Canvas()
        root.addWidget(self.cv.canvas, 1)
        self._busy = False
        self.interval.currentIndexChanged.connect(self._interval_changed)
        self.kind.currentIndexChanged.connect(self._fill_props)
        self.prop.currentIndexChanged.connect(self._prop_changed)
        self.indicator.toggled.connect(self._refresh_vario)
        self.code.currentIndexChanged.connect(self._refresh_vario)
        self.vario.currentIndexChanged.connect(self._vario_changed)
        self.compute_btn.clicked.connect(self.compute)
        self.thick_btn.clicked.connect(self.thickness)
        self.save_btn.clicked.connect(self.save)
        self._interval_changed()

    # -- state --------------------------------------------------------------------------------------------------------------------------
    def _interval_changed(self, *_):
        if self._busy:
            return
        self.ms = MapSession(self.project, self.interval.currentData())
        self.save_btn.setEnabled(False)
        self._fill_props()

    def _fill_props(self, *_):
        props = self.ms.properties(self.kind.currentText())
        self._props = props
        self._busy = True
        self.prop.clear()
        self.prop.addItems(list(props))
        self._busy = False
        self._prop_changed()

    def _prop_changed(self, *_):
        if self._busy or self.ms is None:
            return
        srcs = self._props.get(self.prop.currentText(), [])
        self._busy = True
        self.source.clear()
        self.source.addItems(["(the only one in each well)"] + srcs)
        self.code.clear()
        codes = None
        try:
            d = collect(self.project, self.ms.uid, self.prop.currentText(), kind=self.kind.currentText(), source=None, frame="top") \
                if self.prop.currentText() else None
            codes = d.facies_codes() if d is not None else None
        except ProjectError:
            pass
        for c, p in (codes or {}).items():
            self.code.addItem(f"{c}  ({p:.1%})", c)
        self.indicator.setEnabled(codes is not None)
        if codes is None:
            self.indicator.setChecked(False)
        self.code.setEnabled(self.indicator.isChecked())
        self._busy = False
        self._refresh_vario()

    def _indicator(self):
        return int(self.code.currentData()) if self.indicator.isChecked() and self.code.currentData() is not None else None

    def _refresh_vario(self, *_):
        if self._busy or self.ms is None:
            return
        self.code.setEnabled(self.indicator.isChecked())
        self._choices = self.ms.variogram_choices(self.prop.currentText(), self._indicator()) if self.prop.currentText() else []
        self._busy = True
        self.vario.clear()
        for c in self._choices:
            self.vario.addItem(c.text)
        self._busy = False
        self._vario_changed()

    def _vario_changed(self, *_):
        if self._busy:
            return
        i = self.vario.currentIndex()
        self.vwarn.setText(self.ms.warning_for(self._choices[i]) or "" if 0 <= i < len(self._choices) else
                           "No variogram for this property in this interval yet: compute or import one first (Variograms menu).")

    # -- actions --------------------------------------------------------------------------------------------------------------------------
    def _num(self, w, name, required=True):
        t = w.text().strip()
        if not t:
            if required:
                raise MapError(f"{name} is required")
            return None
        try:
            return float(t)
        except ValueError:
            raise MapError(f"{name}: {t!r} is not a number") from None

    def _grid(self):
        k = self.grid_src.currentIndex()
        if k == 2:
            return self.ms.wells_grid(self._num(self.inc, "grid increment"), self._num(self.margin, "grid margin"))
        return self.ms.surface_grid("top" if k == 0 else "base")

    def _show(self, pm):
        self.cv.show(pm)
        self.notes.setPlainText("\n".join(self.ms.notes))
        self.save_btn.setEnabled(True)
        self.problem.setText("")

    def compute(self):
        try:
            i = self.vario.currentIndex()
            if not 0 <= i < len(self._choices):
                raise MapError("choose a variogram (none stored for this property in this interval)")
            if not self.unit.currentText():
                raise MapError("choose the unit of the values: none is assumed")
            lo, hi = self._num(self.lo, "range minimum", False), self._num(self.hi, "range maximum", False)
            if (lo is None) != (hi is None):
                raise MapError("give both ends of the clipping range, or neither")
            src = self.source.currentText()
            pm = self.ms.compute(self.prop.currentText(), self._choices[i], self._grid(), self.unit.currentText(),
                                 kind=self.kind.currentText(), source=None if src.startswith("(") else src,
                                 indicator=self._indicator(), min_coverage=self._num(self.coverage, "minimum coverage"),
                                 mask="thickness" if self.mask.isChecked() else "none",
                                 bounds=None if lo is None else (lo, hi))
        except (MapError, ProjectError, ValueError) as exc:
            self.problem.setText(str(exc))
            return
        self._show(pm)

    def thickness(self):
        try:
            pm = self.ms.thickness(self._grid())
        except (MapError, ProjectError, ValueError) as exc:
            self.problem.setText(str(exc))
            return
        self._show(pm)

    def _ask_action(self, conflict):
        from PySide6.QtWidgets import QInputDialog
        mode, ok = QInputDialog.getItem(self, "Map exists", f"{conflict.item} already has a map ({conflict.existing}). "
                                        f"What should happen to {conflict.new!r}?", ["replace", "skip", "suffix"], 0, False)
        if not ok:
            return None
        if mode == "suffix":
            suf, ok = QInputDialog.getText(self, "Suffix", "Suffix for the new name:")
            return Action("suffix", suf) if ok else None
        return Action(mode)

    def save(self):
        label = self.label.text().strip()
        try:
            acts = {}
            for c in self.ms.conflicts(label):
                a = self.choose_action(c)
                if a is None:
                    return
                acts[c.key] = a
            res = self.ms.save(label, acts)
        except (ProjectError, MapError) as exc:
            self.problem.setText(str(exc))
            return
        self.saved = res
        self.problem.setText("")
        self.notes.appendPlainText("; ".join(res.applied + res.notes))

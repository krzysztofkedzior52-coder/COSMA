"""Choose what a variogram is computed from: interval, data kind, property, source, frame and,
for a facies curve, the facies code (a 0/1 indicator variogram)."""

from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout, QWidget)

from ..project import Project, ProjectError
from ..variogram_data import FRAMES, SOURCES, available_properties, collect
from ..variogram_session import VariogramSession

ANY = "(the only one in each well)"


class VariogramSetupDialog(QDialog):
    def __init__(self, project: Project, parent: QWidget | None = None, preset: dict | None = None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Variogram: data")
        self.setMinimumWidth(460)
        preset = preset or {}
        self._busy = True
        lay = QVBoxLayout(self)
        form = QFormLayout()
        lay.addLayout(form)
        self.interval = QComboBox()
        for iv in project.chart.intervals:
            self.interval.addItem(iv.name, iv.uid)
        self.kind = QComboBox()
        self.kind.addItems(SOURCES)
        self.prop = QComboBox()
        self.source = QComboBox()
        self.frame = QComboBox()
        self.frame.addItems(FRAMES)
        self.average = QCheckBox("Layer averages: one value per well (the variogram property maps are kriged with)")
        self.indicator = QCheckBox("Facies: variogram of the 0/1 indicator of one code (with averages: of the proportion)")
        self.code = QComboBox()
        form.addRow("Interval", self.interval)
        form.addRow("Data", self.kind)
        form.addRow("Property", self.prop)
        form.addRow("Source", self.source)
        form.addRow("Vertical frame", self.frame)
        form.addRow(self.average)
        form.addRow(self.indicator)
        form.addRow("Facies code", self.code)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        if preset.get("uid"):
            self.interval.setCurrentIndex(max(0, self.interval.findData(preset["uid"])))
        if preset.get("kind"):
            self.kind.setCurrentText(preset["kind"])
        self._fill_props(preset.get("curve"))
        if preset.get("source") and self.source.findText(preset["source"]) >= 0:
            self.source.setCurrentText(preset["source"])
        if preset.get("frame"):
            self.frame.setCurrentText(preset["frame"])
        self._busy = False
        self._preset_code = preset.get("indicator")
        self.indicator.setChecked(self._preset_code is not None)
        self.average.setChecked(bool(preset.get("average")))
        self.average.toggled.connect(self._changed)
        for w in (self.interval, self.prop, self.source, self.frame, self.code):
            w.currentIndexChanged.connect(self._changed)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.indicator.toggled.connect(self._changed)
        self._changed()

    def _fill_props(self, keep=None):
        props = available_properties(self.project, self.kind.currentText())
        self._props = props
        self.prop.blockSignals(True)
        self.prop.clear()
        self.prop.addItems(list(props))
        if keep and self.prop.findText(keep) >= 0:
            self.prop.setCurrentText(keep)
        self.prop.blockSignals(False)
        self._fill_sources()

    def _fill_sources(self):
        srcs = self._props.get(self.prop.currentText(), [])
        cur = self.source.currentText()
        self.source.blockSignals(True)
        self.source.clear()
        self.source.addItems([ANY] + srcs)
        if cur and self.source.findText(cur) >= 0:
            self.source.setCurrentText(cur)
        self.source.blockSignals(False)

    def _kind_changed(self, *_):
        if self._busy:
            return
        self._fill_props()
        self._changed()

    def _args(self):
        src = self.source.currentText()
        return dict(uid=self.interval.currentData(), kind=self.kind.currentText(), variable=self.prop.currentText(),
                    source=None if src in (ANY, "") else src, frame=self.frame.currentText())

    def _changed(self, *_):
        if self._busy:
            return
        self.problem = None
        self.codes = None
        a = self._args()
        if self.sender() is self.prop:
            self._fill_sources()
            a = self._args()
        ok = False
        if not a["variable"]:
            self.problem = f"no {a['kind']} data in the project"
        else:
            try:
                d = collect(self.project, a["uid"], a["variable"], kind=a["kind"], source=a["source"], frame=a["frame"])
                self.codes = d.facies_codes()
                self.info.setText(f"{len(d.used)} well(s), {len(d.values)} samples"
                                  + (f"; not used: {'; '.join(f'{w} ({r})' for w, r in d.skipped.items())}" if d.skipped else ""))
                ok = True
            except ProjectError as exc:
                self.problem = str(exc)
        self.indicator.setEnabled(self.codes is not None)
        if self.codes is None:
            self.indicator.blockSignals(True)
            self.indicator.setChecked(False)
            self.indicator.blockSignals(False)
        self.code.blockSignals(True)
        cur = self.code.currentData()
        self.code.clear()
        for c, p in (self.codes or {}).items():
            self.code.addItem(f"{c}  ({p:.1%})", c)
        want = cur if cur is not None else self._preset_code
        if want is not None and self.code.findData(want) >= 0:
            self.code.setCurrentIndex(self.code.findData(want))
        self.code.blockSignals(False)
        self.code.setEnabled(self.indicator.isChecked())
        if self.problem:
            self.info.setText(self.problem)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(ok)

    def values(self) -> dict:
        v = self._args()
        v["indicator"] = int(self.code.currentData()) if self.indicator.isChecked() else None
        v["average"] = self.average.isChecked()
        return v


def make_session(project: Project, values: dict, wells=None) -> VariogramSession:
    return VariogramSession(project, values["uid"], values["variable"], kind=values["kind"], source=values["source"],
                            frame=values["frame"], indicator=values.get("indicator"), wells=wells,
                            average=bool(values.get("average")), min_coverage=values.get("min_coverage", 0.5))


def ask_variogram_setup(project: Project, parent: QWidget | None = None, preset: dict | None = None):
    dlg = VariogramSetupDialog(project, parent, preset)
    return dlg.values() if dlg.exec() == QDialog.Accepted else None

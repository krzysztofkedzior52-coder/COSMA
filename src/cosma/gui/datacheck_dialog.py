"""Data checks: is there enough data per interval and facies to simulate from? Every threshold is entered by the user
(no defaults). Nothing is stored until the checks have been run on the current entries and Save is pressed; any edit
after the run switches Save off again. The report suggests fallbacks but chooses none."""

from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..datacheck import (AVERAGING, CoreSpec, DataCheckError, DataCheckSetup, PropertySpec, Thresholds, run_checks,
                         to_state)
from ..facies import FaciesError

FIELDS = (("min_hard_cells", "Minimum hard cells per interval and facies", int),
          ("min_wells", "Minimum wells per interval and facies", int),
          ("lag_v", "Vertical variogram lag (m)", float),
          ("nlags_v", "Vertical variogram number of lags", int),
          ("min_pairs_v", "Vertical: minimum pairs per lag", int),
          ("min_usable_lag_v", "Vertical: must be supported to at least this lag (m)", float),
          ("lag_h", "Horizontal variogram lag (m)", float),
          ("nlags_h", "Horizontal variogram number of lags", int),
          ("min_pairs_h", "Horizontal: minimum pairs per lag", int),
          ("min_usable_lag_h", "Horizontal: must be supported to at least this lag (m)", float),
          ("same_layer_tol", "Horizontal pairs must lie within this vertical distance (m)", float),
          ("min_wells_h", "Horizontal: minimum wells", int),
          ("min_core_pairs", "Core: minimum porosity-permeability pairs per regression", int),
          ("min_r2", "Core: minimum R-squared (leave empty for no R-squared test)", float))


def _default_build(project):
    from ..grid_definition import GridDefinitionError, build_model, required_items
    gd = project.grid_def
    if gd is None:
        raise DataCheckError("define the grid first (Grid menu)")
    missing = required_items(project, gd)
    if missing:
        raise DataCheckError("the grid cannot be built yet (" + missing[0] + (", ..." if len(missing) > 1 else "") + ")")
    try:
        return build_model(project, gd)
    except (GridDefinitionError, ValueError) as exc:
        raise DataCheckError(f"the grid could not be built: {exc}") from None


class DataCheckDialog(QDialog):
    def __init__(self, project, parent: QWidget | None = None, build=None):
        super().__init__(parent)
        self.project = project
        self.build = build or _default_build
        self.setWindowTitle("Data checks")
        self.setMinimumSize(900, 860)
        lay = QVBoxLayout(self)
        note = QLabel("Say how much data is enough, then run the checks. They count hard cells per interval and facies, "
                      "test whether vertical and horizontal variograms can be computed from your wells, show cell-scale "
                      "distributions and test core regressions at three levels. The report only suggests a fallback; "
                      "you choose it later. Save is possible only after a run on the current entries.")
        note.setWordWrap(True)
        lay.addWidget(note)
        grid = QGridLayout()
        self.edits = {}
        for r, (key, label, _t) in enumerate(FIELDS):
            grid.addWidget(QLabel(label), r, 0)
            e = QLineEdit()
            e.textChanged.connect(self._edited)
            grid.addWidget(e, r, 1)
            self.edits[key] = e
        lay.addLayout(grid)
        lay.addWidget(QLabel("Properties to check (log curve of the same source as the facies); the averaging rule turns "
                             "log samples into a cell value:"))
        self.props = QTableWidget(3, 3)
        self.props.setHorizontalHeaderLabels(["Name", "Curve", "Averaging"])
        self.props.horizontalHeader().setStretchLastSection(True)
        self.props.setMaximumHeight(130)
        self.props.itemChanged.connect(self._edited)
        lay.addWidget(self.props)
        pb = QHBoxLayout()
        add, rem = QPushButton("Add row"), QPushButton("Remove last")
        add.clicked.connect(lambda: (self.props.setRowCount(self.props.rowCount() + 1), self._edited()))
        rem.clicked.connect(lambda: (self.props.setRowCount(max(0, self.props.rowCount() - 1)), self._edited()))
        pb.addWidget(add)
        pb.addWidget(rem)
        pb.addStretch(1)
        lay.addLayout(pb)
        cg = QGridLayout()
        self.core = {}
        for r, (key, label) in enumerate((("porosity", "Core porosity column (all four core entries empty = no core check)"),
                                          ("permeability", "Core permeability column"),
                                          ("depth_tol", "Core takes the facies of the nearest log sample within (m)"),
                                          ("source", "Core source (only if a well has several)"))):
            cg.addWidget(QLabel(label), r, 0)
            e = QLineEdit()
            e.textChanged.connect(self._edited)
            cg.addWidget(e, r, 1)
            self.core[key] = e
        lay.addLayout(cg)
        self.problem = QLabel("")
        self.problem.setStyleSheet("color: #b00020")
        self.problem.setWordWrap(True)
        lay.addWidget(self.problem)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        lay.addWidget(self.report, 1)
        row = QHBoxLayout()
        self.run_btn = QPushButton("Run checks")
        self.run_btn.clicked.connect(self.run)
        row.addWidget(self.run_btn)
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save)
        row.addWidget(self.save_btn)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)
        self._ran = None
        self.saved = False
        self._load(project.data_checks)

    def _edited(self, *_):
        self._ran = None
        self.save_btn.setEnabled(False)

    def _load(self, s):
        if s is None:
            return
        for e in list(self.edits.values()) + list(self.core.values()):
            e.blockSignals(True)
        self.props.blockSignals(True)
        for key, _l, _t in FIELDS:
            v = getattr(s.thresholds, key)
            self.edits[key].setText("" if v is None else f"{v:g}")
        self.props.setRowCount(max(3, len(s.properties)))
        for r, p in enumerate(s.properties):
            for c, v in enumerate((p.name, p.curve, p.averaging)):
                self.props.setItem(r, c, QTableWidgetItem(v))
        if s.core is not None:
            self.core["porosity"].setText(s.core.porosity)
            self.core["permeability"].setText(s.core.permeability)
            self.core["depth_tol"].setText(f"{s.core.depth_tol:g}")
            self.core["source"].setText(s.core.source or "")
        for e in list(self.edits.values()) + list(self.core.values()):
            e.blockSignals(False)
        self.props.blockSignals(False)

    def read_setup(self) -> DataCheckSetup:
        vals = {}
        for key, label, typ in FIELDS:
            text = self.edits[key].text().strip()
            if not text:
                if key == "min_r2":
                    vals[key] = None
                    continue
                raise DataCheckError(f"{label}: enter a value")
            try:
                v = float(text)
                if typ is int:
                    if v != int(v):
                        raise ValueError
                    v = int(v)
            except ValueError:
                raise DataCheckError(f"{label}: enter {'a whole number' if typ is int else 'a number'}") from None
            vals[key] = v
        props = []
        for r in range(self.props.rowCount()):
            cells = [(self.props.item(r, c).text().strip() if self.props.item(r, c) else "") for c in range(3)]
            if not any(cells):
                continue
            if not all(cells):
                raise DataCheckError(f"property row {r + 1}: fill in name, curve and averaging "
                                     f"({', '.join(AVERAGING)})")
            props.append(PropertySpec(*cells))
        c = {k: e.text().strip() for k, e in self.core.items()}
        core = None
        if any(c[k] for k in ("porosity", "permeability", "depth_tol")):
            if not (c["porosity"] and c["permeability"] and c["depth_tol"]):
                raise DataCheckError("core check: give the porosity column, the permeability column and the depth tolerance")
            try:
                tol = float(c["depth_tol"])
            except ValueError:
                raise DataCheckError("core depth tolerance: enter a number") from None
            core = CoreSpec(c["porosity"], c["permeability"], tol, c["source"] or None)
        elif c["source"]:
            raise DataCheckError("a core source is given but no core columns")
        return DataCheckSetup(Thresholds(**vals), tuple(props), core)

    def run(self):
        self.problem.setText("")
        self._ran = None
        self.save_btn.setEnabled(False)
        try:
            s = self.read_setup()
            if self.project.facies_setup is None:
                raise DataCheckError("define the facies first (Facies menu)")
            model = self.build(self.project)
            rep = run_checks(self.project, model, s)
        except (DataCheckError, FaciesError) as exc:
            self.problem.setText(str(exc))
            self.report.setPlainText("")
            return
        self.report.setPlainText(rep.summary())
        self._ran = to_state(s)
        self.save_btn.setEnabled(True)

    def save(self) -> bool:
        try:
            s = self.read_setup()
        except DataCheckError as exc:
            self.problem.setText(str(exc))
            return False
        if self._ran is None or to_state(s) != self._ran:
            self.problem.setText("Run the checks on these entries first: what is saved must be what was run.")
            self.save_btn.setEnabled(False)
            return False
        self.project.set_data_checks(s)
        self.saved = True
        self.problem.setText("")
        return True

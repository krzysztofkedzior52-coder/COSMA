"""Facies from logs. Nothing is stored until the preview of the current entries has been shown and Save is pressed;
any edit after the preview switches Save off again. Facies are typed by name in the tables and resolved against the
scheme."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QComboBox, QDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..facies import (CONFLICT_POLICIES, DEPTH_TYPES, OPS, CodeMap, Condition, Facies, FaciesError, FaciesSetup, Flag,
                      Rule, classify_all, hard_facies, to_state)

CHOOSE = "- choose -"
M_RULES, M_CODES, M_FLAGS = "Rules on log curves", "Code map (a curve already holds facies codes)", "Flags only"


def _default_build(project):
    """The grid for the hard-cell preview, or (None, reason)."""
    from ..grid_definition import GridDefinitionError, build_model, required_items
    gd = project.grid_def
    if gd is None:
        return None, "define the grid first (Grid menu) to see the hard cells"
    missing = required_items(project, gd)
    if missing:
        return None, "the grid cannot be built yet (" + missing[0] + (", ..." if len(missing) > 1 else "") + ")"
    try:
        return build_model(project, gd).grid, ""
    except (GridDefinitionError, ValueError) as exc:
        return None, f"the grid could not be built: {exc}"


class FaciesDialog(QDialog):
    def __init__(self, project, parent: QWidget | None = None, build=None):
        super().__init__(parent)
        self.project = project
        self.build = build or _default_build
        self.setWindowTitle("Facies from logs")
        self.setMinimumSize(980, 820)
        lay = QVBoxLayout(self)
        note = QLabel("Define the facies, say how a log becomes facies, then press Preview and check the result. "
                      "Save is possible only for entries that have been previewed. A sample no rule matches stays "
                      "undefined; nothing is assigned by guess.")
        note.setWordWrap(True)
        lay.addWidget(note)
        self.scheme = self._table(["Code", "Name"], 90, lay, "Facies scheme (codes are whole numbers >= 1)", 4)
        row = QHBoxLayout()
        row.addWidget(QLabel("How facies are found:"))
        self.method = QComboBox()
        self.method.addItems([CHOOSE, M_RULES, M_CODES, M_FLAGS])
        row.addWidget(self.method)
        row.addWidget(QLabel("Log source (only if a well has several):"))
        self.source = QLineEdit()
        row.addWidget(self.source)
        lay.addLayout(row)
        self.rules = self._table(["Facies", "Curve 1", "Op", "Value", "Curve 2", "Op", "Value"], 110, lay,
                                 "Rules, in priority order: the first rule whose conditions all hold wins (operators < <= > >=)", 4)
        row = QHBoxLayout()
        row.addWidget(QLabel("Code-map curve:"))
        self.map_curve = QLineEdit()
        row.addWidget(self.map_curve)
        lay.addLayout(row)
        self.codes = self._table(["Curve value", "Facies"], 130, lay, "Code map: which curve value is which facies", 3)
        self.flags = self._table(["Well", "Top", "Base", "Depth type (md/tvd/tvdss)", "Facies"], 130, lay,
                                 "Flags: intervals that are a given facies, overriding the log", 3)
        grid = QGridLayout()
        grid.addWidget(QLabel("Flag sampling step (m), for wells without a log:"), 0, 0)
        self.step = QLineEdit()
        grid.addWidget(self.step, 0, 1)
        grid.addWidget(QLabel("Minimum classified share of a cell's well path (0-1):"), 1, 0)
        self.coverage = QLineEdit()
        grid.addWidget(self.coverage, 1, 1)
        grid.addWidget(QLabel("Wells that disagree in one cell:"), 2, 0)
        self.conflict = QComboBox()
        self.conflict.addItems([CHOOSE, "longest path wins", "no hard datum for that cell"])
        grid.addWidget(self.conflict, 2, 1)
        lay.addLayout(grid)
        self.problem = QLabel("")
        self.problem.setStyleSheet("color: #b00020")
        self.problem.setWordWrap(True)
        lay.addWidget(self.problem)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        lay.addWidget(self.report, 1)
        row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self.preview)
        row.addWidget(self.preview_btn)
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save)
        row.addWidget(self.save_btn)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)
        self._previewed = None
        self.saved = False
        self._load(project.facies_setup)
        for t in (self.scheme, self.rules, self.codes, self.flags):
            t.itemChanged.connect(self._edited)
        for e in (self.source, self.map_curve, self.step, self.coverage):
            e.textChanged.connect(self._edited)
        self.method.currentTextChanged.connect(self._edited)
        self.conflict.currentTextChanged.connect(self._edited)

    # -- widgets ----------------------------------------------------------------------------------------------------
    def _table(self, heads, width, lay, title, rows):
        box = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel(title))
        t = QTableWidget(rows, len(heads))
        t.setHorizontalHeaderLabels(heads)
        t.horizontalHeader().setStretchLastSection(True)
        for c in range(len(heads) - 1):
            t.setColumnWidth(c, width)
        t.setMaximumHeight(150)
        left.addWidget(t)
        box.addLayout(left, 1)
        btns = QVBoxLayout()
        add, rem = QPushButton("Add row"), QPushButton("Remove last")
        add.clicked.connect(lambda: (t.setRowCount(t.rowCount() + 1), self._edited()))
        rem.clicked.connect(lambda: (t.setRowCount(max(0, t.rowCount() - 1)), self._edited()))
        btns.addWidget(add)
        btns.addWidget(rem)
        btns.addStretch(1)
        box.addLayout(btns)
        lay.addLayout(box)
        return t

    @staticmethod
    def _txt(t, r, c):
        it = t.item(r, c)
        return it.text().strip() if it is not None else ""

    @staticmethod
    def _put(t, r, c, text):
        t.setItem(r, c, QTableWidgetItem(str(text)))

    def _edited(self, *_):
        self._previewed = None
        self.save_btn.setEnabled(False)

    # -- stored setup -> form ---------------------------------------------------------------------------------------
    def _load(self, s):
        if s is None:
            return
        blocked = [t.blockSignals(True) for t in (self.scheme, self.rules, self.codes, self.flags)]
        name = {f.code: f.name for f in s.scheme}
        self.scheme.setRowCount(len(s.scheme))
        for r, f in enumerate(s.scheme):
            self._put(self.scheme, r, 0, f.code)
            self._put(self.scheme, r, 1, f.name)
        if s.rules:
            self.method.setCurrentText(M_RULES)
            self.rules.setRowCount(len(s.rules))
            for r, rule in enumerate(s.rules):
                self._put(self.rules, r, 0, name[rule.facies])
                for n, c in enumerate(rule.conditions[:2]):
                    for k, v in enumerate((c.curve, c.op, c.value)):
                        self._put(self.rules, r, 1 + 3 * n + k, f"{v:g}" if isinstance(v, float) else v)
        elif s.code_map is not None:
            self.method.setCurrentText(M_CODES)
            self.map_curve.setText(s.code_map.curve)
            self.codes.setRowCount(len(s.code_map.mapping))
            for r, (k, v) in enumerate(s.code_map.mapping.items()):
                self._put(self.codes, r, 0, k)
                self._put(self.codes, r, 1, name[v])
        else:
            self.method.setCurrentText(M_FLAGS)
        self.flags.setRowCount(len(s.flags))
        for r, f in enumerate(s.flags):
            for c, v in enumerate((f.well, f.top, f.base, f.depth_type, name[f.facies])):
                self._put(self.flags, r, c, f"{v:g}" if isinstance(v, float) else v)
        if isinstance(s.source, str):
            self.source.setText(s.source)
        if s.flag_step is not None:
            self.step.setText(f"{s.flag_step:g}")
        if s.min_coverage is not None:
            self.coverage.setText(f"{s.min_coverage:g}")
        if s.conflict is not None:
            self.conflict.setCurrentIndex(1 + CONFLICT_POLICIES.index(s.conflict))
        for t, b in zip((self.scheme, self.rules, self.codes, self.flags), blocked):
            t.blockSignals(b)

    # -- form -> setup -----------------------------------------------------------------------------------------------
    def _num(self, text, what):
        try:
            return float(text)
        except ValueError:
            raise FaciesError(f"{what}: enter a number") from None

    def read_setup(self) -> FaciesSetup:
        scheme = []
        for r in range(self.scheme.rowCount()):
            code, name = self._txt(self.scheme, r, 0), self._txt(self.scheme, r, 1)
            if not code and not name:
                continue
            try:
                scheme.append(Facies(int(code), name))
            except ValueError:
                raise FaciesError(f"facies row {r + 1}: the code must be a whole number") from None
            except FaciesError as exc:
                raise FaciesError(f"facies row {r + 1}: {exc}") from None
        by_name = {f.name.lower(): f.code for f in scheme}

        def fac(text, where):
            if text.lower() not in by_name:
                raise FaciesError(f"{where}: {text!r} is not a facies of the scheme")
            return by_name[text.lower()]

        m = self.method.currentText()
        if m == CHOOSE:
            raise FaciesError("choose how facies are found")
        rules, cmap = [], None
        if m == M_RULES:
            for r in range(self.rules.rowCount()):
                cells = [self._txt(self.rules, r, c) for c in range(7)]
                if not any(cells):
                    continue
                conds = []
                for base in (1, 4):
                    cu, op, v = cells[base:base + 3]
                    if cu or op or v:
                        if not (cu and op and v):
                            raise FaciesError(f"rule {r + 1}: a condition needs a curve, an operator and a value")
                        conds.append(Condition(cu, op, self._num(v, f"rule {r + 1}")))
                if not conds:
                    raise FaciesError(f"rule {r + 1} has no condition")
                rules.append(Rule(fac(cells[0], f"rule {r + 1}"), tuple(conds)))
        elif m == M_CODES:
            mapping = {}
            for r in range(self.codes.rowCount()):
                a, b = self._txt(self.codes, r, 0), self._txt(self.codes, r, 1)
                if not a and not b:
                    continue
                try:
                    key = int(a)
                except ValueError:
                    raise FaciesError(f"code map row {r + 1}: the curve value must be a whole number") from None
                if key in mapping:
                    raise FaciesError(f"code map row {r + 1}: curve value {key} is mapped twice")
                mapping[key] = fac(b, f"code map row {r + 1}")
            cmap = CodeMap(self.map_curve.text().strip(), mapping)
        flags = []
        for r in range(self.flags.rowCount()):
            cells = [self._txt(self.flags, r, c) for c in range(5)]
            if not any(cells):
                continue
            if not all(cells):
                raise FaciesError(f"flag row {r + 1}: fill in all five entries")
            flags.append(Flag(cells[0], self._num(cells[1], f"flag row {r + 1} top"), self._num(cells[2], f"flag row {r + 1} base"),
                              cells[3], fac(cells[4], f"flag row {r + 1}")))
        step = self._num(self.step.text(), "flag step") if self.step.text().strip() else None
        cov = self._num(self.coverage.text(), "minimum classified share") if self.coverage.text().strip() else None
        ci = self.conflict.currentIndex()
        return FaciesSetup(tuple(scheme), tuple(rules), cmap, self.source.text().strip() or None, tuple(flags), step, cov,
                           None if ci == 0 else CONFLICT_POLICIES[ci - 1])

    # -- preview and save --------------------------------------------------------------------------------------------
    def preview(self):
        self.problem.setText("")
        self._previewed = None
        self.save_btn.setEnabled(False)
        try:
            s = self.read_setup()
            wf, skipped = classify_all(self.project, s)
            if not wf:
                raise FaciesError("no well could be classified: " + "; ".join(f"{w}: {r}" for w, r in skipped.items()))
        except FaciesError as exc:
            self.problem.setText(str(exc))
            self.report.setPlainText("")
            return
        names = {f.code: f.name for f in s.scheme}
        lines = ["Log classification (share of well length):"]
        for w, d in wf.items():
            tot = float(d.length.sum())
            parts = [f"{names[c]} {100 * float(d.length[d.code == c].sum()) / tot:.1f}%" for c in sorted(names) if (d.code == c).any()]
            und = 100 * float(d.length[d.code == 0].sum()) / tot
            lines.append(f"  {w} [{d.source or 'flags only'}]: {d.md.size:,} samples: " + (", ".join(parts) or "no facies")
                         + f", undefined {und:.1f}%" + (f"; {d.flagged:,} set by flags" if d.flagged else "")
                         + (f"; {d.outside_path:,} log samples beyond the trajectory dropped" if d.outside_path else "")
                         + ("; curve values with no mapping: " + ", ".join(f"{k} ({v})" for k, v in sorted(d.unmapped.items()))
                            if d.unmapped else ""))
        for w, r in skipped.items():
            lines.append(f"  skipped {w}: {r}")
        if s.min_coverage is None or s.conflict is None:
            lines.append("\nChoose the minimum classified share and the conflict policy to see the hard cells.")
        else:
            grid, why = self.build(self.project)
            if grid is None:
                lines.append("\nHard cells not shown: " + why)
            else:
                try:
                    lines.append("\nHard cells in the grid:\n" + hard_facies(self.project, s, grid).summary())
                except FaciesError as exc:
                    lines.append("\nHard cells not shown: " + str(exc))
        self.report.setPlainText("\n".join(lines))
        self._previewed = to_state(s)
        self.save_btn.setEnabled(True)

    def save(self) -> bool:
        try:
            s = self.read_setup()
        except FaciesError as exc:
            self.problem.setText(str(exc))
            return False
        if self._previewed is None or to_state(s) != self._previewed:
            self.problem.setText("Preview these entries first: what is saved must be what was previewed.")
            self.save_btn.setEnabled(False)
            return False
        self.project.set_facies_setup(s)
        self.saved = True
        self.problem.setText("")
        return True

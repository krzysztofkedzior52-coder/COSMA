"""Fault transmissibility: one explicit choice per fault. Nothing is preselected. 'Calculated' is offered only for
faults whose throw is visible (known after a build); until then it is allowed and the build refuses a mistake.

SGR methods: the built-in published relation, or the user's own stored methods (named expressions for log10 kf in SGR
and D, with their source). A fault's choice keeps its own copy of the expression, so later edits of a stored method
never change an earlier choice silently."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QTableWidget, QVBoxLayout, QWidget)

from ..fault_relations import (LIBRARY, Method, RelationError, check_new_method, methods_from_json, methods_to_json)
from ..fault_sgr import vshale_notice, wells_with_vshale
from ..fault_transmissibility import FaultTransmissibility, TransmissibilityError
from ..project import ProjectError

CHOOSE, OPEN, FIXED, CALC, SGR = ("- choose -", "open (no multiplier)", "fixed multiplier", "calculated (slab, my kf)",
                                  "calculated (SGR from VSHALE)")
BY_THICKNESS, BY_THROW = "thickness (m)", "thickness / throw"
MODE_TEXT = {"open": OPEN, "fixed": FIXED}


def builtin_label(key):
    return f"{key} (built-in)"


def own_label(name):
    return f"{name} (mine)"


class FaultTransmissibilityDialog(QDialog):
    def __init__(self, project, parent: QWidget | None = None, pick_open=None, pick_save=None):
        super().__init__(parent)
        self.project = project
        self.pick_open = pick_open or (lambda title: QFileDialog.getOpenFileName(self, title, "", "SGR methods (*.json)")[0])
        self.pick_save = pick_save or (lambda title: QFileDialog.getSaveFileName(self, title, "sgr_methods.json",
                                                                                 "SGR methods (*.json)")[0])
        self.setWindowTitle("Fault transmissibility")
        self.setMinimumSize(900, 420)
        lay = QVBoxLayout(self)
        note = QLabel(
            "Every fault needs a choice. 'open' passes flow as if there were no fault rock; 'fixed' multiplies the "
            "transmissibility across the fault by your number (0 = sealing). Faults with visible throw can also be "
            "'calculated': a slab of fault rock (permeability and thickness you give) in series with the matrix, "
            "which needs cell permeability and so waits for the property step. 'calculated (SGR from VSHALE)' takes the "
            "fault-rock permeability from the shale gouge ratio of each cell pair, using the built-in published relation or "
            "one of your own methods (an expression for log10 kf in SGR and D; kf in mD, D = throw in m); it needs "
            "VSHALE and permeability on the grid, so it also waits for the property step. Throw is known after a "
            "grid build.")
        note.setWordWrap(True)
        lay.addWidget(note)
        msg = vshale_notice(project)
        self.vshale = QLabel(msg or ("VSHALE curve found in: " + ", ".join(wells_with_vshale(project)) if wells_with_vshale(project)
                                     else "VSHALE: no faults to calculate for."))
        self.vshale.setWordWrap(True)
        if msg:
            self.vshale.setStyleSheet("color: #b36b00")
        lay.addWidget(self.vshale)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Fault", "Throw", "Treatment", "Multiplier / kf (mD)", "Thickness basis",
                                              "Thickness value", "SGR method", "Min throw (m)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 160)
        self.table.setColumnWidth(1, 190)
        self.table.setColumnWidth(2, 160)
        self.table.setColumnWidth(3, 140)
        self.table.setColumnWidth(4, 130)
        lay.addWidget(self.table, 1)
        self.relation_note = QLabel("")
        self.relation_note.setWordWrap(True)
        lay.addWidget(self.relation_note)
        lay.addWidget(QLabel("My SGR methods: log10 kf as an expression in SGR (fraction 0-1) and D (throw, m); allowed: "
                             "numbers, + - * / **, parentheses, log10 ln exp sqrt abs min max. Example: "
                             "-4*SGR - 0.25*log10(D)*(1 - SGR)**5"))
        form = QHBoxLayout()
        self.m_name, self.m_expr = QLineEdit(), QLineEdit()
        self.m_name.setPlaceholderText("name")
        self.m_expr.setPlaceholderText("log10 kf = ...")
        form.addWidget(self.m_name, 1)
        form.addWidget(self.m_expr, 3)
        lay.addLayout(form)
        form = QHBoxLayout()
        self.m_cite, self.m_note = QLineEdit(), QLineEdit()
        self.m_cite.setPlaceholderText("source (required): paper, report or calibration")
        self.m_note.setPlaceholderText("note: basin, valid range, caveats (optional)")
        form.addWidget(self.m_cite, 1)
        form.addWidget(self.m_note, 1)
        lay.addLayout(form)
        form = QHBoxLayout()
        self.m_test_btn, self.m_add_btn = QPushButton("Test"), QPushButton("Add / replace method")
        self.m_list = QComboBox()
        self.m_load_btn, self.m_remove_btn = QPushButton("Load into form"), QPushButton("Remove")
        self.m_export_btn, self.m_import_btn = QPushButton("Export methods..."), QPushButton("Import methods...")
        for w in (self.m_test_btn, self.m_add_btn, self.m_list, self.m_load_btn, self.m_remove_btn, self.m_export_btn,
                  self.m_import_btn):
            form.addWidget(w)
        lay.addLayout(form)
        self.m_report = QPlainTextEdit()
        self.m_report.setReadOnly(True)
        self.m_report.setMaximumHeight(110)
        lay.addWidget(self.m_report)
        self.m_test_btn.clicked.connect(self.test_method)
        self.m_add_btn.clicked.connect(self.add_method)
        self.m_load_btn.clicked.connect(self.load_method)
        self.m_remove_btn.clicked.connect(self.remove_method)
        self.m_export_btn.clicked.connect(self.export_methods)
        self.m_import_btn.clicked.connect(self.import_methods)
        self.problem = QLabel("")
        self.problem.setStyleSheet("color: #b00020")
        self.problem.setWordWrap(True)
        lay.addWidget(self.problem)
        row = QHBoxLayout()
        self.save_btn = QPushButton("Save choices")
        self.save_btn.clicked.connect(self.save)
        row.addWidget(self.save_btn)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)
        self.rows: dict[str, dict] = {}
        self._copies: dict = {}
        self._fill()
        self._refresh_methods()

    def _throw_text(self, name):
        seen = self.project.fault_throw_seen.get(name)
        return "not built yet: unknown" if seen is None else ("visible" if seen else "negligible")

    def _fill(self):
        names = list(self.project.faults)
        self.table.setRowCount(len(names))
        for r, name in enumerate(names):
            seen = self.project.fault_throw_seen.get(name)
            self.table.setCellWidget(r, 0, QLabel(name))
            self.table.setCellWidget(r, 1, QLabel(self._throw_text(name)))
            mode = QComboBox()
            mode.addItems([CHOOSE, OPEN, FIXED] + ([CALC, SGR] if seen is not False else []))
            val = QLineEdit()
            basis = QComboBox()
            basis.addItems([BY_THICKNESS, BY_THROW])
            thick = QLineEdit()
            rel = QComboBox()
            rel.addItems(self._method_labels())
            minthr = QLineEdit()
            for c, w in ((2, mode), (3, val), (4, basis), (5, thick), (6, rel), (7, minthr)):
                self.table.setCellWidget(r, c, w)
            ent = dict(mode=mode, val=val, basis=basis, thick=thick, rel=rel, minthr=minthr)
            self.rows[name] = ent
            cur = self.project.fault_transmissibility.get(name)
            if cur is not None:
                if cur.mode == "calculated":
                    mode.setCurrentText(SGR if cur.method == "sgr" else CALC)
                else:
                    mode.setCurrentText(MODE_TEXT[cur.mode])
                if cur.mode == "fixed":
                    val.setText(f"{cur.multiplier:g}")
                elif cur.mode == "calculated":
                    basis.setCurrentText(BY_THICKNESS if cur.thickness is not None else BY_THROW)
                    thick.setText(f"{cur.thickness if cur.thickness is not None else cur.thickness_per_throw:g}")
                    if cur.method == "slab":
                        val.setText(f"{cur.kf:g}")
                    else:
                        minthr.setText(f"{cur.min_throw:g}")
                        if cur.relation:
                            rel.setCurrentText(builtin_label(cur.relation))
                        else:
                            self._select_own(rel, cur)
            mode.currentTextChanged.connect(lambda _t, n=name: self._enable(n))
            rel.currentTextChanged.connect(lambda _t, n=name: self._enable(n))
            self._enable(name)

    def _enable(self, name):
        e = self.rows[name]
        m = e["mode"].currentText()
        e["val"].setEnabled(m in (FIXED, CALC))
        e["val"].setPlaceholderText({FIXED: "multiplier >= 0", CALC: "kf in mD"}.get(m, ""))
        e["basis"].setEnabled(m in (CALC, SGR))
        e["thick"].setEnabled(m in (CALC, SGR))
        e["rel"].setEnabled(m == SGR)
        e["minthr"].setEnabled(m == SGR)
        if m == SGR:
            self._show_method(e["rel"].currentText())

    def _method_labels(self):
        return [CHOOSE, *(builtin_label(k) for k in LIBRARY), *(own_label(n) for n in self.project.sgr_methods)]

    def _select_own(self, combo, cur):
        """Show a stored choice's own expression: the method it came from if that still says the same, else a
        read-only entry for the copy the choice holds."""
        label = own_label(cur.expression_name)
        sm = self.project.sgr_methods.get(cur.expression_name)
        if sm is None or sm.expression != cur.expression:
            label = f"{cur.expression_name} (saved copy: {cur.expression.strip()})"
            if combo.findText(label) < 0:
                combo.addItem(label)
            self._copies[label] = cur
        combo.setCurrentText(label)

    def _show_method(self, label):
        if label.endswith(" (built-in)"):
            lib = LIBRARY[label[:-len(" (built-in)")]]
            self.relation_note.setText(f"{lib.citation}\nlog10 kf = {lib.expression}\nNote: {lib.note}")
        elif label.endswith(" (mine)"):
            m = self.project.sgr_methods.get(label[:-len(" (mine)")])
            if m is not None:
                self.relation_note.setText(f"Your method {m.name!r}: log10 kf = {m.expression}\nSource: {m.citation}"
                                           + (f"\nNote: {m.note}" if m.note else ""))
        elif label in self._copies:
            c = self._copies[label]
            self.relation_note.setText(f"This fault keeps its own saved copy of method {c.expression_name!r} "
                                       f"(log10 kf = {c.expression}); the stored method has since changed or been removed.")
        else:
            self.relation_note.setText("")

    # -- my methods ----------------------------------------------------------------------------------------------
    def _refresh_methods(self):
        self.m_list.clear()
        self.m_list.addItems(list(self.project.sgr_methods))
        for e in self.rows.values():
            cur = e["rel"].currentText()
            e["rel"].blockSignals(True)
            e["rel"].clear()
            e["rel"].addItems(self._method_labels() + list(self._copies))
            e["rel"].setCurrentText(cur if e["rel"].findText(cur) >= 0 else CHOOSE)
            e["rel"].blockSignals(False)

    def _form_method(self):
        try:
            m = Method(self.m_name.text().strip(), self.m_expr.text().strip(), self.m_cite.text().strip(), self.m_note.text().strip())
            return m
        except RelationError as exc:
            self.m_report.setPlainText(str(exc))
            return None

    def test_method(self):
        m = self._form_method()
        if m is None:
            return
        sg = [0.1, 0.3, 0.5, 0.7, 0.9]
        lines = [f"log10 kf = {m.expression}", "kf (mD) at SGR:   " + "  ".join(f"{x:>9g}" for x in sg)]
        for d in (5.0, 20.0, 100.0):
            lines.append(f"D = {d:>5g} m:       " + "  ".join(f"{10 ** float(m.log10_kf(x, d)):>9.3g}" for x in sg))
        self.m_report.setPlainText("\n".join(lines))

    def add_method(self) -> bool:
        m = self._form_method()
        if m is None:
            return False
        try:
            self.project.add_sgr_method(m)
        except ProjectError as exc:
            self.m_report.setPlainText(str(exc))
            return False
        self._refresh_methods()
        self.m_report.setPlainText(f"Method {m.name!r} stored in this project. A fault's choice keeps its own copy of "
                                   "the expression, so changing a method later does not change choices already saved.")
        return True

    def load_method(self):
        m = self.project.sgr_methods.get(self.m_list.currentText())
        if m is not None:
            self.m_name.setText(m.name)
            self.m_expr.setText(m.expression)
            self.m_cite.setText(m.citation)
            self.m_note.setText(m.note)

    def remove_method(self):
        name = self.m_list.currentText()
        if name in self.project.sgr_methods:
            self.project.remove_sgr_method(name)
            self._refresh_methods()
            self.m_report.setPlainText(f"Method {name!r} removed from the list. Faults that already use it keep their copy.")

    def export_methods(self):
        if not self.project.sgr_methods:
            self.m_report.setPlainText("There are no methods of your own to export.")
            return
        path = self.pick_save("Export my SGR methods")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(methods_to_json(self.project.sgr_methods))
            self.m_report.setPlainText(f"Exported {len(self.project.sgr_methods)} method(s) to {path}")

    def import_methods(self):
        path = self.pick_open("Import SGR methods")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                new = methods_from_json(f.read())
        except (OSError, RelationError, UnicodeDecodeError) as exc:
            self.m_report.setPlainText(f"Could not import: {exc}")
            return
        replaced = [n for n in new if n in self.project.sgr_methods]
        for m in new.values():
            self.project.add_sgr_method(m)
        self._refresh_methods()
        self.m_report.setPlainText(f"Imported {len(new)} method(s): {', '.join(new)}"
                                   + (f". Replaced existing: {', '.join(replaced)}" if replaced else ""))

    def _number(self, text, what):
        try:
            return float(text)
        except ValueError:
            raise TransmissibilityError(f"{what}: enter a number") from None

    def save(self) -> bool:
        """Store every row that has a choice; report the first row that cannot be stored. Returns True when
        every fault has a valid stored choice."""
        self.problem.setText("")
        todo, bad = {}, None
        for name, e in self.rows.items():
            m = e["mode"].currentText()
            try:
                if m == CHOOSE:
                    raise TransmissibilityError("no choice made")
                if m == OPEN:
                    todo[name] = FaultTransmissibility("open")
                elif m == FIXED:
                    todo[name] = FaultTransmissibility("fixed", multiplier=self._number(e["val"].text(), "multiplier"))
                else:
                    t = self._number(e["thick"].text(), "thickness value")
                    kw = dict(thickness=t) if e["basis"].currentText() == BY_THICKNESS else dict(thickness_per_throw=t)
                    if m == CALC:
                        kf = self._number(e["val"].text(), "fault-rock permeability")
                        todo[name] = FaultTransmissibility("calculated", method="slab", kf=kf, **kw)
                    else:
                        rel = e["rel"].currentText()
                        if rel == CHOOSE:
                            raise TransmissibilityError("choose an SGR method (the built-in relation or one of yours)")
                        mt = self._number(e["minthr"].text(), "minimum throw")
                        if rel.endswith(" (built-in)"):
                            todo[name] = FaultTransmissibility("calculated", method="sgr", relation=rel[:-len(" (built-in)")],
                                                               min_throw=mt, **kw)
                        elif rel in self._copies:
                            c = self._copies[rel]
                            todo[name] = FaultTransmissibility("calculated", method="sgr", expression=c.expression,
                                                               expression_name=c.expression_name, min_throw=mt, **kw)
                        else:
                            sm = self.project.sgr_methods[rel[:-len(" (mine)")]]
                            todo[name] = FaultTransmissibility("calculated", method="sgr", expression=sm.expression,
                                                               expression_name=sm.name, min_throw=mt, **kw)
            except TransmissibilityError as exc:
                if bad is None:
                    bad = f"{name}: {exc}"
        for name, choice in todo.items():
            try:
                self.project.set_fault_transmissibility(name, choice)
            except ProjectError as exc:
                bad = bad or f"{name}: {exc}"
        if bad:
            self.problem.setText(bad)
            return False
        return True

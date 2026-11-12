"""Fault ages and interactions. Opens by force when a project has two or more faults whose interactions are
not settled. Nothing is stored until the user presses 'Find interactions' (depths, reach) or picks an answer."""

from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..fault_interactions import SAME_AGE, UNRELATED, age_order, InteractionError, unconfirmed
from ..project import ProjectError

CHOOSE = "- choose -"


class FaultInteractionDialog(QDialog):
    def __init__(self, project, parent: QWidget | None = None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Fault ages and interactions")
        self.setMinimumSize(820, 560)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "Choose the depth at which each (vertical) fault trace is taken and how close a fault end must come to "
            "another fault to count as meeting it. Proposed depths are mid-stick; they count only once you press "
            "'Find interactions'. Geometry cannot say which of two crossing faults is older: you decide."))
        lay.itemAt(0).widget().setWordWrap(True)
        grid = QGridLayout()
        lay.addLayout(grid)
        self.depth_edits: dict[str, QLineEdit] = {}
        for r, (name, f) in enumerate(project.faults.items()):
            grid.addWidget(QLabel(name), r, 0)
            z0, z1 = f.bounds()[4:]
            e = QLineEdit(f"{project.fault_depths.get(name, (z0 + z1) / 2):g}")
            grid.addWidget(e, r, 1)
            grid.addWidget(QLabel(f"trace depth (m); sticks span {z0:.0f}-{z1:.0f} m"), r, 2)
            self.depth_edits[name] = e
        row = QHBoxLayout()
        row.addWidget(QLabel("Reach (m)"))
        self.reach = QLineEdit("" if project.fault_reach is None else f"{project.fault_reach:g}")
        self.reach.setPlaceholderText("e.g. one cell diagonal of the intended grid")
        row.addWidget(self.reach)
        b = QPushButton("Find interactions")
        b.clicked.connect(self.find)
        row.addWidget(b)
        lay.addLayout(row)
        self.problem = QLabel("")
        self.problem.setStyleSheet("color: #b00020")
        self.problem.setWordWrap(True)
        lay.addWidget(self.problem)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Interaction", "Geometry suggests", "Which is older?", "Status"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 330)
        self.table.setColumnWidth(1, 150)
        self.table.setColumnWidth(2, 180)
        lay.addWidget(self.table, 1)
        row2 = QHBoxLayout()
        self.use_btn = QPushButton("Accept all suggestions")
        self.use_btn.clicked.connect(self.use_suggestions)
        row2.addWidget(self.use_btn)
        row2.addStretch(1)
        self.status = QLabel("")
        row2.addWidget(self.status)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        row2.addWidget(self.close_btn)
        lay.addLayout(row2)
        self.combos: list[QComboBox] = []
        self._fill()

    # -- actions ------------------------------------------------------------------------------
    def find(self):
        self.problem.setText("")
        try:
            reach = float(self.reach.text())
            depths = {n: float(e.text()) for n, e in self.depth_edits.items()}
        except ValueError:
            self.problem.setText("Every depth and the reach must be numbers.")
            return
        try:
            if not reach > 0:
                raise ProjectError("the reach must be a positive distance")
            for n, d in depths.items():
                self.project.set_fault_depth(n, d)
            _, dropped, problems = self.project.refresh_fault_interactions(reach)
        except (ProjectError, InteractionError, ValueError) as exc:
            self.problem.setText(str(exc))
            return
        notes = list(problems) + [f"Earlier answer dropped, the faults no longer meet that way: {i.describe()}" for i in dropped]
        self.problem.setText("\n".join(notes))
        self._fill()

    def choose(self, row: int, text: str):
        if text == CHOOSE:
            return
        i = self.project.fault_interactions[row]
        answer = {"same age": SAME_AGE, "unrelated": UNRELATED}.get(text)
        if answer is None:
            answer = text.replace(" is older", "")
        try:
            self.project.confirm_fault_interaction(i.key, answer)
        except (ProjectError, InteractionError) as exc:
            self.problem.setText(str(exc))
        self._refresh_status()

    def use_suggestions(self):
        for i in list(self.project.fault_interactions):
            if i.suggestion is not None and not i.confirmed:
                self.project.confirm_fault_interaction(i.key, i.suggestion)
        self._fill()

    # -- display ----------------------------------------------------------------------------------
    def _answer_text(self, i, answer):
        if answer in (SAME_AGE, UNRELATED):
            return answer.replace("_", " ")
        return f"{answer} is older"

    def _fill(self):
        ints = self.project.fault_interactions
        self.table.setRowCount(len(ints))
        self.combos = []
        for r, i in enumerate(ints):
            it = QTableWidgetItem(i.describe() + (f" ({i.note})" if i.note and i.kind == "crossing" else ""))
            self.table.setItem(r, 0, it)
            sug = "geometry cannot say" if i.suggestion is None else self._answer_text(i, i.suggestion)
            self.table.setItem(r, 1, QTableWidgetItem(sug))
            c = QComboBox()
            c.addItem(CHOOSE)
            for a in i.valid_answers():
                c.addItem(self._answer_text(i, a))
            if i.answer is not None:
                c.setCurrentText(self._answer_text(i, i.answer))
            c.currentTextChanged.connect(lambda text, row=r: self.choose(row, text))
            self.table.setCellWidget(r, 2, c)
            self.combos.append(c)
        self._refresh_status()

    def _refresh_status(self):
        pr = self.project
        ints = pr.fault_interactions
        for r, i in enumerate(ints):
            cur = pr.fault_interactions[r]
            self.table.setItem(r, 3, QTableWidgetItem("confirmed" if cur.confirmed else "NOT CONFIRMED"))
        self.use_btn.setEnabled(any(i.suggestion is not None and not i.confirmed for i in ints))
        if len(pr.faults) < 2:
            self.status.setText("Fewer than two faults: nothing to settle.")
        elif pr.fault_reach is None:
            self.status.setText("Interactions not checked yet.")
        elif unconfirmed(ints):
            self.status.setText(f"{len(unconfirmed(ints))} interaction(s) still need your answer; the grid cannot be built until they have one.")
        elif not ints:
            self.status.setText("No faults meet: no ages needed.")
        else:
            try:
                self.status.setText("Oldest to youngest: " + ", ".join(age_order(list(pr.faults), ints)))
            except InteractionError as exc:
                self.status.setText(str(exc))

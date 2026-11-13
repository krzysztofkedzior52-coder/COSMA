"""Editor for the stratigraphic chart: rename and reorder intervals, set contacts, add/remove."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QComboBox, QDialog, QDialogButtonBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QRadioButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..chart_edit import (Row, apply_chart_edit, check_hand_chart, parse_interval_text, pick_names, rows_from_chart,
                          validate, wells_with_tops)
from ..project import Project, ProjectError
from ..strat import CONTACTS


class ChartEditorDialog(QDialog):
    """Intervals top to bottom. The horizon between two intervals is one object: "Base of A /
    Top of B" - the column shows it, so it is clear that one surface serves both."""

    def __init__(self, project: Project, parent: QWidget | None = None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Stratigraphic chart")
        self.setMinimumSize(760, 420)
        self.rows = rows_from_chart(project.chart)
        self.base_name = project.chart.base_name
        self._building = False
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Intervals from top to bottom. An interval is named after the pick at its top; "
                             "renaming it renames that pick in every well (the old name is remembered)."))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "Interval", "Contact at its top", "Top horizon (shared with the interval above)", "Attached"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.itemChanged.connect(self._item_changed)
        lay.addWidget(self.table)
        bar = QHBoxLayout()
        self.up_btn, self.down_btn = QPushButton("Move up"), QPushButton("Move down")
        self.add_btn, self.remove_btn = QPushButton("Add interval below"), QPushButton("Remove interval")
        for b in (self.up_btn, self.down_btn, self.add_btn, self.remove_btn):
            bar.addWidget(b)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.up_btn.clicked.connect(lambda: self.move(self.selected(), -1))
        self.down_btn.clicked.connect(lambda: self.move(self.selected(), +1))
        self.add_btn.clicked.connect(lambda: self.add(self.selected()))
        self.remove_btn.clicked.connect(lambda: self.remove(self.selected()))
        self.problem = QLabel("")
        self.problem.setWordWrap(True)
        self.problem.setStyleSheet("color: #b00020;")
        lay.addWidget(self.problem)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self._rebuild()

    # -- state ------------------------------------------------------------------------------------------------
    def selected(self) -> int:
        r = self.table.currentRow()
        return r if 0 <= r < len(self.rows) else len(self.rows) - 1

    def _uses(self, row: Row) -> list:
        return self.project.horizon_uses(row.uid) if row.uid else []

    def move(self, i: int, delta: int):
        j = i + delta
        if 0 <= i < len(self.rows) and 0 <= j < len(self.rows):
            self.rows[i], self.rows[j] = self.rows[j], self.rows[i]
            self._rebuild(select=j)

    def add(self, after: int):
        n = 1
        names = {r.name.casefold() for r in self.rows} | {self.base_name.casefold()}
        while f"New interval {n}".casefold() in names:
            n += 1
        self.rows.insert(after + 1, Row(None, f"New interval {n}"))
        self._rebuild(select=after + 1)

    def remove(self, i: int):
        if not 0 <= i < len(self.rows):
            return
        uses = self._uses(self.rows[i])
        if uses:
            self.problem.setText(f"Cannot remove {self.rows[i].name!r}: {', '.join(uses)} attached to its top.")
            return
        if len(self.rows) == 1:
            self.problem.setText("A chart needs at least one interval.")
            return
        del self.rows[i]
        self._rebuild(select=min(i, len(self.rows) - 1))

    # -- table -----------------------------------------------------------------------------------------------------
    def _rebuild(self, select: int | None = None):
        self._building = True
        t = self.table
        t.setRowCount(len(self.rows) + 1)
        for r, row in enumerate(self.rows):
            num = QTableWidgetItem(str(r + 1))
            num.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            t.setItem(r, 0, num)
            nm = QTableWidgetItem(row.name)
            nm.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
            t.setItem(r, 1, nm)
            combo = QComboBox()
            combo.addItems(list(CONTACTS))
            combo.setCurrentText(row.contact)
            combo.currentTextChanged.connect(lambda txt, rr=r: self._contact_changed(rr, txt))
            t.setCellWidget(r, 2, combo)
            above = self.rows[r - 1].name if r else None
            label = f"Base of {above} / Top of {row.name}" if above else f"Top of {row.name}"
            hz = QTableWidgetItem(label)
            hz.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            t.setItem(r, 3, hz)
            uses = self._uses(row)
            at = QTableWidgetItem(", ".join(uses) if uses else ("new" if row.uid is None else "-"))
            at.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            t.setItem(r, 4, at)
        b = len(self.rows)
        for c, text in ((0, ""), (3, f"Base of {self.rows[-1].name} (base horizon)"), (4, "")):
            it = QTableWidgetItem(text)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            t.setItem(b, c, it)
        bn = QTableWidgetItem(self.base_name)
        bn.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
        f = bn.font()
        f.setItalic(True)
        bn.setFont(f)
        t.setItem(b, 1, bn)
        t.setCellWidget(b, 2, QLabel(""))
        self._building = False
        if select is not None:
            t.selectRow(select)
            t.setCurrentCell(select, 1)
        self._validate()

    def _contact_changed(self, r, text):
        if not self._building and r < len(self.rows):
            self.rows[r].contact = text
            self._validate()

    def _item_changed(self, item):
        if self._building:
            return
        r = item.row()
        if item.column() == 1:
            if r < len(self.rows):
                self.rows[r].name = item.text()
            elif r == len(self.rows):
                self.base_name = item.text()
            self._building = True
            # keep the read-only horizon labels in step with the names
            for k in range(len(self.rows)):
                above = self.rows[k - 1].name if k else None
                self.table.item(k, 3).setText(f"Base of {above} / Top of {self.rows[k].name}" if above else f"Top of {self.rows[k].name}")
            self.table.item(len(self.rows), 3).setText(f"Base of {self.rows[-1].name} (base horizon)")
            self._building = False
            self._validate()

    def _validate(self):
        problems = validate(self.rows, self.base_name)
        self.problem.setText("; ".join(problems))
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not problems)
        return problems

    def accept(self):
        try:
            apply_chart_edit(self.project, self.rows, self.base_name)
        except ProjectError as exc:
            self.problem.setText(str(exc))
            return
        super().accept()


def edit_chart_with_dialog(project: Project, parent: QWidget | None = None) -> bool:
    app = QApplication.instance() or QApplication([])      # noqa: F841
    return ChartEditorDialog(project, parent).exec() == QDialog.Accepted


class ChartStartDialog(QDialog):
    """The project has no stratigraphic chart yet. Either build it from the tops of one well, or type the intervals in.
    Nothing is preselected: how the chart is made is the user's choice."""

    def __init__(self, project: Project, parent: QWidget | None = None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Define the stratigraphic chart")
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Surfaces, property maps and variograms attach to the intervals of the stratigraphic chart, so it "
                             "has to exist first. You can change it afterwards (rename, reorder, add)."))
        self.from_well = QRadioButton("Build it from the tops of a well")
        self.by_hand = QRadioButton("Type the intervals in")
        grp = QButtonGroup(self)
        for b in (self.from_well, self.by_hand):
            grp.addButton(b)
        wells = wells_with_tops(project)
        lay.addWidget(self.from_well)
        self.well_combo = QComboBox()
        for n, k in wells.items():
            self.well_combo.addItem(f"{n} ({k} tops)", n)
        self.well_note = QLabel("The picks are ordered by depth: each pick opens an interval, the deepest one is the base. "
                                "The well becomes the project's reference well for the chart."
                                if wells else "No well has two or more tops yet. Import tops first (Import > Formation tops...), "
                                              "or type the intervals in.")
        self.well_note.setWordWrap(True)
        lay.addWidget(self.well_combo)
        lay.addWidget(self.well_note)
        self.from_well.setEnabled(bool(wells))
        self.well_combo.setEnabled(False)
        lay.addWidget(self.by_hand)
        self.names = QPlainTextEdit()
        self.names.setPlaceholderText("One interval per line, from the top down.\nA name is the pick at the top of the interval.")
        self.names.setFixedHeight(100)
        self.base = QLineEdit()
        self.base.setPlaceholderText("Name of the base (the pick below the lowest interval)")
        hint = pick_names(project)
        self.hint = QLabel(("Pick names already in the wells: " + ", ".join(hint) + ". Use the same names so the picks are recognised.")
                           if hint else "")
        self.hint.setWordWrap(True)
        for w in (self.names, self.base, self.hint):
            lay.addWidget(w)
        self.names.setEnabled(False)
        self.base.setEnabled(False)
        self.problem = QLabel("")
        self.problem.setWordWrap(True)
        self.problem.setStyleSheet("color: #b00020;")
        lay.addWidget(self.problem)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        for w in (self.from_well, self.by_hand):
            w.toggled.connect(self._refresh)
        self.well_combo.currentIndexChanged.connect(self._refresh)
        self.names.textChanged.connect(self._refresh)
        self.base.textChanged.connect(self._refresh)
        self._refresh()

    def how(self):
        if self.from_well.isChecked() and self.well_combo.currentData():
            return {"mode": "well", "well": self.well_combo.currentData()}
        if self.by_hand.isChecked():
            return {"mode": "hand", "intervals": parse_interval_text(self.names.toPlainText()), "base": self.base.text().strip()}
        return None

    def _refresh(self, *_):
        fw, bh = self.from_well.isChecked(), self.by_hand.isChecked()
        self.well_combo.setEnabled(fw)
        for w in (self.names, self.base):
            w.setEnabled(bh)
        problems = []
        if not (fw or bh):
            problems.append("choose how the chart is made")
        elif bh:
            problems = check_hand_chart(parse_interval_text(self.names.toPlainText()), self.base.text())
        self.problem.setText("; ".join(problems))
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not problems)


def start_chart_with_dialog(project: Project, parent: QWidget | None = None):
    """The answer {"mode": ...} for chart_edit.start_chart, or None if cancelled."""
    app = QApplication.instance() or QApplication([])      # noqa: F841
    dlg = ChartStartDialog(project, parent)
    return dlg.how() if dlg.exec() == QDialog.Accepted else None

"""New-project flow: ask for a type well, import it first, build and confirm the chart."""

from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (QApplication, QButtonGroup, QDialog, QDialogButtonBox, QFileDialog, QLabel, QLineEdit,
                               QRadioButton, QVBoxLayout, QWidget)

from ..imports import ImportCancelled, import_file
from ..project import Project, ProjectError


class NewProjectDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("New project")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Project name:"))
        self.name = QLineEdit()
        self.name.textChanged.connect(self._check)
        lay.addWidget(self.name)
        lay.addWidget(QLabel("Do you have a type well for this project?"))
        self.yes = QRadioButton("Yes - import it first; its tops define the stratigraphic chart")
        self.no = QRadioButton("No - I will define the stratigraphy later")
        grp = QButtonGroup(self)
        for b in (self.yes, self.no):
            grp.addButton(b)
            lay.addWidget(b)
            b.toggled.connect(self._check)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self._check()

    def _check(self, *_):          # an answer to the type-well question is required, not defaulted
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(bool(self.name.text().strip()) and (self.yes.isChecked() or self.no.isChecked()))

    def result_tuple(self):
        return self.name.text().strip(), self.yes.isChecked()


def ask_new_project(parent: QWidget | None = None):
    app = QApplication.instance() or QApplication([])      # noqa: F841
    d = NewProjectDialog(parent)
    return d.result_tuple() if d.exec() == QDialog.Accepted else None


_FILTERS = {"well_headers": "Well headers (*.csv *.txt)", "deviation": "Deviation surveys (*.csv *.txt)",
            "tops": "Formation tops (*.csv *.txt)"}


def qt_pick_file(parent: QWidget | None = None) -> Callable:
    def pick(kind: str, title: str):
        path, _ = QFileDialog.getOpenFileName(parent, title, "", _FILTERS.get(kind, "All files (*)"))
        return path or None
    return pick


def new_project_workflow(pick_file: Callable, ask: Callable, confirm: Callable, choose_well: Callable,
                         edit_chart: Callable):
    """Create a project.

    pick_file(kind, title) -> path or None; ask() -> (name, has_type_well) or None;
    confirm(preview) -> values or None (the import dialog); choose_well(names) -> name;
    edit_chart(project) -> bool (the chart editor).

    With a type well: headers -> deviation survey (optional; without it only picks in one depth
    type can be ordered) -> tops -> chart built from the tops, shown in the editor. A cancelled
    editor keeps the chart as built; the project is only abandoned if a required import is
    cancelled. Returns the Project, or None if the user backed out.
    """
    answer = ask()
    if answer is None:
        return None
    name, has_type_well = answer
    pr = Project(name)
    if not has_type_well:
        return pr
    for kind, title, required in (("well_headers", "Type well: header", True),
                                  ("deviation", "Type well: deviation survey (cancel to skip)", False),
                                  ("tops", "Type well: formation tops", True)):
        path = pick_file(kind, title)
        if path is None:
            if required:
                return None
            continue
        try:
            import_file(kind, path, confirm, project=pr)
        except ImportCancelled:
            if required:
                return None
    with_tops = [n for n, r in pr.wells.items() if len(r.tops) >= 2]
    if not with_tops:
        raise ProjectError("the type well needs at least two tops to build a stratigraphic chart")
    well = with_tops[0] if len(with_tops) == 1 else choose_well(with_tops)
    pr.set_type_well(well)
    pr.build_chart()
    edit_chart(pr)
    return pr

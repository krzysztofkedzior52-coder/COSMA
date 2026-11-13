"""Qt confirmation dialog for any `ImportPreview` (PySide6)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..imports import (ImportPreview, ImportRejected, advice_for, conflicts_for, finalize, placement_for)
from ..project import Action, WellLink

_TAG = {"file": "from file", "default": "default - please check", "required": "required"}
_PLACEHOLDER = "- choose -"
_NEW = "New well"
_ACTIONS = (("replace", "Replace existing"), ("skip", "Skip (keep existing)"), ("suffix", "Import with suffix"))
_DEFAULT_SUFFIX = "_new"


def _table(headers, stretch_last=True):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(stretch_last)
    return t


def _ro(text):
    it = QTableWidgetItem(text)
    it.setFlags(Qt.ItemIsEnabled)
    it.setToolTip(text)
    return it


class ImportDialog(QDialog):
    """Shows what was read from the file and the parameters to confirm. OK stays disabled
    until every required parameter has a value."""

    def __init__(self, preview: ImportPreview, parent: QWidget | None = None):
        super().__init__(parent)
        self.preview = preview
        self.setWindowTitle(f"Confirm import - {preview.path.name}")
        self.setMinimumWidth(680)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"<b>{preview.path.name}</b> ({preview.kind})"))

        lay.addWidget(QLabel("Read from the file:"))
        self.table = QTableWidget(len(preview.found), 2)
        self.table.setHorizontalHeaderLabels(["Item", "Value"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for r, (k, v) in enumerate(preview.found):
            self.table.setItem(r, 0, QTableWidgetItem(k))
            self.table.setItem(r, 1, QTableWidgetItem(v))
        self.table.setMinimumHeight(min(260, 30 + 26 * len(preview.found)))
        lay.addWidget(self.table)

        # -- well connection ---------------------------------------------------------------------------
        self.link_rows = {}
        if preview.file_wells:
            lay.addWidget(QLabel("Connect each well in the file to an existing well, or import it as a new well:"))
            self.link_table = _table(["Well in file", "Connect to", "Name of new well", "Note"])
            self.link_table.setRowCount(len(preview.file_wells))
            for r, fw in enumerate(preview.file_wells):
                sug, note = preview.links[fw]
                self.link_table.setItem(r, 0, _ro(fw))
                combo = QComboBox()
                combo.addItems([_NEW] + list(preview.existing))
                combo.setCurrentText(sug.name if sug.mode == "existing" else _NEW)
                name = QLineEdit(sug.name if sug.mode == "new" else fw)
                name.setEnabled(sug.mode == "new")
                combo.currentTextChanged.connect(lambda t, n=name: (n.setEnabled(t == _NEW), self._refresh()))
                name.textChanged.connect(self._refresh)
                self.link_table.setCellWidget(r, 1, combo)
                self.link_table.setCellWidget(r, 2, name)
                self.link_table.setItem(r, 3, _ro(note))
                self.link_rows[fw] = (combo, name)
            self.link_table.setMinimumHeight(min(220, 40 + 30 * len(preview.file_wells)))
            lay.addWidget(self.link_table)

        # -- pick names -> stratigraphic chart ------------------------------------------------------------
        self.top_rows = {}
        if preview.top_names:
            lay.addWidget(QLabel("Match each pick in the file to a horizon of the stratigraphic chart "
                                 "(or ignore it):"))
            self.top_table = _table(["Pick in file", "Horizon in chart", "Note"])
            self.top_table.setRowCount(len(preview.top_names))
            for r, n in enumerate(preview.top_names):
                uid, note = preview.top_suggestions[n]
                self.top_table.setItem(r, 0, _ro(n))
                combo = QComboBox()
                self._fill_top_combo(combo, uid)
                combo.currentIndexChanged.connect(self._refresh)
                self.top_table.setCellWidget(r, 1, combo)
                self.top_table.setItem(r, 2, _ro(note))
                self.top_rows[n] = combo
            self.top_table.setMinimumHeight(min(220, 40 + 30 * len(preview.top_names)))
            lay.addWidget(self.top_table)

        # -- parameters --------------------------------------------------------------------------------------------
        lay.addWidget(QLabel("Import parameters (confirm or change):"))
        self.form = QFormLayout()
        self.widgets, self.rows = {}, {}
        self._horizon_touched = False
        self._pl_key, self._pl = None, None
        for p in preview.params:
            if p.kind == "crs":
                w = QComboBox()
                w.setEditable(True)
                w.setInsertPolicy(QComboBox.NoInsert)
                w.lineEdit().setPlaceholderText("choose from the list, or type an EPSG code (e.g. 32616)")
                w.setMinimumWidth(420)
                w.currentTextChanged.connect(self._refresh)
            elif p.kind == "choice":
                w = QComboBox()
                self._fill_choice(w, p, p.value)
                w.currentIndexChanged.connect(self._refresh)
                if p.key == "horizon":
                    w.activated.connect(lambda _i: setattr(self, "_horizon_touched", True))
            else:
                w = QLineEdit("" if p.value is None else str(p.value))
                if p.kind == "number":
                    w.setValidator(QDoubleValidator())
                w.textChanged.connect(self._refresh)
            tip = f"{_TAG[p.source]}" + (f" - {p.help}" if p.help else "")
            w.setToolTip(tip)
            label = QLabel(f"{p.label}  <i>({_TAG[p.source]})</i>")
            label.setToolTip(tip)
            self.form.addRow(label, w)
            self.widgets[p.key], self.rows[p.key] = w, label
        lay.addLayout(self.form)
        self.crs_label = QLabel("")
        self.crs_label.setWordWrap(True)
        lay.addWidget(self.crs_label)
        self._crs_sig = {}
        self._fill_crs_lists()

        self.suggest_label = QLabel("")
        self.suggest_label.setWordWrap(True)
        lay.addWidget(self.suggest_label)
        self.chart_btn = QPushButton("Edit stratigraphic chart...")
        self.chart_btn.clicked.connect(self._edit_chart)
        self.chart_btn.setVisible(bool(preview.project is not None and getattr(preview.project, "chart", None) is not None
                                       and ("horizon" in self.widgets or preview.top_names)))
        lay.addWidget(self.chart_btn, alignment=Qt.AlignLeft)

        # -- conflicts ---------------------------------------------------------------------------------------------
        self.conflict_keys = ()
        self.conflict_rows = {}
        self._choices = {}
        self.conflict_box = QWidget()
        cl = QVBoxLayout(self.conflict_box)
        cl.setContentsMargins(0, 0, 0, 0)
        self.conflict_label = QLabel("")
        self.conflict_label.setWordWrap(True)
        cl.addWidget(self.conflict_label)
        self.all_combo = QComboBox()
        self.all_combo.addItems(["Apply one choice to all conflicts..."] + [t for _, t in _ACTIONS])
        self.all_combo.currentIndexChanged.connect(self._apply_all)
        cl.addWidget(self.all_combo)
        self.conflict_table = _table(["Well", "Item", "Already in project", "In file", "Action", "Suffix"])
        cl.addWidget(self.conflict_table)
        self.conflict_box.setVisible(False)
        lay.addWidget(self.conflict_box)

        self.warn = QLabel("")
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color: #b00020;")
        self.warn.setText("<br>".join(f"&#9888; {w}" for w in preview.warnings))
        lay.addWidget(self.warn)
        self.advice = QLabel("")
        self.advice.setWordWrap(True)
        self.advice.setStyleSheet("color: #b36b00;")
        lay.addWidget(self.advice)
        self.problem = QLabel("")
        self.problem.setWordWrap(True)
        lay.addWidget(self.problem)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Import")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self._refresh()

    # -- combo helpers --------------------------------------------------------------------------------------------
    @staticmethod
    def _fill_choice(w: QComboBox, p, current):
        w.blockSignals(True)
        w.clear()
        if current is None:
            w.addItem(_PLACEHOLDER, None)
        for c in p.choices:
            w.addItem(str(p.choice_labels.get(c, c)), c)
        i = w.findData(current)
        w.setCurrentIndex(i if i >= 0 else 0)
        w.blockSignals(False)

    def _fill_top_combo(self, combo: QComboBox, current):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(_PLACEHOLDER, None)
        for uid, label in self.preview.top_choices.items():
            combo.addItem(label, uid)
        i = combo.findData(current)
        combo.setCurrentIndex(i if i >= 0 else 0)
        combo.blockSignals(False)

    # -- values -------------------------------------------------------------------------------------------------------
    def values(self) -> dict:
        out = {}
        for p in self.preview.params:
            w = self.widgets[p.key]
            out[p.key] = (w.currentText() if p.kind == "crs" else w.currentData()) if isinstance(w, QComboBox) else w.text()
        if self.preview.file_wells:
            out["well_links"] = {}
            for fw, (combo, name) in self.link_rows.items():
                t = combo.currentText()
                out["well_links"][fw] = WellLink("new", name.text().strip()) if t == _NEW else WellLink("existing", t)
        if self.preview.top_names:
            out["top_map"] = {n: c.currentData() for n, c in self.top_rows.items() if c.currentData() is not None}
        if self.preview.file_wells or self.preview.conflict_finder is not None:
            out["conflict_actions"] = self._current_actions()
        return out

    def _current_actions(self) -> dict:
        acts = {}
        for key, (combo, suf) in self.conflict_rows.items():
            mode = combo.currentData()
            if mode:
                acts[key] = Action(mode, suf.text().strip() if mode == "suffix" else "")
        return acts

    # -- surfaces: where does it sit? ------------------------------------------------------------------------------
    def _update_placement(self, vals):
        if "horizon" not in self.widgets:
            return
        key = (vals.get("xy_unit"), vals.get("z_unit"), vals.get("positive"))
        if key != self._pl_key:
            self._pl_key, self._pl = key, placement_for(self.preview, vals)
        pl = self._pl
        combo = self.widgets["horizon"]
        p = next(q for q in self.preview.params if q.key == "horizon")
        if pl is None:
            self.suggest_label.setText("Suggestions for where this surface sits appear once its units are chosen.")
            return
        ev = {s.uid: s for s in pl.candidates}
        combo.blockSignals(True)
        cur = combo.currentData()
        for i in range(combo.count()):
            uid = combo.itemData(i)
            if uid in ev:
                base = p.choice_labels.get(uid, uid)
                combo.setItemText(i, f"{base}  -  {ev[uid].text}")
        if pl.suggested is not None and not self._horizon_touched:
            combo.setCurrentIndex(combo.findData(pl.suggested.uid))
        elif cur is None and not self._horizon_touched:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)
        if pl.suggested is not None:
            self.suggest_label.setText(f"<b>Suggested: {pl.suggested.label}</b> ({pl.suggested.text}). "
                                       "Check it and confirm, or pick another horizon.")
        else:
            self.suggest_label.setText("No horizon fits clearly - choose where this surface sits." +
                                       ("<br>" + "<br>".join(pl.warnings) if pl.warnings else ""))
        self.rows["horizon"].setText(f"{p.label}  <i>({'suggested - check' if pl.suggested and combo.currentData() == pl.suggested.uid and not self._horizon_touched else _TAG['required']})</i>")

    # -- chart editing -------------------------------------------------------------------------------------------------
    def _edit_chart(self):
        from .chart_editor import ChartEditorDialog
        dlg = ChartEditorDialog(self.preview.project, self)
        if dlg.exec() == QDialog.Accepted:
            self._reload_chart()

    def _reload_chart(self):
        chart = self.preview.project.chart
        labels = {h.uid: h.label for h in chart.horizons()}
        if "horizon" in self.widgets:
            p = next(q for q in self.preview.params if q.key == "horizon")
            cur = self.widgets["horizon"].currentData()
            p.choices, p.choice_labels = tuple(labels), labels
            self._fill_choice(self.widgets["horizon"], p, cur if cur in labels else None)
            self._pl_key = None
        if self.preview.top_names:
            self.preview.top_choices = {**labels, "ignore": "Ignore (do not import)"}
            for n, combo in self.top_rows.items():
                cur = combo.currentData()
                self._fill_top_combo(combo, cur if cur in self.preview.top_choices else chart.match_name(n))
        self._refresh()

    # -- conflicts ------------------------------------------------------------------------------------------------------
    def _set_conflicts(self, conflicts):
        keys = tuple(c.key for c in conflicts) if conflicts else ()
        if keys == self.conflict_keys:
            return
        self._choices.update({k: (c.currentData(), s.text()) for k, (c, s) in self.conflict_rows.items()})
        self.conflict_keys = keys
        self.conflict_rows = {}
        t = self.conflict_table
        t.blockSignals(True)
        t.setRowCount(len(keys))
        for r, c in enumerate(conflicts or []):
            for col, text in enumerate([c.well, c.item + ("   [same data]" if c.identical else ""), c.existing, c.new]):
                t.setItem(r, col, _ro(text))
            combo = QComboBox()
            combo.addItem("- choose -", None)
            for mode, label in _ACTIONS:
                if mode in c.allowed:
                    combo.addItem(label, mode)
            suf = QLineEdit()
            mode0, suf0 = self._choices.get(c.key, (None, ""))
            idx = combo.findData(mode0)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            suf.setText(suf0)
            suf.setEnabled(combo.currentData() == "suffix")
            combo.currentIndexChanged.connect(lambda _i, cb=combo, sf=suf: self._on_action(cb, sf))
            suf.textChanged.connect(self._refresh)
            t.setCellWidget(r, 4, combo)
            t.setCellWidget(r, 5, suf)
            self.conflict_rows[c.key] = (combo, suf)
        t.blockSignals(False)
        t.setMinimumHeight(min(240, 40 + 30 * len(keys)))
        n_same = sum(1 for c in (conflicts or []) if c.identical)
        self.conflict_label.setText(
            f"<b>{len(keys)} item(s) already exist in the project.</b> Choose what to do with each"
            + (f" ({n_same} are identical to what is already there)." if n_same else "."))
        self.conflict_box.setVisible(bool(keys))

    def _on_action(self, combo, suf):
        suf.setEnabled(combo.currentData() == "suffix")
        if combo.currentData() == "suffix" and not suf.text().strip():
            suf.setText(_DEFAULT_SUFFIX)
        self._refresh()

    def _apply_all(self, idx):
        if idx <= 0:
            return
        mode = _ACTIONS[idx - 1][0]
        for combo, suf in self.conflict_rows.values():
            i = combo.findData(mode)
            if i >= 0:
                combo.setCurrentIndex(i)
        self.all_combo.blockSignals(True)
        self.all_combo.setCurrentIndex(0)
        self.all_combo.blockSignals(False)

    # -- refresh ---------------------------------------------------------------------------------------------------------
    def _fill_crs_lists(self):
        """The CRS lists: the suggestions for this file first (each with its reason as tooltip), then the built-in catalog.
        A suggestion is put in the box only when the file or the project states it; otherwise the box stays empty."""
        from ..crs_catalog import catalog
        hints = self.preview.crs_hints
        if hints is None or not any(p.kind == "crs" for p in self.preview.params):
            return
        vals = {p.key: self.widgets[p.key].currentText() for p in self.preview.params if p.kind == "crs"}
        h = hints(vals)
        lines = []
        for p in self.preview.params:
            if p.kind != "crs":
                continue
            sugs, note = h.get(p.key, ([], ""))
            sig = (tuple(x.code for x in sugs), note)
            if self._crs_sig.get(p.key) == sig:
                continue
            self._crs_sig[p.key] = sig
            w = self.widgets[p.key]
            keep = w.currentText()
            w.blockSignals(True)
            w.clear()
            w.addItem("")
            for x in sugs:
                w.addItem(f"{x.label}   [suggested]")
                w.setItemData(w.count() - 1, x.reason, Qt.ToolTipRole)
            w.insertSeparator(w.count())
            for e in catalog():
                w.addItem(e.label)
            w.setCurrentText(keep)
            if not keep:
                strong = next((x for x in sugs if x.strong), None)
                if strong is not None:
                    w.setCurrentText(strong.label)
            w.blockSignals(False)
        for p in self.preview.params:
            if p.kind == "crs":
                sugs, note = h.get(p.key, ([], ""))
                for x in sugs:
                    lines.append(f"<b>{p.label}</b> suggestion: {x.label} - {x.reason}")
                if note:
                    lines.append(f"<i>{note}</i>")
        self.crs_label.setText("<br>".join(lines))

    def _refresh(self, *_):
        self._fill_crs_lists()
        vals = self.values()
        self._update_placement(vals)
        vals = self.values()
        self._set_conflicts(conflicts_for(self.preview, vals))
        vals = self.values()
        adv = advice_for(self.preview, vals)
        self.advice.setText("<br>".join(f"&#9888; {m}" for m in adv))
        try:
            finalize(self.preview, vals)
            self.problem.setText("")
            ok = True
        except ImportRejected as exc:
            self.problem.setText(str(exc))
            ok = False
        for p in self.preview.params:      # conditional parameters appear only when relevant
            on = p.active(vals)
            self.widgets[p.key].setVisible(on)
            self.rows[p.key].setVisible(on)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(ok)


def confirm_with_dialog(preview: ImportPreview, parent: QWidget | None = None):
    """`confirm` callable for `cosma.imports.import_file`: modal dialog; None on cancel."""
    app = QApplication.instance() or QApplication([])      # noqa: F841
    dlg = ImportDialog(preview, parent)
    return dlg.values() if dlg.exec() == QDialog.Accepted else None

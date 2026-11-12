"""Variogram editor: experimental variograms with the model drawn over them. A thin view
over `VariogramSession` - all the logic (and its validation) lives there.

Fields that are empty mean "not set": nothing is filled in behind the user's back, except
by the explicit Compute / Suggest azimuth / Auto-fit buttons, whose results are shown.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QComboBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from ..project import Action, ProjectError
from ..variogram import KINDS, TYPES, VariogramError
from ..variogram_session import VariogramSession

NUM = {"major": "Major (along azimuth)", "minor": "Minor (across)", "vertical": "Vertical"}


def _num(text: str):
    text = text.strip()
    if not text:
        return None
    return float(text)


def _fmt(v):
    return "" if v is None else f"{v:.6g}"


class VariogramEditor(QDialog):
    def __init__(self, session: VariogramSession, parent: QWidget | None = None, choose_action=None, label: str = ""):
        super().__init__(parent)
        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.s = session
        self.choose_action = choose_action or self._ask_action
        self.saved = None
        self._busy = False
        self.setWindowTitle(f"Variogram - {session.data.interval} / {session.variable}")
        self.setMinimumSize(1100, 640)
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        root.addLayout(left, 0)

        d = session.data
        info = (f"{len(d.used)} well(s), {len(d.values)} samples; frame '{d.frame}'; mean thickness {d.mean_thickness:.4g} m")
        if session.average:
            info += (f"\nLayer averages: one value per well" + (f" (proportion of facies {session.indicator})" if session.indicator is not None else "")
                     + "; horizontal variograms only")
        elif session.indicator is not None:
            info += f"\nIndicator of facies {session.indicator}: proportion {np.mean(d.values):.1%}"
        left.addWidget(QLabel(info))
        row = QHBoxLayout()
        row.addWidget(QLabel("Data:"))
        self.transform = QComboBox()
        self.transform.addItems(["none", "normal_score"] + (["indicator"] if session.indicator is not None and not session.average else []))
        self.transform.setCurrentText(session.transform)
        self.transform.setEnabled(session.indicator is None or session.average)         # a facies indicator is 0/1 data: no transform
        self.transform.currentTextChanged.connect(self._transform_changed)
        row.addWidget(self.transform)
        left.addLayout(row)

        box = QGroupBox("Experimental variograms")
        g = QGridLayout(box)
        for c, h in enumerate(["", "lag (m)", "n lags", "tol (deg)", "bandwidth (m)", "layer thickness (m)"]):
            g.addWidget(QLabel(h), 0, c)
        self.exp_fields = {}
        for r, k in enumerate(KINDS, start=1):
            g.addWidget(QLabel(NUM[k]), r, 0)
            f = {key: QLineEdit() for key in ("lag", "nlags", "tolerance_deg", "bandwidth", "vertical_tolerance")}
            st = session.settings[k]
            for c, key in enumerate(("lag", "nlags", "tolerance_deg", "bandwidth", "vertical_tolerance"), start=1):
                f[key].setText(_fmt(st[key]))
                f[key].setMaximumWidth(80)
                g.addWidget(f[key], r, c)
            if k == "vertical":
                f["vertical_tolerance"].setEnabled(False)
                f["bandwidth"].setEnabled(False)
                if session.average:
                    for w in f.values():
                        w.setEnabled(False)
            self.exp_fields[k] = f
        left.addWidget(box)
        bar = QHBoxLayout()
        self.compute_btn = QPushButton("Compute")
        self.az_btn = QPushButton("Suggest azimuth")
        self.fit_btn = QPushButton("Auto-fit")
        for b in (self.compute_btn, self.az_btn, self.fit_btn):
            bar.addWidget(b)
        left.addLayout(bar)
        self.compute_btn.clicked.connect(self.compute_all)
        self.az_btn.clicked.connect(self.suggest_azimuth)
        self.fit_btn.clicked.connect(self.auto_fit)

        mbox = QGroupBox("Model")
        mg = QGridLayout(mbox)
        mg.addWidget(QLabel("Nugget"), 0, 0)
        self.nugget = QLineEdit(_fmt(session.nugget))
        mg.addWidget(self.nugget, 0, 1)
        mg.addWidget(QLabel("Azimuth of major axis (deg from north)"), 0, 2)
        self.azimuth = QLineEdit(_fmt(session.azimuth))
        mg.addWidget(self.azimuth, 0, 3)
        for c, h in enumerate(["Type", "Contribution", "Major (m)", "Minor (m)", "Vertical (m)"]):
            mg.addWidget(QLabel(h), 1, c)
        self.struct_rows = []
        for r in range(3):
            t = QComboBox()
            t.addItems(TYPES)
            fields = [QLineEdit() for _ in range(4)]
            mg.addWidget(t, 2 + r, 0)
            for c, f in enumerate(fields, start=1):
                mg.addWidget(f, 2 + r, c)
            self.struct_rows.append((t, fields))
            t.currentTextChanged.connect(self._model_edited)
            for f in fields:
                f.editingFinished.connect(self._model_edited)
        self.nugget.editingFinished.connect(self._model_edited)
        self.azimuth.editingFinished.connect(self._model_edited)
        self.add_btn, self.rem_btn = QPushButton("Add structure"), QPushButton("Remove last structure")
        mg.addWidget(self.add_btn, 5, 0, 1, 2)
        mg.addWidget(self.rem_btn, 5, 2, 1, 2)
        self.add_btn.clicked.connect(self.add_structure)
        self.rem_btn.clicked.connect(self.remove_structure)
        left.addWidget(mbox)

        self.problem = QLabel("")
        self.problem.setStyleSheet("color: #b00020")
        self.problem.setWordWrap(True)
        left.addWidget(self.problem)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        left.addWidget(self.log)
        self.warn = QLabel("")
        self.warn.setWordWrap(True)
        left.addWidget(self.warn)
        srow = QHBoxLayout()
        self.label = QLineEdit(label)
        self.label.setPlaceholderText("name of this variogram")
        self.save_btn = QPushButton("Save to project")
        self.save_btn.clicked.connect(self.save)
        srow.addWidget(self.label)
        srow.addWidget(self.save_btn)
        left.addLayout(srow)

        self.fig = Figure(figsize=(6, 7))
        self.axes = self.fig.subplots(3, 1)
        self.canvas = FigureCanvasQTAgg(self.fig)
        root.addWidget(self.canvas, 1)
        self._load_model_fields()
        self._refresh()

    # -- reading fields into the session ---------------------------------------------------------------------------------------------
    def _set_problem(self, text: str = ""):
        self.problem.setText(text)

    def _read_exp_settings(self):
        for k in KINDS:
            f, st = self.exp_fields[k], self.s.settings[k]
            try:
                st["lag"] = _num(f["lag"].text())
                st["nlags"] = int(_num(f["nlags"].text()) or 0) or st["nlags"]
                st["tolerance_deg"] = _num(f["tolerance_deg"].text()) or st["tolerance_deg"]
                st["bandwidth"] = _num(f["bandwidth"].text())
                st["vertical_tolerance"] = _num(f["vertical_tolerance"].text())
            except ValueError as exc:
                raise VariogramError(f"{NUM[k]}: {exc}") from None

    def _model_edited(self, *_):
        if self._busy:
            return
        try:
            self.s.set_nugget(_num(self.nugget.text()) or 0.0)
            self.s.set_azimuth(_num(self.azimuth.text()) or 0.0)
            for i, (t, fields) in enumerate(self.struct_rows[:len(self.s.structures)]):
                c, a, b, v = (_num(f.text()) for f in fields)
                self.s.set_structure(i, type=t.currentText(), contribution=c, major=a, minor=b, vertical=v)
            self._set_problem()
        except (ValueError, VariogramError) as exc:
            self._set_problem(f"Model: {exc}")
        self._refresh()

    def _load_model_fields(self):
        self._busy = True
        self.nugget.setText(_fmt(self.s.nugget))
        self.azimuth.setText(_fmt(self.s.azimuth))
        for i, (t, fields) in enumerate(self.struct_rows):
            on = i < len(self.s.structures)
            for w in (t, *fields):
                w.setVisible(on)
            if on:
                st = self.s.structures[i]
                t.setCurrentText(st["type"])
                for f, key in zip(fields, ("contribution", "major", "minor", "vertical")):
                    f.setText(_fmt(st[key]))
        self.add_btn.setEnabled(len(self.s.structures) < 3)
        self.rem_btn.setEnabled(len(self.s.structures) > 1)
        self._busy = False

    def _transform_changed(self, how):
        if self._busy:
            return
        self.s.set_transform(how)
        self._refresh()

    # -- actions ------------------------------------------------------------------------------------------------------------------------------
    def compute_all(self):
        try:
            self._read_exp_settings()
            self._model_edited()
            done = []
            for k in KINDS:
                if self.s.average and k == "vertical":
                    continue
                st = self.s.settings[k]
                try:
                    self.s.compute(k, **{key: st[key] for key in st})
                    done.append(k)
                except VariogramError as exc:
                    self.s.exp.pop(k, None)
                    self.log.appendPlainText(f"{k}: {exc}")
            for k in done:                           # show the values that were used (defaults included)
                for key, w in self.exp_fields[k].items():
                    w.setText(_fmt(self.s.settings[k][key]))
            self._set_problem()
        except (ValueError, VariogramError) as exc:
            self._set_problem(str(exc))
        self._refresh()

    def suggest_azimuth(self):
        try:
            self._read_exp_settings()
            a = self.s.suggest_azimuth()
        except (ValueError, VariogramError) as exc:
            self._set_problem(str(exc))
            return
        if a is None:
            self.log.appendPlainText("Azimuth: no clear anisotropy in the data (or too few wells); set it by hand.")
        else:
            self.s.set_azimuth(a)
            self.log.appendPlainText(f"Azimuth: lowest semivariance at {a:g} deg from north (suggestion - check it).")
            self._load_model_fields()
        self._refresh()

    def auto_fit(self):
        try:
            self._read_exp_settings()
            for m in self.s.auto_fit(self.struct_rows[0][0].currentText()):
                self.log.appendPlainText(m)
            self._set_problem()
        except (ValueError, VariogramError) as exc:
            self._set_problem(str(exc))
        self._load_model_fields()
        self._refresh()

    def add_structure(self):
        try:
            self.s.add_structure()
        except VariogramError as exc:
            self._set_problem(str(exc))
        self._load_model_fields()
        self._refresh()

    def remove_structure(self):
        try:
            self.s.remove_structure(len(self.s.structures) - 1)
        except VariogramError as exc:
            self._set_problem(str(exc))
        self._load_model_fields()
        self._refresh()

    def _ask_action(self, conflict):
        mode, ok = QInputDialog.getItem(self, "Variogram exists", f"{conflict.item} already has a variogram "
                                        f"({conflict.existing}). What should happen to {conflict.new!r}?",
                                        ["replace", "skip", "suffix"], 0, False)
        if not ok:
            return None
        if mode == "suffix":
            suf, ok = QInputDialog.getText(self, "Suffix", "Suffix for the new name:")
            return Action("suffix", suf) if ok else None
        return Action(mode)

    def save(self):
        self._model_edited()
        label = self.label.text().strip()
        try:
            acts = {}
            for c in self.s.conflicts(label):
                a = self.choose_action(c)
                if a is None:
                    return
                acts[c.key] = a
            res = self.s.save(label, acts)
        except (ProjectError, VariogramError) as exc:
            self._set_problem(str(exc))
            return
        self.saved = res
        self._set_problem()
        self.log.appendPlainText("; ".join(res.applied + res.notes))

    # -- drawing -------------------------------------------------------------------------------------------------------------------------------
    def _refresh(self):
        if getattr(self, "axes", None) is None:
            return
        complete = not self.s.missing()
        model = None
        try:
            model = self.s.model() if complete else None
        except VariogramError as exc:
            if not self.problem.text():
                self._set_problem(f"Model: {exc}")
        var = self.s.variance
        for ax, k in zip(self.axes, KINDS):
            ax.clear()
            ax.set_title(NUM[k], fontsize=9, loc="left")
            ax.axhline(var, color="0.6", ls=":", lw=1)
            e = self.s.exp.get(k)
            hmax = None
            if e is not None:
                lag, g, n = e.usable()
                if lag.size:
                    ax.scatter(lag, g, s=10 + 90 * n / n.max(), color="C0", zorder=3)
                    for x, y, c in zip(lag, g, n):
                        ax.annotate(str(int(c)), (x, y), textcoords="offset points", xytext=(0, 5), fontsize=6, ha="center")
                hmax = float(e.lag[-1])
            elif model is not None:
                hmax = 1.5 * model.range_of(k)
            if model is not None and hmax:
                h = np.linspace(0, hmax, 200)
                ax.plot(h, model.gamma_along(h, k), color="C3", lw=1.5)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("lag (m)", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.warn.setText("\n".join(self.s.warnings()))


def edit_variogram(session: VariogramSession, parent=None):
    dlg = VariogramEditor(session, parent)
    dlg.exec()
    return dlg.saved

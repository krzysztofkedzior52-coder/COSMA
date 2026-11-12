"""Properties, permeability and realizations in one dialog.

Tab 1: the properties to simulate (name, GRDECL keyword, variogram name pattern, bounds, search) and, for the property selected, where
the distribution of every (interval, facies) comes from. Tab 2: porosity-permeability relation and anisotropy. Tab 3: how many
realizations, seeds, the confidence of the target check, what to do with failures, the cache folder; run, read the report, export one
realization. Every choice starts unset; nothing is preset. "Check setup" makes realization 0 from exactly the entries; the plan can be
saved only after that."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
                               QListWidget)

from ..perm_sim import EXTRAPOLATION, LEVELS, PermSetup, Relation
from ..property_sim import Distribution, PropertySetup
from ..realizations import (MAX_REALIZATIONS, ON_FAIL, REDRAW, PropertyJob, RealizationError, RealizationManager, RealizationPlan,
                            to_state)
from ..progress import format_duration
from ..progress import Cancelled
from ..realization_runner import check_memory, cpu_count, run_parallel
from .progress_widget import ProgressPanel
from .proportions_dialog import _default_build

CHOOSE = "- choose -"
JOB_COLS = ("Name", "Keyword", "Written to GRDECL", "Variogram name ({facies} = facies name)", "Data-check property (blank = name)",
            "Lower bound (blank = none)", "Upper bound (blank = none)", "Min. data per domain", "Neighbours", "Search radius factor",
            "Multigrid levels", "Conflict probability", "Correlate with property (blank = independent)")


class RealizationsDialog(QDialog):
    def __init__(self, project, parent: QWidget | None = None, build=None, pick_dir=None, pick_save=None):
        super().__init__(parent)
        self.project = project
        self.build = build or _default_build
        self.pick_dir = pick_dir or (lambda: QFileDialog.getExistingDirectory(self, "Cache folder"))
        self.pick_save = pick_save or (lambda: QFileDialog.getSaveFileName(self, "Export realization", "", "GRDECL (*.grdecl *.GRDECL)")[0])
        self.setWindowTitle("Properties, permeability and realizations")
        self.setMinimumSize(1100, 800)
        self.intervals = [iv.name for iv in project.chart.intervals] if project.chart is not None else []
        self.scheme = tuple(project.facies_setup.scheme) if project.facies_setup is not None else ()
        self.domains_keys = [(iv, f.code) for iv in self.intervals for f in self.scheme]
        self.fnames = {f.code: f.name for f in self.scheme}
        lay = QVBoxLayout(self)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        self._build_props()
        self._build_perm()
        self._build_real()
        self.panel = ProgressPanel()
        self.panel.finished.connect(self._finished)
        self.panel.timer.timeout.connect(self._sync_list)
        self._done_cb = lambda kind, val: None
        lay.addWidget(self.panel)
        self.problem = QLabel("")
        self.problem.setStyleSheet("color: #b00020")
        self.problem.setWordWrap(True)
        lay.addWidget(self.problem)
        row = QHBoxLayout()
        self.check_btn = QPushButton("Check setup (make realization 0)")
        self.check_btn.clicked.connect(self.check_setup)
        self.save_btn = QPushButton("Save plan")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        for w in (self.check_btn, self.save_btn, close):
            row.addWidget(w)
        lay.addLayout(row)
        self._ran = None
        self.saved = False
        self.manager = None
        self.results = []
        self._cancel = False
        self._load(project.realization_plan)

    # -- tab 1 --------------------------------------------------------------------------------------------------------
    def _build_props(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(self._note("Each property is simulated separately for every interval and facies, from normal scores, with a "
                               "normal-score variogram of that name attached to the interval. Select a property to say where the "
                               "distribution of each interval and facies comes from. Nothing is borrowed from another domain."))
        self.jobs = QTableWidget(0, len(JOB_COLS))
        self.jobs.setHorizontalHeaderLabels(JOB_COLS)
        self.jobs.setMaximumHeight(200)
        v.addWidget(self.jobs)
        b = QHBoxLayout()
        add, rem = QPushButton("Add property"), QPushButton("Remove selected")
        add.clicked.connect(lambda: self.add_job())
        rem.clicked.connect(self.remove_job)
        b.addWidget(add)
        b.addWidget(rem)
        b.addStretch(1)
        v.addLayout(b)
        self.dom_label = QLabel("Select a property to set its distributions.")
        v.addWidget(self.dom_label)
        self.dom = QTableWidget(0, 7)
        self.dom.setHorizontalHeaderLabels(("Interval", "Facies", "Distribution from", "Kind", "Mean (of the property)", "Std. dev. (of the property)",
                                            "Correlation with that property (between -1 and 1)"))
        v.addWidget(self.dom, 1)
        self.domains: list = []
        self._cur = -1
        self.jobs.itemSelectionChanged.connect(self._job_selected)
        self.jobs.itemChanged.connect(self._job_edited)
        self.tabs.addTab(w, "Properties")

    @staticmethod
    def _note(t):
        n = QLabel(t)
        n.setWordWrap(True)
        return n

    def add_job(self, vals=None, dom=None):
        r = self.jobs.rowCount()
        self.jobs.blockSignals(True)
        self.jobs.insertRow(r)
        vals = list(vals or [])
        vals += [""] * (len(JOB_COLS) - len(vals))
        for c, t in enumerate(vals):
            it = QTableWidgetItem("" if c == 2 else str(t))
            if c == 2:
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if t is True else Qt.Unchecked)
            self.jobs.setItem(r, c, it)
        self.jobs.blockSignals(False)
        self.domains.append(dom or {})
        self._job_edited()
        self.jobs.selectRow(r)

    def remove_job(self):
        r = self.jobs.currentRow()
        if r < 0:
            return
        self._store_domains()
        self._cur = -1
        self.jobs.removeRow(r)
        del self.domains[r]
        self._fill_dom(-1)
        self._refresh_porosity()
        self._edited()

    def _job_selected(self):
        self._store_domains()
        r = self.jobs.currentRow()
        self._fill_dom(r)

    def _store_domains(self):
        if 0 <= self._cur < len(self.domains):
            d = {}
            for i, key in enumerate(self.domains_keys):
                d[key] = (self.dom.cellWidget(i, 2).currentText(), self.dom.cellWidget(i, 3).currentText(),
                          self.dom.cellWidget(i, 4).text(), self.dom.cellWidget(i, 5).text(), self.dom.cellWidget(i, 6).text())
            self.domains[self._cur] = d

    def _fill_dom(self, r):
        self._cur = r
        self.dom.setRowCount(0)
        if r < 0:
            self.dom_label.setText("Select a property to set its distributions.")
            return
        self.dom_label.setText(f"Distributions for {self.jobs.item(r, 0).text() or 'the selected property'}:")
        saved = self.domains[r]
        self.dom.setRowCount(len(self.domains_keys))
        for i, key in enumerate(self.domains_keys):
            self.dom.setItem(i, 0, QTableWidgetItem(key[0]))
            self.dom.setItem(i, 1, QTableWidgetItem(self.fnames[key[1]]))
            src = QComboBox()
            src.addItems([CHOOSE, "data", "typed"])
            kind = QComboBox()
            kind.addItems([CHOOSE, "normal", "lognormal"])
            mean, sd, rho = QLineEdit(), QLineEdit(), QLineEdit()
            s = saved.get(key)
            if s:
                src.setCurrentText(s[0])
                kind.setCurrentText(s[1])
                mean.setText(s[2])
                sd.setText(s[3])
                rho.setText(s[4] if len(s) > 4 else "")
            for c, wd in ((2, src), (3, kind), (4, mean), (5, sd), (6, rho)):
                self.dom.setCellWidget(i, c, wd)
            rho.textChanged.connect(self._edited)
            src.currentTextChanged.connect(self._edited)
            kind.currentTextChanged.connect(self._edited)
            mean.textChanged.connect(self._edited)
            sd.textChanged.connect(self._edited)

    # -- tab 2 --------------------------------------------------------------------------------------------------------
    def _build_perm(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.use_perm = QCheckBox("Make permeability from a porosity property")
        v.addWidget(self.use_perm)
        v.addWidget(self._note("log10 PERMX = intercept + slope * porosity + scatter * e. The scatter term e is either independent "
                               "from cell to cell, or a spatially correlated field with the normal-score variogram you name "
                               "(continuous streaks; one more simulation pass). "
                               "PERMY = PERMX * ky/kx and PERMZ = PERMX * kv/kh. A relation from core uses the regression of the "
                               "data checks at the chosen level and must have passed them."))
        g = QGridLayout()
        self.phi_box = QComboBox()
        self.kykx, self.shift, self.extrap, self.plo, self.phi_hi = QLineEdit(), QLineEdit(), QComboBox(), QLineEdit(), QLineEdit()
        self.extrap.addItems([CHOOSE, *EXTRAPOLATION])
        self.scat = QComboBox()
        self.scat.addItems([CHOOSE, "independent", "correlated"])
        self.scat_vg, self.scat_nn, self.scat_sf, self.scat_lv = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        for r, (t, wd) in enumerate((("Porosity property:", self.phi_box), ("ky/kx:", self.kykx),
                                     ("Core shift (log10 units added to a core intercept):", self.shift),
                                     ("Outside the core porosity range:", self.extrap), ("Lower bound of PERMX, mD (blank = none):", self.plo),
                                     ("Upper bound of PERMX, mD (blank = none):", self.phi_hi), ("Scatter about the line:", self.scat),
                                     ("Correlated scatter: variogram name ({facies} = facies name):", self.scat_vg),
                                     ("Correlated scatter: neighbours:", self.scat_nn),
                                     ("Correlated scatter: search radius factor:", self.scat_sf),
                                     ("Correlated scatter: multigrid levels:", self.scat_lv))):
            g.addWidget(QLabel(t), r, 0)
            g.addWidget(wd, r, 1)
        v.addLayout(g)
        self.perm = QTableWidget(len(self.domains_keys), 8)
        self.perm.setHorizontalHeaderLabels(("Interval", "Facies", "Relation", "Intercept", "Slope", "Scatter (log10)", "Core level", "kv/kh"))
        for i, key in enumerate(self.domains_keys):
            self.perm.setItem(i, 0, QTableWidgetItem(key[0]))
            self.perm.setItem(i, 1, QTableWidgetItem(self.fnames[key[1]]))
            rel = QComboBox()
            rel.addItems([CHOOSE, "typed", "core"])
            lvl = QComboBox()
            lvl.addItems([CHOOSE, *LEVELS])
            self.perm.setCellWidget(i, 2, rel)
            self.perm.setCellWidget(i, 6, lvl)
            for c in (3, 4, 5, 7):
                e = QLineEdit()
                e.textChanged.connect(self._edited)
                self.perm.setCellWidget(i, c, e)
            rel.currentTextChanged.connect(self._edited)
            lvl.currentTextChanged.connect(self._edited)
        v.addWidget(self.perm, 1)
        for e in (self.kykx, self.shift, self.plo, self.phi_hi, self.scat_vg, self.scat_nn, self.scat_sf, self.scat_lv):
            e.textChanged.connect(self._edited)
        for c in (self.phi_box, self.extrap, self.scat):
            c.currentTextChanged.connect(self._edited)
        self.use_perm.toggled.connect(self._edited)
        self.tabs.addTab(w, "Permeability")

    def _refresh_porosity(self):
        cur = self.phi_box.currentText()
        self.phi_box.blockSignals(True)
        self.phi_box.clear()
        self.phi_box.addItems([CHOOSE] + [self.jobs.item(r, 0).text() for r in range(self.jobs.rowCount())])
        self.phi_box.setCurrentText(cur)
        self.phi_box.blockSignals(False)

    # -- tab 3 --------------------------------------------------------------------------------------------------------
    def _build_real(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(self._note("Realizations are fixed by the base seed and their number. Each is tested against its targets; the "
                               "confidence is shared over all tests of a realization. Failing realizations are kept (flagged), "
                               "discarded, or redrawn. The cache folder is optional."))
        g = QGridLayout()
        self.count, self.seed, self.conf, self.redraws, self.cache = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        self.workers = QLineEdit()
        self.on_fail, self.redraw = QComboBox(), QComboBox()
        self.on_fail.addItems([CHOOSE, *ON_FAIL])
        self.redraw.addItems([CHOOSE, *REDRAW])
        pick = QPushButton("Choose folder...")
        pick.clicked.connect(self._pick_cache)
        rows = (("Number of realizations (1-%d):" % MAX_REALIZATIONS, self.count), ("Base seed (whole number):", self.seed),
                ("Confidence of the target check (0.5-<1):", self.conf), ("A failing realization is:", self.on_fail),
                ("Largest number of redraws (for 'redraw'):", self.redraws), ("A redraw changes:", self.redraw),
                ("Cache folder (blank = none):", self.cache),
                (f"Worker processes (1 = this process; this computer has {cpu_count()} usable cores):", self.workers))
        for r, (t, wd) in enumerate(rows):
            g.addWidget(QLabel(t), r, 0)
            g.addWidget(wd, r, 1)
        g.addWidget(pick, 6, 2)
        v.addLayout(g)
        b = QHBoxLayout()
        self.run_btn = QPushButton("Run all")
        self.run_btn.clicked.connect(self.run_all)
        stop = QPushButton("Stop")
        stop.clicked.connect(lambda: setattr(self, "_cancel", True))
        b.addWidget(self.run_btn)
        b.addWidget(stop)
        b.addStretch(1)
        v.addLayout(b)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        v.addWidget(self.report, 1)
        e = QHBoxLayout()
        self.list = QListWidget()
        self.list.setMaximumHeight(110)
        e.addWidget(self.list, 1)
        col = QVBoxLayout()
        self.allow = QCheckBox("Allow a flagged realization")
        self.export_btn = QPushButton("Export selected...")
        self.export_btn.clicked.connect(self.export_selected)
        col.addWidget(self.allow)
        col.addWidget(self.export_btn)
        e.addLayout(col)
        v.addLayout(e)
        for wd in (self.count, self.seed, self.conf, self.redraws, self.cache, self.workers):
            wd.textChanged.connect(self._edited)
        for c in (self.on_fail, self.redraw):
            c.currentTextChanged.connect(self._edited)
        self.tabs.addTab(w, "Realizations")

    def _pick_cache(self):
        d = self.pick_dir()
        if d:
            self.cache.setText(d)

    # -- state --------------------------------------------------------------------------------------------------------
    def _edited(self, *_):
        self._ran = None
        if hasattr(self, "save_btn"):
            self.save_btn.setEnabled(False)
        self.manager = None

    def _job_edited(self, *_):
        self._refresh_porosity()
        self._edited()

    def _load(self, p: RealizationPlan | None):
        if p is None:
            self._refresh_porosity()
            return
        for j in p.jobs:
            s = j.setup
            ud = dict(s.user)
            dom = {}
            for key in self.domains_keys:
                src = dict(s.sources).get(key)
                if src is None:
                    continue
                rho = dict(s.correlation).get(key)
                rho = "" if rho is None else f"{rho:g}"
                if src == "data":
                    dom[key] = ("data", CHOOSE, "", "", rho)
                else:
                    d = ud[key]
                    dom[key] = ("typed", d.kind, f"{d.mean:g}", f"{d.std:g}", rho)
            self.add_job([s.name, j.keyword, j.export, s.variogram, j.data_name or "", "" if s.lower is None else f"{s.lower:g}",
                          "" if s.upper is None else f"{s.upper:g}", s.min_data, s.max_neighbours, f"{s.search_factor:g}", s.levels,
                          f"{s.conflict_probability:g}", s.correlate_with or ""], dom)
        self.jobs.clearSelection()
        self._cur = -1
        self._refresh_porosity()
        if p.perm is not None:
            q = p.perm
            self.use_perm.setChecked(True)
            self.phi_box.setCurrentText(q.porosity)
            self.kykx.setText(f"{q.ky_kx:g}")
            self.shift.setText(f"{q.core_shift:g}")
            self.extrap.setCurrentText(q.extrapolation)
            self.plo.setText("" if q.lower is None else f"{q.lower:g}")
            self.phi_hi.setText("" if q.upper is None else f"{q.upper:g}")
            self.scat.setCurrentText(q.scatter_mode)
            if q.scatter_mode == "correlated":
                self.scat_vg.setText(q.scatter_variogram)
                self.scat_nn.setText(str(q.scatter_neighbours))
                self.scat_sf.setText(f"{q.scatter_search_factor:g}")
                self.scat_lv.setText(str(q.scatter_levels))
            rel, kv = dict(q.relations), dict(q.kv_kh)
            for i, key in enumerate(self.domains_keys):
                r = rel.get(key)
                if r is None:
                    continue
                self.perm.cellWidget(i, 2).setCurrentText("typed" if r.source == "user" else "core")
                if r.source == "user":
                    for c, val in ((3, r.intercept), (4, r.slope), (5, r.scatter)):
                        self.perm.cellWidget(i, c).setText(f"{val:g}")
                else:
                    self.perm.cellWidget(i, 6).setCurrentText(r.level)
                if key in kv:
                    self.perm.cellWidget(i, 7).setText(f"{kv[key]:g}")
        self.count.setText(str(p.count))
        self.seed.setText(str(p.base_seed))
        self.conf.setText(f"{p.confidence:g}")
        self.on_fail.setCurrentText(p.on_fail)
        if p.on_fail == "redraw":
            self.redraws.setText(str(p.max_redraws))
            self.redraw.setCurrentText(p.redraw)
        self._ran = None
        self.save_btn.setEnabled(False)

    @staticmethod
    def _num(text, what, integer=False, optional=False):
        text = text.strip()
        if optional and not text:
            return None
        try:
            v = float(text)
            if integer:
                if v != int(v):
                    raise ValueError
                return int(v)
            return v
        except ValueError:
            raise RealizationError(f"{what}: enter {'a whole number' if integer else 'a number'}") from None

    def read_plan(self) -> RealizationPlan:
        try:
            self._store_domains()
            if not self.intervals or not self.scheme:
                raise RealizationError("define the strat chart and the facies first")
            jobs = []
            for r in range(self.jobs.rowCount()):
                t = lambda c: self.jobs.item(r, c).text().strip()          # noqa: E731
                name = t(0) or f"property {r + 1}"
                dom = self.domains[r]
                src, user, corr = [], [], []
                cw = t(12) or None
                for key in self.domains_keys:
                    d = dom.get(key)
                    where = f"{name}, {key[0]} / {self.fnames[key[1]]}"
                    if not d or d[0] == CHOOSE:
                        raise RealizationError(f"{where}: choose where the distribution comes from")
                    if d[0] == "data":
                        src.append((key, "data"))
                    else:
                        if d[1] == CHOOSE:
                            raise RealizationError(f"{where}: choose normal or lognormal")
                        src.append((key, "user"))
                        user.append((key, Distribution(d[1], self._num(d[2], f"{where}: mean"), self._num(d[3], f"{where}: std. dev."))))
                    rho = d[4].strip() if len(d) > 4 else ""
                    if cw is not None:
                        if not rho:
                            raise RealizationError(f"{where}: enter the correlation with {cw}")
                        corr.append((key, self._num(rho, f"{where}: correlation")))
                    elif rho:
                        raise RealizationError(f"{where}: a correlation is typed but no property to correlate with is named")
                setup = PropertySetup(t(0), t(3), tuple(src), tuple(user), self._num(t(7), f"{name}: min. data", True),
                                      self._num(t(11), f"{name}: conflict probability"), self._num(t(5), f"{name}: lower bound", optional=True),
                                      self._num(t(6), f"{name}: upper bound", optional=True), self._num(t(8), f"{name}: neighbours", True),
                                      self._num(t(9), f"{name}: search factor"), self._num(t(10), f"{name}: levels", True), cw, tuple(corr))
                jobs.append(PropertyJob(setup, t(1), self.jobs.item(r, 2).checkState() == Qt.Checked, t(4) or None))
            perm = None
            if self.use_perm.isChecked():
                if self.phi_box.currentText() == CHOOSE:
                    raise RealizationError("choose the porosity property")
                if self.extrap.currentText() == CHOOSE:
                    raise RealizationError("choose what happens outside the core porosity range")
                rels, kv = [], []
                for i, key in enumerate(self.domains_keys):
                    where = f"permeability, {key[0]} / {self.fnames[key[1]]}"
                    s = self.perm.cellWidget(i, 2).currentText()
                    tx = lambda c: self.perm.cellWidget(i, c).text()       # noqa: E731
                    if s == CHOOSE:
                        raise RealizationError(f"{where}: choose the relation")
                    if s == "typed":
                        rels.append((key, Relation("user", self._num(tx(3), f"{where}: intercept"), self._num(tx(4), f"{where}: slope"),
                                                   self._num(tx(5), f"{where}: scatter"))))
                    else:
                        lv = self.perm.cellWidget(i, 6).currentText()
                        if lv == CHOOSE:
                            raise RealizationError(f"{where}: choose the core level")
                        rels.append((key, Relation("core", level=lv)))
                    kv.append((key, self._num(tx(7), f"{where}: kv/kh")))
                sm = self.scat.currentText()
                if sm == CHOOSE:
                    raise RealizationError("choose whether the permeability scatter is independent or spatially correlated")
                extra = {}
                if sm == "correlated":
                    extra = dict(scatter_variogram=self.scat_vg.text().strip(),
                                 scatter_neighbours=self._num(self.scat_nn.text(), "correlated scatter: neighbours", True),
                                 scatter_search_factor=self._num(self.scat_sf.text(), "correlated scatter: search factor"),
                                 scatter_levels=self._num(self.scat_lv.text(), "correlated scatter: levels", True))
                elif any(e.text().strip() for e in (self.scat_vg, self.scat_nn, self.scat_sf, self.scat_lv)):
                    raise RealizationError("independent scatter takes no variogram or search settings; clear those fields")
                perm = PermSetup(self.phi_box.currentText(), tuple(rels), tuple(kv), self._num(self.kykx.text(), "ky/kx"),
                                 self._num(self.shift.text(), "core shift"), self.extrap.currentText(),
                                 self._num(self.plo.text(), "lower bound of PERMX", optional=True),
                                 self._num(self.phi_hi.text(), "upper bound of PERMX", optional=True), scatter_mode=sm, **extra)
            if self.on_fail.currentText() == CHOOSE:
                raise RealizationError("choose what happens to a failing realization")
            of = self.on_fail.currentText()
            mr, rd = 0, "both"
            if of == "redraw":
                if self.redraw.currentText() == CHOOSE:
                    raise RealizationError("choose what a redraw changes")
                mr = self._num(self.redraws.text(), "largest number of redraws", True)
                rd = self.redraw.currentText()
            return RealizationPlan(tuple(jobs), perm, self._num(self.count.text(), "number of realizations", True),
                                   self._num(self.seed.text(), "base seed", True), self._num(self.conf.text(), "confidence"), of, mr, rd)
        except RealizationError:
            raise
        except ValueError as exc:                    # setup classes raise ValueError subclasses
            raise RealizationError(str(exc)) from None

    def _manager(self):
        plan = self.read_plan()
        if self.project.facies_sim is None:
            raise RealizationError("save the facies simulation first (Facies > Simulate facies)")
        model = self.build(self.project)
        cache = self.cache.text().strip()
        return RealizationManager(self.project, model, plan, cache_dir=Path(cache) if cache else None), plan

    # -- actions ------------------------------------------------------------------------------------------------------
    def _set_busy(self, busy: bool):
        self.tabs.setEnabled(not busy)
        for b in (self.check_btn, self.save_btn):
            b.setEnabled(False if busy else b is self.check_btn or self._ran is not None)

    def _start(self, fn, items, skipped, done):
        self.problem.setText("")
        self._set_busy(True)
        self._done_cb = done
        self.panel.start(fn, items, skipped)

    def _finished(self):
        self._set_busy(False)
        self._sync_list()
        kind, val = self.panel.outcome
        if kind == "failed":
            self.problem.setText(val)
        self._done_cb(kind, val)

    def check_setup(self):
        """Makes realization 0 (not stored) in a worker thread; the plan can be saved when it has succeeded."""
        self._ran = None
        self.save_btn.setEnabled(False)
        try:
            plan = self.read_plan()
            if self.project.facies_sim is None:
                raise RealizationError("save the facies simulation first (Facies > Simulate facies)")
        except RealizationError as exc:
            self.problem.setText(str(exc))
            return False
        cache = self.cache.text().strip()

        def job(ctx):
            ctx.progress("building the grid", 0.0)
            model = self.build(self.project)
            mgr = RealizationManager(self.project, model, plan, cache_dir=Path(cache) if cache else None)
            return mgr, mgr.generate(0, 0, ctx.progress)[3]

        def done(kind, val):
            if kind != "ok":
                if kind == "cancelled":
                    self.problem.setText("Stopped; the setup has not been checked.")
                return
            mgr, check = val
            self.manager = mgr
            self._ran = to_state(plan)
            self.save_btn.setEnabled(True)
            self.report.setPlainText("Realization 0 (attempt 0), not stored:\n" + check["text"])
        self._start(job, 1, 0, done)
        return True

    def run_all(self):
        self.list.clear()
        self.results = []
        try:
            plan = self.read_plan()
            if self.project.facies_sim is None:
                raise RealizationError("save the facies simulation first (Facies > Simulate facies)")
            workers = self._num(self.workers.text(), "worker processes", True)
            if workers < 1:
                raise RealizationError("worker processes: enter a whole number of 1 or more (1 = this process only)")
        except RealizationError as exc:
            self.problem.setText(str(exc))
            return
        cache = self.cache.text().strip()
        box = {}

        def job(ctx):
            ctx.progress("building the grid", 0.0)
            model = self.build(self.project)
            mgr = RealizationManager(self.project, model, plan, cache_dir=Path(cache) if cache else None)
            box["mgr"] = mgr
            todo = [n for n in range(plan.count) if not mgr.has_cache(n)]
            ctx.tracker.set_skipped(plan.count - len(todo))   # known only now that the cache folder has been looked at
            if workers == 1 or not todo:
                box["note"] = ""
                for n in range(plan.count):
                    cached = mgr.has_cache(n)
                    r = mgr.realize(n, keep_arrays=False) if cached else mgr.realize(n, ctx.progress, keep_arrays=False)
                    self.results.append(r)
                    if not cached:
                        ctx.item_done()
                return mgr
            w = min(workers, len(todo))
            msg, ok = check_memory(mgr._active_cells(), w)
            box["note"] = msg + (f" ({workers - w} requested worker(s) not needed)" if w < workers else "")
            if not ok:
                raise RealizationError(msg)
            for n in range(plan.count):                       # cached ones first, in this process
                if mgr.has_cache(n):
                    self.results.append(mgr.realize(n, keep_arrays=False))

            def got(r):
                self.results.append(r)
                ctx.label = f"{len(self.results)} of {plan.count} realizations finished, {w} worker processes"
            ctx.label = f"starting {w} worker processes (each loads the model and the compiled kernels)"
            _, stopped = run_parallel(mgr, todo, w, ctx.tracker, lambda: ctx.cancel_requested, got)
            if stopped:
                raise Cancelled()
            return mgr

        def done(kind, val):
            mgr = box.get("mgr")
            if mgr is None:
                return
            self.manager = mgr
            self.results.sort(key=lambda r: r.n)
            self.list.clear()
            self._sync_list()
            head = (box.get("note") or "") + ("\n" if box.get("note") else "")
            if kind == "cancelled":
                head += f"STOPPED after {len(self.results)} of {plan.count} realization(s).\n"
            self.report.setPlainText(head + mgr.summary(self.results) + (
                f"\nTotal time {format_duration(self.panel.tracker.elapsed)}." if kind == "ok" else ""))
        self._start(job, plan.count, 0, done)

    def _sync_list(self, *_):
        while self.list.count() < len(self.results):
            r = self.results[self.list.count()]
            self.list.addItem(f"{r.n}: {r.status}" + (f" (attempt {r.attempt})" if r.attempt else "") + (" [cache]" if r.from_cache else ""))

    def done(self, code):
        if self.panel.running:
            self.panel.stop()
            self.panel.wait()
        super().done(code)

    def export_selected(self):
        """Writes one realization (in a worker thread: without a cache it is regenerated from its seeds first)."""
        self.problem.setText("")
        row = self.list.currentRow()
        if self.manager is None or row < 0 or row >= len(self.results):
            self.problem.setText("Run the realizations and select one.")
            return
        path = self.pick_save()
        if not path:
            return
        n, allow, mgr = self.results[row].n, self.allow.isChecked(), self.manager
        regen = "" if mgr.has_cache(n) else " (not cached: regenerating it from its seeds first)"
        self.report.appendPlainText(f"Exporting realization {n}{regen}...")

        def job(ctx):
            return mgr.export(n, path, allow_flagged=allow, progress=ctx.progress)

        def done(kind, val):
            if kind == "ok":
                self.report.appendPlainText("Written: " + ", ".join(str(p) for p in val))
            elif kind == "cancelled":
                self.report.appendPlainText("Export stopped; nothing was written.")
        self._start(job, 1, 0, done)

    def save(self) -> bool:
        try:
            plan = self.read_plan()
        except RealizationError as exc:
            self.problem.setText(str(exc))
            return False
        if self._ran is None or to_state(plan) != self._ran:
            self.problem.setText("Check the setup with these entries first: what is saved must be what was run.")
            self.save_btn.setEnabled(False)
            return False
        self.project.set_realization_plan(plan)
        self.saved = True
        self.problem.setText("")
        return True

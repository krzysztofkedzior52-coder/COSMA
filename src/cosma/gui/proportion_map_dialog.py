"""Import a facies proportion map from an image. Colours on the map are tied to facies that already exist in the project.
Nothing is preset: every tolerance, radius and coverage is typed (the smoothing radius may be taken from a suggestion).
Processing order: crop, rotate, flip, scale, mask - then control points and clicks refer to the processed image."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QRadioButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..proportion_maps import (Legend, MapImportError, Recipe, apply_ops, georeference, load_raster, recipe_to_state,
                               run_recipe, suggest_sigma)

CHOOSE = "- choose -"
MODES = ("no clicking", "control point", "legend colour", "mask vertex")


def _default_spec(project):
    gd = project.grid_def
    if gd is None:
        raise MapImportError("define the grid first (Grid menu)")
    return gd.spec(getattr(project, "boundary", None))


def _pick_file(parent):
    from PySide6.QtWidgets import QFileDialog
    p, _ = QFileDialog.getOpenFileName(parent, "Proportion map image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff);;All files (*)")
    return p or None


def _confirm(parent, text):
    return QMessageBox.question(parent, "Import proportion map", text) == QMessageBox.StandardButton.Yes


class ProportionMapDialog(QDialog):
    def __init__(self, project, parent: QWidget | None = None, spec=None, pick_file=None, confirm=None):
        super().__init__(parent)
        self.project = project
        self._spec = spec
        self._pick = pick_file or (lambda: _pick_file(self))
        self._confirm = confirm or (lambda text: _confirm(self, text))
        self.setWindowTitle("Import proportion map")
        self.setMinimumSize(1100, 950)
        self.scheme = tuple(project.facies_setup.scheme) if project.facies_setup is not None else ()
        self.names = [iv.name for iv in project.chart.intervals] if project.chart is not None else []
        self.fig = self.canvas = None
        self.raster = None
        self.path = ""
        self._ran = None
        self.saved = False
        self.result = None
        self._img = None
        lay = QVBoxLayout(self)
        note = QLabel("Tie each colour on the map to a facies that already exists in this project. A facies with no colour must be "
                      "declared absent (its fraction is then 0 wherever the map has data). Colours that match nothing become no "
                      "data. The image is not stored, only the resampled fractions and the settings used.")
        note.setWordWrap(True)
        lay.addWidget(note)
        top = QHBoxLayout()
        self.open_btn = QPushButton("Open image...")
        self.open_btn.clicked.connect(self.open_image)
        top.addWidget(self.open_btn)
        self.file_label = QLabel("(no image)")
        top.addWidget(self.file_label, 1)
        top.addWidget(QLabel("Interval:"))
        self.interval = QComboBox()
        self.interval.addItems([CHOOSE] + self.names)
        top.addWidget(self.interval)
        lay.addLayout(top)
        # processing
        g = QGridLayout()
        self.crop = [QLineEdit() for _ in range(4)]
        g.addWidget(QLabel("Crop x0 y0 x1 y1 (pixels, blank = none):"), 0, 0)
        for i, e in enumerate(self.crop):
            g.addWidget(e, 0, 1 + i)
        self.rotate = QComboBox()
        self.rotate.addItems(["no rotation", "90 deg clockwise", "180 deg", "90 deg anticlockwise"])
        self.flip_h, self.flip_v = QCheckBox("flip left-right"), QCheckBox("flip top-bottom")
        self.scale = QLineEdit()
        g.addWidget(self.rotate, 1, 1, 1, 2)
        g.addWidget(self.flip_h, 1, 3)
        g.addWidget(self.flip_v, 1, 4)
        g.addWidget(QLabel("Scale factor (blank = none):"), 2, 0)
        g.addWidget(self.scale, 2, 1)
        self.keep = QComboBox()
        self.keep.addItems(["mask: keep inside the vertices", "mask: keep outside the vertices"])
        g.addWidget(self.keep, 2, 2, 1, 2)
        clr = QPushButton("Clear mask vertices")
        clr.clicked.connect(self._clear_mask)
        g.addWidget(clr, 2, 4)
        lay.addLayout(g)
        modes = QHBoxLayout()
        self.mode_btns = {}
        modes.addWidget(QLabel("A click on the image sets a:"))
        for m in MODES:
            b = QRadioButton(m)
            b.toggled.connect(self._refresh_view)
            modes.addWidget(b)
            self.mode_btns[m] = b
        self.mode_btns[MODES[0]].setChecked(True)
        lay.addLayout(modes)
        self.mask_pts: list = []
        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.fig = Figure(figsize=(9, 3.6))
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.canvas.mpl_connect("button_press_event", self._clicked)
            lay.addWidget(self.canvas, 2)
        except Exception:
            self.fig = self.canvas = None
        # control points and legend
        mid = QHBoxLayout()
        left, right = QVBoxLayout(), QVBoxLayout()
        left.addWidget(QLabel("Control points (image col, row; map X, Y) - at least 3:"))
        self.cp = QTableWidget(0, 4)
        self.cp.setHorizontalHeaderLabels(["col", "row", "X", "Y"])
        self.cp.setMaximumHeight(130)
        left.addWidget(self.cp)
        row = QHBoxLayout()
        a, r = QPushButton("Add point"), QPushButton("Remove last")
        a.clicked.connect(lambda: self.cp.setRowCount(self.cp.rowCount() + 1))
        r.clicked.connect(lambda: self.cp.setRowCount(max(0, self.cp.rowCount() - 1)))
        self.geo_btn = QPushButton("From GeoTIFF corners")
        self.geo_btn.clicked.connect(self._from_geotiff)
        for w in (a, r, self.geo_btn):
            row.addWidget(w)
        left.addLayout(row)
        self.tol = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(QLabel("Largest accepted residual (map units):"))
        row.addWidget(self.tol)
        left.addLayout(row)
        mid.addLayout(left, 1)
        right.addWidget(QLabel("Legend: colour (R G B) tied to a facies of this project:"))
        self.legend = QTableWidget(0, 2)
        self.legend.setHorizontalHeaderLabels(["R G B", "Facies"])
        self.legend.horizontalHeader().setStretchLastSection(True)
        self.legend.setMaximumHeight(130)
        right.addWidget(self.legend)
        row = QHBoxLayout()
        a, r = QPushButton("Add colour"), QPushButton("Remove last")
        a.clicked.connect(self._add_legend)
        r.clicked.connect(lambda: self.legend.setRowCount(max(0, self.legend.rowCount() - 1)))
        row.addWidget(a)
        row.addWidget(r)
        right.addLayout(row)
        self.ctol = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(QLabel("Colour tolerance (RGB distance, 0-441):"))
        row.addWidget(self.ctol)
        right.addLayout(row)
        right.addWidget(QLabel("Facies absent from the map (fraction 0):"))
        self.absent = QListWidget()
        self.absent.setMaximumHeight(70)
        for f in self.scheme:
            it = QListWidgetItem(f.name)
            it.setFlags(it.flags() | it.flags().ItemIsUserCheckable)
            it.setCheckState(it.checkState().Unchecked)
            self.absent.addItem(it)
        right.addWidget(self.absent)
        mid.addLayout(right, 1)
        lay.addLayout(mid)
        row = QHBoxLayout()
        self.sigma, self.cover = QLineEdit(), QLineEdit()
        row.addWidget(QLabel("Smoothing sigma (pixels, 0 = none):"))
        row.addWidget(self.sigma)
        self.suggest_btn = QPushButton("Suggest")
        self.suggest_btn.clicked.connect(self.suggest)
        row.addWidget(self.suggest_btn)
        self.suggestion = QLabel("")
        row.addWidget(self.suggestion, 1)
        row.addWidget(QLabel("Least share of active columns covered (0-1]:"))
        row.addWidget(self.cover)
        lay.addLayout(row)
        self.problem = QLabel("")
        self.problem.setStyleSheet("color: #b00020")
        self.problem.setWordWrap(True)
        lay.addWidget(self.problem)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(150)
        lay.addWidget(self.report)
        row = QHBoxLayout()
        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self.process)
        self.save_btn = QPushButton("Save...")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        for w in (self.process_btn, self.save_btn, close):
            row.addWidget(w)
        lay.addLayout(row)
        for e in (*self.crop, self.scale, self.tol, self.ctol, self.sigma, self.cover):
            e.textChanged.connect(self._edited)
        for c in (self.rotate, self.keep, self.interval):
            c.currentTextChanged.connect(self._edited)
        for c in (self.flip_h, self.flip_v):
            c.toggled.connect(self._edited)
        self.cp.itemChanged.connect(self._edited)
        self.legend.itemChanged.connect(self._edited)
        self.absent.itemChanged.connect(self._edited)

    # -- helpers ------------------------------------------------------------------------------------------------------
    def spec(self):
        if self._spec is None:
            return _default_spec(self.project)
        return self._spec() if callable(self._spec) else self._spec

    def _edited(self, *_):
        self._ran = None
        self.save_btn.setEnabled(False)
        self._refresh_view()

    @staticmethod
    def _num(text, what, integer=False):
        try:
            v = float(text)
            if integer:
                if v != int(v):
                    raise ValueError
                return int(v)
            return v
        except ValueError:
            raise MapImportError(f"{what}: enter {'a whole number' if integer else 'a number'}") from None

    def _cell(self, table, r, c):
        it = table.item(r, c)
        return it.text() if it else ""

    def _add_legend(self):
        r = self.legend.rowCount()
        self.legend.setRowCount(r + 1)
        cb = QComboBox()
        cb.addItems([CHOOSE] + [f.name for f in self.scheme])
        cb.currentTextChanged.connect(self._edited)
        self.legend.setCellWidget(r, 1, cb)
        return r

    def facies_of_row(self, r):
        return self.legend.cellWidget(r, 1)

    def _clear_mask(self):
        self.mask_pts = []
        self._edited()

    # -- processing ops from the form ---------------------------------------------------------------------------------
    def read_ops(self):
        ops = []
        c = [e.text().strip() for e in self.crop]
        if any(c):
            if not all(c):
                raise MapImportError("crop: fill in all four numbers or none")
            ops.append(("crop", *[self._num(v, "crop", True) for v in c]))
        k = self.rotate.currentIndex()
        if k:
            ops.append(("rotate", k))
        if self.flip_h.isChecked() or self.flip_v.isChecked():
            ops.append(("flip", self.flip_h.isChecked(), self.flip_v.isChecked()))
        if self.scale.text().strip():
            ops.append(("scale", self._num(self.scale.text(), "scale factor")))
        if self.mask_pts:
            ops.append(("mask", tuple(tuple(p) for p in self.mask_pts), "inside" if self.keep.currentIndex() == 0 else "outside"))
        return tuple(ops)

    def processed(self):
        if self.raster is None:
            raise MapImportError("open an image first")
        return apply_ops(self.raster, tuple(o for o in self.read_ops() if o[0] != "mask"))

    def read_recipe(self) -> Recipe:
        if self.raster is None:
            raise MapImportError("open an image first")
        iv = self.interval.currentText()
        if iv == CHOOSE:
            raise MapImportError("choose the interval this map belongs to")
        ops = self.read_ops()
        cps = []
        for r in range(self.cp.rowCount()):
            cps.append(tuple(self._num(self._cell(self.cp, r, c), f"control point {r + 1}") for c in range(4)))
        entries = []
        for r in range(self.legend.rowCount()):
            parts = self._cell(self.legend, r, 0).replace(",", " ").split()
            if len(parts) != 3:
                raise MapImportError(f"legend row {r + 1}: type the colour as three numbers R G B")
            rgb = tuple(self._num(p, f"legend row {r + 1}", True) for p in parts)
            fname = self.facies_of_row(r).currentText()
            if fname == CHOOSE:
                raise MapImportError(f"legend row {r + 1}: choose the facies for colour {rgb}")
            entries.append((rgb, next(f.code for f in self.scheme if f.name == fname)))
        absent = tuple(f.code for i, f in enumerate(self.scheme) if self.absent.item(i).checkState() == self.absent.item(i).checkState().Checked)
        legend = Legend(tuple(entries), self._num(self.ctol.text(), "colour tolerance"), absent)
        return Recipe(iv, ops, tuple(cps), self._num(self.tol.text(), "residual tolerance"), legend,
                      self._num(self.sigma.text(), "smoothing sigma"), self._num(self.cover.text(), "minimum coverage"), self.path)

    # -- actions ------------------------------------------------------------------------------------------------------
    def open_image(self):
        p = self._pick()
        if not p:
            return
        try:
            self.raster = load_raster(p)
        except MapImportError as exc:
            self.problem.setText(str(exc))
            return
        self.path = str(p).replace("\\", "/").split("/")[-1]
        self.file_label.setText(f"{self.path}: {self.raster.shape[1]} x {self.raster.shape[0]} pixels"
                                + (" (GeoTIFF tags found)" if self.raster.geo_affine else ""))
        self.problem.setText("")
        self._edited()

    def _from_geotiff(self):
        try:
            if self.raster is None or self.raster.geo_affine is None:
                raise MapImportError("this image has no GeoTIFF georeferencing")
            if self.read_ops():
                raise MapImportError("the GeoTIFF corners refer to the original image: clear crop, rotate, flip, scale and mask first")
            a, b, c, d, e, f = self.raster.geo_affine
            h, w = self.raster.shape
            pts = [(0, 0), (w, 0), (0, h), (w, h)]
        except MapImportError as exc:
            self.problem.setText(str(exc))
            return
        self.cp.setRowCount(0)
        self.cp.setRowCount(len(pts))
        for r, (col, row) in enumerate(pts):
            for c_, v in enumerate((col, row, a * col + b * row + c, d * col + e * row + f)):
                self.cp.setItem(r, c_, QTableWidgetItem(f"{v:.10g}"))
        self.problem.setText("")

    def _clicked(self, ev):
        if ev.inaxes is None or ev.xdata is None or self.raster is None or getattr(ev.inaxes, "_cosma", "") != "image":
            return
        self.click(float(ev.xdata), float(ev.ydata))

    def click(self, col, row):
        mode = next(m for m, b in self.mode_btns.items() if b.isChecked())
        if mode == "control point":
            r = next((i for i in range(self.cp.rowCount()) if not self._cell(self.cp, i, 0) and not self._cell(self.cp, i, 1)), None)
            if r is None:
                r = self.cp.rowCount()
                self.cp.setRowCount(r + 1)
            self.cp.setItem(r, 0, QTableWidgetItem(f"{col:.2f}"))
            self.cp.setItem(r, 1, QTableWidgetItem(f"{row:.2f}"))
        elif mode == "legend colour":
            try:
                pr = self.processed()
                px = pr.rgb[min(int(row), pr.shape[0] - 1), min(int(col), pr.shape[1] - 1)]
            except MapImportError as exc:
                self.problem.setText(str(exc))
                return
            r = next((i for i in range(self.legend.rowCount()) if not self._cell(self.legend, i, 0)), None)
            if r is None:
                r = self._add_legend()
            self.legend.setItem(r, 0, QTableWidgetItem(f"{px[0]} {px[1]} {px[2]}"))
        elif mode == "mask vertex":
            self.mask_pts.append((float(col), float(row)))
            self._edited()

    def suggest(self):
        try:
            cps = [tuple(self._num(self._cell(self.cp, r, c), f"control point {r + 1}") for c in range(4))
                   for r in range(self.cp.rowCount())]
            tol = self._num(self.tol.text(), "residual tolerance")
            g = georeference(cps, tol)
            a, b, c, d, e, f = g.affine
            px = abs(a * e - b * d) ** 0.5
            sp = self.spec()
            cell = min(float(np.diff(sp.x_edges).min()), float(np.diff(sp.y_edges).min()))
            s = suggest_sigma(px, cell)
        except MapImportError as exc:
            self.suggestion.setText("")
            self.problem.setText(f"cannot suggest a radius yet: {exc}")
            return None
        self.problem.setText("")
        self.suggestion.setText(f"suggested {s:.3g} px = half the smallest model column ({cell:g}) at {px:.4g} per pixel; "
                                "you may change it")
        self.sigma.setText(f"{s:.3g}")
        return s

    def process(self):
        self.problem.setText("")
        self._ran = None
        self.save_btn.setEnabled(False)
        try:
            rec = self.read_recipe()
            pm, img = run_recipe(self.raster, rec, self.scheme, self.spec(), with_images=True)
        except MapImportError as exc:
            self.problem.setText(str(exc))
            self.report.setPlainText("")
            return
        self.result, self._img = pm, img
        self.report.setPlainText(pm.report)
        self._ran = recipe_to_state(rec)
        self.save_btn.setEnabled(True)
        self._refresh_view()

    def _refresh_view(self, *_):
        if self.fig is None:
            return
        self.fig.clear()
        try:
            pr = self.processed()
        except MapImportError:
            self.canvas.draw_idle()
            return
        ax = self.fig.add_subplot(1, 2, 1)
        ax._cosma = "image"
        ax.imshow(np.where(pr.valid[..., None], pr.rgb, 255), interpolation="nearest")
        ax.set_title("image after processing (click here)", fontsize=8)
        ax.tick_params(labelsize=6)
        for r in range(self.cp.rowCount()):
            try:
                ax.plot(float(self._cell(self.cp, r, 0)), float(self._cell(self.cp, r, 1)), "k+", ms=10)
                ax.annotate(str(r + 1), (float(self._cell(self.cp, r, 0)), float(self._cell(self.cp, r, 1))), fontsize=7)
            except ValueError:
                pass
        if self.mask_pts:
            p = np.array(self.mask_pts + self.mask_pts[:1])
            ax.plot(p[:, 0], p[:, 1], "m-o", ms=3)
        if self.result is not None and self._ran is not None:
            ax2 = self.fig.add_subplot(1, 2, 2)
            fr = self.result.fractions
            ax2.imshow(fr[..., 0].T, origin="lower", vmin=0, vmax=1, cmap="viridis")
            ax2.set_title(f"model grid: fraction of {self.scheme[0].name}", fontsize=8)
            ax2.tick_params(labelsize=6)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def save(self) -> bool:
        try:
            rec = self.read_recipe()
        except MapImportError as exc:
            self.problem.setText(str(exc))
            return False
        if self._ran is None or recipe_to_state(rec) != self._ran:
            self.problem.setText("Process these entries first: what is saved must be what was processed.")
            self.save_btn.setEnabled(False)
            return False
        replaced = rec.interval in self.project.proportion_maps
        text = self.result.report + ("\n\nThis replaces the existing map of this interval." if replaced else "") + "\n\nImport this map?"
        if not self._confirm(text):
            return False
        self.project.set_proportion_map(self.result)
        self.saved = True
        return True

"""The COSMA application icon: a simplified geological cross section (layers cut by a fault, one well).

Drawn in code, so it is sharp at every size (taskbar, title bar, alt-tab) and needs no image file at run time;
`export_icon_files` writes PNG and ICO copies for shortcuts and installers.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

SIZES = (16, 24, 32, 48, 64, 128, 256)
SKY = QColor("#cfe8f7")
LAYERS = (QColor("#e9c46a"), QColor("#f4a261"), QColor("#a1887f"), QColor("#455a64"))     # sand, reservoir, shale, basement
LINE = QColor("#1b2a33")
TOPS = (0.30, 0.46, 0.64, 0.80)         # top of each layer on the up-thrown (left) side, fraction of the height
THROW = 0.13                            # how far the right block has dropped
FAULT = ((0.60, -0.02), (0.44, 1.02))   # fault line, top to bottom (dipping to the left)


def _fault_x(y: float) -> float:
    (x0, y0), (x1, y1) = FAULT
    return x0 + (x1 - x0) * (y - y0) / (y1 - y0)


def paint(p: QPainter, size: int):
    """Draw the icon into a `size` x `size` square at the painter's origin."""
    s = float(size)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.save()
    p.scale(s, s)
    clip = QPainterPath()
    clip.addRoundedRect(QRectF(0.02, 0.02, 0.96, 0.96), 0.16, 0.16)
    p.setClipPath(clip)
    p.fillRect(QRectF(0, 0, 1, 1), SKY)

    def block(shift: float, right: bool):
        xs = lambda y: _fault_x(y)                                     # noqa: E731
        for k, top in enumerate(TOPS):
            y0 = top + shift
            y1 = TOPS[k + 1] + shift if k + 1 < len(TOPS) else 1.2
            pts = []
            if right:
                pts = [(xs(y0), y0), (1.02, y0), (1.02, y1), (xs(y1), y1)]
            else:
                pts = [(-0.02, y0), (xs(y0), y0), (xs(y1), y1), (-0.02, y1)]
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(LAYERS[k]))
            p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))

    block(0.0, False)
    block(THROW, True)
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(LINE, max(0.045, 1.4 / s), Qt.SolidLine, Qt.RoundCap))
    (x0, y0), (x1, y1) = FAULT
    p.drawLine(QPointF(x0, y0), QPointF(x1, y1))
    if size >= 32:                                                      # the well: too fine to read below this size
        p.setPen(QPen(LINE, 0.03, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(0.2, 0.16), QPointF(0.2, 0.74))
        p.setBrush(QBrush(QColor("#ffffff")))
        p.drawEllipse(QPointF(0.2, 0.16), 0.035, 0.035)
    p.restore()
    p.setClipping(False)
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(LINE, max(0.03 * s, 1.0)))
    p.drawRoundedRect(QRectF(0.02 * s, 0.02 * s, 0.96 * s, 0.96 * s), 0.16 * s, 0.16 * s)


def pixmap(size: int) -> QPixmap:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    paint(p, size)
    p.end()
    return QPixmap.fromImage(img)


def make_icon() -> QIcon:
    icon = QIcon()
    for n in SIZES:
        icon.addPixmap(pixmap(n))
    return icon


def export_icon_files(folder) -> list:
    """Write cosma_icon.png (256 px) and cosma.ico (all sizes, needs Pillow) into `folder`; returns the paths written."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    png = folder / "cosma_icon.png"
    pixmap(256).save(str(png), "PNG")
    out = [png]
    try:
        from PIL import Image
        Image.open(png).save(folder / "cosma.ico", sizes=[(n, n) for n in SIZES if n <= 256])
        out.append(folder / "cosma.ico")
    except ImportError:
        pass
    return out


def set_app_identity(app) -> None:
    """Window/taskbar icon for every window, and on Windows a stable app id so the taskbar shows COSMA's own icon and
    groups its windows, instead of the icon of python.exe."""
    import sys
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("COSMA.GeomodelBuilder")
        except Exception:
            pass
    app.setWindowIcon(make_icon())

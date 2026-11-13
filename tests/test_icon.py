import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from test_main_window import app  # noqa: F401

from cosma.gui import icon


def test_icon_has_every_size_and_is_not_blank(app):
    ic = icon.make_icon()
    assert {s.width() for s in ic.availableSizes()} >= set(icon.SIZES)
    for n in icon.SIZES:
        img = icon.pixmap(n).toImage()
        assert img.width() == n and img.pixelColor(0, 0).alpha() == 0                 # rounded corner is transparent
        colours = {img.pixelColor(x, y).rgb() for x in range(0, n, max(1, n // 16)) for y in range(0, n, max(1, n // 16))}
        assert len(colours) >= 5, n                                                      # sky, several layers, outline


def test_app_identity_sets_the_window_icon(app):
    icon.set_app_identity(app)
    assert not app.windowIcon().isNull()


def test_export_writes_png_and_ico(app, tmp_path):
    out = icon.export_icon_files(tmp_path)
    assert (tmp_path / "cosma_icon.png").stat().st_size > 500
    try:
        import PIL  # noqa: F401
        assert (tmp_path / "cosma.ico").stat().st_size > 500 and len(out) == 2
    except ImportError:
        assert len(out) == 1

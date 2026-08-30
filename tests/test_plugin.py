"""
Basic smoke tests for napari-deepsegclem.
Run with:  pytest tests/ -v
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Core module tests (no GPU / no models needed)
# ---------------------------------------------------------------------------

def test_ensure_u8_gray():
    from napari_deepsegclem._core import _ensure_u8_gray
    m = np.random.rand(64, 64).astype(np.float32)
    out = _ensure_u8_gray(m)
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})


def test_skeleton_layers():
    from napari_deepsegclem._core import build_skeleton_layers
    binary = np.zeros((128, 128), np.uint8)
    binary[60:70, 20:110] = 1  # horizontal bar
    skel, ep, bp = build_skeleton_layers(binary)
    assert skel.any(), "Skeleton should not be empty"
    assert ep.any(),   "Should have endpoints"


def test_quantify_components():
    from napari_deepsegclem._core import quantify_components
    binary = np.zeros((64, 64), np.uint8)
    binary[10:20, 10:20] = 1
    binary[40:50, 40:50] = 1
    df = quantify_components(binary, "LM", "test")
    assert len(df) == 2
    assert "area_px" in df.columns
    assert "circularity" in df.columns
    assert "skeleton_length_px" in df.columns


def test_rotate_with_M():
    from napari_deepsegclem._core import _rotate_with_M
    img = np.zeros((100, 80), np.uint8)
    img[40:60, 30:50] = 255
    rot, M, wh = _rotate_with_M(img, 45)
    assert rot.shape[0] > 0 and rot.shape[1] > 0
    assert M.shape == (2, 3)


def test_map_box_to_original():
    from napari_deepsegclem._core import _rotate_with_M, _map_box_to_original
    img = np.zeros((200, 200), np.uint8)
    _, M, rot_wh = _rotate_with_M(img, 30)
    poly = _map_box_to_original((10, 10), (50, 50), M, rot_wh, 1)
    assert poly.shape == (4, 2)


# ---------------------------------------------------------------------------
# Widget smoke test (requires qtpy / PyQt5 / PySide2)
# ---------------------------------------------------------------------------

@pytest.fixture
def make_napari_viewer(qtbot):
    """Minimal stub so tests work without a real napari install."""
    try:
        import napari
        viewer = napari.Viewer(show=False)
        yield viewer
        viewer.close()
    except ImportError:
        pytest.skip("napari not installed")


def test_widget_instantiation(make_napari_viewer):
    from napari_deepsegclem._widget import DeepSegCLEMWidget
    widget = DeepSegCLEMWidget(make_napari_viewer)
    assert widget is not None
    assert widget.tabs.count() == 5

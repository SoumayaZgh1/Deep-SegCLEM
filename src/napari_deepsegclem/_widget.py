"""
_widget.py
----------
Main napari dock widget for Deep-SegCLEM.
Tabs: Load → Segment → Correlate → Quantify
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from qtpy.QtCore import QThread, Signal, Qt
from qtpy.QtGui import QFont, QColor, QPalette
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QTabWidget, QGroupBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QTextEdit, QProgressBar, QSplitter,
    QSizePolicy, QScrollArea, QFrame, QComboBox, QSlider,
    QMessageBox,
)

# ---------------------------------------------------------------------------
# Colour palette (matches original dark theme)
# ---------------------------------------------------------------------------
ACCENT  = "#5E81AC"
ACCENT2 = "#88C0D0"
SUCCESS = "#98C379"
WARN    = "#E5C07B"
DANGER  = "#E06C75"
MUTED   = "#A7AEBE"
TEXT    = "#E5E9F0"
PANEL   = "#2B2E3B"


def _btn(label: str, color: str = ACCENT, tooltip: str = "") -> QPushButton:
    b = QPushButton(label)
    b.setStyleSheet(
        f"QPushButton {{ background: {color}; color: {TEXT}; border: none; "
        f"border-radius: 4px; padding: 6px 14px; font-weight: bold; }}"
        f"QPushButton:hover {{ background: #6e91bc; }}"
        f"QPushButton:disabled {{ background: #3a3f4b; color: {MUTED}; }}"
    )
    if tooltip:
        b.setToolTip(tooltip)
    return b


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {ACCENT2}; font-weight: bold; font-size: 11px;")
    return lbl


def _info_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
    lbl.setWordWrap(True)
    return lbl


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

class _ModelLoaderThread(QThread):
    done = Signal(object, str)   # (models_tuple | None, message)

    def __init__(self, lm_path: str, em_path: str):
        super().__init__()
        self.lm_path = lm_path
        self.em_path = em_path

    def run(self):
        try:
            from ._core import load_models
            models = load_models(self.lm_path, self.em_path)
            self.done.emit(models, "Models loaded successfully.")
        except Exception:
            self.done.emit(None, traceback.format_exc())


class _SegThread(QThread):
    done = Signal(dict)
    progress = Signal(str)

    def __init__(self, lm_src, em_src, lm_model, em_model, thr_lm, thr_em):
        super().__init__()
        self.lm_src = lm_src; self.em_src = em_src
        self.lm_model = lm_model; self.em_model = em_model
        self.thr_lm = thr_lm; self.thr_em = thr_em

    def run(self):
        try:
            from ._core import segment_lm, segment_em
            self.progress.emit("Segmenting LM image…")
            m1 = segment_lm(self.lm_src, self.lm_model, self.thr_lm)
            self.progress.emit("Segmenting EM image…")
            m2 = segment_em(self.em_src, self.em_model, self.thr_em)
            os.makedirs("segmentation_results", exist_ok=True)
            import cv2
            cv2.imwrite("segmentation_results/mask1_LM_original.png", m1)
            cv2.imwrite("segmentation_results/mask2_EM.png", m2)
            self.done.emit({"ok": True, "lm_mask": m1, "em_mask": m2,
                            "lm_path": "segmentation_results/mask1_LM_original.png",
                            "em_path": "segmentation_results/mask2_EM.png"})
        except Exception:
            self.done.emit({"ok": False, "msg": traceback.format_exc()})


class _CorrThread(QThread):
    done = Signal(dict)
    progress = Signal(int, str)

    def __init__(self, lm_mask, em_mask, lm_src, em_src, out_dir,
                 angle_step, scale_list):
        super().__init__()
        self.lm_mask = lm_mask; self.em_mask = em_mask
        self.lm_src = lm_src;   self.em_src = em_src
        self.out_dir = out_dir
        self.angle_step = angle_step
        self.scale_list = scale_list

    def run(self):
        try:
            from ._core import correlate
            result = correlate(
                self.lm_mask, self.em_mask,
                self.lm_src, self.em_src,
                out_dir=self.out_dir,
                angles=range(0, 360, self.angle_step),
                scale_percents=self.scale_list,
                progress_callback=lambda p, m: self.progress.emit(p, m),
            )
            self.done.emit(result)
        except Exception:
            self.done.emit({"ok": False, "msg": traceback.format_exc()})


class _QuantThread(QThread):
    done = Signal(dict)
    progress = Signal(str)

    def __init__(self, lm_mask, em_mask, lm_src, em_src, out_dir):
        super().__init__()
        self.lm_mask = lm_mask; self.em_mask = em_mask
        self.lm_src = lm_src;   self.em_src = em_src
        self.out_dir = out_dir

    def run(self):
        try:
            from ._core import run_quantification
            self.progress.emit("Computing skeletons and metrics…")
            result = run_quantification(
                self.lm_mask, self.em_mask,
                self.lm_src, self.em_src,
                out_dir=self.out_dir,
            )
            self.done.emit(result)
        except Exception:
            self.done.emit({"ok": False, "msg": traceback.format_exc()})


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class DeepSegCLEMWidget(QWidget):
    """Napari dock widget for Deep-SegCLEM."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # State
        self._lm_source = None   # file path (str)
        self._em_source = None   # file path (str)
        self._lm_mask: Optional[np.ndarray] = None
        self._em_mask: Optional[np.ndarray] = None
        self._lm_model = None
        self._em_model = None
        self._corr_result: Optional[dict] = None
        self._worker = None

        self._build_ui()
        self._apply_stylesheet()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        # ---- Header ----
        hdr = QLabel("Deep-SegCLEM  ·  Mitochondria Analysis Suite")
        hdr.setStyleSheet(f"color: {TEXT}; font-size: 13px; font-weight: bold;")
        hdr.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(hdr)

        sub = QLabel("Max Planck Institute for Biophysics · Zaghbani et al. 2025")
        sub.setStyleSheet(f"color: {MUTED}; font-size: 9px;")
        sub.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(sub)

        # ---- Tabs ----
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: 1px solid #3A3F4B; background: {PANEL}; }}"
            f"QTabBar::tab {{ background: #232634; color: {MUTED}; padding: 6px 14px; }}"
            f"QTabBar::tab:selected {{ background: {PANEL}; color: {TEXT}; }}"
        )
        root_layout.addWidget(self.tabs)

        self.tabs.addTab(self._build_setup_tab(),    "⚙️  Setup")
        self.tabs.addTab(self._build_load_tab(),     "📂  Load")
        self.tabs.addTab(self._build_segment_tab(),  "🔬  Segment")
        self.tabs.addTab(self._build_correlate_tab(),"🔗  Correlate")
        self.tabs.addTab(self._build_quantify_tab(), "📊  Quantify")

        # ---- Progress bar ----
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setStyleSheet(
            f"QProgressBar {{ background: #232634; border-radius: 3px; color: {TEXT}; }}"
            f"QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}"
        )
        root_layout.addWidget(self._progress)

        # ---- Status log ----
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(90)
        self._log.setStyleSheet(
            f"QTextEdit {{ background: #1a1b24; color: {MUTED}; "
            f"border: 1px solid #3A3F4B; font-size: 10px; }}"
        )
        root_layout.addWidget(self._log)

    # ---- Setup tab ----
    def _build_setup_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(10)

        lay.addWidget(_section_label("Model Weights"))
        lay.addWidget(_info_label(
            "Select your trained .h5 weight files. They are loaded once and "
            "stay in memory for all subsequent operations."
        ))

        # LM weights
        lm_grp = QGroupBox("LM Model  (FCN-ResNet50)")
        lm_lay = QHBoxLayout(lm_grp)
        self._lm_weights_lbl = QLabel("fcn_resnet50_best.h5")
        self._lm_weights_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        lm_lay.addWidget(self._lm_weights_lbl, 1)
        btn_lm_w = _btn("Browse", PANEL)
        btn_lm_w.clicked.connect(self._browse_lm_weights)
        lm_lay.addWidget(btn_lm_w)
        lay.addWidget(lm_grp)

        # EM weights
        em_grp = QGroupBox("EM Model  (FCN-CoReNet)")
        em_lay = QHBoxLayout(em_grp)
        self._em_weights_lbl = QLabel("savedcopy.h5")
        self._em_weights_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        em_lay.addWidget(self._em_weights_lbl, 1)
        btn_em_w = _btn("Browse", PANEL)
        btn_em_w.clicked.connect(self._browse_em_weights)
        em_lay.addWidget(btn_em_w)
        lay.addWidget(em_grp)

        # Output directory
        out_grp = QGroupBox("Output Directory")
        out_lay = QHBoxLayout(out_grp)
        self._out_dir_lbl = QLabel(os.path.abspath("."))
        self._out_dir_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        out_lay.addWidget(self._out_dir_lbl, 1)
        btn_out = _btn("Browse", PANEL)
        btn_out.clicked.connect(self._browse_out_dir)
        out_lay.addWidget(btn_out)
        lay.addWidget(out_grp)

        # Load button
        self._btn_load_models = _btn("Load Models", ACCENT,
                                     "Build model architectures and load weights")
        self._btn_load_models.clicked.connect(self._load_models)
        lay.addWidget(self._btn_load_models)

        self._model_status = QLabel("Models: not loaded")
        self._model_status.setStyleSheet(f"color: {DANGER}; font-weight: bold;")
        self._model_status.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._model_status)

        lay.addStretch()
        return w

    # ---- Load tab ----
    def _build_load_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(10)

        lay.addWidget(_section_label("Load Images into Napari"))
        lay.addWidget(_info_label(
            "Images are added as layers in the Napari viewer. "
            "LM = RGB/fluorescence · EM = Grayscale electron micrograph."
        ))

        lm_grp = QGroupBox("Light Microscopy (LM)")
        lm_lay = QVBoxLayout(lm_grp)
        self._lm_path_lbl = QLabel("No file selected")
        self._lm_path_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        lm_lay.addWidget(self._lm_path_lbl)
        row = QHBoxLayout()
        btn_lm = _btn("Browse LM", ACCENT)
        btn_lm.clicked.connect(self._load_lm)
        row.addWidget(btn_lm)
        self._btn_lm_active = _btn("Set Active Layer as LM", PANEL)
        self._btn_lm_active.clicked.connect(self._set_active_as_lm)
        row.addWidget(self._btn_lm_active)
        lm_lay.addLayout(row)
        lay.addWidget(lm_grp)

        em_grp = QGroupBox("Electron Microscopy (EM)")
        em_lay = QVBoxLayout(em_grp)
        self._em_path_lbl = QLabel("No file selected")
        self._em_path_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        em_lay.addWidget(self._em_path_lbl)
        row2 = QHBoxLayout()
        btn_em = _btn("Browse EM", ACCENT)
        btn_em.clicked.connect(self._load_em)
        row2.addWidget(btn_em)
        self._btn_em_active = _btn("Set Active Layer as EM", PANEL)
        self._btn_em_active.clicked.connect(self._set_active_as_em)
        row2.addWidget(self._btn_em_active)
        em_lay.addLayout(row2)
        lay.addWidget(em_grp)

        # Organelle channel
        org_grp = QGroupBox("Organelle Channel (optional, e.g. DRP1)")
        org_lay = QVBoxLayout(org_grp)
        self._org_path_lbl = QLabel("No file selected")
        self._org_path_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        org_lay.addWidget(self._org_path_lbl)
        btn_org = _btn("Browse Organelle Channel", PANEL)
        btn_org.clicked.connect(self._load_organelle)
        org_lay.addWidget(btn_org)
        lay.addWidget(org_grp)

        lay.addStretch()
        return w

    # ---- Segment tab ----
    def _build_segment_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(10)

        lay.addWidget(_section_label("Deep Learning Segmentation"))
        lay.addWidget(_info_label(
            "Runs the trained FCN-ResNet50 (LM) and FCN-CoReNet (EM) models. "
            "Masks are added as label layers in the viewer."
        ))

        # Thresholds
        thr_grp = QGroupBox("Prediction Thresholds")
        thr_lay = QHBoxLayout(thr_grp)
        thr_lay.addWidget(QLabel("LM threshold:"))
        self._thr_lm = QDoubleSpinBox()
        self._thr_lm.setRange(0.1, 0.95); self._thr_lm.setSingleStep(0.05)
        self._thr_lm.setValue(0.5)
        thr_lay.addWidget(self._thr_lm)
        thr_lay.addSpacing(20)
        thr_lay.addWidget(QLabel("EM threshold:"))
        self._thr_em = QDoubleSpinBox()
        self._thr_em.setRange(0.1, 0.95); self._thr_em.setSingleStep(0.05)
        self._thr_em.setValue(0.5)
        thr_lay.addWidget(self._thr_em)
        lay.addWidget(thr_grp)

        # Mask editor hint
        edit_grp = QGroupBox("Mask Editor (after segmentation)")
        edit_lay = QVBoxLayout(edit_grp)
        edit_lay.addWidget(_info_label(
            "Use Napari's built-in paint/erase tools on the label layer directly.\n"
            "Select the mask layer → click the Paint Bucket or Eraser in the toolbar.\n"
            "Changes are reflected instantly in all downstream steps."
        ))
        row_edit = QHBoxLayout()
        self._btn_edit_lm = _btn("Select LM Mask Layer", PANEL)
        self._btn_edit_lm.clicked.connect(lambda: self._select_layer("LM_mask"))
        row_edit.addWidget(self._btn_edit_lm)
        self._btn_edit_em = _btn("Select EM Mask Layer", PANEL)
        self._btn_edit_em.clicked.connect(lambda: self._select_layer("EM_mask"))
        row_edit.addWidget(self._btn_edit_em)
        edit_lay.addLayout(row_edit)
        edit_lay.addWidget(_info_label(
            "💡 After editing, click 'Sync Masks from Layers' to push edits "
            "back into memory before running Correlate or Quantify."
        ))
        self._btn_sync = _btn("Sync Masks from Layers → Memory", ACCENT2)
        self._btn_sync.clicked.connect(self._sync_masks)
        edit_lay.addWidget(self._btn_sync)
        lay.addWidget(edit_grp)

        self._btn_segment = _btn("▶  Run Segmentation", ACCENT)
        self._btn_segment.clicked.connect(self._run_segment)
        lay.addWidget(self._btn_segment)

        lay.addStretch()
        return w

    # ---- Correlate tab ----
    def _build_correlate_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(10)

        lay.addWidget(_section_label("LM ↔ EM Correlation"))
        lay.addWidget(_info_label(
            "Template matching over rotation, flip, and scale to align "
            "the LM segmentation mask with the EM mask."
        ))

        # Search params
        sp_grp = QGroupBox("Search Parameters")
        sp_lay = QHBoxLayout(sp_grp)
        sp_lay.addWidget(QLabel("Angle step (°):"))
        self._angle_step = QSpinBox()
        self._angle_step.setRange(1, 10); self._angle_step.setValue(1)
        sp_lay.addWidget(self._angle_step)
        sp_lay.addSpacing(16)
        sp_lay.addWidget(QLabel("Scale presets:"))
        self._scale_combo = QComboBox()
        self._scale_combo.addItem("Fine  (20–100 %)",   [20,30,40,50,60,70,80,90,100])
        self._scale_combo.addItem("Medium (25–75 %)",   [25,50,75])
        self._scale_combo.addItem("Coarse (20,50)",      [20,50])
        sp_lay.addWidget(self._scale_combo)
        lay.addWidget(sp_grp)

        self._btn_correlate = _btn("▶  Run Correlation", ACCENT)
        self._btn_correlate.clicked.connect(self._run_correlate)
        lay.addWidget(self._btn_correlate)

        # Results summary
        self._corr_result_lbl = QLabel("No correlation run yet.")
        self._corr_result_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        self._corr_result_lbl.setWordWrap(True)
        lay.addWidget(self._corr_result_lbl)

        # View outputs
        self._btn_view_corr = _btn("View Correlation Outputs in Napari", PANEL)
        self._btn_view_corr.clicked.connect(self._view_corr_outputs)
        self._btn_view_corr.setEnabled(False)
        lay.addWidget(self._btn_view_corr)

        # Organelle crop
        org_grp = QGroupBox("Apply Same Transform to Organelle Channel")
        org_lay = QVBoxLayout(org_grp)
        org_lay.addWidget(_info_label(
            "If you loaded an organelle channel (e.g. DRP1), the same rotation/flip "
            "is applied and the matched region is cropped and added as a layer."
        ))
        self._btn_org_crop = _btn("Crop Organelle (same region)", PANEL)
        self._btn_org_crop.clicked.connect(self._crop_organelle)
        org_lay.addWidget(self._btn_org_crop)
        lay.addWidget(org_grp)

        lay.addStretch()
        return w

    # ---- Quantify tab ----
    def _build_quantify_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(10)

        lay.addWidget(_section_label("Quantification & Export"))
        lay.addWidget(_info_label(
            "Computes per-component metrics (area, perimeter, circularity, "
            "skeleton length, endpoints, branch points) for both LM and EM masks.\n"
            "Saves skeleton overlay images and Excel files."
        ))

        out_grp = QGroupBox("Output Directory")
        out_lay = QHBoxLayout(out_grp)
        self._quant_out_lbl = QLabel("quantification_results")
        self._quant_out_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        out_lay.addWidget(self._quant_out_lbl, 1)
        btn_qout = _btn("Browse", PANEL)
        btn_qout.clicked.connect(self._browse_quant_dir)
        out_lay.addWidget(btn_qout)
        lay.addWidget(out_grp)

        self._btn_quantify = _btn("▶  Run Quantify & Export", ACCENT)
        self._btn_quantify.clicked.connect(self._run_quantify)
        lay.addWidget(self._btn_quantify)

        self._quant_result_lbl = QLabel("No quantification run yet.")
        self._quant_result_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        self._quant_result_lbl.setWordWrap(True)
        lay.addWidget(self._quant_result_lbl)

        self._btn_view_quant = _btn("View Skeleton Overlays in Napari", PANEL)
        self._btn_view_quant.clicked.connect(self._view_quant_outputs)
        self._btn_view_quant.setEnabled(False)
        lay.addWidget(self._btn_view_quant)

        lay.addWidget(_section_label("Legend"))
        legend = QLabel(
            "⬜ Skeleton   🟢 Endpoints   🔴 Branch points"
        )
        legend.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        lay.addWidget(legend)

        lay.addStretch()
        return w

    # ------------------------------------------------------------------
    # Stylesheet
    # ------------------------------------------------------------------
    def _apply_stylesheet(self):
        self.setStyleSheet(
            f"QWidget {{ background-color: #1E1F29; color: {TEXT}; "
            f"font-family: 'Segoe UI', Arial, sans-serif; font-size: 11px; }}"
            f"QGroupBox {{ border: 1px solid #3A3F4B; border-radius: 4px; "
            f"margin-top: 8px; padding: 8px; color: {TEXT}; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; "
            f"padding: 0 4px; color: {ACCENT2}; font-weight: bold; }}"
            f"QLabel {{ color: {TEXT}; }}"
            f"QSpinBox, QDoubleSpinBox, QComboBox {{ background: #2B2E3B; "
            f"color: {TEXT}; border: 1px solid #3A3F4B; border-radius: 3px; padding: 2px; }}"
            f"QScrollBar:vertical {{ background: #1E1F29; width: 8px; }}"
            f"QScrollBar::handle:vertical {{ background: #3A3F4B; border-radius: 4px; }}"
        )

    # ------------------------------------------------------------------
    # Helper: log
    # ------------------------------------------------------------------
    def _log_msg(self, msg: str, color: str = TEXT):
        self._log.append(f'<span style="color:{color}">{msg}</span>')
        self._log.ensureCursorVisible()

    def _set_progress(self, pct: int, msg: str = ""):
        self._progress.setValue(pct)
        if msg:
            self._progress.setFormat(f"{msg}  {pct}%")

    # ------------------------------------------------------------------
    # Setup actions
    # ------------------------------------------------------------------
    def _browse_lm_weights(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select LM weights (.h5)", "", "HDF5 (*.h5)")
        if p:
            self._lm_weights_lbl.setText(p)
            self._log_msg(f"LM weights: {p}")

    def _browse_em_weights(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select EM weights (.h5)", "", "HDF5 (*.h5)")
        if p:
            self._em_weights_lbl.setText(p)
            self._log_msg(f"EM weights: {p}")

    def _browse_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self._out_dir_lbl.setText(d)

    def _load_models(self):
        lm_w = self._lm_weights_lbl.text().strip()
        em_w = self._em_weights_lbl.text().strip()

        if not lm_w or not os.path.exists(lm_w):
            # try relative/resource path
            import sys
            base = getattr(sys, "_MEIPASS", os.path.abspath("."))
            lm_w = os.path.join(base, "fcn_resnet50_best.h5")
        if not em_w or not os.path.exists(em_w):
            import sys
            base = getattr(sys, "_MEIPASS", os.path.abspath("."))
            em_w = os.path.join(base, "savedcopy.h5")

        if not os.path.exists(lm_w):
            self._log_msg(f"LM weights not found: {lm_w}", DANGER)
            return
        if not os.path.exists(em_w):
            self._log_msg(f"EM weights not found: {em_w}", DANGER)
            return

        self._log_msg("Loading models… (this may take a moment)", WARN)
        self._btn_load_models.setEnabled(False)
        self._set_progress(10, "Loading models")

        self._model_thread = _ModelLoaderThread(lm_w, em_w)
        self._model_thread.done.connect(self._on_models_loaded)
        self._model_thread.start()

    def _on_models_loaded(self, result, msg):
        self._btn_load_models.setEnabled(True)
        self._set_progress(0)
        if result is None:
            self._log_msg(f"Model loading failed:\n{msg}", DANGER)
            self._model_status.setText("Models: FAILED ✗")
            self._model_status.setStyleSheet(f"color: {DANGER}; font-weight: bold;")
        else:
            self._lm_model, self._em_model = result
            self._log_msg("✅ Models loaded successfully.", SUCCESS)
            self._model_status.setText("Models: loaded ✓")
            self._model_status.setStyleSheet(f"color: {SUCCESS}; font-weight: bold;")

    # ------------------------------------------------------------------
    # Load actions
    # ------------------------------------------------------------------
    def _load_lm(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select LM Image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)"
        )
        if p:
            self._load_image_as_layer(p, "LM_image", "lm")

    def _load_em(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select EM Image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)"
        )
        if p:
            self._load_image_as_layer(p, "EM_image", "em")

    def _load_organelle(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Organelle Channel", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)"
        )
        if p:
            self._organelle_source = p
            import cv2
            self._organelle_img = cv2.imread(p, cv2.IMREAD_COLOR)
            arr = cv2.cvtColor(self._organelle_img, cv2.COLOR_BGR2RGB)
            self._add_or_replace_layer(arr, "Organelle_channel", kind="image")
            self._org_path_lbl.setText(os.path.basename(p))
            self._log_msg(f"Organelle channel loaded: {p}", SUCCESS)

    def _load_image_as_layer(self, path: str, layer_name: str, kind: str):
        try:
            import cv2
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Cannot read: {path}")
            # Convert to RGB for display
            if img.ndim == 2:
                display = img
            elif img.shape[2] == 3:
                display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                display = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

            self._add_or_replace_layer(display, layer_name, kind="image")

            if kind == "lm":
                self._lm_source = path
                self._lm_path_lbl.setText(os.path.basename(path))
                self._log_msg(f"LM image loaded: {path}", SUCCESS)
            else:
                self._em_source = path
                self._em_path_lbl.setText(os.path.basename(path))
                self._log_msg(f"EM image loaded: {path}", SUCCESS)
        except Exception:
            self._log_msg(traceback.format_exc(), DANGER)

    def _set_active_as_lm(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            self._log_msg("No active layer selected.", WARN)
            return
        data = layer.data
        if hasattr(data, 'compute'):
            data = data.compute()
        # store the path if available
        if hasattr(layer, 'source') and layer.source and layer.source.path:
            self._lm_source = str(layer.source.path)
        else:
            # save as temp file
            import cv2, tempfile
            tmp = tempfile.mktemp(suffix=".png")
            if data.ndim == 3:
                cv2.imwrite(tmp, cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(tmp, data.astype(np.uint8))
            self._lm_source = tmp
        self._lm_path_lbl.setText(layer.name)
        self._log_msg(f"Active layer '{layer.name}' set as LM.", SUCCESS)

    def _set_active_as_em(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            self._log_msg("No active layer selected.", WARN)
            return
        data = layer.data
        if hasattr(data, 'compute'):
            data = data.compute()
        if hasattr(layer, 'source') and layer.source and layer.source.path:
            self._em_source = str(layer.source.path)
        else:
            import cv2, tempfile
            tmp = tempfile.mktemp(suffix=".png")
            if data.ndim == 2:
                cv2.imwrite(tmp, data.astype(np.uint8))
            else:
                cv2.imwrite(tmp, cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_RGB2BGR))
            self._em_source = tmp
        self._em_path_lbl.setText(layer.name)
        self._log_msg(f"Active layer '{layer.name}' set as EM.", SUCCESS)

    # ------------------------------------------------------------------
    # Segment actions
    # ------------------------------------------------------------------
    def _run_segment(self):
        if self._lm_model is None or self._em_model is None:
            self._log_msg("Load models first (Setup tab).", DANGER)
            return
        if not self._lm_source or not self._em_source:
            self._log_msg("Load both LM and EM images first.", DANGER)
            return

        self._btn_segment.setEnabled(False)
        self._set_progress(5, "Segmenting")
        self._log_msg("Starting segmentation…", WARN)

        self._seg_thread = _SegThread(
            self._lm_source, self._em_source,
            self._lm_model, self._em_model,
            self._thr_lm.value(), self._thr_em.value()
        )
        self._seg_thread.progress.connect(lambda m: self._log_msg(m))
        self._seg_thread.done.connect(self._on_segment_done)
        self._seg_thread.start()

    def _on_segment_done(self, result: dict):
        self._btn_segment.setEnabled(True)
        self._set_progress(0)
        if not result["ok"]:
            self._log_msg(f"Segmentation failed:\n{result['msg']}", DANGER)
            return

        self._lm_mask = result["lm_mask"]
        self._em_mask = result["em_mask"]

        # Add as label layers
        lm_label = (self._lm_mask > 0).astype(np.uint8)
        em_label = (self._em_mask > 0).astype(np.uint8)
        self._add_or_replace_layer(lm_label, "LM_mask", kind="labels")
        self._add_or_replace_layer(em_label, "EM_mask", kind="labels")

        self._log_msg(
            f"✅ Segmentation done. LM mask saved → {result['lm_path']}\n"
            f"   EM mask saved → {result['em_path']}", SUCCESS
        )
        self._log_msg(
            "💡 Edit masks: select the label layer, use Paint/Erase tools, "
            "then click 'Sync Masks from Layers'.", MUTED
        )

    def _select_layer(self, name: str):
        for layer in self.viewer.layers:
            if layer.name == name:
                self.viewer.layers.selection.active = layer
                self._log_msg(f"Layer '{name}' selected. Use Napari toolbar to paint/erase.", SUCCESS)
                return
        self._log_msg(f"Layer '{name}' not found. Run segmentation first.", WARN)

    def _sync_masks(self):
        """Pull current label layer data back into _lm_mask / _em_mask."""
        updated = []
        for layer in self.viewer.layers:
            data = layer.data
            if hasattr(data, 'compute'):
                data = data.compute()
            if layer.name == "LM_mask":
                self._lm_mask = (data > 0).astype(np.uint8) * 255
                updated.append("LM")
            elif layer.name == "EM_mask":
                self._em_mask = (data > 0).astype(np.uint8) * 255
                updated.append("EM")
        if updated:
            self._log_msg(f"✅ Synced masks from layers: {', '.join(updated)}", SUCCESS)
        else:
            self._log_msg("No LM_mask or EM_mask layers found to sync.", WARN)

    # ------------------------------------------------------------------
    # Correlate actions
    # ------------------------------------------------------------------
    def _run_correlate(self):
        if self._lm_mask is None or self._em_mask is None:
            self._log_msg("Run segmentation first.", DANGER)
            return
        if not self._lm_source or not self._em_source:
            self._log_msg("LM and EM image sources required.", DANGER)
            return

        scale_list = self._scale_combo.currentData()
        self._btn_correlate.setEnabled(False)
        self._set_progress(0, "Correlating")
        self._log_msg("Starting correlation (this can take several minutes)…", WARN)

        out_dir = os.path.join(self._out_dir_lbl.text().strip() or ".", "correlation_results")

        self._corr_thread = _CorrThread(
            self._lm_mask, self._em_mask,
            self._lm_source, self._em_source,
            out_dir,
            self._angle_step.value(),
            scale_list,
        )
        self._corr_thread.progress.connect(
            lambda p, m: (self._set_progress(p, m), self._log_msg(m)))
        self._corr_thread.done.connect(self._on_correlate_done)
        self._corr_thread.start()

    def _on_correlate_done(self, result: dict):
        self._btn_correlate.setEnabled(True)
        if not result["ok"]:
            self._log_msg(f"Correlation failed:\n{result.get('msg','')}", DANGER)
            self._set_progress(0)
            return

        self._corr_result = result
        self._set_progress(100, "Correlation complete")
        self._btn_view_corr.setEnabled(True)

        summary = (
            f"✅ Correlation done!\n"
            f"   Score: {result['score']:.4f}  |  Angle: {result['angle']}°  |  "
            f"Flip: {result['flip']}  |  Scale: {result['scale_percent']}%"
        )
        self._corr_result_lbl.setText(summary)
        self._log_msg(summary, SUCCESS)
        self._set_progress(0)

    def _view_corr_outputs(self):
        if not self._corr_result:
            return
        paths = [
            self._corr_result.get("result_path"),
            self._corr_result.get("overlay_path"),
            self._corr_result.get("cropped_lm_original_path"),
            self._corr_result.get("em_overlay_crop_path"),
            self._corr_result.get("lm_original_poly_path"),
        ]
        for p in paths:
            if p and os.path.exists(p):
                self._open_image_in_napari(p, os.path.basename(p))

    def _crop_organelle(self):
        if not self._corr_result or not self._corr_result.get("ok"):
            self._log_msg("Run Correlate first.", WARN)
            return
        if not hasattr(self, "_organelle_img") or self._organelle_img is None:
            self._log_msg("Load an organelle channel first (Load tab).", WARN)
            return

        import cv2
        angle = self._corr_result["angle"]
        flipCode = self._corr_result["flip"]
        tl = self._corr_result["top_left"]
        br = self._corr_result["bottom_right"]

        from ._core import _rotate_with_M
        rot, _, _ = _rotate_with_M(self._organelle_img, angle)
        trans = cv2.flip(rot, flipCode)
        x1, y1 = tl; x2, y2 = br
        crop = trans[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            self._log_msg("Organelle crop is empty.", DANGER)
            return

        out_dir = os.path.join(self._out_dir_lbl.text().strip() or ".", "correlation_results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "organelle_cropped_region.png")
        cv2.imwrite(out_path, crop)
        self._open_image_in_napari(out_path, "Organelle_crop")
        self._log_msg(f"✅ Organelle crop saved → {out_path}", SUCCESS)

    # ------------------------------------------------------------------
    # Quantify actions
    # ------------------------------------------------------------------
    def _browse_quant_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Quantification Output Directory")
        if d:
            self._quant_out_lbl.setText(d)

    def _run_quantify(self):
        if self._lm_mask is None or self._em_mask is None:
            self._log_msg("Run segmentation first.", DANGER)
            return
        if not self._lm_source or not self._em_source:
            self._log_msg("LM and EM image sources required.", DANGER)
            return

        self._btn_quantify.setEnabled(False)
        self._set_progress(10, "Quantifying")
        self._log_msg("Running quantification…", WARN)

        out_dir = self._quant_out_lbl.text().strip() or "quantification_results"

        self._quant_thread = _QuantThread(
            self._lm_mask, self._em_mask,
            self._lm_source, self._em_source,
            out_dir,
        )
        self._quant_thread.progress.connect(self._log_msg)
        self._quant_thread.done.connect(self._on_quantify_done)
        self._quant_thread.start()

    def _on_quantify_done(self, result: dict):
        self._btn_quantify.setEnabled(True)
        self._set_progress(0)
        if not result["ok"]:
            self._log_msg(f"Quantification failed:\n{result.get('msg','')}", DANGER)
            return

        self._quant_result = result
        self._btn_view_quant.setEnabled(True)
        summary = (
            f"✅ Quantification done!\n"
            f"   LM: {result['lm_components']} components → {result['lm_excel']}\n"
            f"   EM: {result['em_components']} components → {result['em_excel']}"
        )
        self._quant_result_lbl.setText(summary)
        self._log_msg(summary, SUCCESS)

    def _view_quant_outputs(self):
        if not hasattr(self, "_quant_result"):
            return
        for key in ("lm_overlay", "em_overlay"):
            p = self._quant_result.get(key)
            if p and os.path.exists(p):
                self._open_image_in_napari(p, os.path.basename(p))

    # ------------------------------------------------------------------
    # Napari layer helpers
    # ------------------------------------------------------------------
    def _add_or_replace_layer(self, data: np.ndarray, name: str, kind: str = "image"):
        """Add or replace a named layer in the viewer."""
        # Remove existing layer with same name
        to_remove = [l for l in self.viewer.layers if l.name == name]
        for l in to_remove:
            self.viewer.layers.remove(l)

        if kind == "labels":
            self.viewer.add_labels(data.astype(np.int32), name=name)
        else:
            self.viewer.add_image(data, name=name)

    def _open_image_in_napari(self, path: str, layer_name: str):
        try:
            import cv2
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                self._log_msg(f"Cannot read: {path}", DANGER)
                return
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._add_or_replace_layer(img, layer_name, kind="image")
        except Exception:
            self._log_msg(traceback.format_exc(), DANGER)

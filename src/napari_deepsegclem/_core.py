"""
_core.py
--------
All non-UI computation: segmentation, correlation, quantification.
Ported from the original Tkinter app (Zaghbani et al., MPI Biophysics 2025).
"""

from __future__ import annotations

import math
import os
import traceback
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


# ---------------------------------------------------------------------------
# Model architecture helpers (kept identical to original)
# ---------------------------------------------------------------------------

def _build_fcn_resnet50(input_shape=(None, None, 3), num_classes=1):
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import ResNet50

    model_input = layers.Input(shape=input_shape)
    base_model = ResNet50(weights=None, include_top=False, input_tensor=model_input)

    conv1 = base_model.get_layer("conv1_relu").output
    conv2 = base_model.get_layer("conv2_block3_out").output
    conv3 = base_model.get_layer("conv3_block4_out").output
    conv4 = base_model.get_layer("conv4_block6_out").output
    conv5 = base_model.get_layer("conv5_block3_out").output

    x = layers.Conv2DTranspose(512, 3, strides=2, padding="same")(conv5)
    x = layers.Concatenate()([x, conv4])
    x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)

    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, conv3])
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, conv2])
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, conv1])
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    x = layers.Conv2DTranspose(
        num_classes, kernel_size=3, strides=2, padding="same",
        activation="sigmoid" if num_classes == 1 else "softmax"
    )(x)

    return Model(inputs=model_input, outputs=x)


def _se_block(input_tensor, reduction=16):
    from tensorflow.keras import layers
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // reduction, activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([input_tensor, se])


def _build_fcn_corenet(input_shape=(256, 256, 3), num_classes=1):
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import ResNet50

    model_input = layers.Input(shape=input_shape)
    base_model = ResNet50(weights=None, include_top=False, input_tensor=model_input)

    conv1 = base_model.get_layer("conv1_relu").output
    conv2 = base_model.get_layer("conv2_block3_out").output
    conv3 = base_model.get_layer("conv3_block4_out").output
    conv4 = base_model.get_layer("conv4_block6_out").output
    conv5 = base_model.get_layer("conv5_block3_out").output

    x = layers.Conv2DTranspose(512, 3, strides=2, padding="same")(conv5)
    x = layers.Concatenate()([x, conv4])
    x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)
    x = _se_block(x); x = layers.Dropout(0.3)(x)

    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, conv3])
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = _se_block(x); x = layers.Dropout(0.3)(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, conv2])
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = _se_block(x); x = layers.Dropout(0.3)(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, conv1])
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = _se_block(x); x = layers.Dropout(0.3)(x)

    shared_up = layers.UpSampling2D(size=(2, 2))(x)

    seg_output = layers.Conv2D(num_classes, 1, activation="sigmoid", name="segmentation")(shared_up)
    conf = layers.Conv2D(32, 3, padding="same", activation="relu")(shared_up)
    conf = layers.Conv2D(1, 1, activation="sigmoid", name="confidence")(conf)
    refined_output = layers.Multiply(name="refined_output")([seg_output, conf])

    return Model(inputs=model_input, outputs=[refined_output, seg_output, conf])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(lm_weights_path: str, em_weights_path: str):
    """
    Build both model architectures and load weights.
    Returns (lm_model, em_model) or raises on failure.
    """
    lm_model = _build_fcn_resnet50(input_shape=(None, None, 3), num_classes=1)
    lm_model.load_weights(lm_weights_path)

    em_model = _build_fcn_corenet(input_shape=(256, 256, 3), num_classes=1)
    em_model.load_weights(em_weights_path)

    return lm_model, em_model


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _pad_to_multiple(img: np.ndarray, mult: int = 32):
    h, w = img.shape[:2]
    H = int(np.ceil(h / mult) * mult)
    W = int(np.ceil(w / mult) * mult)
    img_p = cv2.copyMakeBorder(img, 0, H - h, 0, W - w, cv2.BORDER_REFLECT_101)
    return img_p, (H - h, W - w)


def _unpad(img_p: np.ndarray, pads: Tuple[int, int]) -> np.ndarray:
    pb, pr = pads
    h, w = img_p.shape[:2]
    return img_p[:h - pb if pb else h, :w - pr if pr else w]


def _to_bgr(img) -> np.ndarray:
    """Accept path / PIL / numpy, return BGR uint8."""
    from PIL import Image as PILImage
    if isinstance(img, str):
        out = cv2.imread(img, cv2.IMREAD_COLOR)
        if out is None:
            raise ValueError(f"Cannot read image: {img}")
        return out
    if isinstance(img, PILImage.Image):
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    arr = np.asarray(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return arr.copy()


def _get_model_hw(model, fallback=(256, 256)):
    try:
        shp = model.input_shape
        if isinstance(shp, (list, tuple)) and isinstance(shp[0], (list, tuple)):
            shp = shp[0]
        H, W = shp[1], shp[2]
        if H is None or W is None:
            return fallback
        return int(H), int(W)
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_lm(image_source, lm_model, threshold: float = 0.5) -> np.ndarray:
    """
    Run LM segmentation on any-size input.
    Returns uint8 mask 0/255 at original resolution.
    """
    bgr = _to_bgr(image_source)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    padded, pads = _pad_to_multiple(rgb, 32)
    x = padded[None, ...]  # (1, Hpad, Wpad, 3)

    pred = lm_model.predict(x, verbose=0)
    # pred shape: (1, Hpad, Wpad, 1)
    p = pred[0, ..., 0]
    p = _unpad(p, pads)
    h0, w0 = rgb.shape[:2]
    p = p[:h0, :w0]
    return (p >= threshold).astype(np.uint8) * 255


def segment_em(image_source, em_model, threshold: float = 0.5) -> np.ndarray:
    """
    Run EM segmentation; resizes input to model size then restores original res.
    Returns uint8 mask 0/255 at original EM resolution.
    """
    bgr = _to_bgr(image_source)
    H0, W0 = bgr.shape[:2]
    Hm, Wm = _get_model_hw(em_model, fallback=(256, 256))

    rgb_rs = cv2.resize(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (Wm, Hm), interpolation=cv2.INTER_AREA
    ).astype(np.float32) / 255.0

    pred = em_model.predict(rgb_rs[None, ...], verbose=0)

    refined = pred[0] if isinstance(pred, (list, tuple)) else pred
    p = refined[0, ..., 0]
    m_small = (p >= threshold).astype(np.uint8) * 255
    return cv2.resize(m_small, (W0, H0), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def _rotate_with_M(image: np.ndarray, angle: float,
                   interp=cv2.INTER_NEAREST, border_value=0):
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
    bw = int(h * abs_sin + w * abs_cos)
    bh = int(h * abs_cos + w * abs_sin)
    M[0, 2] += bw / 2.0 - cx
    M[1, 2] += bh / 2.0 - cy
    rotated = cv2.warpAffine(image, M, (bw, bh), flags=interp,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    return rotated, M, (bw, bh)


def _flip_point(x, y, W, H, flipCode):
    if flipCode == 1:
        return W - 1 - x, y
    if flipCode == 0:
        return x, H - 1 - y
    if flipCode == -1:
        return W - 1 - x, H - 1 - y
    return x, y


def _map_box_to_original(top_left, bottom_right, M, rot_wh, flipCode):
    Wrot, Hrot = rot_wh
    x1, y1 = top_left
    x2, y2 = bottom_right
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    unflipped = np.array([_flip_point(int(c[0]), int(c[1]), Wrot, Hrot, flipCode)
                          for c in corners], dtype=np.float32)
    Minv = cv2.invertAffineTransform(M)
    orig_pts = cv2.transform(unflipped.reshape(-1, 1, 2), Minv).reshape(-1, 2)
    return orig_pts


def _clamp_box(xmin, ymin, xmax, ymax, W, H):
    xmin = max(0, min(int(np.floor(xmin)), W - 1))
    ymin = max(0, min(int(np.floor(ymin)), H - 1))
    xmax = max(0, min(int(np.ceil(xmax)), W))
    ymax = max(0, min(int(np.ceil(ymax)), H))
    if xmax <= xmin: xmax = min(W, xmin + 1)
    if ymax <= ymin: ymax = min(H, ymin + 1)
    return xmin, ymin, xmax, ymax


def _ensure_u8_gray(mask: np.ndarray) -> np.ndarray:
    m = mask.copy()
    if m.dtype != np.uint8:
        m = (m * 255).astype(np.uint8) if m.max() <= 1.0 else m.astype(np.uint8)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return ((m > 0).astype(np.uint8)) * 255


def _downscale(img: np.ndarray, scale_pct: float) -> np.ndarray:
    w = int(img.shape[1] * scale_pct / 100)
    h = int(img.shape[0] * scale_pct / 100)
    return cv2.resize(img, (max(1, w), max(1, h)), interpolation=cv2.INTER_AREA)


def correlate(
    lm_mask: np.ndarray,
    em_mask: np.ndarray,
    lm_image_source,
    em_image_source,
    out_dir: str = "correlation_results",
    angles=range(0, 360, 1),
    flips=(1, 0, -1),
    scale_percents=(20, 30, 40, 50, 60, 70, 80, 90, 100),
    progress_callback=None,
) -> dict:
    """
    Full correlation pipeline.  Returns a rich result dict.
    progress_callback(pct: int, msg: str) is called periodically.
    """
    os.makedirs(out_dir, exist_ok=True)

    lm_full = _ensure_u8_gray(lm_mask)
    em_full = _ensure_u8_gray(em_mask)

    def _resize_max(img, max_dim):
        h, w = img.shape[:2]
        s = min(1.0, float(max_dim) / float(max(h, w)))
        if s >= 1.0:
            return img, 1.0
        return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), s

    lm_small, s_lm = _resize_max(lm_full, 1200)
    em_small, _s_em = _resize_max(em_full, 900)

    best = None
    total = len(scale_percents) * len(list(angles)) * len(flips)
    done = 0

    for sp in scale_percents:
        tmpl = _downscale(em_small, sp)
        th, tw = tmpl.shape[:2]
        if th < 5 or tw < 5:
            continue

        for angle in angles:
            rot, M_small, rot_wh_s = _rotate_with_M(lm_small, angle)
            Hrot, Wrot = rot.shape[:2]

            for flipCode in flips:
                done += 1
                trans = cv2.flip(rot, flipCode)
                Ht, Wt = trans.shape[:2]
                if Ht < th or Wt < tw:
                    continue
                res = cv2.matchTemplate(trans, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if best is None or max_val > best[0]:
                    best = (max_val, max_loc, angle, flipCode, sp, tw, th,
                            M_small, rot_wh_s, trans.copy())

            if progress_callback and done % (len(flips) * 10) == 0:
                pct = int(done * 100 / max(1, total))
                progress_callback(pct, f"Correlating… {pct}%")

    if best is None:
        return {"ok": False, "msg": "No match found."}

    score, top_s, angle, flipCode, sp, wT_s, hT_s, M_small, rot_wh_s, best_trans_s = best
    bot_s = (top_s[0] + wT_s, top_s[1] + hT_s)

    # Scale box up to full-res transformed LM space
    inv_s = 1.0 / max(1e-9, s_lm)
    tl_full = (int(round(top_s[0] * inv_s)), int(round(top_s[1] * inv_s)))
    br_full = (int(round(bot_s[0] * inv_s)), int(round(bot_s[1] * inv_s)))

    # ---- debug QC: small match image ----
    qc = cv2.cvtColor(best_trans_s, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(qc, top_s, bot_s, (0, 0, 255), 2)
    result_path = os.path.join(out_dir, "best_match_result.png")
    cv2.imwrite(result_path, qc)

    # ---- overlay on em_small ----
    em_bgr_s = cv2.cvtColor(em_small, cv2.COLOR_GRAY2BGR)
    mr_s = qc[top_s[1]:bot_s[1], top_s[0]:bot_s[0]]
    overlay_path = None
    if mr_s.size > 0:
        mr_rs = cv2.resize(mr_s, (em_bgr_s.shape[1], em_bgr_s.shape[0]), interpolation=cv2.INTER_NEAREST)
        ov = em_bgr_s.copy()
        ov[:, :, 0] = mr_rs[:, :, 0]
        overlayed = cv2.addWeighted(em_bgr_s, 0.5, ov, 0.5, 0)
        overlay_path = os.path.join(out_dir, "overlayed_match.png")
        cv2.imwrite(overlay_path, overlayed)

    # ---- full-res transform of LM mask + crop ----
    lm_full_rot, M_full, rot_wh_full = _rotate_with_M(lm_full, angle)
    lm_full_trans = cv2.flip(lm_full_rot, flipCode)
    lm_full_bgr = cv2.cvtColor(lm_full_trans, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(lm_full_bgr, tl_full, br_full, (0, 0, 255), 3)
    box_path = os.path.join(out_dir, "lm_image_with_box.png")
    cv2.imwrite(box_path, lm_full_bgr)

    x1, y1 = tl_full; x2, y2 = br_full
    crop_mask = lm_full_trans[y1:y2, x1:x2]
    crop_mask_path = None
    if crop_mask is not None and crop_mask.size > 0:
        crop_mask_path = os.path.join(out_dir, "cropped_lm_mask.png")
        cv2.imwrite(crop_mask_path, crop_mask)

    # ---- original LM image crop ----
    lm_orig_bgr = _to_bgr(lm_image_source)
    H0, W0 = lm_orig_bgr.shape[:2]

    poly_orig = _map_box_to_original(tl_full, br_full, M_full, rot_wh_full, flipCode)
    xmin = poly_orig[:, 0].min(); xmax = poly_orig[:, 0].max()
    ymin = poly_orig[:, 1].min(); ymax = poly_orig[:, 1].max()
    bx_min, by_min, bx_max, by_max = _clamp_box(xmin, ymin, xmax, ymax, W0, H0)

    crop_orig = lm_orig_bgr[by_min:by_max, bx_min:bx_max]
    crop_orig_path = None
    poly_path = None
    if crop_orig is not None and crop_orig.size > 0:
        crop_orig_path = os.path.join(out_dir, "cropped_LM_original_coords.png")
        cv2.imwrite(crop_orig_path, crop_orig)

    poly_vis = lm_orig_bgr.copy()
    poly_int = np.round(poly_orig).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(poly_vis, [poly_int], isClosed=True, color=(0, 0, 255), thickness=3)
    poly_path = os.path.join(out_dir, "lm_original_with_mapped_region.png")
    cv2.imwrite(poly_path, poly_vis)

    # ---- EM + LM crop green overlay ----
    em_bgr_full = _to_bgr(em_image_source)
    em_overlay_path = os.path.join(out_dir, "em_with_lm_crop_overlay.png")
    _save_green_overlay(em_bgr_full, crop_orig if crop_orig is not None else crop_mask,
                        em_overlay_path)

    return {
        "ok": True,
        "score": float(score),
        "angle": int(angle),
        "flip": int(flipCode),
        "scale_percent": int(sp),
        "top_left": (int(tl_full[0]), int(tl_full[1])),
        "bottom_right": (int(br_full[0]), int(br_full[1])),
        "M_full": M_full.tolist(),
        "rot_wh_full": tuple(map(int, rot_wh_full)),
        "poly_orig": poly_orig.tolist(),
        "bbox_orig": (int(bx_min), int(by_min), int(bx_max), int(by_max)),
        # paths
        "result_path": result_path,
        "overlay_path": overlay_path,
        "lm_with_box_path": box_path,
        "cropped_lm_path": crop_mask_path,
        "cropped_lm_original_path": crop_orig_path,
        "lm_original_poly_path": poly_path,
        "em_overlay_crop_path": em_overlay_path,
    }


def _save_green_overlay(em_bgr: np.ndarray, crop_bgr: Optional[np.ndarray],
                        out_path: str, alpha: float = 0.55, thr: int = 15):
    if crop_bgr is None or crop_bgr.size == 0:
        return
    if crop_bgr.ndim == 2:
        crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
    crop_rs = cv2.resize(crop_bgr, (em_bgr.shape[1], em_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(crop_rs.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    mask_sig = (gray > thr).astype(np.uint8)
    green = np.zeros_like(em_bgr, dtype=np.uint8)
    green[:, :, 1] = mask_sig * 255
    blended = cv2.addWeighted(em_bgr.astype(np.uint8), 1.0, green, alpha, 0)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, blended)


# ---------------------------------------------------------------------------
# Quantification
# ---------------------------------------------------------------------------

def _neighbors_count(binary_img: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(binary_img.astype(np.uint8), -1, kernel)


def build_skeleton_layers(binary_mask: np.ndarray):
    """Returns (skeleton bool, endpoints bool, branchpoints bool) for the full image."""
    skel = skeletonize(binary_mask.astype(bool)).astype(np.uint8)
    nbh = _neighbors_count(skel)
    return (skel.astype(bool),
            (skel == 1) & (nbh == 1),
            (skel == 1) & (nbh >= 3))


def make_skeleton_overlay(orig_bgr: np.ndarray, skel, endpoints, branchpoints,
                          out_path: str) -> np.ndarray:
    skel_u8 = skel.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    skel_thick = cv2.dilate(skel_u8, kernel, iterations=1).astype(bool)

    overlay = orig_bgr.copy()
    overlay[skel_thick] = (255, 0, 255)
    ep_y, ep_x = np.where(endpoints)
    overlay[ep_y, ep_x] = (0, 255, 0)
    bp_y, bp_x = np.where(branchpoints)
    overlay[bp_y, bp_x] = (0, 0, 255)

    blended = cv2.addWeighted(orig_bgr, 0.6, overlay, 0.4, 0)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, blended)
    return blended


def quantify_components(binary_mask: np.ndarray, source: str,
                        image_name: str = "") -> pd.DataFrame:
    labeled = label((binary_mask > 0).astype(np.uint8), connectivity=2)
    props = regionprops(labeled)

    records = []
    for prop in props:
        comp = (labeled == prop.label).astype(np.uint8)
        area = int(prop.area)

        cnts, _ = cv2.findContours((comp * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perim = sum(cv2.arcLength(c, True) for c in cnts)
        circ = (4.0 * math.pi * area / (perim * perim)) if perim > 0 else 0.0

        skel = skeletonize(comp.astype(bool)).astype(np.uint8)
        nbh = _neighbors_count(skel)
        n_end = int(np.count_nonzero((skel == 1) & (nbh == 1)))
        n_branch = int(np.count_nonzero((skel == 1) & (nbh >= 3)))

        minr, minc, maxr, maxc = prop.bbox
        records.append({
            "image": image_name, "source": source,
            "component_id": int(prop.label),
            "area_px": area,
            "perimeter_px": float(perim),
            "circularity": float(circ),
            "skeleton_length_px": int(np.count_nonzero(skel)),
            "n_endpoints": n_end,
            "n_branchpoints": n_branch,
            "bbox_minr": int(minr), "bbox_minc": int(minc),
            "bbox_maxr": int(maxr), "bbox_maxc": int(maxc),
        })
    return pd.DataFrame.from_records(records)


def append_excel(df: pd.DataFrame, source: str, out_dir: str = "quantification_results") -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = "LM_metrics.xlsx" if source.upper() == "LM" else "EM_metrics.xlsx"
    path = os.path.join(out_dir, fname)
    if os.path.exists(path):
        old = pd.read_excel(path)
        df = pd.concat([old, df], ignore_index=True)
    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, index=False)
    return path


def run_quantification(
    lm_mask: np.ndarray,
    em_mask: np.ndarray,
    lm_image_source,
    em_image_source,
    out_dir: str = "quantification_results",
) -> dict:
    """Full quantification pipeline. Returns result dict with paths."""
    os.makedirs(out_dir, exist_ok=True)

    # ---- LM ----
    lm_name = (os.path.splitext(os.path.basename(lm_image_source))[0]
               if isinstance(lm_image_source, str) else "LM")
    lm_bgr = _to_bgr(lm_image_source)
    if lm_bgr.shape[:2] != lm_mask.shape[:2]:
        lm_bgr = cv2.resize(lm_bgr, (lm_mask.shape[1], lm_mask.shape[0]))

    lm_bin = (lm_mask > 0).astype(np.uint8)
    lm_skel, lm_ep, lm_bp = build_skeleton_layers(lm_bin)
    lm_overlay = os.path.join(out_dir, f"{lm_name}_LM_overlay.png")
    make_skeleton_overlay(lm_bgr, lm_skel, lm_ep, lm_bp, lm_overlay)
    lm_df = quantify_components(lm_bin, "LM", lm_name)
    lm_excel = append_excel(lm_df, "LM", out_dir)

    # ---- EM ----
    em_bgr = _to_bgr(em_image_source)
    em_mask_rs = cv2.resize(
        (em_mask > 0).astype(np.uint8) * 255,
        (em_bgr.shape[1], em_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    em_bin = (em_mask_rs > 0).astype(np.uint8)
    em_skel, em_ep, em_bp = build_skeleton_layers(em_bin)
    em_overlay = os.path.join(out_dir, "EM_overlay.png")
    make_skeleton_overlay(em_bgr, em_skel, em_ep, em_bp, em_overlay)
    em_df = quantify_components(em_bin, "EM", "EM")
    em_excel = append_excel(em_df, "EM", out_dir)

    return {
        "ok": True,
        "lm_overlay": lm_overlay,
        "em_overlay": em_overlay,
        "lm_excel": lm_excel,
        "em_excel": em_excel,
        "lm_components": len(lm_df),
        "em_components": len(em_df),
    }

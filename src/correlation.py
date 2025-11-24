import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
import os
import matplotlib.pyplot as plt
# --- NEW IMPORTS FOR QUANTIFICATION ---
import pandas as pd
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
import math
import traceback
from tkinter import ttk



# ================== BRAND LOGO & STATUS/OVERLAY HELPERS ==================
import sys
import platform
import subprocess
import threading
import time

import os
import sys

def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller exe.
    """
    try:
        # When running as a bundled exe
        base_path = sys._MEIPASS
    except Exception:
        # When running as a normal script
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# ===================== MASK EDITOR =====================
from PIL import ImageOps

def _composite_overlay_rgb(base_rgb: np.ndarray, mask_bin: np.ndarray, rgba=(255,0,0,110)):
    """
    base_rgb: HxWx3 uint8
    mask_bin: HxW {0,255}
    returns composited PIL.Image for display
    """
    base = Image.fromarray(base_rgb, mode="RGB")
    h, w = mask_bin.shape[:2]
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    ov_np = np.array(overlay)
    # put red in alpha only where mask>0
    alpha = (mask_bin > 0).astype(np.uint8) * rgba[3]
    ov_np[..., 0] = rgba[0]; ov_np[..., 1] = rgba[1]; ov_np[..., 2] = rgba[2]; ov_np[..., 3] = alpha
    overlay = Image.fromarray(ov_np, mode="RGBA")
    comp = base.convert("RGBA")
    comp.alpha_composite(overlay)
    return comp.convert("RGB")

# ======================= MASK EDITOR (LM/EM) =======================
# Requirements: cv2, PIL (Image, ImageTk), globals: mask1, mask2, image1, image2, label1, label2, IMAGE_DISPLAY_SIZE, THEME, root

def refresh_preview_from_mask(kind: str):
    """Refresh left/right small preview label after editing."""
    global mask1, mask2, label1, label2, IMAGE_DISPLAY_SIZE
    if kind == "LM":
        m = mask1
        target_label = label1
    else:
        m = mask2
        target_label = label2

    # normalize to 0..255 uint8
    if m is None:
        return
    mm = (m * 255).astype(np.uint8) if m.max() <= 1 else m.astype(np.uint8)
    vis = cv2.resize(mm, IMAGE_DISPLAY_SIZE, interpolation=cv2.INTER_NEAREST)
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)))
    target_label.configure(image=imgtk)
    target_label.image = imgtk  # keep ref


class MaskEditorWindow:
    """
    Interactive mask editor with accurate letterbox mapping:
    - Left mouse = draw (white, 255)
    - Right mouse = erase (black, 0)
    - Brush size slider 5..25
    - Live hover circle shows brush footprint in canvas coords
    - Apply overwrites global mask1/mask2, writes PNG, refreshes main preview
    """
    def __init__(self, kind: str, parent):
        assert kind in ("LM", "EM")
        self.kind = kind
        self.parent = parent

        # ---- get original image as background (for context) and the mask
        self.mask = self._get_mask().copy()  # uint8 0/255
        self.h, self.w = self.mask.shape[:2]

        bg = self._get_background_image()  # BGR same size as mask
        self.bg = bg

        # ---- editor window
        self.win = tk.Toplevel(self.parent)
        self.win.title(f"Edit {self.kind} Mask")
        self.win.configure(bg=THEME["bg"])
        self.win.geometry("1000x720")
        self.win.transient(self.parent)

        # toolbar
        top = tk.Frame(self.win, bg=THEME["panel2"])
        top.pack(fill="x", padx=8, pady=6)

        tk.Label(top, text=f"{self.kind} — Draw: Left | Erase: Right",
                 fg=THEME["text"], bg=THEME["panel2"],
                 font=("Segoe UI", 10, "bold")).pack(side="left", padx=(4,16))

        tk.Label(top, text="Brush:", fg=THEME["text"], bg=THEME["panel2"]).pack(side="left")
        self.brush = tk.IntVar(value=12)
        tk.Scale(top, from_=5, to=25, orient="horizontal", variable=self.brush,
                 length=180, showvalue=True, bg=THEME["panel2"],
                 highlightthickness=0, troughcolor=THEME["panel"],
                 fg=THEME["text"]).pack(side="left", padx=(6,16))

        self.btn_apply = tk.Button(top, text="Apply & Overwrite",
                                   fg=THEME["text"], bg=THEME["button"],
                                   activebackground=THEME["button-h"],
                                   relief="flat", padx=12, pady=6,
                                   command=self._apply_and_close)
        self.btn_apply.pack(side="right", padx=(6,8))
        tk.Button(top, text="Cancel", fg=THEME["text"], bg=THEME["button"],
                  activebackground=THEME["button-h"], relief="flat",
                  padx=12, pady=6, command=self.win.destroy).pack(side="right")

        # canvas area
        self.canvas = tk.Canvas(self.win, bg=THEME["bg"], highlightthickness=0,
                                width=960, height=600)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # state for mapping
        self.view_w = 960
        self.view_h = 600
        self.scale = 1.0
        self.pad_x = 0
        self.pad_y = 0

        self._imgtk = None
        self._cursor_id = None
        self._last = None  # last (x,y) canvas for smooth stroke

        # initial render
        self._compute_layout()
        self._render()

        # bindings
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._down_draw)
        self.canvas.bind("<Button-3>", self._down_erase)
        self.canvas.bind("<B1-Motion>", self._drag_draw)
        self.canvas.bind("<B3-Motion>", self._drag_erase)
        self.canvas.bind("<ButtonRelease-1>", self._up)
        self.canvas.bind("<ButtonRelease-3>", self._up)
        self.canvas.bind("<Motion>", self._hover)

    # ---------- data sources ----------
    def _get_mask(self) -> np.ndarray:
        global mask1, mask2
        m = mask1 if self.kind == "LM" else mask2
        if m is None:
            # create empty mask same size as background later
            raise ValueError(f"No {self.kind} mask in memory.")
        m = (m * 255).astype(np.uint8) if m.max() <= 1 else m.astype(np.uint8)
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        return m

    def _get_background_image(self) -> np.ndarray:
        """Return BGR image same size as mask."""
        global image1, image2
        if self.kind == "LM":
            if isinstance(image1, str):
                bg = cv2.imread(image1)  # BGR
            elif isinstance(image1, Image.Image):
                bg = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
            else:
                bg = image1
        else:
            if isinstance(image2, Image.Image):
                bg = cv2.cvtColor(np.array(image2.convert("RGB")), cv2.COLOR_RGB2BGR)
            else:
                bg = image2
        if bg is None:
            bg = np.zeros((self.h, self.w, 3), np.uint8)
        if bg.shape[:2] != (self.h, self.w):
            bg = cv2.resize(bg, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return bg

    # ---------- layout & rendering ----------
    def _compute_layout(self):
        """Letterbox image inside canvas; compute scale & padding."""
        cw = self.canvas.winfo_width() or self.view_w
        ch = self.canvas.winfo_height() or self.view_h
        self.view_w, self.view_h = cw, ch
        if self.w == 0 or self.h == 0:
            self.scale = 1.0; self.pad_x = self.pad_y = 0; return
        s = min((cw - 2) / self.w, (ch - 2) / self.h)
        s = max(0.01, s)
        self.scale = s
        disp_w = int(self.w * s)
        disp_h = int(self.h * s)
        self.pad_x = (cw - disp_w) // 2
        self.pad_y = (ch - disp_h) // 2

    def _compose_display(self):
        """Blend bg + mask (white) and letterbox to canvas bitmap with safe clipping."""
    # overlay mask in white on bg
        overlay = self.bg.copy()
        overlay[self.mask > 0] = (255, 50, 50)
        disp = cv2.addWeighted(self.bg, 0.6, overlay, 0.4, 0)

    # resize to display area
        disp_res = cv2.resize(
           disp,
           (int(self.w * self.scale), int(self.h * self.scale)),
           interpolation=cv2.INTER_NEAREST
        )

    # prepare empty canvas background
        full = np.zeros((self.view_h, self.view_w, 3), np.uint8)
        full[:] = (30, 30, 35)

        y0 = max(0, self.pad_y)
        x0 = max(0, self.pad_x)

    # Compute where insert should end safely
        h_res, w_res = disp_res.shape[:2]
        y1 = min(y0 + h_res, self.view_h)
        x1 = min(x0 + w_res, self.view_w)

    # Compute crop dimensions in disp_res
        crop_h = max(0, y1 - y0)
        crop_w = max(0, x1 - x0)

    # ✅ Only draw if crop is valid
        if crop_h > 0 and crop_w > 0:
            full[y0:y1, x0:x1] = disp_res[:crop_h, :crop_w]

        return full


    def _render(self):
        bm = self._compose_display()
        self._imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bm, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._imgtk, anchor="nw")
        # keep cursor on top if present
        self._cursor_id = None

    # ---------- coords ----------
    def _canvas_to_mask(self, cx, cy):
        """Canvas (pixels) -> mask (pixels)."""
        mx = int((cx - self.pad_x) / self.scale)
        my = int((cy - self.pad_y) / self.scale)
        return mx, my

    def _valid_canvas_point(self, cx, cy):
        return (self.pad_x <= cx < self.view_w - self.pad_x) and \
               (self.pad_y <= cy < self.view_h - self.pad_y)

    # ---------- painting operations ----------
    def _stamp(self, cx, cy, value):
        """Draw a filled circle in mask at brush radius."""
        if not self._valid_canvas_point(cx, cy): return
        mx, my = self._canvas_to_mask(cx, cy)
        if mx < 0 or my < 0 or mx >= self.w or my >= self.h: return
        r = int(self.brush.get())
        cv2.circle(self.mask, (mx, my), r, int(value), thickness=-1)
        self._render()  # refresh display

    def _line_stroke(self, cx0, cy0, cx1, cy1, value):
        """Interpolate between two canvas points and stamp circles (smooth stroke)."""
        # clip early if outside
        steps = max(1, int(np.hypot(cx1 - cx0, cy1 - cy0)))
        for t in range(steps + 1):
            x = int(cx0 + (cx1 - cx0) * t / steps)
            y = int(cy0 + (cy1 - cy0) * t / steps)
            self._stamp(x, y, value)

    # ---------- mouse handlers ----------
    def _down_draw(self, e):
        self._last = (e.x, e.y)
        self._stamp(e.x, e.y, 255)

    def _down_erase(self, e):
        self._last = (e.x, e.y)
        self._stamp(e.x, e.y, 0)

    def _drag_draw(self, e):
        if self._last is None: self._last = (e.x, e.y)
        x0, y0 = self._last
        self._line_stroke(x0, y0, e.x, e.y, 255)
        self._last = (e.x, e.y)

    def _drag_erase(self, e):
        if self._last is None: self._last = (e.x, e.y)
        x0, y0 = self._last
        self._line_stroke(x0, y0, e.x, e.y, 0)
        self._last = (e.x, e.y)

    def _up(self, e):
        self._last = None

    def _hover(self, e):
        # live circle cursor in canvas coords
        self.canvas.delete(self._cursor_id)
        r = max(2, int(self.brush.get() * self.scale))
        x0, y0, x1, y1 = e.x - r, e.y - r, e.x + r, e.y + r
        self._cursor_id = self.canvas.create_oval(
            x0, y0, x1, y1,
            outline=THEME["accent-2"], width=1
        )

    def _on_resize(self, _evt):
        self._compute_layout()
        self._render()

    # ---------- apply ----------
    def _apply_and_close(self):
        """Overwrite global mask, save PNG, refresh preview, close."""
        global mask1, mask2
        # ensure 0/1 for program use; save 0/255 to disk
        out_u8 = (self.mask > 0).astype(np.uint8) * 255

        if self.kind == "LM":
            mask1 = out_u8.copy()
            out_path = os.path.join("segmentation_results", "mask1_LM_original.png")
        else:
            # keep EM mask as 0/1 in memory like before
            mask2 = (out_u8 > 0).astype(np.uint8)
            out_path = os.path.join("segmentation_results", "mask2_EM.png")

        os.makedirs("segmentation_results", exist_ok=True)
        cv2.imwrite(out_path, out_u8)

        # refresh preview panel in main UI
        refresh_preview_from_mask(self.kind)

        messagebox.showinfo("Mask Editor", f"{self.kind} mask updated and saved:\n{out_path}")
        self.win.destroy()


def open_mask_editor(kind: str):
    """Public helper to open LM/EM editor ('LM' or 'EM')."""
    MaskEditorWindow(kind.upper(), root)




def make_brand_logo(size: int = 22, stroke: int = 2):
    """
    Combined microscope + mitochondria outline icon, monochrome, to match toolbar icons.
    Returns a Tk PhotoImage.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    col = (220, 225, 235, 255)
    r = stroke
    s = size

    # Lens circle
    d.ellipse([s*0.08, s*0.08, s*0.52, s*0.52], outline=col, width=r)
    # Mito inner squiggle inside lens
    d.arc([s*0.16, s*0.18, s*0.44, s*0.46], start=210, end=330, fill=col, width=r)
    d.arc([s*0.2, s*0.22, s*0.4, s*0.42], start=30, end=150, fill=col, width=r)

    # Microscope body
    d.line([(s*0.50, s*0.25), (s*0.82, s*0.57)], fill=col, width=r)    # arm
    d.line([(s*0.60, s*0.44), (s*0.84, s*0.44)], fill=col, width=r)    # eyepiece line
    d.rectangle([s*0.60, s*0.37, s*0.68, s*0.44], outline=col, width=r) # eyepiece
    d.line([(s*0.70, s*0.58), (s*0.74, s*0.70)], fill=col, width=r)    # stage post
    d.line([(s*0.60, s*0.70), (s*0.86, s*0.70)], fill=col, width=r)    # stage
    d.line([(s*0.52, s*0.82), (s*0.90, s*0.82)], fill=col, width=r)    # base

    return ImageTk.PhotoImage(img)

# Extend your existing outline icon factory with 'help' and 'folderopen'
# (ADD these cases INSIDE your make_outline_icon(kind,...) function)
# ---
# elif kind == "help":
#     drw.ellipse([r+2, r+2, s-r-2, s-r-2], outline=col, width=r)
#     drw.arc([s*0.28, s*0.22, s*0.72, s*0.6], start=200, end=340, fill=col, width=r)
#     drw.ellipse([s*0.45, s*0.68, s*0.55, s*0.78], fill=col)
# elif kind == "folderopen":
#     drw.rounded_rectangle([r, s*0.42, s-r, s-r], radius=4, outline=col, width=r)
#     drw.polygon([(r+2, s*0.42), (s*0.42, s*0.42), (s*0.52, s*0.58), (r+2, s*0.58)], outline=col, width=r)
#     drw.line([(s*0.24, s*0.24), (s*0.40, s*0.24)], fill=col, width=r)
#     drw.line([(s*0.22, s*0.28), (s*0.38, s*0.28)], fill=col, width=r)




# ===== Professional UI: theme + icon generator =====
from PIL import ImageDraw, ImageFont


# ============= HELP WINDOW FUNCTION =============
def open_help_window():
    help_win = tk.Toplevel(root)
    help_win.title("User Manual – Mitochondria Analysis Suite")
    help_win.configure(bg=THEME["bg"])
    help_win.geometry("650x600")
    
    # Make scrollable
    container = tk.Frame(help_win, bg=THEME["bg"])
    canvas = tk.Canvas(container, bg=THEME["bg"], highlightthickness=0)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas, bg=THEME["bg"])

    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    container.pack(fill="both", expand=True)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Styled text sections
    def section(title, content):
        tk.Label(scroll_frame, text=title, font=("Segoe UI", 12, "bold"), fg=THEME["accent"], bg=THEME["bg"], pady=8).pack(anchor="w", padx=10)
        tk.Label(scroll_frame, text=content, font=("Segoe UI", 10), fg=THEME["text"], bg=THEME["bg"], justify="left", wraplength=600).pack(anchor="w", padx=20)

    # --- Manual Content ---
    section("🧭 Workflow Overview",
    "1. Click 'Load LM' to load Light Microscopy image.\n"
    "2. Click 'Load EM' to load Electron Microscopy image.\n"
    "3. Click 'Segment' to generate masks for LM and EM.\n"
    "4. Click 'Correlate' to align LM segmentation with EM.\n"
    "5. Click 'Quantify & Export' to generate Excel metrics and overlays.")

    section("🎨 Overlay Color Legend",
    "• White = Skeleton structure of mitochondria\n"
    "• Green Dots = Endpoints detected in mitochondrial segments\n"
    "• Red Dots = Branch points (junctions) in mitochondrial structures")

    section("📊 Excel Metrics Explanation",
    "• area_px — Number of pixels inside each mitochondria region.\n"
    "• perimeter_px — Pixel-length of mitochondrial boundary.\n"
    "• circularity — Computed as 4π × Area / Perimeter². Values near 1 = round.\n"
    "• skeleton_length_px — Length (in pixels) of skeleton line.\n"
    "• n_endpoints — Number of segment ends detected.\n"
    "• n_branchpoints — Indicator of mitochondrial network branching complexity.\n"
    "\nUse these to distinguish isolated vs networked mitochondria biologically.")

    section("💾 Output Location",
    "• segmentation_results/ — Raw masks from LM and EM.\n"
    "• correlation_results/ — Aligned LM-EM overlays.\n"
    "• quantification_results/ — Excel files + global skeleton overlays.")

    section("📌 Citation & Credit",
    "If this software assists in your research, please cite:\n"
    "“Mitochondria Segmentation & Correlation Tool — Zaghbani et al., Max Planck Institute for Biophysics (2025)”")

    section("👤 Developer & Contact",
    "Developed by: Soumaya Zaghbani, Max Planck Institute for Biophysics (2025)\n"
    "Contact: soumaya.yaghbanigmail.com\n"
    "\nFeel free to cite or reach out for collaboration or support.")


THEME = {
    "bg":        "#1E1F29",  # main background (Napari-ish)
    "bg-alt":    "#2A2C39",  # panels
    "panel":     "#2B2E3B",
    "panel2":    "#232634",
    "accent":    "#5E81AC",  # bluish
    "accent-2":  "#88C0D0",
    "text":      "#E5E9F0",
    "muted":     "#A7AEBE",
    "success":   "#98C379",  # green
    "warn":      "#E5C07B",  # amber
    "danger":    "#E06C75",  # red
    "button":    "#3B4252",
    "button-h":  "#4C566A",
    "border":    "#3A3F4B",
}

def make_outline_icon(kind:str, size:int=22, stroke:int=2):
    """
    Creates a monochrome outline icon (Pillow) and returns a Tk PhotoImage.
    Icons: 'load', 'load2', 'segment', 'correlate', 'quantify'
    """
    img = Image.new("RGBA", (size, size), (0,0,0,0))
    drw = ImageDraw.Draw(img)
    s, c = size, THEME["text"]
    col = (220, 225, 235, 255)
    r = stroke

    if kind in ("load","load2"):
        # folder + arrow
        drw.rounded_rectangle([r, s*0.45, s-r, s-r], radius=4, outline=col, width=r)
        drw.polygon([(r+2, s*0.45), (s*0.4, s*0.45), (s*0.48, s*0.6), (r+2, s*0.6)], outline=col, width=r)
        # arrow down
        drw.line([(s*0.5, s*0.15), (s*0.5, s*0.38)], fill=col, width=r, joint="curve")
        drw.polygon([(s*0.5, s*0.38), (s*0.43, s*0.3), (s*0.57, s*0.3)], fill=col)
    elif kind == "segment":
        # magic wand
        drw.line([(s*0.2, s*0.8), (s*0.75, s*0.25)], fill=col, width=r)
        drw.line([(s*0.65, s*0.15), (s*0.85, s*0.35)], fill=col, width=r)
        drw.line([(s*0.6, s*0.35), (s*0.8, s*0.55)], fill=col, width=r)
        # sparkles
        drw.line([(s*0.15, s*0.2), (s*0.25, s*0.3)], fill=col, width=r)
        drw.line([(s*0.25, s*0.2), (s*0.15, s*0.3)], fill=col, width=r)
    elif kind == "correlate":
        # rotate arrows
        drw.arc([r+2, r+2, s-r-2, s-r-2], start=30, end=210, fill=col, width=r)
        drw.polygon([(s*0.82, s*0.5), (s*0.65, s*0.45), (s*0.7, s*0.6)], fill=col)
    elif kind == "quantify":
        # bar chart + small dot
        bx = s*0.18
        drw.rectangle([bx, s*0.6, bx+s*0.12, s*0.82], outline=col, width=r)
        drw.rectangle([bx+s*0.18, s*0.45, bx+s*0.3, s*0.82], outline=col, width=r)
        drw.rectangle([bx+s*0.36, s*0.3, bx+s*0.48, s*0.82], outline=col, width=r)
        drw.ellipse([s*0.72, s*0.2, s*0.82, s*0.3], outline=col, width=r)

    elif kind == "help":
        drw.ellipse([r+2, r+2, s-r-2, s-r-2], outline=col, width=r)
    # question curve
        drw.arc([s*0.28, s*0.22, s*0.72, s*0.6], start=200, end=340, fill=col, width=r)
    # dot
        drw.ellipse([s*0.45, s*0.68, s*0.55, s*0.78], fill=col)
    else:
        drw.rectangle([r, r, s-r, s-r], outline=col, width=r)
            # Add inside make_outline_icon() — extend icons:


    return ImageTk.PhotoImage(img)

# Small status badge factory
def status_badge(text:str, fg:str, bg:str):
    lbl = tk.Label(text=text, font=("Segoe UI", 10, "bold"), fg=fg, bg=bg, padx=8, pady=2)
    return lbl


# ---------- QUANTIFICATION HELPERS (FULL-IMAGE) ----------

def ensure_binary(mask):
    m = (mask > 0).astype(np.uint8)
    return m

def compute_perimeter_px(binary_mask):
    cnts, _ = cv2.findContours((binary_mask*255).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = 0.0
    for c in cnts:
        perim += cv2.arcLength(c, True)
    return perim

def neighbors_count(binary_img):
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.uint8)
    nbh = cv2.filter2D(binary_img.astype(np.uint8), -1, kernel)
    return nbh

def build_full_skeleton_layers(full_binary_mask):
    """Return (skeleton, endpoints, branchpoints) as boolean masks for the FULL image."""
    skel = skeletonize(full_binary_mask.astype(bool)).astype(np.uint8)
    nbh = neighbors_count(skel)
    endpoints = ((skel == 1) & (nbh == 1))
    branchpoints = ((skel == 1) & (nbh >= 3))
    return skel.astype(bool), endpoints, branchpoints

def overlay_full_skeleton(original_bgr, skel, endpoints, branchpoints, out_path):
    """
    Draw full-image overlay:
    - original image (BGR)
    - skeleton (white)
    - endpoints (green)
    - branchpoints (red)
    """
    skel_u8 = skel.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)   # increase to (5,5) for even thicker lines
    skel_thick = cv2.dilate(skel_u8, kernel, iterations=1).astype(bool)
    overlay = original_bgr.copy()
    # Skeleton (white)
    overlay[skel_thick] = (255, 0, 255)
    # Endpoints (green)
    ep_y, ep_x = np.where(endpoints)
    overlay[ep_y, ep_x] = (0, 255, 0)
    # Branchpoints (red)
    bp_y, bp_x = np.where(branchpoints)
    overlay[bp_y, bp_x] = (0, 0, 255)

    # Blend lightly so the original is visible
    blended = cv2.addWeighted(original_bgr, 0.6, overlay, 0.4, 0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, blended)
    return blended

def quantify_components(mask_for_metrics, source_name, image_name=""):
    """
    Compute per-component metrics on the given mask (in ORIGINAL image size):
    area, perimeter, circularity, skeleton length, #endpoints, #branchpoints, bbox.
    Returns a DataFrame.
    """
    bin_mask = ensure_binary(mask_for_metrics)

    # Optional: remove tiny specks (tune min_size if needed)
    # bin_mask = remove_small_objects(bin_mask.astype(bool), min_size=20).astype(np.uint8)

    labeled = label(bin_mask, connectivity=2)
    props = regionprops(labeled)

    records = []
    for prop in props:
        comp_mask = (labeled == prop.label).astype(np.uint8)
        area = int(prop.area)
        perim = compute_perimeter_px(comp_mask)
        circularity = (4.0 * math.pi * area / (perim*perim)) if perim > 0 else 0.0

        skel = skeletonize(comp_mask.astype(bool)).astype(np.uint8)
        skel_len = int(np.count_nonzero(skel))

        nbh = neighbors_count(skel)
        num_end = int(np.count_nonzero((skel == 1) & (nbh == 1)))
        num_branch = int(np.count_nonzero((skel == 1) & (nbh >= 3)))

        minr, minc, maxr, maxc = prop.bbox

        records.append({
            "image": image_name,
            "source": source_name,  # "LM" or "EM"
            "component_id": int(prop.label),
            "area_px": area,
            "perimeter_px": float(perim),
            "circularity": float(circularity),
            "skeleton_length_px": skel_len,
            "n_endpoints": num_end,
            "n_branchpoints": num_branch,
            "bbox_minr": int(minr), "bbox_minc": int(minc),
            "bbox_maxr": int(maxr), "bbox_maxc": int(maxc),
        })

    df = pd.DataFrame.from_records(records)
    return df

def append_to_excel_separate(df, source, out_dir="quantification_results"):
    """
    Save metrics to separate Excel files:
      LM -> LM_metrics.xlsx
      EM -> EM_metrics.xlsx
    Appends if file exists.
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = "LM_metrics.xlsx" if source.upper() == "LM" else "EM_metrics.xlsx"
    excel_path = os.path.join(out_dir, fname)

    if os.path.exists(excel_path):
        old = pd.read_excel(excel_path)
        all_df = pd.concat([old, df], ignore_index=True)
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
            all_df.to_excel(writer, index=False)
    else:
        df.to_excel(excel_path, index=False)
    return excel_path




def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)
    M[0, 2] += bound_w / 2 - center[0]
    M[1, 2] += bound_h / 2 - center[1]



    # Ensure borderValue is white (255) for masks
    rotated = cv2.warpAffine(image, M, (bound_w, bound_h), flags=cv2.INTER_NEAREST)
    return rotated


def flip_image(image, flipCode):
    flipped = cv2.flip(image, flipCode)
    return flipped

def downscale_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def display_image_with_matplotlib(image, title):
    """Display an image with Matplotlib while handling grayscale and color images correctly."""
    plt.figure(figsize=(6, 6))
    
    # Check if the image has 3 channels (Color)
    if len(image.shape) == 3 and image.shape[2] == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    elif len(image.shape) == 2:  # If Grayscale
        plt.imshow(image, cmap='gray')
    else:  # If it's a single-channel but mistakenly has 3rd dimension
        plt.imshow(image[:, :, 0], cmap='gray')
    
    plt.title(title)
    plt.axis('off')
    plt.show()


def correlate_images():
    global mask1, mask2  

    if mask1 is None or mask2 is None:
        messagebox.showwarning("Warning", "Please segment both images before running correlation.")
        return

    try:
        # Convert from [0,1] to [0,255] if needed and ensure uint8
        lm_mask = (mask1 * 255).astype(np.uint8) if mask1.max() <= 1 else mask1.astype(np.uint8)
        em_mask = (mask2 * 255).astype(np.uint8) if mask2.max() <= 1 else mask2.astype(np.uint8)

        # Ensure grayscale format
        if len(lm_mask.shape) == 3:
            lm_mask = cv2.cvtColor(lm_mask, cv2.COLOR_BGR2GRAY)
        if len(em_mask.shape) == 3:
            em_mask = cv2.cvtColor(em_mask, cv2.COLOR_BGR2GRAY)

        # 🔹 Binarize masks to 0/255 (like in the standalone script)
        lm_mask = ((lm_mask > 0).astype(np.uint8)) * 255
        em_mask = ((em_mask > 0).astype(np.uint8)) * 255

        # Save original (now binarized) masks for debugging
        cv2.imwrite("debug_original_lm_mask.png", lm_mask)
        cv2.imwrite("debug_original_em_mask.png", em_mask)
        print("em_mask shape:", em_mask.shape)
        print("lm_mask shape:", lm_mask.shape)

        angles = range(0, 360, 1)
        flips = [1, 0, -1]
        scale_percents = [25, 50, 100]

        best_match = None

        for scale_percent in scale_percents:
            for angle in angles:
                for flipCode in flips:
                    downscaled_em_mask = downscale_image(em_mask, scale_percent)
                    trans_lm_mask = rotate_image(lm_mask, angle)
                    trans_lm_mask = flip_image(trans_lm_mask, flipCode)
                    
                    result = cv2.matchTemplate(trans_lm_mask, downscaled_em_mask, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

                    if best_match is None or max_val > best_match[0]:
                        best_match = (
                            max_val, max_loc, angle, flipCode, scale_percent,
                            downscaled_em_mask.shape[1], downscaled_em_mask.shape[0],
                            trans_lm_mask.copy()
                        )

        if best_match:
            max_val, max_loc, angle, flipCode, scale_percent, w, h, best_trans_img = best_match
            print(f"Best Match: Max Value: {max_val}, Location: {max_loc}, Angle: {angle}, Flip: {flipCode}, Scale: {scale_percent}%")

            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            best_trans_img = cv2.cvtColor(best_trans_img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(best_trans_img, top_left, bottom_right, (0, 0, 255), 3)

            os.makedirs("correlation_results", exist_ok=True)
            result_path = "correlation_results/best_match_result.png"
            cv2.imwrite(result_path, best_trans_img)

            display_image_with_matplotlib(best_trans_img, 'Best Match Result (Red Box)')

            matched_region = best_trans_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            if matched_region.size == 0:
                print("Matched region is empty, skipping overlay.")
                return

            resized_matched_region = cv2.resize(
                matched_region,
                (em_mask.shape[1], em_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            if len(em_mask.shape) == 2:  
                em_mask = cv2.cvtColor(em_mask, cv2.COLOR_GRAY2BGR)

            overlay = em_mask.copy()
            if len(overlay.shape) == 2:  
                overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

            if len(resized_matched_region.shape) == 2:
                resized_matched_region = cv2.cvtColor(resized_matched_region, cv2.COLOR_GRAY2BGR)

            overlay[:, :, 0] = resized_matched_region[:, :, 0]
            overlay[:, :, 1] = em_mask[:, :, 1]
            overlay[:, :, 2] = em_mask[:, :, 2]
            print("resized_matched_region shape:", resized_matched_region.shape)
            print("overlay shape:", overlay.shape)

            overlayed_result = cv2.addWeighted(em_mask, 0.5, overlay, 0.5, 0)
            overlay_path = "correlation_results/overlayed_match.png"
            cv2.imwrite(overlay_path, overlayed_result)
            display_image_with_matplotlib(overlayed_result, 'Overlayed Match (Blue Highlight)')

            # --- APPLY TRANSFORMATION TO THE LM IMAGE (unchanged) ---
            if isinstance(image1, str):
                lm_image = cv2.imread(image1)
            elif isinstance(image1, Image.Image):
                lm_image = np.array(image1)
            else:
                lm_image = image1

            if lm_image is None:
                raise ValueError("Error: LM image could not be loaded!")

            if len(lm_image.shape) == 2:
                lm_image = cv2.cvtColor(lm_image, cv2.COLOR_GRAY2BGR)

            trans_lm_image = rotate_image(lm_image, angle)
            trans_lm_image = flip_image(trans_lm_image, flipCode)

            cv2.rectangle(trans_lm_image, top_left, bottom_right, (0, 0, 255), 3)

            lm_with_box_path = "correlation_results/lm_image_with_box.png"
            cv2.imwrite(lm_with_box_path, trans_lm_image)
            print(f"✅ LM image with red box saved at: {lm_with_box_path}")

            # Re-apply transform for cropping
            trans_lm_image = rotate_image(lm_image, angle)
            trans_lm_image = flip_image(trans_lm_image, flipCode)

            x1, y1 = top_left
            x2, y2 = bottom_right

            cropped_lm_region = trans_lm_image[y1:y2, x1:x2]

            if cropped_lm_region.size == 0: 
                print("Error: Cropped LM region is empty. Check coordinates.")
            else:
                cropped_lm_path = "correlation_results/cropped_lm_image.png"
                cv2.imwrite(cropped_lm_path, cropped_lm_region)
                print(f"✅ Corrected Cropped LM region saved at: {cropped_lm_path}")

            cropped_lm_path = "correlation_results/cropped_lm_image.png"
            cv2.imwrite(cropped_lm_path, cropped_lm_region)
            print(f"✅ Cropped LM image saved at: {cropped_lm_path}")

            messagebox.showinfo(
                "Correlation Result",
                f"Correlation result saved:\n{result_path}\nOverlay saved at {overlay_path}\n"
                f"LM Image with Red Box saved at {lm_with_box_path}\nCropped LM region saved at {cropped_lm_path}"
            )
        else:
            messagebox.showinfo("Correlation Result", "No match found.")

    except Exception as e:
        traceback.print_exc()
        messagebox.showerror("Error", f"An error occurred during correlation: {e}")


def preprocess_for_lm(image_path):
    """
    Preprocess the LM image for segmentation using the working logic.
    """
    try:
        image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
        image = tf.image.resize(image, (1024, 1024)) / 255.0
        input_image = np.expand_dims(image.numpy(), axis=0)  # Add batch dimension
        return input_image
    except Exception as e:
        print(f"Preprocessing error for LM: {e}")
        return None


def postprocess_lm_output(prediction, original_size, threshold=0.5):
    """
    Postprocess the LM segmentation output to match the original size.
    """
    try:
        prediction = prediction.squeeze()  # Remove batch dimension
        binary_mask = (prediction > threshold).astype(np.uint8)  # Binarize
        resized_mask = cv2.resize(binary_mask, original_size[::-1], interpolation=cv2.INTER_NEAREST)
        return resized_mask * 255  # Scale to 0-255
    except Exception as e:
        print(f"Postprocessing error for LM: {e}")
        return None


def pad_to_multiple(img, mult=32, pad_value=0):
    h, w = img.shape[:2]
    nh = math.ceil(h / mult) * mult
    nw = math.ceil(w / mult) * mult
    pad_bottom = nh - h
    pad_right  = nw - w
    if img.ndim == 3:
        padded = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right,
                                    cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))
    else:
        padded = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right,
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded, (pad_bottom, pad_right)

def preprocess_for_lm_native(image_path):
    # read as RGB float32 in [0,1]
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read LM image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    padded, (pb, pr) = pad_to_multiple(rgb, mult=32, pad_value=0)
    x = np.expand_dims(padded, axis=0)  # (1,H',W',3)
    return x, rgb.shape[:2], (pb, pr)

def postprocess_lm_native(pred, orig_size, pads, thr=0.5):
    # pred: (1,H',W',1)
    h0, w0 = orig_size
    pb, pr = pads
    y = pred[0, ...]  # (H',W',1) or (H',W')
    if y.ndim == 3: y = y[..., 0]
    if pb or pr:
        y = y[:y.shape[0]-pb, :y.shape[1]-pr]
    y = y[:h0, :w0]  # safety crop
    mask = (y >= thr).astype(np.uint8) * 255
    return mask  # native size



def _to_3c_bgr(img):
    # img can be path, PIL, or np
    if isinstance(img, str):
        im = cv2.imread(img, cv2.IMREAD_COLOR)   # BGR uint8
    elif isinstance(img, Image.Image):
        im = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        im = img
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        elif im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    return im

def pad_to_multiple(img, mult=32):
    h, w = img.shape[:2]
    H = int(np.ceil(h / mult) * mult)
    W = int(np.ceil(w / mult) * mult)
    pad_bottom = H - h
    pad_right  = W - w
    # pad on bottom/right to preserve top-left alignment
    img_p = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101)
    return img_p, (pad_bottom, pad_right)

def unpad(img_p, pad_info):
    pad_bottom, pad_right = pad_info
    if pad_bottom == 0 and pad_right == 0:
        return img_p
    h, w = img_p.shape[:2]
    return img_p[:h - pad_bottom, :w - pad_right]


def preprocess_for_lm_variable(image_path_or_obj):
    im_bgr = _to_3c_bgr(image_path_or_obj)           # HxWx3 uint8
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    im_padded, pad_info = pad_to_multiple(im_rgb, 32)
    x = (im_padded.astype(np.float32) / 255.0)[None, ...]  # (1, Hpad, Wpad, 3)
    return x, pad_info, im_rgb.shape[:2]  # keep original H,W for sanity

def postprocess_lm_output_variable(pred, pad_info, orig_hw, threshold=0.5):
    # pred is model output, shape (1, Hpad, Wpad, 1)
    p = pred[0, ..., 0]
    p_unpad = unpad(p, pad_info)
    # (optional) sanity crop to original size in case image had been pre-cropped elsewhere
    H, W = orig_hw
    p_unpad = p_unpad[:H, :W]
    mask = (p_unpad > threshold).astype(np.uint8) * 255
    return mask


def segment_image_lmm(image_in, model):
    try:
        x, pad_info, orig_hw = preprocess_for_lm_variable(image_in)
        pred = model.predict(x)
        # pred shape: (1, Hpad, Wpad, 1) for num_classes=1
        binary_mask = postprocess_lm_output_variable(pred, pad_info, orig_hw, threshold=0.5)
        return binary_mask  # uint8 0/255 at ORIGINAL resolution
    except Exception as e:
        print(f"Error in LM segmentation: {e}")
        return None


def preprocess_image_em(image, target_size=(256, 256)):

    if isinstance(image, Image.Image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # ✅ Convert grayscale → RGB (3-channel)
    if len(image.shape) == 2:  
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    else:  # If it's BGR (OpenCV loaded), convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, target_size) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # (1,256,256,3)
    return image



def segment_image_em(image, model):
    """
    Segment the EM image using the provided model and return the binary mask.
    """
    try:
        if isinstance(image, Image.Image):  # If PIL Image, convert to NumPy
            image = np.array(image)

        preprocessed_image = preprocess_image_em(image)  # Must output (1,256,256,3)
        prediction = model.predict(preprocessed_image)

        # ✅ Model returns [refined_output, seg_output, conf]
        refined = prediction[0]  # Take ONLY refined output

        # Now refined is shape (1, 256, 256, 1)
        binary_mask = (refined[0, :, :, 0] > 0.5).astype(np.uint8)
        binary_mask = cv2.resize(binary_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        return binary_mask

    except Exception as e:
        print(f"Error in EM segmentation: {e}")
        return None





def display_mask(mask, title):
    """
    Display a segmentation mask in a new Tkinter window.
    """
    mask_pil = Image.fromarray(mask).resize((300, 300), Image.LANCZOS)
    mask_tk = ImageTk.PhotoImage(mask_pil)
    mask_window = tk.Toplevel(root)
    mask_window.title(title)
    label = tk.Label(mask_window, image=mask_tk)
    label.image = mask_tk
    label.pack()


def segment_images():
    global mask1, mask2

    if image1 is not None and image2 is not None:
        try:
            # Segment the LM image
            mask1 = segment_image_lmm(image1, model1)  # Updated LM segmentation
            if mask1 is None:
                raise ValueError("Segmentation for the first image (LM) failed.")

            # Save the LM mask
            lm_mask_save_path = "segmentation_results/mask1_LM_original.png"
            cv2.imwrite(lm_mask_save_path, mask1)

            # Segment the EM image
            resized_image2 = image2.resize((512, 512))  # Ensure correct size for processing
            mask2 = segment_image_em(np.array(resized_image2), unet_model_em)
            if mask2 is None:
                raise ValueError("Segmentation for the second image (EM) failed.")

            # Save the EM mask
            em_mask_save_path = "segmentation_results/mask2_EM.png"

            cv2.imwrite(em_mask_save_path, mask2 * 255)

            # Display segmentation masks in GUI
            display_mask(cv2.resize(mask1, (300, 300)), "LM Mask")
            display_mask(cv2.resize(mask2 * 255, (300, 300)), "EM Mask")

            messagebox.showinfo(
                "Segmentation Result",
                f"Segmentation masks saved:\n{lm_mask_save_path} (LM)\n{em_mask_save_path} (EM)",
            )
        except Exception as e:
            messagebox.showerror("Segmentation Error", f"Error in segmentation: {e}")
    else:
        messagebox.showwarning("Warning", "Please load both images before running segmentation.")


IMAGE_DISPLAY_SIZE = (300, 300)  # Adjust image size

def load_first_image():
    global image1
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).convert("RGB")
        img_resized = img.resize(IMAGE_DISPLAY_SIZE, Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)
        label1.config(image=img_tk, text="")
        label1.image = img_tk
        image1 = file_path

# Function to load the second image (EM)
def load_second_image():
    global image2
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).convert("L")
        img_resized = img.resize(IMAGE_DISPLAY_SIZE, Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)
        label2.config(image=img_tk, text="")
        label2.image = img_tk
        image2 = img

def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // reduction, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([input_tensor, se])

def FCN_CoReNet(input_shape=(256, 256, 3), num_classes=1):
    model_input = layers.Input(shape=input_shape)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=model_input)

    # Encoder
    conv1 = base_model.get_layer("conv1_relu").output
    conv2 = base_model.get_layer("conv2_block3_out").output
    conv3 = base_model.get_layer("conv3_block4_out").output
    conv4 = base_model.get_layer("conv4_block6_out").output
    conv5 = base_model.get_layer("conv5_block3_out").output

    # Decoder with SE and Dropout
    x = layers.Conv2DTranspose(512, kernel_size=3, strides=2, padding="same")(conv5)
    x = layers.Concatenate()([x, conv4])
    x = layers.Conv2D(512, kernel_size=3, padding="same", activation="relu")(x)
    x = se_block(x)
    x = layers.Dropout(0.3)(x)  # 🔥 Dropout after SE block

    x = layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, conv3])
    x = layers.Conv2D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = se_block(x)
    x = layers.Dropout(0.3)(x)  # 🔥 Dropout

    x = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, conv2])
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = se_block(x)
    x = layers.Dropout(0.3)(x)  # 🔥 Dropout

    x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, conv1])
    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = se_block(x)
    x = layers.Dropout(0.3)(x)  # 🔥 Dropout

    # Upsample to original size
    shared_features_up = layers.UpSampling2D(size=(2, 2))(x)

    # Output branches
    seg_output = layers.Conv2D(num_classes, kernel_size=1, activation='sigmoid', name='segmentation')(shared_features_up)

    conf = layers.Conv2D(32, 3, padding='same', activation='relu')(shared_features_up)
    conf = layers.Conv2D(1, 1, activation='sigmoid', name='confidence')(conf)

    refined_output = layers.Multiply(name='refined_output')([seg_output, conf])

    return Model(inputs=model_input, outputs=[refined_output, seg_output, conf])




def FCN_ResNet50(input_shape=(None, None, 3), num_classes=1):
    """
    Fully convolutional ResNet50-based FCN that accepts arbitrary H, W.
    """
    model_input = layers.Input(shape=input_shape)

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=model_input
    )

    # Encoder feature maps (same as your original)
    conv1 = base_model.get_layer("conv1_relu").output           # ~ H/4
    conv2 = base_model.get_layer("conv2_block3_out").output     # ~ H/4 or H/8
    conv3 = base_model.get_layer("conv3_block4_out").output     # ~ H/8
    conv4 = base_model.get_layer("conv4_block6_out").output     # ~ H/16
    conv5 = base_model.get_layer("conv5_block3_out").output     # ~ H/32

    # Decoder
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

    # FINAL LAYER: MUST CALL IT ON x
    x = layers.Conv2DTranspose(
        num_classes,
        kernel_size=3,
        strides=2,
        padding="same",
        activation="sigmoid" if num_classes == 1 else "softmax"
    )(x)   # <--- THIS (x) WAS MISSING

    return Model(inputs=model_input, outputs=x)


def quantify_now():
    if 'mask1' not in globals() or mask1 is None:
        messagebox.showwarning("Warning", "No LM mask in memory. Segment first.")
        return
    if 'mask2' not in globals() or mask2 is None:
        messagebox.showwarning("Warning", "No EM mask in memory. Segment first.")
        return

    try:
        os.makedirs("quantification_results", exist_ok=True)

        # ---------- LM Full Overlay ----------
        lm_img_name = os.path.splitext(os.path.basename(image1))[0] if isinstance(image1, str) else "LM"
        lm_orig = cv2.imread(image1) if isinstance(image1, str) else None
        if lm_orig is None:
            raise ValueError("LM original image could not be loaded.")
        if lm_orig.shape[:2] != mask1.shape[:2]:
            lm_orig = cv2.resize(lm_orig, (mask1.shape[1], mask1.shape[0]))

        lm_bin = ensure_binary(mask1)
        lm_skel, lm_end, lm_branch = build_full_skeleton_layers(lm_bin)
        lm_overlay_path = os.path.join("quantification_results", f"{lm_img_name}_LM_overlay.png")
        overlay_full_skeleton(lm_orig, lm_skel, lm_end, lm_branch, lm_overlay_path)

        lm_df = quantify_components(lm_bin, "LM", lm_img_name)
        lm_excel_path = append_to_excel_separate(lm_df, "LM", out_dir="quantification_results")

        # ---------- EM Full Overlay ----------
        em_img_name = "EM"
        em_orig_pil = image2  # PIL image
        em_orig_rgb = np.array(em_orig_pil.convert("RGB"))
        em_orig_bgr = cv2.cvtColor(em_orig_rgb, cv2.COLOR_RGB2BGR)

        em_mask_resized = cv2.resize((mask2 * 255).astype(np.uint8),
                                     (em_orig_bgr.shape[1], em_orig_bgr.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
        em_bin = ensure_binary(em_mask_resized)
        em_skel, em_end, em_branch = build_full_skeleton_layers(em_bin)
        em_overlay_path = os.path.join("quantification_results", f"{em_img_name}_EM_overlay.png")
        overlay_full_skeleton(em_orig_bgr, em_skel, em_end, em_branch, em_overlay_path)

        em_df = quantify_components(em_bin, "EM", em_img_name)
        em_excel_path = append_to_excel_separate(em_df, "EM", out_dir="quantification_results")

        messagebox.showinfo(
            "Quantification Done",
            f"✅ LM overlay saved → {lm_overlay_path}\n"
            f"✅ EM overlay saved → {em_overlay_path}\n\n"
            f"📊 LM metrics → {lm_excel_path}\n"
            f"📊 EM metrics → {em_excel_path}"
        )
    except Exception as qe:
        traceback.print_exc()
        messagebox.showerror("Quantification Error", f"{qe}")

# ======== THREAD-SAFE COMPUTE-ONLY VERSIONS (NO TK, NO MATPLOTLIB) ========

def segment_images_compute():
    """
    Runs segmentation and writes masks to disk. No UI calls.
    Returns dict with paths and shapes; sets global mask1, mask2.
    """
    global mask1, mask2, image1, image2
    if image1 is None or image2 is None:
        return {"ok": False, "msg": "Please load both LM and EM images before running segmentation."}

    os.makedirs("segmentation_results", exist_ok=True)

    # LM
    m1 = segment_image_lmm(image1, model1)
    if m1 is None:
        return {"ok": False, "msg": "Segmentation failed for LM image."}
    mask1 = m1
    lm_mask_save_path = os.path.join("segmentation_results", "mask1_LM_original.png")
    cv2.imwrite(lm_mask_save_path, mask1)

    # EM (compute at model size then save)
    resized_image2 = image2.resize((512, 512))
    m2 = segment_image_em(np.array(resized_image2), unet_model_em)
    if m2 is None:
        return {"ok": False, "msg": "Segmentation failed for EM image."}
    mask2 = m2
    em_mask_save_path = os.path.join("segmentation_results", "mask2_EM.png")
    cv2.imwrite(em_mask_save_path, (mask2 * 255).astype(np.uint8))

    return {
        "ok": True,
        "lm_path": lm_mask_save_path,
        "em_path": em_mask_save_path,
        "lm_shape": tuple(mask1.shape),
        "em_shape": tuple(mask2.shape),
    }




def correlate_images_compute():
    """
    Runs correlation and saves outputs. No UI calls or matplotlib.
    Returns dict with saved paths and transform info.
    """
    global mask1, mask2, image1
    if mask1 is None or mask2 is None:
        return {"ok": False, "msg": "Please segment both images before running correlation."}

    # ---------- PREPROCESS MASKS (like in correlate_lm_em) ----------
    # Convert to uint8 and rescale if in [0,1]
    lm_mask = (mask1 * 255).astype(np.uint8) if mask1.max() <= 1 else mask1.astype(np.uint8)
    em_mask = (mask2 * 255).astype(np.uint8) if mask2.max() <= 1 else mask2.astype(np.uint8)

    # Ensure single-channel
    if len(lm_mask.shape) == 3:
        lm_mask = cv2.cvtColor(lm_mask, cv2.COLOR_BGR2GRAY)
    if len(em_mask.shape) == 3:
        em_mask = cv2.cvtColor(em_mask, cv2.COLOR_BGR2GRAY)

    # Binarize to 0/255 like in the standalone script
    lm_mask = ((lm_mask > 0).astype(np.uint8)) * 255
    em_mask = ((em_mask > 0).astype(np.uint8)) * 255
    # ------------------------------------------------------ #

    os.makedirs("correlation_results", exist_ok=True)

    # Same search space as correlate_lm_em
    angles = range(0, 360, 1)
    flips = [1, 0, -1]
    scale_percents = [20, 30, 40, 50, 60, 70, 80, 90, 100]

    best_match = None  # (score, loc, angle, flip, scale, w, h, trans_lm_mask)

    for scale_percent in scale_percents:
        # Downscale EM mask once per scale (like in correlate_lm_em)
        downscaled_em_mask = downscale_image(em_mask, scale_percent)
        if downscaled_em_mask is None:
            continue

        th, tw = downscaled_em_mask.shape[:2]

        for angle in angles:
            for flipCode in flips:
                # Transform LM mask
                trans_lm_mask = rotate_image(lm_mask, angle)
                trans_lm_mask = flip_image(trans_lm_mask, flipCode)

                H, W = trans_lm_mask.shape[:2]
                if H < th or W < tw:
                    continue

                # Template matching on LM mask (shape-based)
                result = cv2.matchTemplate(trans_lm_mask, downscaled_em_mask, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if best_match is None or max_val > best_match[0]:
                    best_match = (
                        max_val,           # score
                        max_loc,           # top-left
                        angle,
                        flipCode,
                        scale_percent,
                        tw,                # width of template at this scale
                        th,                # height
                        trans_lm_mask.copy()
                    )

    if not best_match:
        return {"ok": False, "msg": "No match found during correlation."}

    max_val, max_loc, angle, flipCode, scale_percent, w, h, best_trans_img = best_match
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # ---------- SAVE BEST MATCH ON TRANSFORMED LM MASK ----------
    best_trans_img_bgr = cv2.cvtColor(best_trans_img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(best_trans_img_bgr, top_left, bottom_right, (0, 0, 255), 3)
    result_path = "correlation_results/best_match_result.png"
    cv2.imwrite(result_path, best_trans_img_bgr)

    # ---------- OVERLAY BLUE CHANNEL WITH MATCHED REGION (your original logic) ----------
    matched_region = best_trans_img_bgr[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    if matched_region.size > 0:
        resized_matched_region = cv2.resize(
            matched_region,
            (em_mask.shape[1], em_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        em_mask_bgr = cv2.cvtColor(em_mask, cv2.COLOR_GRAY2BGR)
        overlay = em_mask_bgr.copy()
        overlay[:, :, 0] = resized_matched_region[:, :, 0]
        overlay[:, :, 1] = em_mask_bgr[:, :, 1]
        overlay[:, :, 2] = em_mask_bgr[:, :, 2]
        overlayed_result = cv2.addWeighted(em_mask_bgr, 0.5, overlay, 0.5, 0)
        overlay_path = "correlation_results/overlayed_match.png"
        cv2.imwrite(overlay_path, overlayed_result)
    else:
        overlay_path = None

    # ---------- SAVE TRANSFORMED LM ORIGINAL WITH BOX + CROP (unchanged) ----------
    if isinstance(image1, str):
        lm_image = cv2.imread(image1)
    elif isinstance(image1, Image.Image):
        lm_image = np.array(image1)
    else:
        lm_image = image1

    if lm_image is None:
        return {"ok": False, "msg": "LM original image could not be loaded."}

    if len(lm_image.shape) == 2:
        lm_image = cv2.cvtColor(lm_image, cv2.COLOR_GRAY2BGR)

    trans_lm_image = rotate_image(lm_image, angle)
    trans_lm_image = flip_image(trans_lm_image, flipCode)
    cv2.rectangle(trans_lm_image, top_left, bottom_right, (0, 0, 255), 3)
    lm_with_box_path = "correlation_results/lm_image_with_box.png"
    cv2.imwrite(lm_with_box_path, trans_lm_image)

    x1, y1 = top_left
    x2, y2 = bottom_right
    cropped_lm_region = trans_lm_image[y1:y2, x1:x2]
    cropped_lm_path = None
    if cropped_lm_region.size > 0:
        cropped_lm_path = "correlation_results/cropped_lm_image.png"
        cv2.imwrite(cropped_lm_path, cropped_lm_region)

    return {
        "ok": True,
        "result_path": result_path,
        "overlay_path": overlay_path,
        "lm_with_box_path": lm_with_box_path,
        "cropped_lm_path": cropped_lm_path,
        "angle": angle,
        "flip": flipCode,
        "scale_percent": scale_percent,  # <-- fixed: single best scale, not the list
        "score": float(max_val)
    }



def quantify_compute():
    """
    Quantifies LM & EM, writes overlays + Excel. No UI calls.
    """
    global mask1, mask2, image1, image2
    if mask1 is None or mask2 is None:
        return {"ok": False, "msg": "Please segment first. No masks in memory."}

    os.makedirs("quantification_results", exist_ok=True)

    # LM
    lm_img_name = os.path.splitext(os.path.basename(image1))[0] if isinstance(image1, str) else "LM"
    lm_orig = cv2.imread(image1) if isinstance(image1, str) else None
    if lm_orig is None:
        return {"ok": False, "msg": "LM original image could not be loaded."}
    if lm_orig.shape[:2] != mask1.shape[:2]:
        lm_orig = cv2.resize(lm_orig, (mask1.shape[1], mask1.shape[0]))

    lm_bin = ensure_binary(mask1)
    lm_skel, lm_end, lm_branch = build_full_skeleton_layers(lm_bin)
    lm_overlay_path = os.path.join("quantification_results", f"{lm_img_name}_LM_overlay.png")
    overlay_full_skeleton(lm_orig, lm_skel, lm_end, lm_branch, lm_overlay_path)

    lm_df = quantify_components(lm_bin, "LM", lm_img_name)
    lm_excel_path = append_to_excel_separate(lm_df, "LM", out_dir="quantification_results")

    # EM
    em_img_name = "EM"
    em_orig_rgb = np.array(image2.convert("RGB"))  # image2 is PIL
    em_orig_bgr = cv2.cvtColor(em_orig_rgb, cv2.COLOR_RGB2BGR)

    em_mask_resized = cv2.resize((mask2 * 255).astype(np.uint8),
                                 (em_orig_bgr.shape[1], em_orig_bgr.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
    em_bin = ensure_binary(em_mask_resized)
    em_skel, em_end, em_branch = build_full_skeleton_layers(em_bin)
    em_overlay_path = os.path.join("quantification_results", f"{em_img_name}_EM_overlay.png")
    overlay_full_skeleton(em_orig_bgr, em_skel, em_end, em_branch, em_overlay_path)

    em_df = quantify_components(em_bin, "EM", em_img_name)
    em_excel_path = append_to_excel_separate(em_df, "EM", out_dir="quantification_results")

    return {
        "ok": True,
        "lm_overlay": lm_overlay_path,
        "em_overlay": em_overlay_path,
        "lm_excel": lm_excel_path,
        "em_excel": em_excel_path
    }

# # Load models
# model1 = FCN_ResNet50(input_shape=(None, None, 3), num_classes=1)
# model1.load_weights("fcn_resnet50_best.h5")
# unet_model_em = FCN_CoReNet(input_shape=(256, 256, 3))
# unet_model_em.load_weights("savedcopy.h5")



model1 = FCN_ResNet50(input_shape=(None, None, 3), num_classes=1)
model1.load_weights(resource_path("fcn_resnet50_best.h5"))

unet_model_em = FCN_CoReNet(input_shape=(256, 256, 3))
unet_model_em.load_weights(resource_path("savedcopy.h5"))



# ===================== IMAGE PREVIEW & FULL VIEWER =====================

def open_full_image(path):
    """Opens a full resolution image in a scrollable viewer window."""
    try:
        img = Image.open(path)
    except:
        messagebox.showerror("Error", f"Cannot open image: {path}")
        return

    top = tk.Toplevel(root)
    top.title(f"Full View - {os.path.basename(path)}")
    top.configure(bg=THEME["bg"])

    # Scrollable Canvas
    canvas = tk.Canvas(top, bg=THEME["bg"], highlightthickness=0)
    v_scroll = tk.Scrollbar(top, orient="vertical", command=canvas.yview)
    h_scroll = tk.Scrollbar(top, orient="horizontal", command=canvas.xview)
    canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

    v_scroll.pack(side="right", fill="y")
    h_scroll.pack(side="bottom", fill="x")
    canvas.pack(fill="both", expand=True)

    # Add image inside canvas
    img_tk = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image = img_tk  # prevent garbage collection
    canvas.config(scrollregion=canvas.bbox("all"))


def show_preview_window(title, image_paths):
    """Opens a themed preview window with clickable thumbnails."""
    prev = tk.Toplevel(root)
    prev.title(title)
    prev.configure(bg=THEME["bg"])
    prev.geometry("900x600")

    frame = tk.Frame(prev, bg=THEME["bg"])
    frame.pack(fill="both", expand=True, padx=10, pady=10)

    max_thumb_size = 350  # Thumbnail max size

    for idx, path in enumerate(image_paths):
        if not os.path.exists(path):  # Skip if missing
            continue
        try:
            img = Image.open(path)
            # Resize to thumbnail
            img.thumbnail((max_thumb_size, max_thumb_size))
            img_tk = ImageTk.PhotoImage(img)
        except:
            continue

        thumb = tk.Label(frame, image=img_tk, bg=THEME["panel"], bd=2, relief="ridge", cursor="hand2")
        thumb.image = img_tk  # keep ref
        thumb.grid(row=0, column=idx, padx=10, pady=10)

        # Click event to open full image
        thumb.bind("<Button-1>", lambda e, p=path: open_full_image(p))







# ===================== Modern Professional UI =====================
root = tk.Tk()
root.title("Mitochondria Analysis Suite")
root.configure(bg=THEME["bg"])
root.geometry("1320x780")


# -------- Status bar controller --------
status_text = tk.StringVar(value="Status: Idle")
def set_status(msg: str):
    status_text.set(f"Status: {msg}")
    # Force UI refresh
    root.update_idletasks()

# -------- Modal processing overlay --------

class ProcessingOverlay:
    """
    Busy indicator that can run as a full-window shade (old behavior)
    or a small floating popup (non-blocking visual).
    """
    def __init__(self, parent, text="Processing…", fullscreen=True):
        self.parent = parent
        self.text = text
        self.fullscreen = fullscreen
        self.win = None
        self._spin = True

    def show(self):
        if self.win:
            return

        if self.fullscreen:
            # ==== OLD: dim the whole root ====
            self.win = tk.Toplevel(self.parent)
            self.win.overrideredirect(True)
            self.win.attributes("-topmost", True)
            self.win.configure(bg=THEME["bg"])
            self.win.geometry(
                f"{self.parent.winfo_width()}x{self.parent.winfo_height()}"
                f"+{self.parent.winfo_rootx()}+{self.parent.winfo_rooty()}"
            )

            # Center card
            card_parent = self.win
        else:
            # ==== NEW: small floating popup ====
            self.win = tk.Toplevel(self.parent)
            self.win.overrideredirect(True)  # frameless
            self.win.attributes("-topmost", True)
            self.win.configure(bg=THEME["panel"])  # card is the window
            self.win.update_idletasks()

            w, h = 360, 120
            # place near top-center of the main window
            px = self.parent.winfo_rootx() + max(0, (self.parent.winfo_width() - w) // 2)
            py = self.parent.winfo_rooty() + 80
            self.win.geometry(f"{w}x{h}+{px}+{py}")

            card_parent = self.win  # the window itself acts as the card

        # Build the card content
        card = tk.Frame(card_parent, bg=THEME["panel"], bd=0,
                        highlightthickness=1, highlightbackground=THEME["border"])
        if self.fullscreen:
            card.place(relx=0.5, rely=0.5, anchor="center")
        else:
            # fill the small popup
            card.pack(fill="both", expand=True)

        lbl = tk.Label(card, text=self.text, font=("Segoe UI", 12, "bold"),
                       fg=THEME["text"], bg=THEME["panel"], padx=20, pady=12)
        lbl.pack()

        self.spinner = tk.Label(card, text="⏳", font=("Segoe UI", 16),
                                fg=THEME["accent"], bg=THEME["panel"])
        self.spinner.pack(pady=(0, 10))

        # animate spinner
        def animate():
            glyphs = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
            i = 0
            while self._spin and self.win and self.win.winfo_exists():
                self.spinner.config(text=glyphs[i % len(glyphs)])
                i += 1
                time.sleep(0.07)
        threading.Thread(target=animate, daemon=True).start()

        # size card to content if fullscreen
        if self.fullscreen:
            card.update_idletasks()
            w, h = max(220, card.winfo_width()), max(90, card.winfo_height())
            card.config(width=w, height=h)

    def close(self):
        self._spin = False
        if self.win and self.win.winfo_exists():
            self.win.destroy()
        self.win = None




# -------- Open folder utility --------
def open_folder(path: str):
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["explorer", path])
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        set_status(f"Opened folder: {path}")
    except Exception as e:
        messagebox.showerror("Open Folder", f"Cannot open folder:\n{path}\n\n{e}")
        set_status("Open folder failed")


# Global stateful status (no logic change to your functions)
status_lm  = tk.StringVar(value="Not loaded")
status_em  = tk.StringVar(value="Not loaded")
status_seg = tk.StringVar(value="Pending")
status_corr= tk.StringVar(value="Pending")
status_q   = tk.StringVar(value="Ready")

# ---------- Top Title Bar ----------
# ---------- Top Title Bar (with brand) ----------
title_bar = tk.Frame(root, bg=THEME["panel"], height=56, bd=0, highlightthickness=0)
title_bar.pack(fill="x", side="top")

brand_logo = make_brand_logo(22, 2)
tk.Label(
    title_bar, image=brand_logo, bg=THEME["panel"]
).pack(side="left", padx=(14, 6), pady=10)

tk.Label(
    title_bar, text="Deep-SegCLEM – Mitochondria Analysis Suite",
    font=("Segoe UI", 16, "bold"), fg=THEME["text"], bg=THEME["panel"]
).pack(side="left", padx=4, pady=10)

# Optional institute tag on the right
tk.Label(
    title_bar, text="Max Planck Institute for Biophysics", 
    font=("Segoe UI", 10), fg=THEME["muted"], bg=THEME["panel"]
).pack(side="right", padx=16, pady=10)


# ---------- Toolbar with Icons (flat minimal) ----------
toolbar = tk.Frame(root, bg=THEME["panel2"], height=48)
toolbar.pack(fill="x", padx=0, pady=(0, 10))

# Icon set
ico_load_lm   = make_outline_icon("load")
ico_load_em   = make_outline_icon("load2")
ico_segment   = make_outline_icon("segment")
ico_correlate = make_outline_icon("correlate")
ico_quantify  = make_outline_icon("quantify")

def mk_toolbtn(parent, icon, text, cmd):
    btn = tk.Button(
        parent, image=icon, compound="left", text="  "+text,
        font=("Segoe UI", 10, "bold"), fg=THEME["text"], bg=THEME["panel2"],
        activebackground=THEME["button-h"], activeforeground=THEME["text"],
        relief="flat", bd=0, padx=10, pady=6, cursor="hand2", command=cmd
    )
    # hover
    def _ent(e): btn.configure(bg=THEME["button-h"])
    def _lev(e): btn.configure(bg=THEME["panel2"])
    btn.bind("<Enter>", _ent); btn.bind("<Leave>", _lev)
    return btn

# Wrap existing handlers to update status without changing your logic
def on_load_lm():
    set_status("Loading LM image…")
    load_first_image()
    status_lm.set("Loaded ✓")
    set_status("LM loaded")

def on_load_em():
    set_status("Loading EM image…")
    load_second_image()
    status_em.set("Loaded ✓")
    set_status("EM loaded")



def show_segmentation_preview_with_edit(lm_path, em_path):
    prev = tk.Toplevel(root)
    prev.title("Segmentation Preview")
    prev.configure(bg=THEME["bg"])
    prev.geometry("980x600")

    wrap = tk.Frame(prev, bg=THEME["bg"])
    wrap.pack(fill="both", expand=True, padx=12, pady=12)

    cols = []
    for idx, (title, path, kind) in enumerate([
        ("LM Mask", lm_path, "LM"),
        ("EM Mask", em_path, "EM"),
    ]):
        col = tk.Frame(wrap, bg=THEME["bg-alt"], bd=0, highlightthickness=1, highlightbackground=THEME["border"])
        col.grid(row=0, column=idx, padx=10, pady=10, sticky="nsew")
        cols.append(col)

        tk.Label(col, text=title, font=("Segoe UI", 12, "bold"),
                 fg=THEME["text"], bg=THEME["bg-alt"]).pack(anchor="w", padx=10, pady=(10,4))

        # thumbnail
        if os.path.exists(path):
            try:
                im = Image.open(path)
                im.thumbnail((420, 420))
                imgtk = ImageTk.PhotoImage(im)
                lbl = tk.Label(col, image=imgtk, bg=THEME["panel"], bd=1, relief="ridge", cursor="hand2")
                lbl.image = imgtk
                lbl.pack(padx=10, pady=8)
                lbl.bind("<Button-1>", lambda e, p=path: open_full_image(p))
            except:
                tk.Label(col, text="(Cannot open image)", fg=THEME["danger"], bg=THEME["bg-alt"]).pack(padx=10, pady=8)
        else:
            tk.Label(col, text="(File not found)", fg=THEME["danger"], bg=THEME["bg-alt"]).pack(padx=10, pady=8)

        # action row
        row = tk.Frame(col, bg=THEME["bg-alt"])
        row.pack(fill="x", padx=10, pady=(4,12))
        tk.Button(row, text=f"✍️  Edit {kind} Mask",
                  command=lambda k=kind: MaskEditorWindow(k, root),
                  font=("Segoe UI", 10, "bold"),
                  fg=THEME["text"], bg=THEME["panel"], activebackground=THEME["button-h"],
                  relief="flat", padx=12, pady=6).pack(side="left", padx=4)
        tk.Button(row, text="Open File",
                  command=lambda p=path: open_full_image(p),
                  font=("Segoe UI", 10),
                  fg=THEME["text"], bg=THEME["panel"], activebackground=THEME["button-h"],
                  relief="flat", padx=12, pady=6).pack(side="left", padx=4)

    # make columns expand
    wrap.grid_columnconfigure(0, weight=1)
    wrap.grid_columnconfigure(1, weight=1)






def on_segment():
    overlay = ProcessingOverlay(root, "Segmenting…", fullscreen=False)
    overlay.show()
    root.update_idletasks()

    def worker():
        try:
            set_status("Running segmentation…")
            result = segment_images_compute()  # your segmentation wrapper
        finally:
            def _finish():
                overlay.close()
                if not result["ok"]:
                    set_status("Segmentation error")
                    messagebox.showerror("Segmentation", result.get("msg", "Unknown error"))
                else:
                    status_seg.set("Segmented ✓")
                    set_status("Segmentation complete")
                    messagebox.showinfo(
                        "Segmentation Done",
                        f"Saved:\n• {result['lm_path']}\n• {result['em_path']}"
                    )

                    # ✅ REPLACED with advanced editor preview
                    show_segmentation_preview_with_edit(result["lm_path"], result["em_path"])

                set_status("Idle")

            root.after(0, _finish)

    threading.Thread(target=worker, daemon=True).start()




def on_correlate():
    overlay = ProcessingOverlay(root, "Correlating…", fullscreen=False)
    overlay.show()
    root.update_idletasks()

    def worker():
        result = {"ok": False}
        try:
            set_status("Running correlation…")
            result = correlate_images()  # ✅ Use compute version (no UI blocking)
        except Exception as e:
            traceback.print_exc()
            set_status("Correlation failed")
        finally:
            def _finish():
                try:
                    overlay.close()  # ✅ FORCE CLOSE NO MATTER WHAT
                except:
                    pass

                # ✅ Use returned paths if available
                if result.get("ok") and result.get("result_path") and result.get("overlay_path"):
                    status_corr.set("Correlated ✓")
                    set_status("Correlation complete")
                    show_preview_window("Correlation Preview", [
                        result["result_path"], result["overlay_path"]
                    ])
                else:
                    set_status("No correlation output found")

                set_status("Idle")

            root.after(0, _finish)

    threading.Thread(target=worker, daemon=True).start()





def quantify_now_return_paths():
    """Runs quantification and RETURNS overlay paths for preview window."""
    lm_overlay_path = None
    em_overlay_path = None

    try:
        os.makedirs("quantification_results", exist_ok=True)

        # ---------- LM Full Overlay ----------
        lm_img_name = os.path.splitext(os.path.basename(image1))[0] if isinstance(image1, str) else "LM"
        lm_orig = cv2.imread(image1) if isinstance(image1, str) else None
        if lm_orig is None:
            raise ValueError("LM original image could not be loaded.")
        if lm_orig.shape[:2] != mask1.shape[:2]:
            lm_orig = cv2.resize(lm_orig, (mask1.shape[1], mask1.shape[0]))

        lm_bin = ensure_binary(mask1)
        lm_skel, lm_end, lm_branch = build_full_skeleton_layers(lm_bin)
        lm_overlay_path = os.path.join("quantification_results", f"{lm_img_name}_LM_overlay.png")
        overlay_full_skeleton(lm_orig, lm_skel, lm_end, lm_branch, lm_overlay_path)

        lm_df = quantify_components(lm_bin, "LM", lm_img_name)
        append_to_excel_separate(lm_df, "LM", out_dir="quantification_results")

        # ---------- EM Full Overlay ----------
        em_img_name = "EM"
        em_orig_rgb = np.array(image2.convert("RGB"))
        em_orig_bgr = cv2.cvtColor(em_orig_rgb, cv2.COLOR_RGB2BGR)

        em_mask_resized = cv2.resize((mask2 * 255).astype(np.uint8),
                                     (em_orig_bgr.shape[1], em_orig_bgr.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
        em_bin = ensure_binary(em_mask_resized)
        em_skel, em_end, em_branch = build_full_skeleton_layers(em_bin)
        em_overlay_path = os.path.join("quantification_results", f"{em_img_name}_EM_overlay.png")
        overlay_full_skeleton(em_orig_bgr, em_skel, em_end, em_branch, em_overlay_path)

        em_df = quantify_components(em_bin, "EM", em_img_name)
        append_to_excel_separate(em_df, "EM", out_dir="quantification_results")

        return lm_overlay_path, em_overlay_path

    except Exception as qe:
        traceback.print_exc()
        messagebox.showerror("Quantification Error", f"{qe}")
        return None, None


def on_quantify():
    overlay = ProcessingOverlay(root, "Quantifying…")
    overlay.show()
    root.update_idletasks()

    def worker():
        lm_overlay_path = None
        em_overlay_path = None
        try:
            set_status("Exporting Excel + skeleton overlays…")
            lm_overlay_path, em_overlay_path = quantify_now_return_paths()  # ✅ returns paths now
        finally:
            def _finish():
                overlay.close()
                if not lm_overlay_path or not em_overlay_path:
                    set_status("Quantification error")
                else:
                    status_q.set("Exported ✓")
                    set_status("Quantification done")
                    messagebox.showinfo(
                        "Quantification Done",
                        f"Saved:\n• {lm_overlay_path}\n• {em_overlay_path}"
                    )
                    # ✅ Preview
                    show_preview_window("Skeleton Overlay Preview", [
                        lm_overlay_path, em_overlay_path
                    ])
                set_status("Idle")
            root.after(0, _finish)

    threading.Thread(target=worker, daemon=True).start()


mk_toolbtn(toolbar, ico_load_lm,   "Load LM", on_load_lm).pack(side="left", padx=(10,4), pady=6)
mk_toolbtn(toolbar, ico_load_em,   "Load EM", on_load_em).pack(side="left", padx=4, pady=6)
sep1 = tk.Frame(toolbar, bg=THEME["border"], width=1, height=28); sep1.pack(side="left", padx=10, pady=10)
mk_toolbtn(toolbar, ico_segment,   "Segment", on_segment).pack(side="left", padx=4, pady=6)
mk_toolbtn(toolbar, ico_correlate, "Correlate", on_correlate).pack(side="left", padx=4, pady=6)
sep2 = tk.Frame(toolbar, bg=THEME["border"], width=1, height=28); sep2.pack(side="left", padx=10, pady=10)
mk_toolbtn(toolbar, ico_quantify,  "Quantify & Export", on_quantify).pack(side="left", padx=4, pady=6)
mk_toolbtn(toolbar, ico_quantify, "Quantify & Export", on_quantify)


# ---- Folder group selection + Open button ----
from tkinter import ttk as _ttk  # ensure ttk imported

folder_group_var = tk.StringVar(value="quantification_results")
folder_choices = ["segmentation_results", "correlation_results", "quantification_results"]

# A small dark-themed ttk Style for combobox
sty = _ttk.Style()
try:
    sty.theme_use("clam")
except:
    pass
sty.configure("Dark.TCombobox",
              fieldbackground=THEME["panel2"],
              background=THEME["panel2"],
              foreground=THEME["text"],
              arrowcolor=THEME["text"])

folder_combo = _ttk.Combobox(toolbar, textvariable=folder_group_var, values=folder_choices, width=24, state="readonly", style="Dark.TCombobox")
folder_combo.pack(side="left", padx=(12, 4), pady=8)

ico_open = make_outline_icon("folderopen")
def on_open_results():
    target = folder_group_var.get()
    os.makedirs(target, exist_ok=True)
    open_folder(os.path.abspath(target))

mk_toolbtn(toolbar, ico_open, "Open Results Folder", on_open_results).pack(side="left", padx=4, pady=6)

# Add HELP button at far right
sepH = tk.Frame(toolbar, bg=THEME["border"], width=1, height=28); sepH.pack(side="left", padx=10, pady=10)
ico_help = make_outline_icon("help")
mk_toolbtn(toolbar, ico_help, " Help / Manual", lambda: open_help_window()).pack(side="left", padx=4, pady=6)



# ---------- Main Content: two image panels ----------
content = tk.Frame(root, bg=THEME["bg"])
content.pack(fill="both", expand=True, padx=16, pady=(0, 16))

# LM Panel
panel_lm = tk.Frame(content, bg=THEME["bg-alt"], bd=0, highlightthickness=1, highlightbackground=THEME["border"])
panel_lm.place(relx=0.02, rely=0.02, relwidth=0.46, relheight=0.86)

hdr_lm = tk.Frame(panel_lm, bg=THEME["bg-alt"])
hdr_lm.pack(fill="x")
tk.Label(hdr_lm, text="Light Microscopy (LM)", font=("Segoe UI", 12, "bold"), fg=THEME["text"], bg=THEME["bg-alt"]).pack(side="left", padx=12, pady=10)

lm_status_lbl = tk.Label(hdr_lm, textvariable=status_lm, font=("Segoe UI", 9, "bold"),
                         fg=THEME["bg"], bg=THEME["success"], padx=8, pady=2)
lm_status_lbl.pack(side="right", padx=12, pady=10)

lm_canvas = tk.Label(panel_lm, bg=THEME["panel"])
lm_canvas.pack(fill="both", expand=True, padx=12, pady=(0,12))
# We re-point your existing label1 to this styled container:
label1 = lm_canvas

# EM Panel
panel_em = tk.Frame(content, bg=THEME["bg-alt"], bd=0, highlightthickness=1, highlightbackground=THEME["border"])
panel_em.place(relx=0.52, rely=0.02, relwidth=0.46, relheight=0.86)

hdr_em = tk.Frame(panel_em, bg=THEME["bg-alt"])
hdr_em.pack(fill="x")
tk.Label(hdr_em, text="Electron Microscopy (EM)", font=("Segoe UI", 12, "bold"),
         fg=THEME["text"], bg=THEME["bg-alt"]).pack(side="left", padx=12, pady=10)

em_status_lbl = tk.Label(hdr_em, textvariable=status_em, font=("Segoe UI", 9, "bold"),
                         fg=THEME["bg"], bg=THEME["warn"], padx=8, pady=2)
em_status_lbl.pack(side="right", padx=12, pady=10)

em_canvas = tk.Label(panel_em, bg=THEME["panel"])
em_canvas.pack(fill="both", expand=True, padx=12, pady=(0,12))
# We re-point your existing label2 to this styled container:
label2 = em_canvas

# ---------- Bottom Tip / Legend ----------
legend = tk.Frame(root, bg=THEME["bg"])
legend.pack(fill="x", padx=16, pady=(0, 10))
tk.Label(
    legend,
    text="Legend: Skeleton = white  •  Endpoints = green  •  Branch points = red",
    font=("Segoe UI", 10), fg=THEME["muted"], bg=THEME["bg"]
).pack(side="left")


# ---------- Bottom Status Bar ----------
status_bar = tk.Frame(root, bg=THEME["panel2"], height=28)
status_bar.pack(fill="x", side="bottom")
tk.Label(status_bar, textvariable=status_text, font=("Segoe UI", 10),
         fg=THEME["text"], bg=THEME["panel2"]).pack(side="left", padx=12, pady=4)

tk.Label(status_bar, text="Deep-SegCLEM v1.0", font=("Segoe UI", 10),
         fg=THEME["muted"], bg=THEME["panel2"]).pack(side="right", padx=12, pady=4)


# Keep the app running
root.mainloop()


# 🔬 Deep-SegCLEM

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-brightgreen?logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.10%2B-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/napari-plugin-blue?logo=napari" />
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
  <img src="https://img.shields.io/badge/Institute-MPI%20Biophysics-blueviolet" />
</p>

<p align="center">
  <b>Deep-SegCLEM</b> — Automated mitochondria segmentation, LM↔EM correlation,<br>
  and morphological quantification for CLEM data.
</p>

<p align="center">
  Max Planck Institute for Biophysics &nbsp;·&nbsp; Zaghbani et al., 2025
</p>

---

## 📦 Two Ways to Use Deep-SegCLEM

| | Standalone GUI (`.exe`) | Napari Plugin |
|---|---|---|
| **Best for** | Windows users, no coding required | Any OS, interactive layer editing |
| **Download** | [⬇️ Zenodo](https://zenodo.org/uploads/17702251) or `Executable/` folder | `pip install -e .` |
| **Requires** | Nothing — just run the `.exe` | Python + napari + TensorFlow |
| **Mask editing** | Built-in draw/erase editor | Napari paint/erase brush |
| **Jump to** | [Standalone GUI docs](#-standalone-gui) | [Napari plugin docs](#-napari-plugin) |

---

## 🖥️ Standalone GUI

### Download & Run (Windows)

**Option 1 — Zenodo (recommended, permanent DOI link):**
> 📥 [https://zenodo.org/uploads/17702251](https://zenodo.org/uploads/17702251)

**Option 2 — GitHub repository:**
1. Go to the [`Executable/`](./Executable) folder in this repository
2. Download `Deep-SegCLEM.exe`
3. Double-click to launch — **no installation required**

> ⚠️ On first launch Windows may show a SmartScreen warning.  
> Click **"More info" → "Run anyway"** to proceed.

> ⚠️ Place the `.exe` in the same folder as your model weight files:
> - `fcn_resnet50_best.h5`
> - `savedcopy.h5`

---

### Interface Overview

The GUI has a single window with a toolbar across the top and two image preview panels (LM on the left, EM on the right).

```
┌─────────────────────────────────────────────────────────────┐
│  Deep-SegCLEM – Mitochondria Analysis Suite                 │
├──────────────────────────────────────────────────────────────┤
│ [Load LM] [Load EM] │ [Segment] [Correlate] │ [Quantify]    │
├───────────────────────┬──────────────────────────────────────┤
│                       │                                      │
│   LM Image Preview    │       EM Image Preview               │
│                       │                                      │
└───────────────────────┴──────────────────────────────────────┘
│ Status bar                                                   │
└─────────────────────────────────────────────────────────────┘
```

---

### Step-by-Step Workflow

#### Step 1 — Load LM Image
- Click **Load LM** in the toolbar
- Select your fluorescence microscopy image (`.png`, `.jpg`, `.tif`)
- The image appears in the **left preview panel**

#### Step 2 — Load EM Image
- Click **Load EM** in the toolbar
- Select your electron microscopy image (`.png`, `.jpg`, `.tif`)
- The image appears in the **right preview panel**

#### Step 3 — Segment
- Click **Segment**
- The software runs both deep learning models automatically:
  - **FCN-ResNet50** → segments the LM image
  - **FCN-CoReNet** → segments the EM image
- Two mask preview windows pop up showing the results
- Masks are saved automatically to `segmentation_results/`

> 💡 **Manual mask correction:** After segmentation, click **Edit LM Mask** or **Edit EM Mask** in the preview window.  
> - **Left mouse** = draw (add to mask)  
> - **Right mouse** = erase (remove from mask)  
> - Use the **Brush size slider** to adjust brush radius  
> - Click **Apply & Overwrite** to save your edits

#### Step 4 — Correlate
- Click **Correlate**
- The software searches for the best alignment of the LM mask onto the EM mask by testing:
  - All rotations (0°–360°)
  - Three flip modes (horizontal, vertical, both)
  - Multiple scale factors (20%–100%)
- A result window shows the best match with a red bounding box
- All outputs are saved to `correlation_results/`

> ⏱️ Correlation searches 360° × 3 flips × multiple scales — this can take **2–10 minutes** depending on image size.

#### Step 5 — Quantify & Export
- Click **Quantify & Export**
- The software computes per-mitochondrion metrics for both LM and EM masks
- Skeleton overlay images and Excel files are saved to `quantification_results/`

---

### Output Files

```
segmentation_results/
├── mask1_LM_original.png        ← LM binary mask at original resolution
└── mask2_EM.png                 ← EM binary mask at original resolution

correlation_results/
├── best_match_result.png        ← Best match with red bounding box
├── overlayed_match.png          ← LM crop blended onto EM
├── lm_image_with_box.png        ← Transformed LM with match region
├── cropped_lm_image.png         ← Cropped LM region
└── em_with_lm_crop_overlay.png  ← EM with green LM signal overlay

quantification_results/
├── <name>_LM_overlay.png        ← LM skeleton overlay image
├── EM_overlay.png               ← EM skeleton overlay image
├── LM_metrics.xlsx              ← Per-mitochondrion LM metrics
└── EM_metrics.xlsx              ← Per-mitochondrion EM metrics
```

---

### Overlay Color Legend

| Color | Meaning |
|---|---|
| 🟣 Magenta | Skeleton (medial axis) |
| 🟢 Green | Endpoints (open tips) |
| 🔴 Red | Branch points (network junctions) |

---

### Excel Metrics Reference

| Column | Description |
|---|---|
| `area_px` | Mitochondrion area in pixels |
| `perimeter_px` | Boundary length in pixels |
| `circularity` | 4π × Area / Perimeter² — near **1.0** = round; lower = elongated |
| `skeleton_length_px` | Medial axis length (tubularity proxy) |
| `n_endpoints` | Number of open skeleton tips |
| `n_branchpoints` | Junction count (network complexity indicator) |

---

### Troubleshooting (Standalone GUI)

| Problem | Solution |
|---|---|
| App won't open / SmartScreen warning | Click "More info" → "Run anyway" |
| "Model weights not found" error | Place `.exe` and `.h5` files in the same folder |
| Segmentation produces empty mask | Check that the image loaded correctly; try a different threshold |
| Correlation takes very long | This is expected — the full search covers 360° × 3 flips × 5 scales |
| Excel file not generated | Make sure `quantification_results/` folder is writable |

---

---

## 🔌 Napari Plugin

The same pipeline is also available as a **napari dock widget**, enabling interactive mask editing directly on image layers.

### Installation

**1 — Clone the repository**
```bash
git clone https://github.com/SoumayaZgh1/Deep-SegCLEM.git
cd Deep-SegCLEM
```

**2 — Create a conda environment** (recommended)
```bash
conda create -n deepsegclem python=3.9
conda activate deepsegclem
```

**3 — Install TensorFlow**
```bash
# CPU only
pip install "tensorflow>=2.10,<2.16"

# GPU (CUDA 11.x)
pip install "tensorflow-gpu>=2.10,<2.16"
```

**4 — Fix NumPy version** (required for TF compatibility)
```bash
pip install "numpy<2"
```

**5 — Install napari**
```bash
pip install "napari[all]" pyqt5
```

**6 — Install the plugin**
```bash
cd src
pip install -e .
```

**7 — Launch**
```bash
python -m napari
```

Then go to **Plugins → Deep-SegCLEM**.

---

### Plugin Tabs

| Tab | Function |
|---|---|
| ⚙️ **Setup** | Browse and load `.h5` model weight files |
| 📂 **Load** | Open LM, EM and optional organelle channel images as napari layers |
| 🔬 **Segment** | Run deep learning segmentation → masks appear as label layers |
| 🔗 **Correlate** | Template-matching alignment of LM mask to EM |
| 📊 **Quantify** | Compute metrics and export to Excel + overlay images |

---

### Interactive Mask Editing (Napari)

After segmentation, masks appear as **label layers** in napari:

1. Click `LM_mask` or `EM_mask` in the layer list
2. Select the **Paint** tool (brush icon) in the napari toolbar
3. Draw corrections directly on the canvas
4. Select the **Erase** tool to remove false positives
5. Click **"Sync Masks from Layers → Memory"** in the plugin panel when done

---

### Troubleshooting (Napari Plugin)

| Error | Solution |
|---|---|
| `AttributeError: _ARRAY_API not found` | `pip install "numpy<2"` |
| `'napari' is not recognized` | Use `python -m napari` |
| Plugin not in Plugins menu | Run `pip install -e .` from the repo folder, then restart napari |
| Models load but mask is empty | Lower threshold to 0.3 in the Segment tab |

---

## 🗂️ Repository Structure

```
Deep-SegCLEM/
├── Executable/                  ← Standalone Windows .exe
├── Example_LM-EM_images/        ← Example input images to test the software
├── Example-results/             ← Example output files
├── models/                      ← Model weight files (.h5)
├── dataset/                     ← Dataset links and references
├── src/                         ← Napari plugin source code
│   └── napari_deepsegclem/
│       ├── _core.py             ← All computation (segmentation, correlation, quantification)
│       ├── _widget.py           ← Napari dock widget UI
│       └── napari.yaml          ← Napari plugin manifest
├── pyproject.toml               ← Plugin package config
└── README.md                    ← This file
```

---

## 🧠 Model Architectures

### LM Model — FCN-ResNet50
Fully convolutional ResNet50 with a U-Net-style skip-connection decoder. Accepts **any input resolution** via reflective padding. Predicts at native image size.

### EM Model — FCN-CoReNet
ResNet50 backbone extended with **Squeeze-and-Excitation (SE) blocks**, dropout regularisation, and a dual-branch output: a segmentation head multiplied element-wise by a confidence head to produce a refined prediction.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 📚 Citation

```bibtex
@software{zaghbani2025deepsegclem,
  author    = {Zaghbani, Soumaya},
  title     = {Deep-SegCLEM: Automated mitochondria segmentation and
               correlation for CLEM data},
  year      = {2025},
  publisher = {Max Planck Institute for Biophysics},
  doi       = {10.5281/zenodo.17702251},
  url       = {https://github.com/SoumayaZgh1/Deep-SegCLEM}
}
```

---

## 👤 Contact

**Soumaya Zaghbani**  
Max Planck Institute for Biophysics, Frankfurt am Main, Germany  
soumaya.zaghbani@biophys.mpg.de

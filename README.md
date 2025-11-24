# Deep-SegCLEM
**README – Deep-SegCLEM

Mitochondria Segmentation, Correlation & Quantification Suite**

Deep-SegCLEM is a standalone application for automated mitochondria segmentation, fiducial-free LM-EM correlation, and morphometric quantification, developed at the Max Planck Institute for Biophysics (García-Saez Group).

The tool combines deep learning–based segmentation for both LM and EM modalities with automated alignment and skeleton-based morphological analysis.
It is designed for researchers working with CLEM datasets, regulated cell death, mitochondrial ultrastructure, and fluorescence–EM correlation workflows.

🔧 Features
1. Light Microscopy (LM) Segmentation

Uses a fully convolutional ResNet-50 FCN model

Accepts arbitrary LM image sizes

Automatically resizes to segmentation resolution

Produces accurate mitochondria masks in native resolution

Includes a built-in mask editor

2. Electron Microscopy (EM) Segmentation

Uses the custom FCN-CoReNet model (refined + segmentation + confidence branch)

Standardized processing at 256×256 with resolution recovery

Fast inference, robust boundary detection

3. LM ↔ EM Correlation (Fiducial-free)

Exhaustive transform search (rotation, flip, scaling)

Template matching for optimal alignment

Saves:

Transformed LM mask

Best-match bounding box

Blue-channel overlay

Cropped LM region corresponding to EM mitochondria

4. Morphological Quantification

For both LM and EM masks, the tool computes:

Area (px)

Perimeter (px)

Circularity

Skeleton length

Number of endpoints

Number of branchpoints

Bounding boxes

Full-image skeleton overlays

Exports to:

LM_metrics.xlsx

EM_metrics.xlsx

Skeleton overlay images

5. Graphical Interface

Toolbar with icons (Load, Segment, Correlate, Quantify)

Preview windows for masks, overlays, and correlation results

Scrollable help manual included in-app

📦 Installation

No installation required.
The application is distributed as a single Windows executable (.exe) created via PyInstaller.

Simply download and run:

Deep-SegCLEM.exe


No Python environment needed.

📁 Folder Structure (Auto-generated)
Deep-SegCLEM/
 ├── segmentation_results/
 │     ├── mask1_LM_original.png
 │     └── mask2_EM.png
 │
 ├── correlation_results/
 │     ├── best_match_result.png
 │     ├── overlayed_match.png
 │     ├── lm_image_with_box.png
 │     └── cropped_lm_image.png
 │
 ├── quantification_results/
 │     ├── <LM_name>_LM_overlay.png
 │     ├── EM_EM_overlay.png
 │     ├── LM_metrics.xlsx
 │     └── EM_metrics.xlsx

Usage
1. Load images

Click Load LM → Select LM fluorescence image

Click Load EM → Select EM grayscale image

2. Segment

Press Segment.
The app will:

Run deep-learning segmentation

Save masks

Show preview

Allow manual mask editing (optional)

3. Correlate

Click Correlate to run fiducial-free alignment.
The app outputs:

Best match

Overlay

LM crop

4. Quantify

Click Quantify & Export to compute:

Per-mitochondrion metrics

Full-image skeleton overlays

Excel spreadsheets

🧠 Models Used
LM Model

Architecture: FCN-ResNet50 (fully convolutional)

Input: Arbitrary size

Output: 1-channel mitochondria mask

Trained on CellDeathPred-compatible fluorescence data

EM Model

Architecture: FCN-CoReNet

Output branches:

Refined mask

Raw segmentation

Confidence

Trained on high-resolution mitochondrial EM datasets


👩‍💻 Author

Soumaya Zaghbani
Max Planck Institute for Biophysics
Frankfurt am Main, Germany

For questions or support:
📧 soumaya.zaghbani@biophys.mpg.de

📄 Citation

If this tool supports your research, please cite:



📝 License

This software is intended for academic and research use only.
Redistribution without permission is prohibited.

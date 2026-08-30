"""
napari-deepsegclem
==================
Napari plugin for CLEM mitochondria segmentation, correlation, and quantification.
Max Planck Institute for Biophysics – Zaghbani et al. (2025)
"""

from ._version import __version__
from ._widget import DeepSegCLEMWidget

__all__ = ["DeepSegCLEMWidget", "__version__"]

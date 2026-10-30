"""Multiclass semantic segmentation on SuperTuxKart via frozen SAM 2 + a light decoder."""

__version__ = "0.1.0"

from stkseg.constants import CLASS_NAMES, COLORS, NUM_CLASSES

__all__ = ["CLASS_NAMES", "COLORS", "NUM_CLASSES", "__version__"]

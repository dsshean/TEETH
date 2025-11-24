"""
TEETH: Dental Disease Detection with PyTorch

A complete PyTorch implementation for detecting and classifying dental diseases.
"""

__version__ = "1.0.0"
__author__ = "TEETH Project"

# Make submodules easily importable
from . import detection
from . import classification

__all__ = ["detection", "classification"]

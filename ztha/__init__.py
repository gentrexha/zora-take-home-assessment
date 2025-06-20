"""
Zora Take-Home Assessment: Collector Churn Analysis

A machine learning pipeline for predicting collector churn in NFT marketplaces.
"""

from .config import CONFIG
from .pipeline import ChurnPredictor, main

__version__ = "0.1.0"
__author__ = "Gent Rexha"

__all__ = ["ChurnPredictor", "main", "CONFIG"]

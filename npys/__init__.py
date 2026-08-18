"""n-PYS: Data-Driven Variable-Exponent Analysis for Photoemission Yield Spectroscopy.

Note: `fitter.FitResult` and `robust_piecewise_segmenter.FitResult` are two
distinct dataclasses that happen to share the same name. To avoid ambiguity,
only the former is re-exported here as `FitResult`; the segmenter's result
type is available as `robust_piecewise_segmenter.FitResult` if needed.
"""

from .fitter import (
    OneOverNScanFitter,
    ThreeLSFitter,
    BaseYieldFitter,
    FitResult,
    DataGenerator,
)
from .rk_pys_roi import pys_roi, PeakEstimator, SpectrumROIExtractor
from .robust_piecewise_segmenter import (
    PiecewiseLinearRegression,
    PiecewiseQuadraticRegression,
    PiecewiseHybridRegression,
)
from .datconv import AcConv, AdvAcConv
from .reportmaker import ReportBuilder
from . import batch_analyzer

__all__ = [
    "OneOverNScanFitter",
    "ThreeLSFitter",
    "BaseYieldFitter",
    "FitResult",
    "DataGenerator",
    "pys_roi",
    "PeakEstimator",
    "SpectrumROIExtractor",
    "PiecewiseLinearRegression",
    "PiecewiseQuadraticRegression",
    "PiecewiseHybridRegression",
    "AcConv",
    "AdvAcConv",
    "ReportBuilder",
    "batch_analyzer",
]

"""
wavexplain: causal, counterfactual attribution for multi-series
time-series forecasters.

Quickstart:
    from wavexplain import MultiSeriesWaveNet, CounterfactualExplainer, render_card_html

See README.md for a full example.
"""

from .model import MultiSeriesWaveNet, FixedSeriesWrapper, GatedDilatedCausalConv1d
from .attribution import CounterfactualExplainer
from .cards import render_card_html

__version__ = "0.1.0"

__all__ = [
    "MultiSeriesWaveNet",
    "FixedSeriesWrapper",
    "GatedDilatedCausalConv1d",
    "CounterfactualExplainer",
    "render_card_html",
]

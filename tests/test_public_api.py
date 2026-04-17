"""Checks for the supported top-level API surface."""

from __future__ import annotations

import sccircuits
from sccircuits import BBQ, Circuit, FitAnalysis, PointPicker, TransitionFitter


def test_get_info_lists_current_public_classes():
    info = sccircuits.get_info()

    assert info["version"] == sccircuits.get_version()
    assert info["main_classes"] == [
        "Circuit",
        "BBQ",
        "TransitionFitter",
        "FitAnalysis",
    ]
    assert "CircuitFitter" not in info["main_classes"]


def test_current_top_level_exports_are_importable():
    assert Circuit is sccircuits.Circuit
    assert BBQ is sccircuits.BBQ
    assert TransitionFitter is sccircuits.TransitionFitter
    assert FitAnalysis is sccircuits.FitAnalysis
    assert PointPicker is sccircuits.PointPicker
    assert not hasattr(sccircuits, "CircuitFitter")

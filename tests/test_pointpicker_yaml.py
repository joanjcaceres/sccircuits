"""Tests for PointPicker YAML serialization and backward compatibility."""

from __future__ import annotations

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sccircuits.pointpicker import PointPicker


def test_save_and_load_yaml_preserves_unlabeled_points(tmp_path):
    fig, ax = plt.subplots()
    picker = PointPicker(ax, use_widgets=False)

    picker._add_point(1.0, 2.0)
    picker._add_point(3.0, 4.0)
    picker._apply_tag(0, 0, 1, 0.25)
    picker._apply_tag(1, 1, 2, 0.5)
    picker._remove_tag(1)

    output = tmp_path / "points.yaml"
    picker.save_yaml(str(output), include_sigma=True)
    plt.close(fig)

    with open(output, "r") as f:
        payload = yaml.safe_load(f)

    assert "unlabeled" in payload
    assert len(payload["unlabeled"]) == 1

    fig2, ax2 = plt.subplots()
    loaded = PointPicker.load_yaml(ax2, str(output))

    assert loaded.points.shape == (2, 2)
    np.testing.assert_allclose(loaded.points[0], np.array([1.0, 2.0]))
    np.testing.assert_allclose(loaded.points[1], np.array([3.0, 4.0]))
    assert loaded.labels == [(0, 1), None]
    assert loaded.sigmas == [0.25, None]
    plt.close(fig2)


def test_load_yaml_supports_legacy_format_without_unlabeled(tmp_path):
    legacy_payload = {
        "metadata": {
            "format_version": "1.0",
            "axis_lock": False,
            "x_scale": 1.0,
            "include_sigma": True,
        },
        "data": {
            "2,3": [[5.0, 6.0, 0.2]],
            "1,1": [[7.0, 8.0]],
        },
    }

    input_path = tmp_path / "legacy_points.yaml"
    with open(input_path, "w") as f:
        yaml.dump(legacy_payload, f, default_flow_style=False)

    fig, ax = plt.subplots()
    loaded = PointPicker.load_yaml(ax, str(input_path))

    assert loaded.points.shape == (2, 2)
    entries = {
        (label, tuple(float(v) for v in point), sigma)
        for point, label, sigma in zip(loaded.points, loaded.labels, loaded.sigmas)
    }
    assert entries == {
        ((2, 3), (5.0, 6.0), 0.2),
        ((1, 1), (7.0, 8.0), None),
    }
    plt.close(fig)

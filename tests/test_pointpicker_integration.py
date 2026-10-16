"""Compatibility between the standalone digitizer and spectroscopy fitting."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from mpl_pointpicker import PointPicker as StandalonePointPicker
from sccircuits import PointPicker, TransitionFitter
from sccircuits.pointpicker import PointPicker as ModulePointPicker


def test_old_imports_reexport_the_standalone_class():
    assert PointPicker is ModulePointPicker is StandalonePointPicker


def test_transition_fitter_consumes_standalone_exports(tmp_path):
    fig, ax = plt.subplots()
    picker = PointPicker(ax, use_widgets=False)
    picker._add_point(0.25, 5.1)
    picker._add_point(0.50, 3.2)
    picker._add_point(0.75, 4.0)
    picker.tag_point(0, (0, 1), 0.02)
    picker.tag_point(1, "series A", 0.1)

    def model(phi, params):
        return np.array([0.0, params[0] + phi])

    data = picker.to_transition_data(include_sigma=True)
    fitter = TransitionFitter(model, data, returns_eigenvalues=True)
    assert list(fitter.data) == [(0, 1)]
    assert fitter.data[(0, 1)][0].sigma == 0.02

    filename = tmp_path / "standalone.yaml"
    picker.save_yaml(filename)
    loaded_data = TransitionFitter.load_from_yaml(filename)
    assert loaded_data == data
    restored = TransitionFitter(model, loaded_data, returns_eigenvalues=True)
    assert restored.data == fitter.data
    picker.disconnect()
    plt.close(fig)

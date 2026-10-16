"""Compatibility imports for the standalone :mod:`mpl_pointpicker` package.

Both ``from sccircuits import PointPicker`` and the historical module import
remain supported. The implementation is maintained in mpl-pointpicker.
"""

from mpl_pointpicker import PointDict, PointPicker

__all__ = ["PointPicker", "PointDict"]

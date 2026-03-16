# Point Picking

`PointPicker` provides an interactive Matplotlib layer for collecting and
tagging spectroscopy points.

## Basic Usage

```python
import matplotlib.pyplot as plt
import numpy as np

from sccircuits import PointPicker

fig, ax = plt.subplots()
x = np.linspace(0.0, 2.0 * np.pi, 400)
y = np.sin(x)
ax.plot(x, y)

picker = PointPicker(ax, axis_lock=True)
plt.show()
```

The picker is active while the figure window is open. It uses keyboard-driven
modes:

| Key | Mode | Action |
| --- | --- | --- |
| `a` or `Esc` | add | Add new points |
| `m` | move | Drag existing points |
| `d` | delete | Remove a point |
| `t` | tag | Assign a transition label and optional sigma |
| `i` | inspect | Print point metadata |
| `n` | new column | Release the x-lock in `axis_lock=True` mode |

## Exporting Data

Save tagged points to YAML:

```python
picker.save_yaml("spectroscopy_points.yaml", include_sigma=True, x_scale=2.0 * np.pi)
```

Load them later:

```python
loaded = PointPicker.load_yaml(ax, "spectroscopy_points.yaml")
transition_data = loaded.to_transition_data(include_sigma=True)
```

## Typical Spectroscopy Workflow

1. plot the measured spectrum in Matplotlib
2. tag transitions with `PointPicker`
3. save the result as YAML
4. load the YAML with `TransitionFitter.load_from_yaml()`
5. fit your model

This keeps the manual labeling step separate from the fitting code while using
the same transition data structure end to end.

# BBQ Circuit Designer App

The file
[`apps/bbq_circuit_designer.py`](https://github.com/joanjcaceres/sccircuits/blob/main/apps/bbq_circuit_designer.py)
contains a Tk application for sketching a circuit graph and exporting the
matrix-building code needed by `BBQ`.

## Launch

Run the app from the repository root:

```bash
python apps/bbq_circuit_designer.py
```

The window title is **BBQ Matrix Builder**.

## Core Editing Modes

Toolbar buttons and shortcuts:

| Tool | Shortcut | Use |
| --- | --- | --- |
| Node | `n` | Create nodes by clicking empty canvas space |
| Edge | `c` | Connect two nodes and define capacitance/inductance |
| Ground | `g` | Connect a node to ground |
| Reset | none | Clear the canvas |

Useful canvas interactions:

- drag nodes to reposition them
- double-click a node to rename it
- shift-click to build a multi-node selection
- press `Delete` or `Backspace` to remove the focused node
- use `Ctrl` + mouse wheel to zoom
- use the middle mouse button to pan

## Graph Editing Features

The app also supports:

- **Copy** / **Paste** for selected subgraphs
- **Concatenate** to duplicate a selected block to the right
- **Merge** (`m`) to merge selected nodes into the focused node
- **Save** / **Load** for project snapshots
- **Undo** with `Ctrl+z` or `Cmd+z`

Ground connections can be edited or dragged like regular edges.

## Exporting Matrices

Use the **Snippet** button to copy a Python snippet to the clipboard. The
generated code includes helpers such as:

- `C_matrix_entries(...)`
- `C_matrix_sparse(...)`
- `C_matrix_func(...)`
- `L_inv_matrix_entries(...)`
- `L_inv_matrix_sparse(...)`
- `L_inv_matrix_func(...)`

Those functions evaluate the symbolic edge parameters and produce either dense
or sparse matrix forms.

## Feeding the Snippet into `BBQ`

The exported snippet is meant to be pasted into a normal Python script or
notebook:

```python
from sccircuits import BBQ

C_matrix = C_matrix_func(C1=40e-15, C2=32.9e-15, Cg=3e-15)
L_inv_matrix = L_inv_matrix_func(L1_inv=1 / 1.23e-9, Lg_inv=0.0)

bbq = BBQ(C_matrix, L_inv_matrix, non_linear_nodes=(-1, 0))
```

That gives you a reproducible bridge from the GUI sketch to the numerical model.

## Why This Guide Is Manual

The docs intentionally describe the app as a workflow rather than generating API
docs for it. The app imports Tk at runtime, so keeping it out of autodoc makes
Read the Docs builds more reliable.

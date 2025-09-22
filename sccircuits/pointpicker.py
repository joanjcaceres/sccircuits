import csv
import io
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

try:  # pragma: no cover - fallback when run as a script
    from ._pointpicker_tagui import BaseTagUI, create_tag_ui
except ImportError:  # pragma: no cover
    from _pointpicker_tagui import BaseTagUI, create_tag_ui  # type: ignore

# Type alias for load_csv_to_dict return type
PointDict = Dict[
    Tuple[int, int],
    List[Union[Tuple[float, float], Tuple[float, float, Optional[float]]]],
]

# Disable Matplotlib's default "l" key (toggle y‑log scale) so we can use it freely
if "l" in plt.rcParams.get("keymap.yscale", []):
    lst = plt.rcParams["keymap.yscale"]
    plt.rcParams["keymap.yscale"] = [k for k in lst if k != "l"]


class PointPicker:
    """Interactive point picker for matplotlib axes (cross‑platform friendly).

    Works with any content in the axes: images, plots, scatter plots, etc.

    Modes (switch with keyboard)
    ---------------------------
    | Key | Mode    | Click action                                |
    |-----|---------|---------------------------------------------|
    | **a** or **Esc** | **Add**    | Left‑click on axes → add point        |
    | **m** | **Move**   | Left‑click–drag a point → move          |
    | **d** | **Delete** | Left‑click on a point → delete          |
    | **t** | **Tag**    | Left‑click on point → assign label via prompt   |
    | **i** | **Inspect** | Left‑click on point → show label info (non-blocking) |

     *En Jupyter con %matplotlib widget se abre un cuadro interactivo para la tecla **t**.*

    Other behavior
    --------------
    • Antes del primer clic se muestra la línea; tras fijar el primer punto deja de moverse.
    • While the Matplotlib toolbar is in *zoom* or *pan* mode, clicks are **ignored**.
    • Pick radius can be tuned via *pick_radius* (in display pixels).
    • Works with multiple images/plots in the same axes.
    • save_project("file.npz") (incl. snapshot de la figura) y load_project(ax,"file.npz") para retomarlo.

    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(img1, alpha=0.7)
    >>> ax.contour(img2)
    >>> picker = PointPicker(ax)
    >>> plt.show()          # interact with a/m/d/t keys + left clicks
    >>> picker.save_to_csv('my_points.csv')
    >>> x_coords, y_coords = picker.get_coordinates()
    """

    def __init__(
        self,
        ax: plt.Axes,
        use_widgets: bool = True,
        *,
        pick_radius: int = 6,
        axis_lock: bool = False,
    ):
        self.ax = ax
        self.pick_radius = pick_radius

        # Tagging UI (ipywidgets when available, fallback to matplotlib controls)
        self._prefer_widgets = bool(use_widgets)
        self._tag_ui: BaseTagUI = create_tag_ui(
            self, prefer_widgets=self._prefer_widgets
        )

        # Axis‑lock configuration
        self.axis_lock = axis_lock
        self._fixed_x: Optional[float] = None  # Locked x for current column
        self._vline: Optional[plt.Line2D] = None  # Vertical guide line artist
        self.labels: list[Optional[Tuple[int, int]]] = []  # label metadata
        self._current_label: Optional[Tuple[int, int]] = None
        self.sigmas: list[Optional[float]] = []  # per-point uncertainty (None = unset)
        self._current_sigma: Optional[float] = None  # last used sigma value

        # Data containers: store points as an N×2 NumPy array
        self.points: np.ndarray = np.empty((0, 2))
        self._markers: list[plt.Line2D] = []

        # Interaction state
        self.mode: str = "add"  # 'add' | 'move' | 'delete' | 'tag' | 'inspect'
        self._selected_idx: Optional[int] = None  # index of point currently dragged

        # Connect canvas events
        canvas = ax.figure.canvas
        self._cids: list[int] = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("button_release_event", self._on_release),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("key_press_event", self._on_key),
        ]

        self._update_title()

    # ------------------------------------------------------------------
    def _set_marker_color(self, idx: int):
        """Apply marker color according to label/sigma availability."""
        if idx < 0 or idx >= len(self._markers):
            return
        lab = self.labels[idx] if idx < len(self.labels) else None
        sigma = self.sigmas[idx] if idx < len(self.sigmas) else None
        if lab is None:
            color = "r"
        else:
            color = "g" if sigma is not None else "orange"
        marker = self._markers[idx]
        marker.set_markerfacecolor(color)
        marker.set_markeredgecolor(color)

    # ------------------------------------------------------------------
    # Tagging helpers
    # ------------------------------------------------------------------
    def _tag_defaults_for(
        self, idx: int
    ) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        current_label = self.labels[idx] if idx < len(self.labels) else None
        sigma = self.sigmas[idx] if idx < len(self.sigmas) else None
        if current_label is None:
            current_label = self._current_label
        if sigma is None:
            sigma = self._current_sigma
        return current_label, sigma

    def _apply_tag(self, idx: int, i: int, j: int, sigma: Optional[float]) -> None:
        if idx < 0 or idx >= len(self.points):
            raise ValueError(f"Invalid point index {idx}")
        if sigma is not None and sigma <= 0:
            raise ValueError("sigma must be positive when provided")
        label = (int(i), int(j))
        self.labels[idx] = label
        self._current_label = label
        self.sigmas[idx] = sigma
        self._current_sigma = sigma
        self._set_marker_color(idx)
        self.ax.figure.canvas.draw_idle()

    def _remove_tag(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.points):
            return
        self.labels[idx] = None
        self.sigmas[idx] = None
        self._set_marker_color(idx)
        self.ax.figure.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Terminal input helper for tagging
    # ------------------------------------------------------------------
    def _inspect_point(self, idx: int):
        """
        Non-blocking inspection of point labels - just prints info to console.
        """
        current_label = self.labels[idx]
        x, y = self.points[idx]

        if current_label is not None:
            print(
                f"Point #{idx} at ({x:.2f}, {y:.2f}) → labeled as ({current_label[0]}, {current_label[1]})"
            )
        else:
            print(f"Point #{idx} at ({x:.2f}, {y:.2f}) → unlabeled")

    # Note: _terminal_tag_input removed to avoid blocking matplotlib interaction
    # Tag mode now delegates to an environment-specific UI (ipywidgets or matplotlib)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _toolbar_active(self) -> bool:
        """Return *True* if toolbar's *pan/zoom* tool is active."""
        tb = getattr(self.ax.figure.canvas, "toolbar", None)
        return bool(getattr(tb, "mode", ""))

    # -- Mouse press ----------------------------------------------------
    def _on_press(self, event):
        if event.inaxes is not self.ax or event.button != 1 or self._toolbar_active():
            return  # ignore right‑clicks and toolbar interactions

        idx = self._index_near(event)

        if self.mode == "add":
            if idx is None:
                # Regular add
                self._add_point(float(event.xdata), float(event.ydata))
            else:
                # Clicked an existing point → (re)activate its column when axis_lock is on
                if self.axis_lock:
                    self._fixed_x = self.points[idx, 0]
                    self._update_vline(self._fixed_x)
                    print(
                        f"Column at x={self._fixed_x:.2f} re‑selected; add more points."
                    )

        elif self.mode == "delete":
            if idx is not None:
                self._remove_point(idx)

        elif self.mode == "move":
            if idx is not None:
                self._selected_idx = idx  # start dragging

        elif self.mode == "tag":
            if idx is not None:
                self._tag_point(idx)

        elif self.mode == "inspect":
            if idx is not None:
                self._inspect_point(idx)

    # -- Mouse motion ---------------------------------------------------
    def _on_motion(self, event):
        # Show guide line only *before* first click in a column
        if (
            self.axis_lock
            and self.mode == "add"
            and self._fixed_x is None
            and event.inaxes is self.ax
            and event.xdata is not None
        ):
            self._update_vline(float(event.xdata))

        # Drag‑to‑move logic
        if self.mode != "move" or self._selected_idx is None:
            return
        if event.inaxes is not self.ax or event.ydata is None:
            return

        idx = self._selected_idx
        # In axis‑lock: allow vertical motion only (keep x constant)
        if self.axis_lock:
            x = self.points[idx, 0]
            y = float(event.ydata)
        else:
            if event.xdata is None:
                return
            x, y = float(event.xdata), float(event.ydata)

        # Update data + marker
        self.points[idx, :] = [x, y]
        m = self._markers[idx]
        m.set_data([x], [y])
        self.ax.figure.canvas.draw_idle()

    # -- Mouse release --------------------------------------------------
    def _on_release(self, _event):
        self._selected_idx = None  # stop dragging

    # -- Key presses ----------------------------------------------------
    def _on_key(self, event):
        # Track previous mode to close tagging UI when switching away from tag mode
        old_mode = self.mode

        if event.key in {"a", "escape"}:
            self.mode = "add"
        elif event.key == "m":
            self.mode = "move"
        elif event.key == "d":
            self.mode = "delete"
        elif event.key == "t":
            self.mode = "tag"
        elif event.key == "i":
            self.mode = "inspect"
        elif event.key == "n" and self.axis_lock:
            # Finish current column
            self._fixed_x = None
            if self._vline is not None:
                self._vline.remove()
                self._vline = None

        # Close tagging UI when leaving tag mode
        if old_mode == "tag" and self.mode != "tag":
            if self._tag_ui is not None:
                self._tag_ui.close()

        self._update_title()

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _update_title(self):
        self.ax.set_title(
            f"Mode: {self.mode}   |   keys → a:Add  m:Move  d:Delete  t:Tag  i:Inspect   (Esc returns to Add)",
            fontsize=9,
        )
        self.ax.figure.canvas.draw_idle()

    # -- Guide line helper ------------------------------------------------
    def _update_vline(self, x: float):
        """Draw or update the vertical guide line (axis‑lock mode)."""
        if self._vline is None:
            self._vline = self.ax.axvline(x, linestyle=":", color="k", alpha=0.5)
        else:
            self._vline.set_xdata([x, x])
        self.ax.figure.canvas.draw_idle()

    def _add_point(self, x: float, y: float):
        # Axis‑lock: lock x at first click in a column
        if self.axis_lock:
            if self._fixed_x is None:
                self._fixed_x = x
                self._update_vline(self._fixed_x)
            x = self._fixed_x

        # Append point
        self.points = np.vstack([self.points, [x, y]])
        # Newly added points start without label or sigma; tagging is explicit
        self.labels.append(None)
        self.sigmas.append(None)

        (marker,) = self.ax.plot(x, y, "ro", markersize=6)
        self._markers.append(marker)
        self._set_marker_color(len(self._markers) - 1)
        self.ax.figure.canvas.draw_idle()
        print(f"Added → ({x:.2f}, {y:.2f})")

    def _remove_point(self, idx: int):
        # Remove point from array and marker
        self.points = np.delete(self.points, idx, axis=0)
        self._markers[idx].remove()
        self._markers.pop(idx)
        self.labels.pop(idx)
        self.sigmas.pop(idx)
        self.ax.figure.canvas.draw_idle()
        print(f"Deleted point #{idx}")

    def _tag_point(self, idx: int):
        if idx < 0 or idx >= len(self.points):
            return
        self._tag_ui.show(idx)

    def _index_near(self, event) -> Optional[int]:
        # Return None if no points are stored
        if self.points.shape[0] == 0:
            return None
        # Transform all points to display coordinates
        display = self.ax.transData.transform(self.points)
        # Compute distances to mouse event
        d = np.hypot(display[:, 0] - event.x, display[:, 1] - event.y)
        idx = int(np.argmin(d))
        # Only return an index if within pick radius
        return idx if d[idx] <= self.pick_radius else None

    # ------------------------------------------------------------------
    # Public utilities
    # ------------------------------------------------------------------
    def disconnect(self):
        # Close any active tagging UI
        if self._tag_ui is not None:
            self._tag_ui.close()

        if self._vline is not None:
            self._vline.remove()
        canvas = self.ax.figure.canvas
        for cid in self._cids:
            canvas.mpl_disconnect(cid)

    @property
    def xy(self) -> np.ndarray:
        return self.points

    @property
    def x_coords(self) -> np.ndarray:
        return self.points[:, 0] if self.points.size else np.array([])

    @property
    def y_coords(self) -> np.ndarray:
        return self.points[:, 1] if self.points.size else np.array([])

    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns a tuple with (x_coords, y_coords)."""
        return self.x_coords, self.y_coords

    # ------------------------------------------------------------------
    # Tagging ---------------------------------------------------------
    # ------------------------------------------------------------------
    def set_current_label(self, label: Tuple[int, int] | None):
        """
        Define the (i, j) label attached to subsequently added points.
        Pass *None* to stop tagging.
        """
        self._current_label = label

    def set_current_sigma(self, sigma: float | None):
        """Define the sigma pre-filled when tagging new points. Use None to clear."""
        if sigma is not None and sigma <= 0:
            raise ValueError(f"sigma must be positive when provided, got {sigma}")
        self._current_sigma = sigma

    def summary(self):
        """Prints a summary of the selected points."""
        n = self.points.shape[0]
        print(f"Selected points: {n}")
        if n:
            print(f"X coords: {self.x_coords}")
            print(f"Y coords: {self.y_coords}")
        else:
            print("No points selected.")

    def view_points(self):
        """Show a table of points with their labels and coordinates."""
        n = self.points.shape[0]
        if n == 0:
            print("No points selected.")
            return
        # Header
        print("Idx |   i |   j |       x |       y |  sigma")
        print("--- | --- | --- | -------- | -------- | ------")
        # Rows
        for idx, (x, y) in enumerate(self.points):
            lab = self.labels[idx]
            sigma_val = self.sigmas[idx]
            i = str(lab[0]) if lab is not None else ""
            j = str(lab[1]) if lab is not None else ""
            sigma_str = f"{sigma_val:6.3f}" if sigma_val is not None else "      "
            print(f"{idx:3d} | {i:3s} | {j:3s} | {x:8.4f} | {y:8.4f} | {sigma_str}")

    def save_to_csv(self, filename: str = "picked_points.csv"):
        """Save points to CSV.

        • Standard mode → columns: x, y
        • Axis‑lock mode  → columns: i, j, x, y (labels optional, rows sorted)
        """
        if self.points.shape[0] == 0:
            print("No points to save.")
            return

        if not self.axis_lock:
            # Legacy behaviour
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "sigma"])
                for (x, y), sigma in zip(self.points, self.sigmas):
                    writer.writerow(
                        [
                            float(x),
                            float(y),
                            "" if sigma is None else float(sigma),
                        ]
                    )
            print(f"Saved {self.points.shape[0]} points to {filename}")
            return

        # Axis‑lock: include labels & order
        records: list[tuple[int | str, int | str, float, float]] = []
        for (x, y), lab, sigma in zip(self.points, self.labels, self.sigmas):
            sigma_entry: float | str = "" if sigma is None else float(sigma)
            if lab is None:
                records.append(("", "", float(x), float(y), sigma_entry))
            else:
                records.append((lab[0], lab[1], float(x), float(y), sigma_entry))
        # Sort by label (ints first; unlabeled last) then by y
        records.sort(
            key=lambda r: (
                r[0] if isinstance(r[0], int) else float("inf"),
                r[1] if isinstance(r[1], int) else float("inf"),
                r[3],
            )
        )

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["i", "j", "x", "y", "sigma"])
            writer.writerows(records)
        print(f"Saved {len(records)} points to {filename}")

    @staticmethod
    def load_csv_to_dict(
        filename: str,
        *,
        include_sigma: bool = False,
        x_scale: float = 1.0,
    ) -> PointDict:
        """Load points saved via :meth:`save_to_csv`, with optional x scaling.

        Parameters
        ----------
        filename : str
            CSV produced by :meth:`save_to_csv` in axis-lock mode.
        include_sigma : bool, default ``False``
            When ``True``, include the stored uncertainty per point.
        x_scale : float, default ``1.0``
            Multiply each x-coordinate by this factor while loading.
        """
        if x_scale <= 0:
            raise ValueError("x_scale must be positive.")
        data: dict[tuple[int, int], list] = {}
        with open(filename, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            has_sigma = header == ["i", "j", "x", "y", "sigma"]
            if not has_sigma and header != ["i", "j", "x", "y"]:
                raise ValueError(
                    f"Expected header ['i','j','x','y'] or ['i','j','x','y','sigma'], got {header}"
                )
            for row in reader:
                if has_sigma:
                    i_str, j_str, x_str, y_str, sigma_str = row
                else:
                    i_str, j_str, x_str, y_str = row
                    sigma_str = ""
                i = int(i_str)
                j = int(j_str)
                x = float(x_str) * x_scale
                y = float(y_str)
                sigma_val = float(sigma_str) if sigma_str.strip() else None
                key = (i, j)
                if include_sigma:
                    data.setdefault(key, []).append((x, y, sigma_val))
                else:
                    data.setdefault(key, []).append((x, y))
        return data

    def to_transition_data(
        self,
        *,
        include_sigma: bool = False,
        x_scale: float = 1.0,
    ) -> PointDict:
        """Return picked points grouped by transition, scaling x if needed.

        Parameters
        ----------
        include_sigma : bool, default ``False``
            Include the stored uncertainty alongside each point.
        x_scale : float, default ``1.0``
            Multiply each x-coordinate before packaging the data.
        """
        if x_scale <= 0:
            raise ValueError("x_scale must be positive.")
        data: PointDict = {}
        for (x, y), label, sigma in zip(self.points, self.labels, self.sigmas):
            if label is None:
                continue
            scaled_x = float(x) * x_scale
            entry: Union[Tuple[float, float], Tuple[float, float, Optional[float]]]
            if include_sigma:
                entry = (scaled_x, float(y), sigma if sigma is not None else None)
            else:
                entry = (scaled_x, float(y))
            data.setdefault((int(label[0]), int(label[1])), []).append(entry)

        return data

    def save_project(self, filename: str = "pointpicker_project.npz"):
        """
        Save the current points, labels, and optionally a snapshot of the figure to a compressed .npz file.

        Args:
            filename (str): Output filename (.npz).
        """

        np.savez(
            filename,
            points=self.points,
            labels=np.array(self.labels, dtype=object),
            sigmas=np.array(self.sigmas, dtype=object),
            axis_lock=self.axis_lock,
        )
        print(f"Project saved to {filename}")

    @classmethod
    def load_project(cls, ax: plt.Axes, filename: str = "pointpicker_project.npz"):
        """
        Load a project file saved by save_project. Restores points, labels, axis_lock, and (if present) shows a snapshot image.
        """
        data = np.load(filename, allow_pickle=True)
        picker = cls(ax, axis_lock=bool(data["axis_lock"]))
        picker.points = np.array(data["points"])
        picker.labels = list(data["labels"])
        if "sigmas" in data.files:
            picker.sigmas = []
            for val in data["sigmas"]:
                if val is None:
                    picker.sigmas.append(None)
                    continue
                if isinstance(val, str):
                    val = val.strip()
                    if val == "":
                        picker.sigmas.append(None)
                        continue
                try:
                    picker.sigmas.append(float(val))
                except Exception:
                    picker.sigmas.append(None)
        else:
            picker.sigmas = [None] * len(picker.points)
        for idx, (x, y) in enumerate(picker.points):
            (marker,) = ax.plot(x, y, "o", markersize=6)
            picker._markers.append(marker)
            picker._set_marker_color(idx)
        if "figure" in data.files and data["figure"].size > 0:
            buf = io.BytesIO(data["figure"].tobytes())
            img = mpimg.imread(buf, format="png")
            ax.imshow(img)
        ax.figure.canvas.draw_idle()
        return picker

    def to_dict(
        self,
        *,
        include_sigma: bool = True,
        x_scale: float = 1.0,
    ) -> dict[
        tuple[int, int],
        list[tuple[float, float] | tuple[float, float, Optional[float]]],
    ]:
        """
        Return a dict mapping (i, j) labels to lists of (x, y[, sigma]) coordinates
        data: dict[tuple[int, int], list[tuple[float, float] | tuple[float, float, Optional[float]]]] = {}
        Set include_sigma=True to retrieve sigma values alongside coordinates.
        """
        if x_scale <= 0:
            raise ValueError("x_scale must be positive.")
        data: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for (x, y), lab, sigma in zip(self.points, self.labels, self.sigmas):
            if lab is None:
                continue
            if isinstance(lab, np.ndarray):
                key = tuple(int(val) for val in lab)
            else:
                key = lab
            x_val = float(x) * x_scale
            if include_sigma:
                sigma_val = float(sigma) if sigma is not None else None
                data.setdefault(key, []).append((x_val, float(y), sigma_val))
            else:
                data.setdefault(key, []).append((x_val, float(y)))
        return data


# ----------------------------------------------------------------------
# Demo usage -----------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = rng.random((200, 300))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(img, cmap="gray", origin="lower")

    picker = PointPicker(ax)
    # axis‑lock example
    # picker = PointPicker(ax, axis_lock=True)

    plt.show()  # interact with the a/m/d/t keys and left clicks

    picker.disconnect()

    print("\nPicked points (x, y):\n", picker.xy)

    # Example of saving the points to a custom file name
    if picker.xy.any():
        picker.save_to_csv("my_picked_points.csv")

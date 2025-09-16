import csv
import io
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox

# Type alias for load_csv_to_dict return type
PointDict = Dict[
    Tuple[int, int],
    List[Union[Tuple[float, float], Tuple[float, float, Optional[float]]]],
]

# Optional: interactive label input for Jupyter widgets
try:
    import ipywidgets as _ipyw
    from IPython.display import clear_output as _ipy_clear
    from IPython.display import display as _ipy_display

    _IPYW_AVAILABLE = True
except Exception:
    _IPYW_AVAILABLE = False

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

        # Widget configuration
        self._use_widgets = bool(use_widgets and _IPYW_AVAILABLE)
        self._widget_box = None  # ipywidgets container
        self._pending_tag_idx: Optional[int] = None  # idx awaiting widget confirm
        self._mpl_box_ax = self._mpl_button_ax = None
        self._mpl_i_ax = self._mpl_j_ax = None  # Separate axes for i and j textboxes
        self._mpl_sigma_ax = None  # Axis for sigma textbox
        self._mpl_remove_ax = None  # Axis for remove tag button
        self._mpl_textbox = self._mpl_button = None
        self._mpl_i_textbox = self._mpl_j_textbox = None  # Separate textbox references
        self._mpl_sigma_textbox = None  # Sigma textbox reference
        self._mpl_remove_button = None  # Remove tag button reference
        self._input_in_progress = False  # Flag to prevent multiple input() calls

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
    # Widget helper for tagging
    # ------------------------------------------------------------------
    def _build_tag_widget(self):
        """Build (or show) a little widget panel to input i,j tags."""
        if not self._use_widgets:
            return
        if self._widget_box is None:
            i_field = _ipyw.IntText(
                description="i:", layout=_ipyw.Layout(width="140px")
            )
            j_field = _ipyw.IntText(
                description="j:", layout=_ipyw.Layout(width="140px")
            )
            sigma_field = _ipyw.Text(
                description="sigma:", layout=_ipyw.Layout(width="160px")
            )
            # Prefill fields: if the clicked point already has a label/sigma, show it;
            # otherwise fall back to the last‑used values.
            initial_label = None
            if self._pending_tag_idx is not None:
                lab = self.labels[self._pending_tag_idx]
                if lab is not None:
                    initial_label = lab
            if initial_label is None:
                initial_label = self._current_label
            if initial_label is not None:
                i_field.value, j_field.value = initial_label

            initial_sigma: Optional[float] = None
            if self._pending_tag_idx is not None:
                sigma_val = self.sigmas[self._pending_tag_idx]
                if sigma_val is not None:
                    initial_sigma = sigma_val
            if initial_sigma is None:
                initial_sigma = self._current_sigma
            if initial_sigma is not None:
                sigma_field.value = f"{initial_sigma:g}"
            else:
                sigma_field.value = ""

            btn = _ipyw.Button(description="OK", layout=_ipyw.Layout(width="80px"))
            remove_btn = _ipyw.Button(
                description="Remove Tag", layout=_ipyw.Layout(width="120px")
            )
            lbl_out = _ipyw.Output(layout=_ipyw.Layout(font_size="12px"))
            box = _ipyw.VBox(
                [_ipyw.HBox([i_field, j_field, sigma_field, btn, remove_btn]), lbl_out]
            )
            box.layout = _ipyw.Layout(width="640px")
            self._widget_box = (
                box,
                i_field,
                j_field,
                sigma_field,
                btn,
                remove_btn,
                lbl_out,
            )

            def _on_ok(_b):
                if self._pending_tag_idx is None:
                    with lbl_out:
                        _ipy_clear()
                        print("No point selected.")
                    return
                idx = self._pending_tag_idx
                try:
                    i = int(i_field.value)
                    j = int(j_field.value)
                    sigma_raw = str(sigma_field.value).strip()
                    if sigma_raw:
                        sigma_val = float(sigma_raw)
                    else:
                        sigma_val = None
                except Exception:
                    with lbl_out:
                        _ipy_clear()
                        print("Valores inválidos.")
                    return
                if sigma_val is not None and sigma_val <= 0:
                    with lbl_out:
                        _ipy_clear()
                        print("sigma debe ser > 0 o dejar vacío.")
                    return
                self.labels[idx] = (i, j)
                self._current_label = (i, j)
                self.sigmas[idx] = sigma_val
                self._current_sigma = sigma_val
                # update marker color depending on sigma availability
                self._set_marker_color(idx)
                self.ax.figure.canvas.draw_idle()
                self._pending_tag_idx = None
                with lbl_out:
                    _ipy_clear()
                    if sigma_val is None:
                        print(f"Punto #{idx} etiquetado con ({i},{j}) sin sigma")
                    else:
                        print(
                            f"Punto #{idx} etiquetado con ({i},{j}) y sigma={sigma_val}"
                        )
                # close widget after labeling
                box.close()
                self._widget_box = None

            def _on_remove(_b):
                if self._pending_tag_idx is None:
                    with lbl_out:
                        _ipy_clear()
                        print("No point selected.")
                    return
                idx = self._pending_tag_idx
                self.labels[idx] = None  # Remove the tag
                self.sigmas[idx] = None
                # update marker color to red for unlabeled point
                self._set_marker_color(idx)
                self.ax.figure.canvas.draw_idle()
                with lbl_out:
                    _ipy_clear()
                    print(f"Punto #{idx} tag removed ✓")
                # Clear the text boxes
                i_field.value = 0
                j_field.value = 0
                sigma_field.value = ""
                self._pending_tag_idx = None
                # close widget after removing tag
                box.close()
                self._widget_box = None

            btn.on_click(_on_ok)
            remove_btn.on_click(_on_remove)
            # Display widget under the current figure
            _ipy_display(box)

    # ------------------------------------------------------------------
    # Matplotlib widget helper (non-Jupyter)
    # ------------------------------------------------------------------
    def _build_mpl_tag_box(self, initial: Optional[Tuple[int, int]] = None):
        """
        Two separate TextBoxes (i: and j:) + Button under the figure for tagging in plain .py scripts.
        Similar to the Jupyter widget interface.
        """
        if self._mpl_box_ax is not None:  # already visible
            return

        fig = self.ax.figure
        # Create axes for i textbox, j textbox, OK button, and Remove Tag button
        self._mpl_i_ax = fig.add_axes([0.08, 0.02, 0.10, 0.05])
        self._mpl_j_ax = fig.add_axes([0.20, 0.02, 0.10, 0.05])
        self._mpl_sigma_ax = fig.add_axes([0.32, 0.02, 0.14, 0.05])
        self._mpl_button_ax = fig.add_axes([0.48, 0.02, 0.08, 0.05])
        self._mpl_remove_ax = fig.add_axes([0.58, 0.02, 0.14, 0.05])

        # Create the two text boxes
        i_txt = TextBox(self._mpl_i_ax, "i:", initial="")
        j_txt = TextBox(self._mpl_j_ax, "j:", initial="")
        sigma_txt = TextBox(self._mpl_sigma_ax, "sigma:", initial="")

        # Pre-fill with initial values if provided
        if initial is not None:
            i_txt.set_val(str(initial[0]))
            j_txt.set_val(str(initial[1]))

        sigma_initial: Optional[float] = None
        if self._pending_tag_idx is not None:
            sigma_initial = self.sigmas[self._pending_tag_idx]
        if sigma_initial is None:
            sigma_initial = self._current_sigma
        if sigma_initial is not None:
            sigma_txt.set_val(f"{sigma_initial:g}")
        else:
            sigma_txt.set_val("")

        btn = Button(self._mpl_button_ax, "OK")
        remove_btn = Button(self._mpl_remove_ax, "Remove Tag")

        def _apply(_=None):
            if self._pending_tag_idx is None:
                return
            try:
                i = int(i_txt.text.strip())
                j = int(j_txt.text.strip())
            except Exception:
                print("Invalid format. Please enter integers in both i and j fields.")
                return
            sigma_raw = sigma_txt.text.strip()
            sigma_val: Optional[float]
            if sigma_raw:
                try:
                    sigma_val = float(sigma_raw)
                except Exception:
                    print("Invalid sigma. Please enter a numeric value or leave blank.")
                    return
                if sigma_val <= 0:
                    print("Invalid sigma. Enter a positive value or leave blank.")
                    return
            else:
                sigma_val = None
            idx = self._pending_tag_idx
            self.labels[idx] = (i, j)
            self._current_label = (i, j)
            self.sigmas[idx] = sigma_val
            self._current_sigma = sigma_val
            self._set_marker_color(idx)
            self.ax.figure.canvas.draw_idle()
            if sigma_val is None:
                print(f"Point #{idx} labeled as ({i}, {j}) without sigma ✓")
            else:
                print(
                    f"Point #{idx} labeled as ({i}, {j}) with sigma={sigma_val:.4g} ✓"
                )

            # Don't close the widgets - keep them open for next tagging
            # Just reset the pending index
            self._pending_tag_idx = None

            # Redraw the figure
            fig.canvas.draw_idle()

        def _remove_tag(_=None):
            if self._pending_tag_idx is None:
                return
            idx = self._pending_tag_idx
            self.labels[idx] = None  # Remove the tag
            self.sigmas[idx] = None
            self._set_marker_color(idx)
            self.ax.figure.canvas.draw_idle()
            print(f"Point #{idx} tag removed ✓")

            # Clear the text boxes
            i_txt.set_val("")
            j_txt.set_val("")
            sigma_txt.set_val("")

            # Don't close the widgets - keep them open for next tagging
            # Just reset the pending index
            self._pending_tag_idx = None

            # Redraw the figure
            fig.canvas.draw_idle()

        # Connect events to both text boxes and buttons
        i_txt.on_submit(_apply)
        j_txt.on_submit(_apply)
        sigma_txt.on_submit(_apply)
        btn.on_clicked(_apply)
        remove_btn.on_clicked(_remove_tag)

        # Store references
        self._mpl_i_textbox = i_txt
        self._mpl_j_textbox = j_txt
        self._mpl_sigma_textbox = sigma_txt
        self._mpl_button = btn
        self._mpl_remove_button = remove_btn

        # For compatibility with cleanup methods, keep the first axis reference
        self._mpl_box_ax = self._mpl_i_ax

        fig.canvas.draw_idle()

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
    # Tag mode now uses matplotlib widgets exclusively for better user experience

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
        # Clean up matplotlib widgets when switching away from tag mode
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

        # Clean up matplotlib widgets when leaving tag mode
        if old_mode == "tag" and self.mode != "tag":
            self._cleanup_mpl_widgets()

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
        # Save label metadata
        self.labels.append(self._current_label)
        self.sigmas.append(self._current_sigma)

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
        """
        When pressing 't' over a point:
        - In Jupyter (%matplotlib widget), uses ipywidgets to pop up a dialog.
        - In a plain .py script, uses matplotlib TextBox + Button below the figure.
        """
        # Check if we're really in a Jupyter environment (not just a .py script)
        try:
            # This will only work in a real Jupyter/IPython environment
            from IPython import get_ipython

            ipython = get_ipython()
            is_jupyter = ipython is not None and hasattr(ipython, "kernel")
        except ImportError:
            is_jupyter = False

        # 1) Jupyter / ipywidgets path (only if truly in Jupyter)
        if self._use_widgets and is_jupyter:
            # Mark this index as awaiting confirmation
            self._pending_tag_idx = idx

            if self._widget_box is None:
                # First time → create and pre-fill the widget automatically
                self._build_tag_widget()
            else:
                # Widget already visible → just update the fields
                (box, i_field, j_field, sigma_field, btn, remove_btn, out) = (
                    self._widget_box
                )
                # Pick up the existing label/sigma if set, else the last-used ones
                lab = (
                    self.labels[idx]
                    if self.labels[idx] is not None
                    else self._current_label
                )
                if lab is not None:
                    i_field.value, j_field.value = lab
                sigma_val = (
                    self.sigmas[idx]
                    if self.sigmas[idx] is not None
                    else self._current_sigma
                )
                if sigma_val is not None:
                    sigma_field.value = f"{sigma_val:g}"
                else:
                    sigma_field.value = ""
                # Provide user feedback inside the widget
                with out:
                    _ipy_clear()
                    print(
                        "Point #{} selected. Modify i, j, sigma or press OK. Or remove tag.".format(
                            idx
                        )
                    )

            print(
                "Please enter i, j and optional sigma in the widget and press OK to tag point #{}, or Remove Tag to remove label".format(
                    idx
                )
            )
            return

        # 2) Script (.py) → Matplotlib TextBox + Button (non-blocking)
        self._pending_tag_idx = idx
        current_label = self.labels[idx]
        x, y = self.points[idx]

        # Pre-fill with this point's existing label or the last-used label
        initial = (
            self.labels[idx] if self.labels[idx] is not None else self._current_label
        )

        # Always clean up existing widgets first to prevent mouse grab conflicts
        if self._mpl_i_textbox is not None or self._mpl_j_textbox is not None:
            self._cleanup_mpl_widgets()

        # Create fresh widgets
        self._build_mpl_tag_box(initial)

        # Provide feedback about the selected point
        if current_label is not None:
            print(
                f"Point #{idx} at ({x:.2f}, {y:.2f}) currently labeled as ({current_label[0]}, {current_label[1]})"
            )
        else:
            print(f"Point #{idx} at ({x:.2f}, {y:.2f}) currently unlabeled")
        print(
            "Enter i, j and optional sigma in the text boxes below the figure and press Enter or click OK."
        )

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
        # Clean up any pending matplotlib widgets first
        self._cleanup_mpl_widgets()

        if self._vline is not None:
            self._vline.remove()
        # Close widget if open (optional)
        if self._widget_box is not None and hasattr(self._widget_box[0], "close"):
            self._widget_box[0].close()
        canvas = self.ax.figure.canvas
        for cid in self._cids:
            canvas.mpl_disconnect(cid)

    def _cleanup_mpl_widgets(self):
        """Safe cleanup of matplotlib widgets to prevent callback errors."""
        if (
            self._mpl_box_ax is not None
            or self._mpl_button_ax is not None
            or self._mpl_i_ax is not None
            or self._mpl_j_ax is not None
            or self._mpl_remove_ax is not None
        ):
            fig = self.ax.figure

            # First, disconnect all widget events to prevent mouse grab conflicts
            try:
                if self._mpl_i_textbox is not None:
                    self._mpl_i_textbox.disconnect_events()
                if self._mpl_j_textbox is not None:
                    self._mpl_j_textbox.disconnect_events()
                if self._mpl_sigma_textbox is not None:
                    self._mpl_sigma_textbox.disconnect_events()
                if self._mpl_button is not None:
                    # Button widgets don't have disconnect_events, but we can clear callbacks
                    self._mpl_button.on_clicked(lambda x: None)
                if self._mpl_remove_button is not None:
                    # Clear remove button callbacks
                    self._mpl_remove_button.on_clicked(lambda x: None)
            except (AttributeError, Exception):
                pass  # Some widgets might not have disconnect_events method

            # Then remove the axes
            try:
                if self._mpl_i_ax is not None:
                    fig.delaxes(self._mpl_i_ax)
                if self._mpl_j_ax is not None:
                    fig.delaxes(self._mpl_j_ax)
                if self._mpl_sigma_ax is not None:
                    fig.delaxes(self._mpl_sigma_ax)
                if self._mpl_button_ax is not None:
                    fig.delaxes(self._mpl_button_ax)
                if self._mpl_remove_ax is not None:
                    fig.delaxes(self._mpl_remove_ax)
                # Legacy cleanup for old single textbox (if present)
                if self._mpl_box_ax is not None and self._mpl_box_ax != self._mpl_i_ax:
                    fig.delaxes(self._mpl_box_ax)
            except Exception:
                # Fallback cleanup
                try:
                    if self._mpl_i_ax is not None:
                        self._mpl_i_ax.remove()
                    if self._mpl_j_ax is not None:
                        self._mpl_j_ax.remove()
                    if self._mpl_sigma_ax is not None:
                        self._mpl_sigma_ax.remove()
                    if self._mpl_button_ax is not None:
                        self._mpl_button_ax.remove()
                    if self._mpl_remove_ax is not None:
                        self._mpl_remove_ax.remove()
                    if (
                        self._mpl_box_ax is not None
                        and self._mpl_box_ax != self._mpl_i_ax
                    ):
                        self._mpl_box_ax.remove()
                except Exception:
                    pass  # Ignore cleanup errors

            # Reset all references
            self._mpl_box_ax = self._mpl_button_ax = None
            self._mpl_i_ax = self._mpl_j_ax = None
            self._mpl_sigma_ax = None
            self._mpl_remove_ax = None
            self._mpl_textbox = self._mpl_button = None
            self._mpl_i_textbox = self._mpl_j_textbox = None
            self._mpl_sigma_textbox = None
            self._mpl_remove_button = None
            self._pending_tag_idx = None

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
    ) -> PointDict:
        """
        Load points from a CSV file saved by PointPicker.save_to_csv (axis_lock mode).
        data: dict[tuple[int, int], list[tuple[float, float] | tuple[float, float, Optional[float]]]] = {}
        Set include_sigma=True to obtain sigma values when available.
        """
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
                x = float(x_str)
                y = float(y_str)
                sigma_val = float(sigma_str) if sigma_str.strip() else None
                key = (i, j)
                if include_sigma:
                    data.setdefault(key, []).append((x, y, sigma_val))
                else:
                    data.setdefault(key, []).append((x, y))
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
        self, *, include_sigma: bool = True
    ) -> dict[
        tuple[int, int],
        list[tuple[float, float] | tuple[float, float, Optional[float]]],
    ]:
        """
        Return a dict mapping (i, j) labels to lists of (x, y[, sigma]) coordinates
        data: dict[tuple[int, int], list[tuple[float, float] | tuple[float, float, Optional[float]]]] = {}
        Set include_sigma=True to retrieve sigma values alongside coordinates.
        """
        data: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for (x, y), lab, sigma in zip(self.points, self.labels, self.sigmas):
            if lab is None:
                continue
            if isinstance(lab, np.ndarray):
                key = tuple(int(val) for val in lab)
            else:
                key = lab
            if include_sigma:
                sigma_val = float(sigma) if sigma is not None else None
                data.setdefault(key, []).append((float(x), float(y), sigma_val))
            else:
                data.setdefault(key, []).append((float(x), float(y)))
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

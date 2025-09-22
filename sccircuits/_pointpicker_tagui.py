from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from matplotlib.widgets import Button, TextBox

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .pointpicker import PointPicker


class BaseTagUI:
    """Interface for tag input backends."""

    def __init__(self, picker: "PointPicker") -> None:
        self.picker = picker

    def is_available(self) -> bool:
        return True

    def show(self, idx: int) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:
        """Release UI resources if any."""
        # Default implementation just forgets about pending state.
        return


def _running_inside_jupyter() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        return False
    shell = get_ipython()
    return bool(shell and hasattr(shell, "kernel"))


# ---------------------------------------------------------------------------
# Jupyter ipywidgets implementation
# ---------------------------------------------------------------------------
@dataclass
class _JupyterWidgetState:
    box: object
    i_field: object
    j_field: object
    sigma_field: object
    ok_button: object
    remove_button: object
    output: object


class JupyterTagUI(BaseTagUI):
    """ipywidgets-based tagging UI for notebook environments."""

    def __init__(self, picker: "PointPicker") -> None:
        super().__init__(picker)
        self._widget: Optional[_JupyterWidgetState] = None
        self._pending_idx: Optional[int] = None
        self._available = False
        self._ipyw = None
        self._display = None
        self._clear_output = None

        if not _running_inside_jupyter():
            return
        try:
            import ipywidgets as ipyw  # type: ignore
            from IPython.display import clear_output, display  # type: ignore
        except Exception:
            return

        self._ipyw = ipyw
        self._display = display
        self._clear_output = clear_output
        self._available = True

    # Public API ---------------------------------------------------------
    def is_available(self) -> bool:
        return bool(self._available)

    def show(self, idx: int) -> None:
        if not self.is_available():
            raise RuntimeError("ipywidgets backend is unavailable")
        self._pending_idx = idx
        if self._widget is None:
            self._build_widget()
            assert self._widget is not None
            self._display(self._widget.box)  # type: ignore[arg-type]
        self._populate_defaults(idx)
        print(
            "Please enter i, j and optional sigma in the widget and press OK to tag point #{}, "
            "or Remove Tag to remove label".format(idx)
        )

    def close(self) -> None:
        if self._widget is None:
            self._pending_idx = None
            return
        if hasattr(self._widget.box, "close"):
            self._widget.box.close()  # type: ignore[attr-defined]
        self._widget = None
        self._pending_idx = None

    # Internal helpers ---------------------------------------------------
    def _build_widget(self) -> None:
        assert self._ipyw is not None and self._display is not None
        ipyw = self._ipyw
        i_field = ipyw.IntText(description="i:", layout=ipyw.Layout(width="140px"))
        j_field = ipyw.IntText(description="j:", layout=ipyw.Layout(width="140px"))
        sigma_field = ipyw.Text(description="sigma:", layout=ipyw.Layout(width="160px"))
        ok_button = ipyw.Button(description="OK", layout=ipyw.Layout(width="80px"))
        remove_button = ipyw.Button(
            description="Remove Tag", layout=ipyw.Layout(width="120px")
        )
        output = ipyw.Output(layout=ipyw.Layout(font_size="12px"))
        box = ipyw.VBox(
            [ipyw.HBox([i_field, j_field, sigma_field, ok_button, remove_button]), output]
        )
        box.layout = ipyw.Layout(width="640px")

        self._widget = _JupyterWidgetState(
            box=box,
            i_field=i_field,
            j_field=j_field,
            sigma_field=sigma_field,
            ok_button=ok_button,
            remove_button=remove_button,
            output=output,
        )
        ok_button.on_click(self._handle_submit)
        remove_button.on_click(self._handle_remove)

    def _populate_defaults(self, idx: int) -> None:
        assert self._widget is not None
        lab, sigma = self.picker._tag_defaults_for(idx)
        if lab is not None:
            self._widget.i_field.value = int(lab[0])
            self._widget.j_field.value = int(lab[1])
        elif self.picker._current_label is not None:
            i_def, j_def = self.picker._current_label
            self._widget.i_field.value = int(i_def)
            self._widget.j_field.value = int(j_def)
        else:
            self._widget.i_field.value = 0
            self._widget.j_field.value = 0
        if sigma is not None:
            self._widget.sigma_field.value = f"{float(sigma):g}"
        elif self.picker._current_sigma is not None:
            self._widget.sigma_field.value = f"{float(self.picker._current_sigma):g}"
        else:
            self._widget.sigma_field.value = ""
        self._write_feedback(
            f"Point #{idx} selected. Modify i, j, sigma or press OK. Or remove tag."
        )

    def _handle_submit(self, _button=None) -> None:
        if self._pending_idx is None:
            self._write_feedback("No point selected.")
            return
        assert self._widget is not None
        idx = self._pending_idx
        try:
            i = int(self._widget.i_field.value)
            j = int(self._widget.j_field.value)
        except Exception:
            self._write_feedback("Valores inválidos.")
            return
        sigma_raw = str(self._widget.sigma_field.value).strip()
        sigma_val: Optional[float]
        if sigma_raw:
            try:
                sigma_val = float(sigma_raw)
            except Exception:
                self._write_feedback("Valores inválidos.")
                return
            if sigma_val <= 0:
                self._write_feedback("sigma debe ser > 0 o dejar vacío.")
                return
        else:
            sigma_val = None
        try:
            self.picker._apply_tag(idx, i, j, sigma_val)
        except ValueError as exc:
            self._write_feedback(str(exc))
            return
        msg = (
            f"Punto #{idx} etiquetado con ({i},{j}) sin sigma"
            if sigma_val is None
            else f"Punto #{idx} etiquetado con ({i},{j}) y sigma={sigma_val}"
        )
        self._write_feedback(msg)
        self._pending_idx = None
        self.close()

    def _handle_remove(self, _button=None) -> None:
        if self._pending_idx is None:
            self._write_feedback("No point selected.")
            return
        idx = self._pending_idx
        self.picker._remove_tag(idx)
        self._write_feedback(f"Punto #{idx} tag removed ✓")
        assert self._widget is not None
        self._widget.i_field.value = 0
        self._widget.j_field.value = 0
        self._widget.sigma_field.value = ""
        self._pending_idx = None
        self.close()

    def _write_feedback(self, message: str) -> None:
        if self._widget is None or self._clear_output is None:
            print(message)
            return
        with self._widget.output:
            self._clear_output()
            print(message)


# ---------------------------------------------------------------------------
# Matplotlib widgets implementation
# ---------------------------------------------------------------------------
@dataclass
class _MatplotlibWidgetBundle:
    i_ax: object
    j_ax: object
    sigma_ax: object
    ok_ax: object
    remove_ax: object
    i_box: TextBox
    j_box: TextBox
    sigma_box: TextBox
    ok_button: Button
    remove_button: Button
    ok_click_cid: int
    remove_click_cid: int


class MatplotlibTagUI(BaseTagUI):
    """Matplotlib TextBox/Button tagging UI for script environments."""

    def __init__(self, picker: "PointPicker") -> None:
        super().__init__(picker)
        self._widgets: Optional[_MatplotlibWidgetBundle] = None
        self._pending_idx: Optional[int] = None
        self._updating_fields = False

    def show(self, idx: int) -> None:
        self._pending_idx = idx
        if self._widgets is None:
            self._widgets = self._build_widgets()
        self._populate_defaults(idx)
        current_label = self.picker.labels[idx]
        x, y = self.picker.points[idx]
        if current_label is not None:
            print(
                f"Point #{idx} at ({x:.2f}, {y:.2f}) currently labeled as ({current_label[0]}, {current_label[1]})"
            )
        else:
            print(f"Point #{idx} at ({x:.2f}, {y:.2f}) currently unlabeled")
        print(
            "Enter i, j and optional sigma in the text boxes below the figure and press Enter or click OK."
        )
        self.picker.ax.figure.canvas.draw_idle()

    def close(self) -> None:
        if self._widgets is None:
            self._pending_idx = None
            return
        fig = self.picker.ax.figure
        widgets = self._widgets
        for box in (widgets.i_box, widgets.j_box, widgets.sigma_box):
            try:
                box.disconnect_events()
            except Exception:
                pass
        for button, cid in (
            (widgets.ok_button, widgets.ok_click_cid),
            (widgets.remove_button, widgets.remove_click_cid),
        ):
            try:
                button.disconnect(cid)
            except Exception:
                pass
        for ax in (widgets.i_ax, widgets.j_ax, widgets.sigma_ax, widgets.ok_ax, widgets.remove_ax):
            try:
                fig.delaxes(ax)
            except Exception:
                try:
                    ax.remove()  # type: ignore[attr-defined]
                except Exception:
                    pass
        self._widgets = None
        self._pending_idx = None
        self.picker.ax.figure.canvas.draw_idle()

    # Internal helpers ---------------------------------------------------
    def _build_widgets(self) -> _MatplotlibWidgetBundle:
        fig = self.picker.ax.figure
        # Ensure there is enough room for the extra controls below the axes
        desired_bottom = 0.15
        if fig.subplotpars.bottom < desired_bottom:
            fig.subplots_adjust(bottom=desired_bottom)

        bottom = 0.02
        height = 0.05
        i_ax = fig.add_axes([0.06, bottom, 0.11, height])
        j_ax = fig.add_axes([0.19, bottom, 0.11, height])
        sigma_ax = fig.add_axes([0.34, bottom, 0.18, height])
        ok_ax = fig.add_axes([0.55, bottom, 0.10, height])
        remove_ax = fig.add_axes([0.68, bottom, 0.20, height])

        i_box = TextBox(i_ax, "i:", initial="")
        j_box = TextBox(j_ax, "j:", initial="")
        sigma_box = TextBox(sigma_ax, "sigma:", initial="")
        ok_button = Button(ok_ax, "OK")
        remove_button = Button(remove_ax, "Remove Tag")

        ok_click_cid = ok_button.on_clicked(self._handle_submit)
        remove_click_cid = remove_button.on_clicked(self._handle_remove)

        return _MatplotlibWidgetBundle(
            i_ax=i_ax,
            j_ax=j_ax,
            sigma_ax=sigma_ax,
            ok_ax=ok_ax,
            remove_ax=remove_ax,
            i_box=i_box,
            j_box=j_box,
            sigma_box=sigma_box,
            ok_button=ok_button,
            remove_button=remove_button,
            ok_click_cid=ok_click_cid,
            remove_click_cid=remove_click_cid,
        )

    def _populate_defaults(self, idx: int) -> None:
        assert self._widgets is not None
        lab, sigma = self.picker._tag_defaults_for(idx)
        self._updating_fields = True
        try:
            if lab is not None:
                self._widgets.i_box.set_val(str(int(lab[0])))
                self._widgets.j_box.set_val(str(int(lab[1])))
            elif self.picker._current_label is not None:
                i_def, j_def = self.picker._current_label
                self._widgets.i_box.set_val(str(int(i_def)))
                self._widgets.j_box.set_val(str(int(j_def)))
            else:
                self._widgets.i_box.set_val("")
                self._widgets.j_box.set_val("")

            if sigma is not None:
                self._widgets.sigma_box.set_val(f"{float(sigma):g}")
            elif self.picker._current_sigma is not None:
                self._widgets.sigma_box.set_val(f"{float(self.picker._current_sigma):g}")
            else:
                self._widgets.sigma_box.set_val("")
        finally:
            self._updating_fields = False

    def _handle_submit(self, _event=None) -> None:
        if self._updating_fields:
            return
        if self._pending_idx is None or self._widgets is None:
            print("No point selected.")
            return
        try:
            i = int(self._widgets.i_box.text.strip())
            j = int(self._widgets.j_box.text.strip())
        except Exception:
            print("Invalid format. Please enter integers in both i and j fields.")
            return
        sigma_raw = self._widgets.sigma_box.text.strip()
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
        idx = self._pending_idx
        try:
            self.picker._apply_tag(idx, i, j, sigma_val)
        except ValueError as exc:
            print(str(exc))
            return
        if sigma_val is None:
            print(f"Point #{idx} labeled as ({i}, {j}) without sigma ✓")
        else:
            print(f"Point #{idx} labeled as ({i}, {j}) with sigma={sigma_val:.4g} ✓")
        self.close()

    def _handle_remove(self, _event=None) -> None:
        if self._pending_idx is None or self._widgets is None:
            print("No point selected.")
            return
        idx = self._pending_idx
        self.picker._remove_tag(idx)
        print(f"Point #{idx} tag removed ✓")
        self._updating_fields = True
        try:
            self._widgets.i_box.set_val("")
            self._widgets.j_box.set_val("")
            self._widgets.sigma_box.set_val("")
        finally:
            self._updating_fields = False
        self.close()


def create_tag_ui(picker: "PointPicker", prefer_widgets: bool) -> BaseTagUI:
    """Return the best available tag UI for the current environment."""
    if prefer_widgets:
        jupyter_ui = JupyterTagUI(picker)
        if jupyter_ui.is_available():
            return jupyter_ui
    return MatplotlibTagUI(picker)

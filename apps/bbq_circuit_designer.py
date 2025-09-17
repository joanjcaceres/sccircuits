import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import sympy as sp
from sympy.printing.pycode import pycode


@dataclass
class Node:
    identifier: int
    name: str
    x: float
    y: float
    circle_id: int
    label_id: int


@dataclass
class Edge:
    identifier: int
    nodes: Tuple[int, int]
    line_id: int
    center_circle_id: Optional[int]
    label_id: int
    capacitance_expr: Optional[sp.Expr]
    capacitance_text: Optional[str]
    inductance_expr: Optional[sp.Expr]
    inductance_text: Optional[str]
    l_inverse_expr: Optional[sp.Expr]
    is_ground: bool = False
    ground_marker_id: Optional[int] = None


@dataclass
class EdgeParameters:
    capacitance_expr: Optional[sp.Expr]
    capacitance_text: Optional[str]
    inductance_expr: Optional[sp.Expr]
    inductance_text: Optional[str]


class EdgeDialog:
    def __init__(
        self,
        parent: tk.Tk,
        first: str,
        second: str,
        default_cap: Optional[str] = None,
        default_ind: Optional[str] = None,
    ):
        self.parent = parent
        self.value: Optional[EdgeParameters] = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.title("Valores del enlace")
        ttk.Label(self.dialog, text=f"Entre {first} y {second}").grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5))

        ttk.Label(self.dialog, text="Capacitancia (F)").grid(row=1, column=0, sticky=tk.W, padx=10)
        self.cap_entry = ttk.Entry(self.dialog, width=18)
        self.cap_entry.grid(row=1, column=1, padx=10, pady=2)
        if default_cap:
            self.cap_entry.insert(0, default_cap)

        ttk.Label(self.dialog, text="Inductancia (H)").grid(row=2, column=0, sticky=tk.W, padx=10)
        self.ind_entry = ttk.Entry(self.dialog, width=18)
        self.ind_entry.grid(row=2, column=1, padx=10, pady=2)
        if default_ind:
            self.ind_entry.insert(0, default_ind)

        buttons = ttk.Frame(self.dialog)
        buttons.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(buttons, text="Cancelar", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(buttons, text="Aceptar", command=self._on_accept).pack(side=tk.RIGHT, padx=5)

        self.cap_entry.focus_set()
        self.dialog.bind("<Return>", lambda _: self._on_accept())
        self.dialog.bind("<Escape>", lambda _: self.dialog.destroy())
        self.dialog.wait_window()

    def _on_accept(self) -> None:
        try:
            cap_expr, cap_text = self._parse_expression(self.cap_entry.get())
            ind_expr, ind_text = self._parse_expression(self.ind_entry.get())
        except ValueError as exc:
            messagebox.showerror("Entrada invalida", str(exc), parent=self.dialog)
            return
        if ind_expr is not None:
            if ind_expr.is_zero is True:
                messagebox.showerror("Entrada invalida", "La inductancia no puede ser cero.", parent=self.dialog)
                return
            if ind_expr.is_number and float(ind_expr.evalf()) == 0.0:
                messagebox.showerror("Entrada invalida", "La inductancia no puede ser cero.", parent=self.dialog)
                return
        self.value = EdgeParameters(
            capacitance_expr=cap_expr,
            capacitance_text=cap_text,
            inductance_expr=ind_expr,
            inductance_text=ind_text,
        )
        self.dialog.destroy()

    @staticmethod
    def _parse_expression(text: str) -> Tuple[Optional[sp.Expr], Optional[str]]:
        stripped = text.strip()
        if not stripped:
            return None, None
        try:
            expr = sp.sympify(stripped)
        except (sp.SympifyError, TypeError) as exc:
            raise ValueError("Introduce un numero o expresion valida.") from exc
        if expr.is_real is False:
            raise ValueError("Solo se admiten valores reales.")
        expr = sp.simplify(expr)
        return expr, stripped


class CircuitGraphApp:
    NODE_RADIUS = 6
    EDGE_CENTER_RADIUS = 16
    GROUND_LINE_LENGTH = 26
    GROUND_TRIANGLE_WIDTH = 18
    GROUND_TRIANGLE_HEIGHT = 12

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("BBQ Matrix Builder")
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[int, Edge] = {}
        self.node_counter = 0
        self.edge_counter = 0
        self.mode: Optional[str] = None
        self.selected_node: Optional[int] = None
        self.dragging_node: Optional[int] = None
        self.drag_offset: Tuple[float, float] = (0.0, 0.0)
        self.ground_node_id: Optional[int] = None
        self.focus_node: Optional[int] = None
        self.status_var = tk.StringVar(value="Pulsa 'n' para crear nodos, 'c' para conectar.")

        self._build_ui()
        self._bind_shortcuts()
        self.root.after_idle(self._ensure_ground_node)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.root, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._handle_canvas_click)

        overlay = ttk.Frame(self.root, padding=8)
        overlay.place(relx=0.02, rely=0.02)

        status_label = ttk.Label(overlay, textvariable=self.status_var)
        status_label.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 6))

        ttk.Button(overlay, text="Modo nodo", command=lambda: self._set_mode("node")).grid(row=1, column=0, padx=2)
        ttk.Button(overlay, text="Modo enlace", command=lambda: self._set_mode("edge")).grid(row=1, column=1, padx=2)
        ttk.Button(overlay, text="A masa", command=lambda: self._set_mode("ground")).grid(row=1, column=2, padx=2)
        ttk.Button(overlay, text="Reiniciar", command=self._reset_all).grid(row=1, column=3, padx=2)

        ttk.Button(overlay, text="Copiar snippet", command=self._copy_snippet).grid(row=2, column=0, columnspan=4, sticky="ew", pady=(6, 0))

    def _bind_shortcuts(self) -> None:
        self.root.bind("n", lambda _: self._set_mode("node"))
        self.root.bind("c", lambda _: self._set_mode("edge"))
        self.root.bind("g", lambda _: self._set_mode("ground"))
        self.root.bind("<Escape>", lambda _: self._set_mode(None))
        self.root.bind("<Delete>", lambda _: self._delete_focused_node())
        self.root.bind("<BackSpace>", lambda _: self._delete_focused_node())

    def _handle_canvas_click(self, event: tk.Event) -> None:
        if self.mode != "node":
            return
        current = self.canvas.find_withtag("current")
        if current:
            tags = self.canvas.gettags(current[0])
            if "node" in tags:
                self._update_status("Haz click en el fondo para agregar un nodo nuevo.")
                return
        self._set_focus_node(None)
        name = self._generate_default_node_name()
        self._add_node(event.x, event.y, name)

    def _generate_default_node_name(self) -> str:
        existing_numbers = {
            int(node.name[1:])
            for node in self.nodes.values()
            if node.name.startswith("N") and node.name[1:].isdigit()
        }
        next_index = 1
        while next_index in existing_numbers:
            next_index += 1
        return f"N{next_index}"

    def _add_node(
        self,
        x: float,
        y: float,
        name: str,
        *,
        silent: bool = False,
        color: str = "#1976d2",
    ) -> int:
        node_id = self.node_counter
        self.node_counter += 1
        tag = f"node_{node_id}"
        circle = self.canvas.create_oval(
            x - self.NODE_RADIUS,
            y - self.NODE_RADIUS,
            x + self.NODE_RADIUS,
            y + self.NODE_RADIUS,
            fill=color,
            outline="black",
            width=2,
            tags=("node", tag),
        )
        label = self.canvas.create_text(
            x + self.NODE_RADIUS + 6,
            y,
            text=name,
            fill="#212121",
            anchor=tk.W,
            tags=("node", tag),
        )
        self.canvas.tag_bind(
            tag,
            "<ButtonPress-1>",
            lambda event, nid=node_id: self._handle_node_press(event, nid),
        )
        self.canvas.tag_bind(
            tag,
            "<B1-Motion>",
            lambda event, nid=node_id: self._handle_node_drag(event, nid),
        )
        self.canvas.tag_bind(
            tag,
            "<ButtonRelease-1>",
            lambda event, nid=node_id: self._handle_node_release(event, nid),
        )
        self.canvas.tag_bind(
            tag,
            "<Double-Button-1>",
            lambda event, nid=node_id: self._rename_node(nid),
        )

        self.nodes[node_id] = Node(node_id, name, x, y, circle, label)
        if not silent:
            self._update_status(f"Nodo {name} creado. Pulsa 'c' para conectar.")
        return node_id

    def _handle_node_press(self, event: tk.Event, node_id: int) -> None:
        if self.mode == "ground":
            self._connect_node_to_ground(node_id)
            return
        if self.mode == "edge":
            self._handle_edge_mode_click(node_id)
            return
        if self.mode != "edge":
            self._set_focus_node(node_id)
        self.dragging_node = node_id
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        node = self.nodes[node_id]
        self.drag_offset = (node.x - canvas_x, node.y - canvas_y)
        self.canvas.tag_raise(node.circle_id)
        self.canvas.tag_raise(node.label_id)
        self._update_status(f"Moviendo nodo {node.name}.")

    def _handle_edge_mode_click(self, node_id: int) -> None:
        if self.selected_node is None:
            self.selected_node = node_id
            self._highlight_node(node_id, True)
            self._update_status("Selecciona el segundo nodo para crear la conexion.")
            return
        if self.selected_node == node_id:
            self._update_status("Selecciona un nodo distinto.")
            return
        first = self.selected_node
        second = node_id
        self._highlight_node(first, False)
        self.selected_node = None
        self._create_edge(first, second)

    def _connect_node_to_ground(self, node_id: int) -> None:
        if self.ground_node_id is None:
            self._update_status("No se encontro el nodo de masa.")
            return
        if node_id == self.ground_node_id:
            self._update_status("Selecciona un nodo distinto de GND.")
            return
        existing = self._find_edge(node_id, self.ground_node_id)
        default_cap = ""
        default_ind = ""
        if existing is not None:
            default_cap = existing.capacitance_text or (
                str(existing.capacitance_expr) if existing.capacitance_expr is not None else ""
            )
            default_ind = existing.inductance_text or (
                str(existing.inductance_expr) if existing.inductance_expr is not None else ""
            )
        node_name = self.nodes[node_id].name
        dialog = EdgeDialog(self.root, node_name, "GND", default_cap or None, default_ind or None)
        if dialog.value is None:
            self._update_status("Conexion a masa cancelada.")
            return
        if existing is not None:
            self._apply_edge_parameters(existing, dialog.value)
            self._update_status(f"Conexion a masa de {node_name} actualizada.")
        else:
            self._create_ground_edge(node_id, dialog.value)
            self._update_status(f"Nodo {node_name} conectado a masa.")

    def _handle_node_drag(self, event: tk.Event, node_id: int) -> None:
        if self.dragging_node != node_id or self.mode == "edge":
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        offset_x, offset_y = self.drag_offset
        new_x = canvas_x + offset_x
        new_y = canvas_y + offset_y
        self._move_node(node_id, new_x, new_y)

    def _handle_node_release(self, event: tk.Event, node_id: int) -> None:
        if self.dragging_node != node_id:
            return
        node = self.nodes[node_id]
        self.dragging_node = None
        self._update_status(f"Nodo {node.name} reposicionado.")

    def _move_node(self, node_id: int, x: float, y: float) -> None:
        node = self.nodes[node_id]
        node.x = x
        node.y = y
        self.canvas.coords(
            node.circle_id,
            x - self.NODE_RADIUS,
            y - self.NODE_RADIUS,
            x + self.NODE_RADIUS,
            y + self.NODE_RADIUS,
        )
        self.canvas.coords(node.label_id, x + self.NODE_RADIUS + 6, y)
        for edge_id, edge in self.edges.items():
            if node_id in edge.nodes:
                self._update_edge_geometry(edge_id)

    def _update_edge_geometry(self, edge_id: int) -> None:
        edge = self.edges[edge_id]
        if edge.is_ground:
            if self.ground_node_id is None:
                return
            primary_id = edge.nodes[0] if edge.nodes[1] == self.ground_node_id else edge.nodes[1]
            node = self.nodes[primary_id]
            x = node.x
            y = node.y
            line_end_y = y + self.GROUND_LINE_LENGTH
            self.canvas.coords(edge.line_id, x, y, x, line_end_y)
            mid_y = y + self.GROUND_LINE_LENGTH / 2
            radius = self.EDGE_CENTER_RADIUS
            if edge.center_circle_id is not None:
                self.canvas.coords(
                    edge.center_circle_id,
                    x - radius,
                    mid_y - radius,
                    x + radius,
                    mid_y + radius,
                )
                self.canvas.tag_raise(edge.label_id, edge.center_circle_id)
            if edge.ground_marker_id is not None:
                triangle_points = [
                    x - self.GROUND_TRIANGLE_WIDTH / 2,
                    line_end_y,
                    x + self.GROUND_TRIANGLE_WIDTH / 2,
                    line_end_y,
                    x,
                    line_end_y + self.GROUND_TRIANGLE_HEIGHT,
                ]
                self.canvas.coords(edge.ground_marker_id, *triangle_points)
            self.canvas.coords(edge.label_id, x, mid_y)
            return

        node_a = self.nodes[edge.nodes[0]]
        node_b = self.nodes[edge.nodes[1]]
        self.canvas.coords(edge.line_id, node_a.x, node_a.y, node_b.x, node_b.y)
        center_x = (node_a.x + node_b.x) / 2
        center_y = (node_a.y + node_b.y) / 2
        radius = self.EDGE_CENTER_RADIUS
        if edge.center_circle_id is not None:
            self.canvas.coords(
                edge.center_circle_id,
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            )
            self.canvas.tag_raise(edge.label_id, edge.center_circle_id)
        self.canvas.coords(edge.label_id, center_x, center_y)

    def _edit_edge(self, edge_id: int) -> None:
        edge = self.edges[edge_id]
        first_name = self.nodes[edge.nodes[0]].name
        second_name = self.nodes[edge.nodes[1]].name
        default_cap = edge.capacitance_text or (
            str(edge.capacitance_expr) if edge.capacitance_expr is not None else ""
        )
        default_ind = edge.inductance_text or (
            str(edge.inductance_expr) if edge.inductance_expr is not None else ""
        )
        dialog = EdgeDialog(self.root, first_name, second_name, default_cap, default_ind)
        if dialog.value is None:
            self._update_status("Edicion de conexion cancelada.")
            return
        self._apply_edge_parameters(edge, dialog.value)
        self._update_status("Conexion actualizada.")

    def _delete_focused_node(self) -> None:
        if self.focus_node is None:
            self._update_status("Selecciona un nodo y presiona Supr para eliminarlo.")
            return
        if self.ground_node_id is not None and self.focus_node == self.ground_node_id:
            messagebox.showinfo("No permitido", "El nodo GND no puede eliminarse.", parent=self.root)
            return
        node_id = self.focus_node
        node = self.nodes.get(node_id)
        if node is None:
            return
        node_name = node.name
        self._remove_node(node_id)
        self._set_focus_node(None)
        self._update_status(f"Nodo {node_name} eliminado.")

    def _remove_node(self, node_id: int) -> None:
        node = self.nodes.pop(node_id, None)
        if node is None:
            return
        self.canvas.delete(node.circle_id)
        self.canvas.delete(node.label_id)
        edges_to_remove = [edge_id for edge_id, edge in self.edges.items() if node_id in edge.nodes]
        for edge_id in edges_to_remove:
            self._remove_edge(edge_id)

    def _remove_edge(self, edge_id: int) -> None:
        edge = self.edges.pop(edge_id, None)
        if edge is None:
            return
        self.canvas.delete(edge.line_id)
        if edge.center_circle_id is not None:
            self.canvas.delete(edge.center_circle_id)
        if edge.ground_marker_id is not None:
            self.canvas.delete(edge.ground_marker_id)
        self.canvas.delete(edge.label_id)

    def _apply_edge_parameters(self, edge: Edge, params: EdgeParameters) -> None:
        edge.capacitance_expr = params.capacitance_expr
        edge.capacitance_text = params.capacitance_text
        edge.inductance_expr = params.inductance_expr
        edge.inductance_text = params.inductance_text
        edge.l_inverse_expr = (
            sp.simplify(sp.Integer(1) / edge.inductance_expr)
            if edge.inductance_expr is not None
            else None
        )
        self.canvas.itemconfigure(
            edge.label_id,
            text=self._edge_label(
                edge.capacitance_expr,
                edge.capacitance_text,
                edge.inductance_expr,
                edge.inductance_text,
            ),
        )
        self._update_edge_geometry(edge.identifier)

    def _rename_node(self, node_id: int) -> None:
        if self.ground_node_id is not None and node_id == self.ground_node_id:
            messagebox.showinfo("No permitido", "El nodo GND no puede renombrarse.", parent=self.root)
            return
        node = self.nodes[node_id]
        new_name = simpledialog.askstring(
            "Renombrar nodo",
            "Nuevo nombre:",
            initialvalue=node.name,
            parent=self.root,
        )
        if not new_name:
            return
        new_name = new_name.strip()
        if not new_name or new_name == node.name:
            return
        node.name = new_name
        self.canvas.itemconfigure(node.label_id, text=new_name)
        self._update_status(f"Nodo renombrado a {new_name}.")

    def _create_edge(self, first: int, second: int) -> None:
        first_name = self.nodes[first].name
        second_name = self.nodes[second].name
        existing = self._find_edge(first, second)
        if existing is not None:
            if not messagebox.askyesno(
                "Enlace existente",
                "Ya existe una conexion entre estos nodos.\n¿Deseas crear otra en paralelo?",
                parent=self.root,
            ):
                self._update_status("Se mantuvo la conexion original.")
                return
        dialog = EdgeDialog(self.root, first_name, second_name)
        if dialog.value is None:
            self._update_status("Conexion cancelada.")
            return
        params = dialog.value
        capacitance_expr = params.capacitance_expr
        capacitance_text = params.capacitance_text
        inductance_expr = params.inductance_expr
        inductance_text = params.inductance_text
        l_inverse_expr: Optional[sp.Expr] = None
        if inductance_expr is not None:
            l_inverse_expr = sp.simplify(sp.Integer(1) / inductance_expr)
        edge_id = self.edge_counter
        self.edge_counter += 1
        tag = f"edge_{edge_id}"
        x1, y1 = self.nodes[first].x, self.nodes[first].y
        x2, y2 = self.nodes[second].x, self.nodes[second].y
        line = self.canvas.create_line(x1, y1, x2, y2, width=2, fill="#424242", tags=("edge", tag))
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        radius = self.EDGE_CENTER_RADIUS
        circle = self.canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            fill="#f5f5f5",
            outline="#424242",
            width=2,
            tags=("edge", tag),
        )
        label = self.canvas.create_text(
            center_x,
            center_y,
            text=self._edge_label(
                capacitance_expr,
                capacitance_text,
                inductance_expr,
                inductance_text,
            ),
            fill="#212121",
            justify=tk.CENTER,
            tags=("edge", tag),
        )
        if circle is not None:
            self.canvas.tag_raise(label, circle)
        self.edges[edge_id] = Edge(
            identifier=edge_id,
            nodes=(first, second),
            line_id=line,
            center_circle_id=circle,
            label_id=label,
            capacitance_expr=capacitance_expr,
            capacitance_text=capacitance_text,
            inductance_expr=inductance_expr,
            inductance_text=inductance_text,
            l_inverse_expr=l_inverse_expr,
        )
        self.canvas.tag_bind(
            tag,
            "<Double-Button-1>",
            lambda event, eid=edge_id: self._edit_edge(eid),
        )
        self._apply_edge_parameters(self.edges[edge_id], params)
        self._update_status("Conexion creada. Pulsa 'c' para otra o Escape para salir del modo.")

    def _create_ground_edge(self, node_id: int, params: EdgeParameters) -> None:
        if self.ground_node_id is None:
            return
        capacitance_expr = params.capacitance_expr
        capacitance_text = params.capacitance_text
        inductance_expr = params.inductance_expr
        inductance_text = params.inductance_text
        l_inverse_expr: Optional[sp.Expr] = None
        if inductance_expr is not None:
            l_inverse_expr = sp.simplify(sp.Integer(1) / inductance_expr)
        edge_id = self.edge_counter
        self.edge_counter += 1
        tag = f"edge_{edge_id}"
        node = self.nodes[node_id]
        x = node.x
        y = node.y
        line_end_y = y + self.GROUND_LINE_LENGTH
        line = self.canvas.create_line(
            x,
            y,
            x,
            line_end_y,
            width=2,
            fill="#424242",
            tags=("edge", tag),
        )
        mid_y = y + self.GROUND_LINE_LENGTH / 2
        radius = self.EDGE_CENTER_RADIUS
        circle = self.canvas.create_oval(
            x - radius,
            mid_y - radius,
            x + radius,
            mid_y + radius,
            fill="#f5f5f5",
            outline="#424242",
            width=2,
            tags=("edge", tag),
        )
        triangle_points = [
            x - self.GROUND_TRIANGLE_WIDTH / 2,
            line_end_y,
            x + self.GROUND_TRIANGLE_WIDTH / 2,
            line_end_y,
            x,
            line_end_y + self.GROUND_TRIANGLE_HEIGHT,
        ]
        triangle = self.canvas.create_polygon(
            triangle_points,
            fill="#f5f5f5",
            outline="#424242",
            width=2,
            tags=("edge", tag),
        )
        label = self.canvas.create_text(
            x,
            mid_y,
            text="",
            fill="#212121",
            justify=tk.CENTER,
            anchor=tk.CENTER,
            tags=("edge", tag),
        )
        self.canvas.tag_raise(label, circle)
        self.edges[edge_id] = Edge(
            identifier=edge_id,
            nodes=(node_id, self.ground_node_id),
            line_id=line,
            center_circle_id=circle,
            label_id=label,
            capacitance_expr=capacitance_expr,
            capacitance_text=capacitance_text,
            inductance_expr=inductance_expr,
            inductance_text=inductance_text,
            l_inverse_expr=l_inverse_expr,
            is_ground=True,
            ground_marker_id=triangle,
        )
        self.canvas.tag_bind(
            tag,
            "<Double-Button-1>",
            lambda event, eid=edge_id: self._edit_edge(eid),
        )
        self._apply_edge_parameters(self.edges[edge_id], params)
        self._update_edge_geometry(edge_id)

    def _edge_label(
        self,
        capacitance_expr: Optional[sp.Expr],
        capacitance_text: Optional[str],
        inductance_expr: Optional[sp.Expr],
        inductance_text: Optional[str],
    ) -> str:
        parts: list[str] = []
        cap_display = self._expression_to_display(capacitance_expr, capacitance_text)
        ind_display = self._expression_to_display(inductance_expr, inductance_text)
        if cap_display is not None:
            parts.append(f"C={cap_display}")
        if ind_display is not None:
            parts.append(f"L={ind_display}")
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return "\n".join(parts)

    def _expression_to_display(
        self, expr: Optional[sp.Expr], raw_text: Optional[str]
    ) -> Optional[str]:
        if expr is None:
            return None
        if expr.free_symbols:
            return raw_text or str(expr)
        try:
            numerical = float(expr.evalf())
        except (TypeError, ValueError):
            return raw_text or str(expr)
        return f"{numerical:g}"

    def _find_edge(self, first: int, second: int) -> Edge | None:
        for edge in self.edges.values():
            if set(edge.nodes) == {first, second}:
                return edge
        return None

    def _highlight_node(self, node_id: int, active: bool) -> None:
        base_color = "#455a64" if self.ground_node_id is not None and node_id == self.ground_node_id else "#1976d2"
        color = "#d32f2f" if active else base_color
        self.canvas.itemconfigure(self.nodes[node_id].circle_id, fill=color)

    def _set_focus_node(self, node_id: Optional[int]) -> None:
        if self.focus_node is not None:
            if self.focus_node != self.selected_node:
                self._highlight_node(self.focus_node, False)
        self.focus_node = node_id
        if node_id is not None and node_id != self.selected_node:
            self._highlight_node(node_id, True)

    def _set_mode(self, mode: Optional[str]) -> None:
        if self.selected_node is not None:
            self._highlight_node(self.selected_node, False)
        self.selected_node = None
        self.mode = mode
        if mode == "node":
            self._update_status("Modo nodo: haz click en el lienzo para crear uno.")
        elif mode == "edge":
            self._set_focus_node(None)
            self._update_status("Modo enlace: selecciona dos nodos existentes.")
        elif mode == "ground":
            self._set_focus_node(None)
            self._update_status("Modo masa: selecciona un nodo para conectarlo a GND.")
        else:
            self._update_status("Modo neutro. Pulsa 'n', 'c' o 'g' para continuar.")

    def _reset_all(self) -> None:
        if not self.nodes and not self.edges:
            return
        if not messagebox.askyesno("Reiniciar", "¿Eliminar todos los nodos y conexiones?", parent=self.root):
            return
        self.canvas.delete("all")
        self.nodes.clear()
        self.edges.clear()
        self.node_counter = 0
        self.edge_counter = 0
        self.ground_node_id = None
        self._set_mode(None)
        self._update_status("Todo reiniciado. Pulsa 'n' para crear nodos.")
        self.root.after_idle(self._ensure_ground_node)

    def _compute_matrices(self) -> Tuple[sp.Matrix, sp.Matrix]:
        node_ids = sorted(self.nodes.keys())
        index_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
        size = len(node_ids)
        c_matrix = sp.zeros(size)
        l_inv_matrix = sp.zeros(size)
        for edge in self.edges.values():
            i, j = (index_map[nid] for nid in edge.nodes)
            if edge.capacitance_expr is not None:
                value = edge.capacitance_expr
                c_matrix[i, i] += value
                c_matrix[j, j] += value
                c_matrix[i, j] -= value
                c_matrix[j, i] -= value
            if edge.l_inverse_expr is not None:
                value = edge.l_inverse_expr
                l_inv_matrix[i, i] += value
                l_inv_matrix[j, j] += value
                l_inv_matrix[i, j] -= value
                l_inv_matrix[j, i] -= value
        return sp.Matrix(c_matrix), sp.Matrix(l_inv_matrix)

    def _matrix_function_snippet(self, func_name: str, matrix: sp.Matrix) -> Tuple[list[str], list[str]]:
        symbols = sorted(matrix.free_symbols, key=lambda sym: sym.name)
        param_names = [symbol.name for symbol in symbols]
        args = ", ".join(param_names)
        indent = " " * 4
        signature = f"def {func_name}({args}):" if args else f"def {func_name}():"
        lines: list[str] = [signature, f"{indent}return np.array(["]

        rows = matrix.tolist()
        for idx, row in enumerate(rows):
            exprs = [pycode(entry) for entry in row]
            suffix = "," if idx < len(rows) - 1 else ""
            lines.append(f"{indent * 2}[{', '.join(exprs)}]{suffix}")

        lines.append(f"{indent}], dtype=float)")
        return lines, param_names

    def _copy_snippet(self) -> None:
        if not self.nodes:
            messagebox.showinfo("Sin datos", "Crea al menos un nodo para generar las matrices.", parent=self.root)
            return
        c_matrix, l_inv_matrix = self._compute_matrices()
        snippet_lines = [
            "import math",
            "import numpy as np",
            "",
        ]

        c_func_lines, c_params = self._matrix_function_snippet("C_matrix_func", c_matrix)
        l_func_lines, l_params = self._matrix_function_snippet("L_inv_matrix_func", l_inv_matrix)

        if c_params:
            snippet_lines.append(f"# C_matrix_func parameters: {', '.join(c_params)}")
        if l_params:
            snippet_lines.append(f"# L_inv_matrix_func parameters: {', '.join(l_params)}")
        if c_params or l_params:
            snippet_lines.append("")

        snippet_lines.extend(c_func_lines)
        snippet_lines.append("")
        snippet_lines.extend(l_func_lines)
        snippet = "\n".join(snippet_lines)
        self.root.clipboard_clear()
        self.root.clipboard_append(snippet)
        self._update_status("Snippet copiado al portapapeles.")

    def _ensure_ground_node(self) -> None:
        if any(node.name == "GND" for node in self.nodes.values()):
            for nid, node in self.nodes.items():
                if node.name == "GND":
                    self.ground_node_id = nid
                    break
            return
        self.root.update_idletasks()
        width = self.canvas.winfo_width() or 600
        height = self.canvas.winfo_height() or 400
        x = width * 0.12
        y = height * 0.85
        self.ground_node_id = self._add_node(x, y, "GND", silent=True, color="#455a64")
        self._update_status("Nodo GND creado automaticamente. Pulsa 'n' para agregar otros nodos.")

    def _update_status(self, message: str) -> None:
        self.status_var.set(message)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = CircuitGraphApp()
    app.run()


if __name__ == "__main__":
    main()

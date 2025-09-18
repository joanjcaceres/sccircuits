import copy
import json
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Dict, Optional, Tuple

import sympy as sp
from sympy.printing.pycode import pycode

GROUND_NODE_ID = -1


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
    ground_offset_x: float = 0.0
    ground_offset_y: float = 0.0


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
        ttk.Label(self.dialog, text=f"Entre {first} y {second}").grid(
            row=0, column=0, columnspan=2, padx=10, pady=(10, 5)
        )

        ttk.Label(self.dialog, text="Capacitancia (F)").grid(
            row=1, column=0, sticky=tk.W, padx=10
        )
        self.cap_entry = ttk.Entry(self.dialog, width=18)
        self.cap_entry.grid(row=1, column=1, padx=10, pady=2)
        if default_cap:
            self.cap_entry.insert(0, default_cap)

        ttk.Label(self.dialog, text="Inductancia (H)").grid(
            row=2, column=0, sticky=tk.W, padx=10
        )
        self.ind_entry = ttk.Entry(self.dialog, width=18)
        self.ind_entry.grid(row=2, column=1, padx=10, pady=2)
        if default_ind:
            self.ind_entry.insert(0, default_ind)

        buttons = ttk.Frame(self.dialog)
        buttons.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(buttons, text="Cancelar", command=self.dialog.destroy).pack(
            side=tk.RIGHT, padx=5
        )
        ttk.Button(buttons, text="Aceptar", command=self._on_accept).pack(
            side=tk.RIGHT, padx=5
        )

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
                messagebox.showerror(
                    "Entrada invalida",
                    "La inductancia no puede ser cero.",
                    parent=self.dialog,
                )
                return
            if ind_expr.is_number and float(ind_expr.evalf()) == 0.0:
                messagebox.showerror(
                    "Entrada invalida",
                    "La inductancia no puede ser cero.",
                    parent=self.dialog,
                )
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
        self.focus_node: Optional[int] = None
        self.dragging_ground_edge: Optional[int] = None
        self.ground_drag_offset: Tuple[float, float] = (0.0, 0.0)
        self.selected_nodes: set[int] = set()
        self.view_scale: float = 1.0
        self.history: list[dict] = []
        self._history_suspended = False
        self._node_drag_moved = False
        self._ground_drag_moved = False
        self.status_var = tk.StringVar(
            value="Press 'n' to create nodes, 'c' to connect."
        )

        self._build_ui()
        self._bind_shortcuts()
        self._push_history()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.root, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._handle_canvas_click)
        self.canvas.bind("<Control-MouseWheel>", self._handle_zoom)
        self.canvas.bind(
            "<Control-Button-4>", lambda event: self._handle_zoom(event, factor=1.1)
        )
        self.canvas.bind(
            "<Control-Button-5>", lambda event: self._handle_zoom(event, factor=0.9)
        )
        self.canvas.bind("<ButtonPress-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._perform_pan)
        self.canvas.bind("<ButtonRelease-2>", self._end_pan)

        overlay = ttk.Frame(self.root, padding=8)
        overlay.place(relx=0.02, rely=0.02)

        status_label = ttk.Label(overlay, textvariable=self.status_var)
        status_label.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 6))

        ttk.Button(
            overlay, text="Node mode", command=lambda: self._set_mode("node")
        ).grid(row=1, column=0, padx=2)
        ttk.Button(
            overlay, text="Edge mode", command=lambda: self._set_mode("edge")
        ).grid(row=1, column=1, padx=2)
        ttk.Button(
            overlay, text="Ground mode", command=lambda: self._set_mode("ground")
        ).grid(row=1, column=2, padx=2)
        ttk.Button(overlay, text="Reset", command=self._reset_all).grid(
            row=1, column=3, padx=2
        )

        ttk.Button(overlay, text="Copy snippet", command=self._copy_snippet).grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=(6, 0)
        )
        ttk.Button(
            overlay, text="Concatenate selection", command=self._duplicate_selection
        ).grid(row=3, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Button(overlay, text="Save project", command=self._save_project).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )
        ttk.Button(overlay, text="Load project", command=self._load_project).grid(
            row=4, column=2, columnspan=2, sticky="ew", pady=(6, 0)
        )

    def _current_node_radius(self) -> float:
        return self.NODE_RADIUS * self.view_scale

    def _current_edge_center_radius(self) -> float:
        return self.EDGE_CENTER_RADIUS * self.view_scale

    def _update_scrollregion(self) -> None:
        bbox = self.canvas.bbox("all")
        if bbox is not None:
            self.canvas.configure(scrollregion=bbox)

    def _bind_shortcuts(self) -> None:
        self.root.bind("n", lambda _: self._set_mode("node"))
        self.root.bind("c", lambda _: self._set_mode("edge"))
        self.root.bind("g", lambda _: self._set_mode("ground"))
        self.root.bind("<Escape>", lambda _: self._set_mode(None))
        self.root.bind("<Delete>", lambda _: self._delete_focused_node())
        self.root.bind("<BackSpace>", lambda _: self._delete_focused_node())
        self.root.bind("<Control-z>", lambda event: self._undo())
        self.root.bind("<Command-z>", lambda event: self._undo())
        self.root.bind("<Control-s>", lambda event: self._save_project())
        self.root.bind("<Command-s>", lambda event: self._save_project())
        self.root.bind("<Control-o>", lambda event: self._load_project())
        self.root.bind("<Command-o>", lambda event: self._load_project())

    def _handle_canvas_click(self, event: tk.Event) -> None:
        if self.mode != "node":
            return
        current = self.canvas.find_withtag("current")
        if current:
            tags = self.canvas.gettags(current[0])
            if "node" in tags:
                self._update_status("Click on empty canvas to add a new node.")
                return
        self._set_focus_node(None)
        self._clear_selection()
        name = self._generate_default_node_name()
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        new_node_id = self._add_node(canvas_x, canvas_y, name)
        self.selected_nodes = {new_node_id}
        self._set_focus_node(new_node_id)
        self._push_history()

    def _handle_zoom(self, event: tk.Event, factor: Optional[float] = None) -> None:
        if factor is None:
            if event.delta == 0:
                return
            factor = 1.1 if event.delta > 0 else 0.9
        new_scale = self.view_scale * factor
        min_scale, max_scale = 0.05, 6.0
        if new_scale < min_scale:
            factor = min_scale / self.view_scale
            new_scale = min_scale
        elif new_scale > max_scale:
            factor = max_scale / self.view_scale
            new_scale = max_scale

        anchor_x = self.canvas.canvasx(event.x)
        anchor_y = self.canvas.canvasy(event.y)
        self.canvas.scale("all", anchor_x, anchor_y, factor, factor)

        for node in self.nodes.values():
            node.x = anchor_x + (node.x - anchor_x) * factor
            node.y = anchor_y + (node.y - anchor_y) * factor

        for edge in self.edges.values():
            if edge.is_ground:
                edge.ground_offset_x *= factor
                edge.ground_offset_y *= factor

        self.view_scale = new_scale

        for node_id, node in self.nodes.items():
            self._move_node(node_id, node.x, node.y)

        self._refresh_all_node_appearances()
        self._update_scrollregion()
        self._push_history()

    def _start_pan(self, event: tk.Event) -> None:
        self.canvas.scan_mark(event.x, event.y)

    def _perform_pan(self, event: tk.Event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _end_pan(self, event: tk.Event) -> None:
        # No additional state to update, but method kept for completeness.
        pass

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
        forced_id: Optional[int] = None,
    ) -> int:
        if forced_id is None:
            node_id = self.node_counter
            self.node_counter += 1
        else:
            node_id = forced_id
            self.node_counter = max(self.node_counter, node_id + 1)
        tag = f"node_{node_id}"
        radius = self._current_node_radius()
        circle = self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=color,
            outline="black",
            width=2,
            tags=("node", tag),
        )
        label = self.canvas.create_text(
            x + radius + 6,
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
            self._update_status(f"Node {name} created. Press 'c' to connect.")
        self._update_scrollregion()
        return node_id

    def _handle_node_press(self, event: tk.Event, node_id: int) -> None:
        if self.mode == "ground":
            self._connect_node_to_ground(node_id)
            return
        if self.mode == "edge":
            self._handle_edge_mode_click(node_id)
            return
        shift_held = bool(event.state & 0x0001)
        if shift_held:
            self._toggle_selection(node_id)
        else:
            self._ensure_selected(node_id)
        self._set_focus_node(node_id)
        self._node_drag_moved = False
        self.dragging_node = node_id
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        node = self.nodes[node_id]
        self.drag_offset = (node.x - canvas_x, node.y - canvas_y)
        self.canvas.tag_raise(node.circle_id)
        self.canvas.tag_raise(node.label_id)
        self._update_status(f"Moving node {node.name}.")

    def _handle_edge_mode_click(self, node_id: int) -> None:
        if self.selected_node is None:
            self.selected_node = node_id
            self._set_focus_node(node_id)
            self._update_status("Select the second node to create the connection.")
            return
        if self.selected_node == node_id:
            self._update_status("Select a different node.")
            return
        first = self.selected_node
        second = node_id
        self._set_focus_node(None)
        self.selected_node = None
        self._create_edge(first, second)

    def _connect_node_to_ground(self, node_id: int) -> None:
        existing = self._find_edge(node_id, GROUND_NODE_ID)
        default_cap = ""
        default_ind = ""
        if existing is not None:
            default_cap = existing.capacitance_text or (
                str(existing.capacitance_expr)
                if existing.capacitance_expr is not None
                else ""
            )
            default_ind = existing.inductance_text or (
                str(existing.inductance_expr)
                if existing.inductance_expr is not None
                else ""
            )
        node_name = self.nodes[node_id].name
        dialog = EdgeDialog(
            self.root, node_name, "GND", default_cap or None, default_ind or None
        )
        if dialog.value is None:
            self._update_status("Ground connection cancelled.")
            return
        changed = False
        if existing is not None:
            self._apply_edge_parameters(existing, dialog.value)
            self._update_status(f"Ground connection for {node_name} updated.")
            changed = True
        else:
            self._create_ground_edge(node_id, dialog.value)
            self._update_status(f"Nodo {node_name} conectado a masa.")
            changed = True
        if changed:
            self._push_history()

    def _handle_node_drag(self, event: tk.Event, node_id: int) -> None:
        if self.dragging_node != node_id or self.mode == "edge":
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        offset_x, offset_y = self.drag_offset
        new_x = canvas_x + offset_x
        new_y = canvas_y + offset_y
        self._move_node(node_id, new_x, new_y)
        self._node_drag_moved = True

    def _handle_node_release(self, event: tk.Event, node_id: int) -> None:
        if self.dragging_node != node_id:
            return
        node = self.nodes[node_id]
        self.dragging_node = None
        self._update_status(f"Node {node.name} moved.")
        if self._node_drag_moved:
            self._node_drag_moved = False
            self._push_history()

    def _handle_ground_press(self, event: tk.Event, edge_id: int) -> None:
        if self.mode in {"edge", "ground"}:
            return
        edge = self.edges.get(edge_id)
        if edge is None or not edge.is_ground:
            return
        node = self.nodes.get(edge.nodes[0])
        if node is None:
            return
        self._ground_drag_moved = False
        self._set_focus_node(edge.nodes[0])
        self.dragging_ground_edge = edge_id
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        end_x = node.x + edge.ground_offset_x
        end_y = node.y + edge.ground_offset_y
        self.ground_drag_offset = (canvas_x - end_x, canvas_y - end_y)
        self._update_status("Moving ground connection.")

    def _handle_ground_drag(self, event: tk.Event, edge_id: int) -> None:
        if self.dragging_ground_edge != edge_id:
            return
        edge = self.edges.get(edge_id)
        if edge is None or not edge.is_ground:
            return
        node = self.nodes.get(edge.nodes[0])
        if node is None:
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        offset_x, offset_y = self.ground_drag_offset
        new_end_x = canvas_x - offset_x
        new_end_y = canvas_y - offset_y
        edge.ground_offset_x = new_end_x - node.x
        edge.ground_offset_y = new_end_y - node.y
        self._update_edge_geometry(edge_id)
        self._update_scrollregion()
        self._ground_drag_moved = True

    def _handle_ground_release(self, event: tk.Event, edge_id: int) -> None:
        if self.dragging_ground_edge != edge_id:
            return
        self.dragging_ground_edge = None
        self._update_status("Ground connection moved.")
        if self._ground_drag_moved:
            self._ground_drag_moved = False
            self._push_history()

    def _move_node(self, node_id: int, x: float, y: float) -> None:
        node = self.nodes[node_id]
        node.x = x
        node.y = y
        radius = self._current_node_radius()
        self.canvas.coords(
            node.circle_id,
            x - radius,
            y - radius,
            x + radius,
            y + radius,
        )
        self.canvas.coords(node.label_id, x + radius + 6, y)
        for edge_id, edge in self.edges.items():
            if node_id in edge.nodes:
                self._update_edge_geometry(edge_id)
        self._update_scrollregion()

    def _update_edge_geometry(self, edge_id: int) -> None:
        edge = self.edges[edge_id]
        if edge.is_ground:
            primary_id = edge.nodes[0]
            if primary_id not in self.nodes:
                return
            node = self.nodes[primary_id]
            x = node.x
            y = node.y
            end_x = node.x + edge.ground_offset_x
            end_y = node.y + edge.ground_offset_y
            self.canvas.coords(edge.line_id, x, y, end_x, end_y)
            mid_x = (x + end_x) / 2
            mid_y = (y + end_y) / 2
            radius = self._current_edge_center_radius()
            if edge.center_circle_id is not None:
                self.canvas.coords(
                    edge.center_circle_id,
                    mid_x - radius,
                    mid_y - radius,
                    mid_x + radius,
                    mid_y + radius,
                )
                self.canvas.tag_raise(edge.label_id, edge.center_circle_id)
            if edge.ground_marker_id is not None:
                triangle_points = [
                    end_x - self.GROUND_TRIANGLE_WIDTH / 2,
                    end_y,
                    end_x + self.GROUND_TRIANGLE_WIDTH / 2,
                    end_y,
                    end_x,
                    end_y + self.GROUND_TRIANGLE_HEIGHT,
                ]
                self.canvas.coords(edge.ground_marker_id, *triangle_points)
            self.canvas.coords(edge.label_id, mid_x, mid_y)
            return

        node_a = self.nodes[edge.nodes[0]]
        node_b = self.nodes[edge.nodes[1]]
        self.canvas.coords(edge.line_id, node_a.x, node_a.y, node_b.x, node_b.y)
        center_x = (node_a.x + node_b.x) / 2
        center_y = (node_a.y + node_b.y) / 2
        radius = self._current_edge_center_radius()
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
        dialog = EdgeDialog(
            self.root, first_name, second_name, default_cap, default_ind
        )
        if dialog.value is None:
            self._update_status("Connection edit cancelled.")
            return
        self._apply_edge_parameters(edge, dialog.value)
        self._update_status("Connection updated.")
        self._push_history()

    def _delete_focused_node(self) -> None:
        if self.focus_node is None:
            if self.selected_nodes:
                candidate = next(iter(self.selected_nodes))
                self._set_focus_node(candidate)
            else:
                self._update_status("Select a node and press Delete to remove it.")
                return
        node_id = self.focus_node
        node = self.nodes.get(node_id)
        if node is None:
            return
        node_name = node.name
        self._remove_node(node_id)
        self._set_focus_node(None)
        self._push_history()
        self._update_status(f"Node {node_name} removed.")

    def _remove_node(self, node_id: int) -> None:
        node = self.nodes.pop(node_id, None)
        if node is None:
            return
        self.selected_nodes.discard(node_id)
        self.canvas.delete(node.circle_id)
        self.canvas.delete(node.label_id)
        edges_to_remove = [
            edge_id for edge_id, edge in self.edges.items() if node_id in edge.nodes
        ]
        for edge_id in edges_to_remove:
            self._remove_edge(edge_id)
        self._update_scrollregion()
        self._refresh_all_node_appearances()

    def _remove_edge(self, edge_id: int) -> None:
        edge = self.edges.pop(edge_id, None)
        if edge is None:
            return
        if self.dragging_ground_edge == edge_id:
            self.dragging_ground_edge = None
        self.canvas.delete(edge.line_id)
        if edge.center_circle_id is not None:
            self.canvas.delete(edge.center_circle_id)
        if edge.ground_marker_id is not None:
            self.canvas.delete(edge.ground_marker_id)
        self.canvas.delete(edge.label_id)
        self._update_scrollregion()

    def _duplicate_selection(self) -> None:
        if not self.selected_nodes:
            messagebox.showinfo(
                "Empty selection",
                "Select at least one node to concatenate.",
                parent=self.root,
            )
            return

        selected_nodes = sorted(self.selected_nodes)

        copies = simpledialog.askinteger(
            "Concatenate selection",
            "Number of repeats:",
            initialvalue=1,
            minvalue=1,
            parent=self.root,
        )
        if copies is None:
            return

        min_x = min(self.nodes[nid].x for nid in selected_nodes)
        max_x = max(self.nodes[nid].x for nid in selected_nodes)

        block_width = max_x - min_x
        min_spacing = max(self._current_node_radius() * 4, 40.0 * self.view_scale)
        dx = (
            block_width if block_width > 0 else self._current_node_radius() * 6
        ) + min_spacing
        dy = 0.0

        original_edges = [
            edge
            for edge in self.edges.values()
            if (
                (edge.is_ground and edge.nodes[0] in selected_nodes)
                or (
                    not edge.is_ground
                    and edge.nodes[0] in selected_nodes
                    and edge.nodes[1] in selected_nodes
                )
            )
        ]

        left_nodes = sorted(
            [nid for nid in selected_nodes if abs(self.nodes[nid].x - min_x) < 1e-6],
            key=lambda nid: self.nodes[nid].y,
        )
        right_nodes = sorted(
            [nid for nid in selected_nodes if abs(self.nodes[nid].x - max_x) < 1e-6],
            key=lambda nid: self.nodes[nid].y,
        )
        pair_count = min(len(left_nodes), len(right_nodes))
        left_nodes = left_nodes[:pair_count]
        right_nodes = right_nodes[:pair_count]
        left_index_map = {node_id: idx for idx, node_id in enumerate(left_nodes)}
        current_tail_map: Dict[int, int] = {
            idx: right_nodes[idx] for idx in range(pair_count)
        }
        left_boundary_set = set(left_nodes)

        all_new_nodes: list[int] = []

        for replica_index in range(1, copies + 1):
            mapping: Dict[int, int] = {}
            shift_x = dx * replica_index
            shift_y = dy * replica_index

            for node_id in selected_nodes:
                original = self.nodes[node_id]
                idx = left_index_map.get(node_id)
                if idx is not None:
                    mapping[node_id] = current_tail_map[idx]
                else:
                    new_name = self._generate_default_node_name()
                    new_node_id = self._add_node(
                        original.x + shift_x,
                        original.y + shift_y,
                        new_name,
                        silent=True,
                    )
                    mapping[node_id] = new_node_id
                    all_new_nodes.append(new_node_id)

            for edge in original_edges:
                params = EdgeParameters(
                    capacitance_expr=edge.capacitance_expr,
                    capacitance_text=edge.capacitance_text,
                    inductance_expr=edge.inductance_expr,
                    inductance_text=edge.inductance_text,
                )
                if edge.is_ground:
                    source_idx = left_index_map.get(edge.nodes[0])
                    source_id = mapping.get(edge.nodes[0])
                    if source_id is not None and not (
                        source_idx is not None
                        and source_id == current_tail_map[source_idx]
                    ):
                        self._instantiate_ground_edge(
                            source_id,
                            params,
                            offset_x=edge.ground_offset_x,
                            offset_y=edge.ground_offset_y,
                        )
                else:
                    first_new = mapping.get(edge.nodes[0])
                    second_new = mapping.get(edge.nodes[1])
                    if first_new is not None and second_new is not None:
                        self._instantiate_edge(first_new, second_new, params)

            for idx, right_original in enumerate(right_nodes):
                tail_candidate = mapping.get(right_original)
                if tail_candidate is not None:
                    current_tail_map[idx] = tail_candidate

        self.selected_nodes = set(all_new_nodes)
        if all_new_nodes:
            self._set_focus_node(all_new_nodes[-1])
        else:
            self._set_focus_node(None)
        self._refresh_all_node_appearances()
        self._update_scrollregion()
        self._push_history()
        self._update_status("Concatenation complete.")

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
        self._update_status(f"Node renamed to {new_name}.")
        self._push_history()

    def _instantiate_edge(
        self,
        first: int,
        second: int,
        params: EdgeParameters,
        *,
        forced_id: Optional[int] = None,
    ) -> int:
        capacitance_expr = params.capacitance_expr
        capacitance_text = params.capacitance_text
        inductance_expr = params.inductance_expr
        inductance_text = params.inductance_text
        l_inverse_expr: Optional[sp.Expr] = None
        if inductance_expr is not None:
            l_inverse_expr = sp.simplify(sp.Integer(1) / inductance_expr)
        if forced_id is None:
            edge_id = self.edge_counter
            self.edge_counter += 1
        else:
            edge_id = forced_id
            self.edge_counter = max(self.edge_counter, edge_id + 1)
        tag = f"edge_{edge_id}"
        x1, y1 = self.nodes[first].x, self.nodes[first].y
        x2, y2 = self.nodes[second].x, self.nodes[second].y
        line = self.canvas.create_line(
            x1, y1, x2, y2, width=2, fill="#424242", tags=("edge", tag)
        )
        radius = self._current_edge_center_radius()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
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
            text="",
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
        self._update_scrollregion()
        return edge_id

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
            self._update_status("Connection cancelled.")
            return
        self._instantiate_edge(first, second, dialog.value)
        self._update_status(
            "Connection created. Press 'c' for another or Esc to exit mode."
        )
        self._push_history()

    def _instantiate_ground_edge(
        self,
        node_id: int,
        params: EdgeParameters,
        *,
        offset_x: float = 0.0,
        offset_y: Optional[float] = None,
        forced_id: Optional[int] = None,
    ) -> int:
        capacitance_expr = params.capacitance_expr
        capacitance_text = params.capacitance_text
        inductance_expr = params.inductance_expr
        inductance_text = params.inductance_text
        l_inverse_expr: Optional[sp.Expr] = None
        if inductance_expr is not None:
            l_inverse_expr = sp.simplify(sp.Integer(1) / inductance_expr)
        if forced_id is None:
            edge_id = self.edge_counter
            self.edge_counter += 1
        else:
            edge_id = forced_id
            self.edge_counter = max(self.edge_counter, edge_id + 1)
        tag = f"edge_{edge_id}"
        node = self.nodes[node_id]
        start_x = node.x
        start_y = node.y
        if offset_y is None:
            offset_y = self.GROUND_LINE_LENGTH
        end_x = start_x + offset_x
        end_y = start_y + offset_y
        line = self.canvas.create_line(
            start_x,
            start_y,
            end_x,
            end_y,
            width=2,
            fill="#424242",
            tags=("edge", tag),
        )
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        radius = self._current_edge_center_radius()
        circle = self.canvas.create_oval(
            mid_x - radius,
            mid_y - radius,
            mid_x + radius,
            mid_y + radius,
            fill="#f5f5f5",
            outline="#424242",
            width=2,
            tags=("edge", tag),
        )
        triangle_points = [
            end_x - self.GROUND_TRIANGLE_WIDTH / 2,
            end_y,
            end_x + self.GROUND_TRIANGLE_WIDTH / 2,
            end_y,
            end_x,
            end_y + self.GROUND_TRIANGLE_HEIGHT,
        ]
        triangle = self.canvas.create_polygon(
            triangle_points,
            fill="#f5f5f5",
            outline="#424242",
            width=2,
            tags=("edge", tag),
        )
        label = self.canvas.create_text(
            mid_x,
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
            nodes=(node_id, GROUND_NODE_ID),
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
            ground_offset_x=offset_x,
            ground_offset_y=offset_y,
        )
        self.canvas.tag_bind(
            tag,
            "<Double-Button-1>",
            lambda event, eid=edge_id: self._edit_edge(eid),
        )
        for item in (line, circle, triangle, label):
            self.canvas.tag_bind(
                item,
                "<ButtonPress-1>",
                lambda event, eid=edge_id: self._handle_ground_press(event, eid),
            )
            self.canvas.tag_bind(
                item,
                "<B1-Motion>",
                lambda event, eid=edge_id: self._handle_ground_drag(event, eid),
            )
            self.canvas.tag_bind(
                item,
                "<ButtonRelease-1>",
                lambda event, eid=edge_id: self._handle_ground_release(event, eid),
            )
        self._apply_edge_parameters(self.edges[edge_id], params)
        self._update_scrollregion()
        return edge_id

    def _create_ground_edge(self, node_id: int, params: EdgeParameters) -> None:
        self._instantiate_ground_edge(node_id, params)

    def _snapshot_state(self) -> dict:
        nodes_snapshot = [
            {
                "identifier": node_id,
                "name": node.name,
                "x": node.x,
                "y": node.y,
            }
            for node_id, node in sorted(self.nodes.items())
        ]
        edges_snapshot = [
            {
                "identifier": edge_id,
                "nodes": list(edge.nodes),
                "capacitance_expr": edge.capacitance_expr,
                "capacitance_text": edge.capacitance_text,
                "inductance_expr": edge.inductance_expr,
                "inductance_text": edge.inductance_text,
                "l_inverse_expr": edge.l_inverse_expr,
                "is_ground": edge.is_ground,
                "ground_offset_x": edge.ground_offset_x,
                "ground_offset_y": edge.ground_offset_y,
            }
            for edge_id, edge in sorted(self.edges.items())
        ]
        return {
            "node_counter": self.node_counter,
            "edge_counter": self.edge_counter,
            "view_scale": self.view_scale,
            "nodes": nodes_snapshot,
            "edges": edges_snapshot,
            "selected_nodes": sorted(self.selected_nodes),
            "focus_node": self.focus_node,
            "selected_node": self.selected_node,
            "mode": self.mode,
        }

    def _restore_state(self, snapshot: dict) -> None:
        self._history_suspended = True
        try:
            self.canvas.delete("all")
            self.nodes.clear()
            self.edges.clear()
            self.node_counter = 0
            self.edge_counter = 0
            self.view_scale = snapshot.get("view_scale", 1.0)
            self.mode = snapshot.get("mode")
            self.selected_node = snapshot.get("selected_node")
            self.dragging_node = None
            self.dragging_ground_edge = None
            self._node_drag_moved = False
            self._ground_drag_moved = False

            for node_data in snapshot.get("nodes", []):
                self._add_node(
                    node_data["x"],
                    node_data["y"],
                    node_data["name"],
                    silent=True,
                    forced_id=node_data["identifier"],
                )

            self.node_counter = snapshot.get("node_counter", self.node_counter)

            for edge_data in snapshot.get("edges", []):
                params = EdgeParameters(
                    capacitance_expr=edge_data["capacitance_expr"],
                    capacitance_text=edge_data["capacitance_text"],
                    inductance_expr=edge_data["inductance_expr"],
                    inductance_text=edge_data["inductance_text"],
                )
                if edge_data.get("is_ground"):
                    self._instantiate_ground_edge(
                        edge_data["nodes"][0],
                        params,
                        offset_x=edge_data.get("ground_offset_x", 0.0),
                        offset_y=edge_data.get(
                            "ground_offset_y", self.GROUND_LINE_LENGTH
                        ),
                        forced_id=edge_data["identifier"],
                    )
                else:
                    self._instantiate_edge(
                        edge_data["nodes"][0],
                        edge_data["nodes"][1],
                        params,
                        forced_id=edge_data["identifier"],
                    )

            self.edge_counter = snapshot.get("edge_counter", self.edge_counter)
            self.selected_nodes = set(snapshot.get("selected_nodes", []))
            self.focus_node = snapshot.get("focus_node")
            self._refresh_all_node_appearances()
            self._update_scrollregion()
        finally:
            self._history_suspended = False

    def _expr_to_string(self, expr: Optional[sp.Expr]) -> Optional[str]:
        if expr is None:
            return None
        return sp.srepr(expr)

    def _expr_from_string(self, text: Optional[str]) -> Optional[sp.Expr]:
        if text in (None, ""):
            return None
        try:
            return sp.sympify(text, evaluate=False)
        except Exception as exc:  # type: ignore[catching-non-exception]
            messagebox.showerror(
                "Load project",
                f"Failed to parse expression '{text}':\n{exc}",
                parent=self.root,
            )
            return None

    def _push_history(self) -> None:
        if self._history_suspended:
            return
        snapshot = self._snapshot_state()
        if self.history and snapshot == self.history[-1]:
            return
        self.history.append(copy.deepcopy(snapshot))
        if len(self.history) > 100:
            self.history = self.history[-100:]

    def _undo(self) -> None:
        if len(self.history) <= 1:
            self._update_status("No hay acciones para deshacer.")
            return
        self.history.pop()
        snapshot = copy.deepcopy(self.history[-1])
        self._restore_state(snapshot)
        self._refresh_all_node_appearances()
        self._update_status("Action undone.")

    def _save_project(self) -> None:
        filename = filedialog.asksaveasfilename(
            title="Save project",
            defaultextension=".json",
            filetypes=[("Circuit project", "*.json"), ("All files", "*.*")],
            parent=self.root,
        )
        if not filename:
            return

        snapshot = copy.deepcopy(self._snapshot_state())
        for edge in snapshot.get("edges", []):
            edge["capacitance_expr"] = self._expr_to_string(
                edge.get("capacitance_expr")
            )
            edge["inductance_expr"] = self._expr_to_string(edge.get("inductance_expr"))
            edge["l_inverse_expr"] = self._expr_to_string(edge.get("l_inverse_expr"))

        data = {"version": 1, "state": snapshot}
        try:
            Path(filename).write_text(json.dumps(data, indent=2))
        except OSError as exc:
            messagebox.showerror(
                "Save project", f"Could not save file:\n{exc}", parent=self.root
            )
            return

        self._update_status(f"Project saved to {Path(filename).name}.")

    def _load_project(self) -> None:
        filename = filedialog.askopenfilename(
            title="Load project",
            defaultextension=".json",
            filetypes=[("Circuit project", "*.json"), ("All files", "*.*")],
            parent=self.root,
        )
        if not filename:
            return

        try:
            data = json.loads(Path(filename).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror(
                "Load project", f"Could not load file:\n{exc}", parent=self.root
            )
            return

        state = data.get("state", data)
        state.setdefault("selected_nodes", [])
        for edge in state.get("edges", []):
            edge["capacitance_expr"] = self._expr_from_string(
                edge.get("capacitance_expr")
            )
            edge["inductance_expr"] = self._expr_from_string(
                edge.get("inductance_expr")
            )
            edge["l_inverse_expr"] = self._expr_from_string(edge.get("l_inverse_expr"))

        current_snapshot = copy.deepcopy(self._snapshot_state())
        self._restore_state(state)
        new_snapshot = copy.deepcopy(self._snapshot_state())
        self.history = [current_snapshot, new_snapshot]
        self._update_status(f"Project loaded from {Path(filename).name}.")

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
            if edge.is_ground:
                if {first, second} == {edge.nodes[0], edge.nodes[1]}:
                    return edge
            else:
                if set(edge.nodes) == {first, second}:
                    return edge
        return None

    def _refresh_node_appearance(self, node_id: int) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            return
        if node_id == self.focus_node:
            color = "#d32f2f"
        elif node_id in self.selected_nodes:
            color = "#ff9800"
        else:
            color = "#1976d2"
        self.canvas.itemconfigure(node.circle_id, fill=color)

    def _refresh_all_node_appearances(self) -> None:
        for node_id in self.nodes:
            self._refresh_node_appearance(node_id)

    def _clear_selection(self) -> None:
        if not self.selected_nodes:
            return
        for node_id in list(self.selected_nodes):
            self.selected_nodes.discard(node_id)
            self._refresh_node_appearance(node_id)

    def _toggle_selection(self, node_id: int) -> None:
        if node_id in self.selected_nodes:
            self.selected_nodes.remove(node_id)
        else:
            self.selected_nodes.add(node_id)
        self._refresh_node_appearance(node_id)

    def _ensure_selected(self, node_id: int) -> None:
        if node_id not in self.selected_nodes or len(self.selected_nodes) > 1:
            self._clear_selection()
            self.selected_nodes.add(node_id)
            self._refresh_node_appearance(node_id)

    def _set_focus_node(self, node_id: Optional[int]) -> None:
        previous = self.focus_node
        self.focus_node = node_id
        if previous is not None:
            self._refresh_node_appearance(previous)
        if node_id is not None:
            self._refresh_node_appearance(node_id)

    def _set_mode(self, mode: Optional[str]) -> None:
        if self.selected_node is not None:
            self._set_focus_node(None)
        self.selected_node = None
        self.mode = mode
        if mode == "node":
            self._update_status("Node mode: click on the canvas to create one.")
        elif mode == "edge":
            self._set_focus_node(None)
            self._update_status("Edge mode: select two existing nodes.")
        elif mode == "ground":
            self._set_focus_node(None)
            self._update_status("Ground mode: select a node to connect to ground.")
        else:
            self._update_status("Neutral mode. Press 'n', 'c' or 'g' to continue.")

    def _reset_all(self) -> None:
        if not self.nodes and not self.edges:
            return
        if not messagebox.askyesno(
            "Reiniciar", "¿Eliminar todos los nodos y conexiones?", parent=self.root
        ):
            return
        self.canvas.delete("all")
        self.nodes.clear()
        self.edges.clear()
        self.node_counter = 0
        self.edge_counter = 0
        self._set_mode(None)
        self._update_status("Workspace cleared. Press 'n' to create nodes.")
        self._push_history()

    def _compute_matrices(self) -> Tuple[sp.Matrix, sp.Matrix]:
        node_ids = sorted(self.nodes.keys())
        index_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
        size = len(node_ids)
        c_matrix = sp.zeros(size)
        l_inv_matrix = sp.zeros(size)
        for edge in self.edges.values():
            first_node, second_node = edge.nodes
            if first_node not in index_map:
                continue
            i = index_map[first_node]

            if second_node == GROUND_NODE_ID:
                if edge.capacitance_expr is not None:
                    value = edge.capacitance_expr
                    c_matrix[i, i] += value
                if edge.l_inverse_expr is not None:
                    value = edge.l_inverse_expr
                    l_inv_matrix[i, i] += value
                continue

            if second_node not in index_map:
                continue
            j = index_map[second_node]
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

    def _matrix_function_snippet(
        self, func_name: str, matrix: sp.Matrix
    ) -> Tuple[list[str], list[str]]:
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
            messagebox.showinfo(
                "Sin datos",
                "Crea al menos un nodo para generar las matrices.",
                parent=self.root,
            )
            return
        c_matrix, l_inv_matrix = self._compute_matrices()
        snippet_lines = [
            "import math",
            "import numpy as np",
            "",
        ]

        c_func_lines, c_params = self._matrix_function_snippet(
            "C_matrix_func", c_matrix
        )
        l_func_lines, l_params = self._matrix_function_snippet(
            "L_inv_matrix_func", l_inv_matrix
        )

        if c_params:
            snippet_lines.append(f"# C_matrix_func parameters: {', '.join(c_params)}")
        if l_params:
            snippet_lines.append(
                f"# L_inv_matrix_func parameters: {', '.join(l_params)}"
            )
        if c_params or l_params:
            snippet_lines.append("")

        snippet_lines.extend(c_func_lines)
        snippet_lines.append("")
        snippet_lines.extend(l_func_lines)
        snippet = "\n".join(snippet_lines)
        self.root.clipboard_clear()
        self.root.clipboard_append(snippet)
        self._update_status("Snippet copiado al portapapeles.")

    def _update_status(self, message: str) -> None:
        self.status_var.set(message)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = CircuitGraphApp()
    app.run()


if __name__ == "__main__":
    main()

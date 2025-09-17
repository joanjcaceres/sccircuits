import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np


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
    label_id: int
    capacitance: Optional[float]
    inductance: Optional[float]
    l_inverse: Optional[float]


class EdgeDialog:
    def __init__(self, parent: tk.Tk, first: str, second: str):
        self.parent = parent
        self.value: Optional[Tuple[Optional[float], Optional[float]]] = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.title("Valores del enlace")
        ttk.Label(self.dialog, text=f"Entre {first} y {second}").grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5))

        ttk.Label(self.dialog, text="Capacitancia (F)").grid(row=1, column=0, sticky=tk.W, padx=10)
        self.cap_entry = ttk.Entry(self.dialog, width=18)
        self.cap_entry.grid(row=1, column=1, padx=10, pady=2)

        ttk.Label(self.dialog, text="Inductancia (H)").grid(row=2, column=0, sticky=tk.W, padx=10)
        self.ind_entry = ttk.Entry(self.dialog, width=18)
        self.ind_entry.grid(row=2, column=1, padx=10, pady=2)

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
            cap = self._parse_float(self.cap_entry.get())
            ind = self._parse_float(self.ind_entry.get())
        except ValueError as exc:
            messagebox.showerror("Entrada invalida", str(exc), parent=self.dialog)
            return
        if ind is not None and np.isclose(ind, 0.0):
            messagebox.showerror("Entrada invalida", "La inductancia no puede ser cero.", parent=self.dialog)
            return
        self.value = (cap, ind)
        self.dialog.destroy()

    @staticmethod
    def _parse_float(text: str) -> Optional[float]:
        stripped = text.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError as exc:
            raise ValueError("Introduce un numero valido.") from exc


class CircuitGraphApp:
    NODE_RADIUS = 18

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("BBQ Matrix Builder")
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[int, Edge] = {}
        self.node_counter = 0
        self.edge_counter = 0
        self.mode: Optional[str] = None
        self.selected_node: Optional[int] = None
        self.status_var = tk.StringVar(value="Pulsa 'n' para crear nodos, 'c' para conectar.")

        self._build_ui()
        self._bind_shortcuts()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.root, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._handle_canvas_click)

        sidebar = ttk.Frame(self.root, padding=10)
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)

        ttk.Label(sidebar, textvariable=self.status_var, wraplength=260).grid(row=0, column=0, sticky="ew", pady=(0, 10))

        controls = ttk.Frame(sidebar)
        controls.grid(row=1, column=0, sticky="ew", pady=5)
        ttk.Button(controls, text="Modo nodo", command=lambda: self._set_mode("node")).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="Modo enlace", command=lambda: self._set_mode("edge")).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="Reiniciar", command=self._reset_all).pack(side=tk.LEFT, padx=2)

        ttk.Label(sidebar, text="Nodos").grid(row=2, column=0, sticky=tk.W, pady=(10, 2))
        self.node_tree = ttk.Treeview(sidebar, columns=("nombre", "pos"), show="headings", height=6)
        self.node_tree.heading("nombre", text="Nombre")
        self.node_tree.heading("pos", text="Posicion")
        self.node_tree.column("nombre", width=100, anchor=tk.CENTER)
        self.node_tree.column("pos", width=140, anchor=tk.CENTER)
        self.node_tree.grid(row=3, column=0, sticky="ew")

        ttk.Label(sidebar, text="Conexiones").grid(row=4, column=0, sticky=tk.W, pady=(10, 2))
        self.edge_tree = ttk.Treeview(sidebar, columns=("nodos", "cap", "ind"), show="headings", height=8)
        self.edge_tree.heading("nodos", text="Entre")
        self.edge_tree.heading("cap", text="C (F)")
        self.edge_tree.heading("ind", text="L (H)")
        self.edge_tree.column("nodos", width=120, anchor=tk.CENTER)
        self.edge_tree.column("cap", width=80, anchor=tk.CENTER)
        self.edge_tree.column("ind", width=80, anchor=tk.CENTER)
        self.edge_tree.grid(row=5, column=0, sticky="ew")

        matrix_frame = ttk.LabelFrame(sidebar, text="Matrices")
        matrix_frame.grid(row=6, column=0, sticky="nsew", pady=(10, 0))
        matrix_frame.columnconfigure(0, weight=1)
        matrix_frame.rowconfigure(1, weight=1)

        ttk.Label(matrix_frame, text="C").grid(row=0, column=0, sticky=tk.W)
        self.c_matrix_text = tk.Text(matrix_frame, height=6, width=40)
        self.c_matrix_text.grid(row=1, column=0, sticky="nsew", padx=2)
        self.c_matrix_text.configure(state="disabled")

        ttk.Label(matrix_frame, text="L^-1").grid(row=2, column=0, sticky=tk.W)
        self.linv_matrix_text = tk.Text(matrix_frame, height=6, width=40)
        self.linv_matrix_text.grid(row=3, column=0, sticky="nsew", padx=2)
        self.linv_matrix_text.configure(state="disabled")

        ttk.Button(sidebar, text="Copiar snippet para BBQ", command=self._copy_snippet).grid(row=7, column=0, sticky="ew", pady=10)

    def _bind_shortcuts(self) -> None:
        self.root.bind("n", lambda _: self._set_mode("node"))
        self.root.bind("c", lambda _: self._set_mode("edge"))
        self.root.bind("<Escape>", lambda _: self._set_mode(None))

    def _handle_canvas_click(self, event: tk.Event) -> None:
        if self.mode != "node":
            return
        current = self.canvas.find_withtag("current")
        if current:
            tags = self.canvas.gettags(current[0])
            if "node" in tags:
                self._update_status("Haz click en el fondo para agregar un nodo nuevo.")
                return
        name = self._ask_node_name()
        self._add_node(event.x, event.y, name)

    def _ask_node_name(self) -> str:
        default = f"N{self.node_counter + 1}"
        prompt = simpledialog.askstring("Nuevo nodo", "Nombre del nodo:", initialvalue=default, parent=self.root)
        if not prompt:
            return default
        return prompt.strip()

    def _add_node(self, x: float, y: float, name: str) -> None:
        node_id = self.node_counter
        self.node_counter += 1
        tag = f"node_{node_id}"
        circle = self.canvas.create_oval(
            x - self.NODE_RADIUS,
            y - self.NODE_RADIUS,
            x + self.NODE_RADIUS,
            y + self.NODE_RADIUS,
            fill="#1976d2",
            outline="black",
            width=2,
            tags=("node", tag),
        )
        label = self.canvas.create_text(x, y, text=name, fill="white", tags=("node", tag))
        self.canvas.tag_bind(tag, "<Button-1>", lambda event, nid=node_id: self._on_node_click(event, nid))

        self.nodes[node_id] = Node(node_id, name, x, y, circle, label)
        self._refresh_nodes_view()
        self._update_status(f"Nodo {name} creado. Pulsa 'c' para conectar.")

    def _on_node_click(self, event: tk.Event, node_id: int) -> None:
        if self.mode != "edge":
            self._update_status(f"Nodo seleccionado: {self.nodes[node_id].name}")
            return
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
        capacitance, inductance = dialog.value
        l_inverse = None
        if inductance is not None:
            l_inverse = 1.0 / inductance
        edge_id = self.edge_counter
        self.edge_counter += 1
        tag = f"edge_{edge_id}"
        x1, y1 = self.nodes[first].x, self.nodes[first].y
        x2, y2 = self.nodes[second].x, self.nodes[second].y
        line = self.canvas.create_line(x1, y1, x2, y2, width=2, fill="#424242", tags=("edge", tag))
        label = self.canvas.create_text(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            text=self._edge_label(capacitance, inductance),
            fill="black",
            tags=("edge", tag),
        )
        self.edges[edge_id] = Edge(edge_id, (first, second), line, label, capacitance, inductance, l_inverse)
        self._refresh_edges_view()
        self._update_matrices()
        self._update_status("Conexion creada. Pulsa 'c' para otra o Escape para salir del modo.")

    def _edge_label(self, capacitance: Optional[float], inductance: Optional[float]) -> str:
        parts: list[str] = []
        if capacitance is not None:
            parts.append(f"C={capacitance:g}")
        if inductance is not None:
            parts.append(f"L={inductance:g}")
        return " | ".join(parts) if parts else "(sin valores)"

    def _find_edge(self, first: int, second: int) -> Edge | None:
        for edge in self.edges.values():
            if set(edge.nodes) == {first, second}:
                return edge
        return None

    def _highlight_node(self, node_id: int, active: bool) -> None:
        color = "#d32f2f" if active else "#1976d2"
        self.canvas.itemconfigure(self.nodes[node_id].circle_id, fill=color)

    def _set_mode(self, mode: Optional[str]) -> None:
        if self.selected_node is not None:
            self._highlight_node(self.selected_node, False)
        self.selected_node = None
        self.mode = mode
        if mode == "node":
            self._update_status("Modo nodo: haz click en el lienzo para crear uno.")
        elif mode == "edge":
            self._update_status("Modo enlace: selecciona dos nodos existentes.")
        else:
            self._update_status("Modo neutro. Pulsa 'n' o 'c' para continuar.")

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
        self._set_mode(None)
        self._refresh_nodes_view()
        self._refresh_edges_view()
        self._clear_matrices()

    def _refresh_nodes_view(self) -> None:
        for item in self.node_tree.get_children():
            self.node_tree.delete(item)
        for node in self.nodes.values():
            position = f"({int(node.x)}, {int(node.y)})"
            self.node_tree.insert("", tk.END, values=(node.name, position))
        self._update_matrices()

    def _refresh_edges_view(self) -> None:
        for item in self.edge_tree.get_children():
            self.edge_tree.delete(item)
        for edge in self.edges.values():
            first, second = edge.nodes
            node_pair = f"{self.nodes[first].name}-{self.nodes[second].name}"
            cap_text = f"{edge.capacitance:g}" if edge.capacitance is not None else "-"
            ind_text = f"{edge.inductance:g}" if edge.inductance is not None else "-"
            self.edge_tree.insert("", tk.END, values=(node_pair, cap_text, ind_text))

    def _clear_matrices(self) -> None:
        self._write_text(self.c_matrix_text, "")
        self._write_text(self.linv_matrix_text, "")

    def _update_matrices(self) -> None:
        if not self.nodes:
            self._clear_matrices()
            return
        c_matrix, l_inv_matrix = self._compute_matrices()
        self._write_text(self.c_matrix_text, self._matrix_to_str(c_matrix))
        self._write_text(self.linv_matrix_text, self._matrix_to_str(l_inv_matrix))

    def _compute_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        node_ids = sorted(self.nodes.keys())
        index_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
        size = len(node_ids)
        c_matrix = np.zeros((size, size))
        l_inv_matrix = np.zeros((size, size))
        for edge in self.edges.values():
            i, j = (index_map[nid] for nid in edge.nodes)
            if edge.capacitance is not None:
                value = edge.capacitance
                c_matrix[i, i] += value
                c_matrix[j, j] += value
                c_matrix[i, j] -= value
                c_matrix[j, i] -= value
            if edge.l_inverse is not None:
                value = edge.l_inverse
                l_inv_matrix[i, i] += value
                l_inv_matrix[j, j] += value
                l_inv_matrix[i, j] -= value
                l_inv_matrix[j, i] -= value
        return c_matrix, l_inv_matrix

    def _matrix_to_str(self, matrix: np.ndarray) -> str:
        if matrix.size == 0:
            return "[]"
        return np.array2string(matrix, precision=4, suppress_small=True)

    def _write_text(self, widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content)
        widget.configure(state="disabled")

    def _copy_snippet(self) -> None:
        if not self.nodes:
            messagebox.showinfo("Sin datos", "Crea al menos un nodo para generar las matrices.", parent=self.root)
            return
        c_matrix, l_inv_matrix = self._compute_matrices()
        snippet = (
            "import numpy as np\n"
            "from sccircuits.bbq import BBQ\n\n"
            f"C_matrix = np.array({repr(c_matrix.tolist())}, dtype=float)\n"
            f"L_inv_matrix = np.array({repr(l_inv_matrix.tolist())}, dtype=float)\n\n"
            "bbq = BBQ(C_matrix=C_matrix, L_inv_matrix=L_inv_matrix)\n"
        )
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

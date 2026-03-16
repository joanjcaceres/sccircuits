import numpy as np
import sympy as sp
from scipy import sparse

from apps.bbq_circuit_designer import CircuitGraphApp, Edge, GROUND_NODE_ID, Node


def _make_app() -> CircuitGraphApp:
    app = CircuitGraphApp.__new__(CircuitGraphApp)
    app.nodes = {
        10: Node(10, "N1", 0.0, 0.0, 0, 0),
        11: Node(11, "N2", 0.0, 0.0, 0, 0),
        15: Node(15, "N3", 0.0, 0.0, 0, 0),
    }

    c1 = sp.Symbol("C1")
    c2 = sp.Symbol("C2")
    cg = sp.Symbol("Cg")
    l1_inv = sp.Symbol("L1_inv")
    lg_inv = sp.Symbol("Lg_inv")

    app.edges = {
        0: Edge(
            0,
            (10, 11),
            0,
            None,
            0,
            c1,
            "C1",
            None,
            None,
            l1_inv,
        ),
        1: Edge(
            1,
            (11, 10),
            0,
            None,
            0,
            c2,
            "C2",
            None,
            None,
            None,
        ),
        2: Edge(
            2,
            (15, GROUND_NODE_ID),
            0,
            None,
            0,
            cg,
            "Cg",
            None,
            None,
            lg_inv,
            is_ground=True,
        ),
    }
    return app


def test_compute_matrix_entries_accumulates_sparse_contributions():
    app = _make_app()

    size, c_entries, l_inv_entries = app._compute_matrix_entries()

    c_total = sp.Symbol("C1") + sp.Symbol("C2")
    assert size == 3
    assert sp.simplify(c_entries[(0, 0)] - c_total) == 0
    assert sp.simplify(c_entries[(1, 1)] - c_total) == 0
    assert sp.simplify(c_entries[(0, 1)] + c_total) == 0
    assert sp.simplify(c_entries[(1, 0)] + c_total) == 0
    assert sp.simplify(c_entries[(2, 2)] - sp.Symbol("Cg")) == 0

    assert sp.simplify(l_inv_entries[(0, 0)] - sp.Symbol("L1_inv")) == 0
    assert sp.simplify(l_inv_entries[(1, 1)] - sp.Symbol("L1_inv")) == 0
    assert sp.simplify(l_inv_entries[(0, 1)] + sp.Symbol("L1_inv")) == 0
    assert sp.simplify(l_inv_entries[(1, 0)] + sp.Symbol("L1_inv")) == 0
    assert sp.simplify(l_inv_entries[(2, 2)] - sp.Symbol("Lg_inv")) == 0


def _numeric_matrix(
    sympy_matrix: sp.Matrix, substitutions: dict[sp.Symbol, float]
) -> np.ndarray:
    return np.array(sympy_matrix.subs(substitutions).tolist(), dtype=float)


def test_matrix_function_snippet_uses_nonzero_entries_and_materializes_correctly():
    app = _make_app()
    size, c_entries, _ = app._compute_matrix_entries()

    lines, params = app._matrix_function_snippet(
        "C_matrix_entries",
        "C_matrix_triplets",
        "C_matrix_sparse",
        "C_matrix_func",
        size,
        c_entries,
    )

    assert params == ["C1", "C2", "Cg"]
    assert any("def C_matrix_entries" in line for line in lines)
    assert any("def C_matrix_triplets" in line for line in lines)
    assert any("def C_matrix_sparse" in line for line in lines)
    assert any(
        "return _sparse_matrix_from_entries(entries, (3, 3))" in line for line in lines
    )
    assert any(
        "return _dense_matrix_from_entries(entries, (3, 3))" in line for line in lines
    )

    namespace: dict[str, object] = {}
    exec(
        "import math\nimport numpy as np\nfrom scipy import sparse\n\n"
        + "\n".join(app._matrix_snippet_support_functions())
        + "\n\n"
        + "\n".join(lines),
        namespace,
    )

    kwargs = {"C1": 1.0, "C2": 2.0, "Cg": 3.0}
    entries = namespace["C_matrix_entries"](**kwargs)
    rows, cols, data, shape = namespace["C_matrix_triplets"](**kwargs)
    sparse_matrix = namespace["C_matrix_sparse"](**kwargs)
    matrix = namespace["C_matrix_func"](**kwargs)

    assert entries == [
        (0, 0, 3.0),
        (0, 1, -3.0),
        (1, 0, -3.0),
        (1, 1, 3.0),
        (2, 2, 3.0),
    ]
    assert shape == (3, 3)
    assert np.array_equal(rows, np.array([0, 0, 1, 1, 2]))
    assert np.array_equal(cols, np.array([0, 1, 0, 1, 2]))
    assert np.allclose(data, np.array([3.0, -3.0, -3.0, 3.0, 3.0]))
    assert sparse.isspmatrix_csr(sparse_matrix)
    assert np.allclose(sparse_matrix.toarray(), matrix)
    assert np.allclose(
        matrix,
        np.array(
            [
                [3.0, -3.0, 0.0],
                [-3.0, 3.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        ),
    )


def test_build_snippet_matches_computed_c_and_l_inverse_matrices():
    app = _make_app()
    c_matrix, l_inv_matrix = app._compute_matrices()
    snippet = app._build_snippet()

    assert "from scipy import sparse" in snippet
    assert "def C_matrix_entries" in snippet
    assert "def C_matrix_sparse" in snippet
    assert "def L_inv_matrix_entries" in snippet
    assert "def L_inv_matrix_sparse" in snippet

    namespace: dict[str, object] = {}
    exec(snippet, namespace)

    c_kwargs = {"C1": 1.0, "C2": 2.0, "Cg": 3.0}
    l_kwargs = {"L1_inv": 4.0, "Lg_inv": 5.0}

    c_expected = _numeric_matrix(
        c_matrix,
        {
            sp.Symbol("C1"): c_kwargs["C1"],
            sp.Symbol("C2"): c_kwargs["C2"],
            sp.Symbol("Cg"): c_kwargs["Cg"],
        },
    )
    l_expected = _numeric_matrix(
        l_inv_matrix,
        {
            sp.Symbol("L1_inv"): l_kwargs["L1_inv"],
            sp.Symbol("Lg_inv"): l_kwargs["Lg_inv"],
        },
    )

    c_sparse = namespace["C_matrix_sparse"](**c_kwargs)
    c_dense = namespace["C_matrix_func"](**c_kwargs)
    l_sparse = namespace["L_inv_matrix_sparse"](**l_kwargs)
    l_dense = namespace["L_inv_matrix_func"](**l_kwargs)

    assert sparse.isspmatrix_csr(c_sparse)
    assert sparse.isspmatrix_csr(l_sparse)
    assert np.allclose(c_sparse.toarray(), c_expected)
    assert np.allclose(c_dense, c_expected)
    assert np.allclose(l_sparse.toarray(), l_expected)
    assert np.allclose(l_dense, l_expected)


def test_merge_nodes_in_snapshot_rewires_edges_and_combines_ground_connections():
    c12 = sp.Symbol("C12")
    c13 = sp.Symbol("C13")
    c23 = sp.Symbol("C23")
    cg1 = sp.Symbol("Cg1")
    cg2 = sp.Symbol("Cg2")
    l1 = sp.Symbol("L1")
    l2 = sp.Symbol("L2")
    l1_inv = sp.simplify(1 / l1)
    l2_inv = sp.simplify(1 / l2)

    snapshot = {
        "node_counter": 4,
        "edge_counter": 15,
        "view_scale": 1.0,
        "nodes": [
            {"identifier": 1, "name": "N1", "x": 0.0, "y": 0.0},
            {"identifier": 2, "name": "N2", "x": 10.0, "y": 0.0},
            {"identifier": 3, "name": "N3", "x": 20.0, "y": 0.0},
        ],
        "edges": [
            {
                "identifier": 10,
                "nodes": [1, 2],
                "capacitance_expr": c12,
                "capacitance_text": "C12",
                "inductance_expr": None,
                "inductance_text": None,
                "l_inverse_expr": None,
                "is_ground": False,
                "ground_offset_x": 0.0,
                "ground_offset_y": 0.0,
            },
            {
                "identifier": 11,
                "nodes": [2, 3],
                "capacitance_expr": c23,
                "capacitance_text": "C23",
                "inductance_expr": None,
                "inductance_text": None,
                "l_inverse_expr": None,
                "is_ground": False,
                "ground_offset_x": 0.0,
                "ground_offset_y": 0.0,
            },
            {
                "identifier": 12,
                "nodes": [1, 3],
                "capacitance_expr": c13,
                "capacitance_text": "C13",
                "inductance_expr": None,
                "inductance_text": None,
                "l_inverse_expr": None,
                "is_ground": False,
                "ground_offset_x": 0.0,
                "ground_offset_y": 0.0,
            },
            {
                "identifier": 13,
                "nodes": [1, GROUND_NODE_ID],
                "capacitance_expr": cg1,
                "capacitance_text": "Cg1",
                "inductance_expr": l1,
                "inductance_text": "L1",
                "l_inverse_expr": l1_inv,
                "is_ground": True,
                "ground_offset_x": 0.0,
                "ground_offset_y": 104.0,
            },
            {
                "identifier": 14,
                "nodes": [2, GROUND_NODE_ID],
                "capacitance_expr": cg2,
                "capacitance_text": "Cg2",
                "inductance_expr": l2,
                "inductance_text": "L2",
                "l_inverse_expr": l2_inv,
                "is_ground": True,
                "ground_offset_x": 20.0,
                "ground_offset_y": 104.0,
            },
        ],
        "selected_nodes": [1, 2],
        "focus_node": 1,
        "selected_node": None,
        "mode": None,
    }

    merged_snapshot, summary = CircuitGraphApp._merge_nodes_in_snapshot(
        snapshot, 1, {1, 2}
    )

    assert {node["identifier"] for node in merged_snapshot["nodes"]} == {1, 3}
    assert merged_snapshot["selected_nodes"] == [1]
    assert merged_snapshot["focus_node"] == 1
    assert merged_snapshot["selected_node"] is None

    non_ground_edges = [
        edge for edge in merged_snapshot["edges"] if not edge.get("is_ground")
    ]
    ground_edges = [edge for edge in merged_snapshot["edges"] if edge.get("is_ground")]

    assert len(non_ground_edges) == 2
    assert sorted(tuple(edge["nodes"]) for edge in non_ground_edges) == [(1, 3), (1, 3)]

    assert len(ground_edges) == 1
    ground_edge = ground_edges[0]
    assert ground_edge["identifier"] == 13
    assert ground_edge["nodes"] == [1, GROUND_NODE_ID]
    assert sp.simplify(ground_edge["capacitance_expr"] - (cg1 + cg2)) == 0
    assert sp.simplify(ground_edge["l_inverse_expr"] - (l1_inv + l2_inv)) == 0
    assert (
        sp.simplify(ground_edge["inductance_expr"] - sp.simplify(1 / (l1_inv + l2_inv)))
        == 0
    )

    assert summary == {
        "merged_nodes": 1,
        "rewired_edges": 3,
        "removed_self_loops": 1,
        "combined_ground_edges": 1,
    }


def test_merge_nodes_in_snapshot_keeps_parallel_non_ground_edges():
    c13 = sp.Symbol("C13")
    c23 = sp.Symbol("C23")

    snapshot = {
        "node_counter": 4,
        "edge_counter": 13,
        "view_scale": 1.0,
        "nodes": [
            {"identifier": 1, "name": "N1", "x": 0.0, "y": 0.0},
            {"identifier": 2, "name": "N2", "x": 10.0, "y": 0.0},
            {"identifier": 3, "name": "N3", "x": 20.0, "y": 0.0},
        ],
        "edges": [
            {
                "identifier": 11,
                "nodes": [2, 3],
                "capacitance_expr": c23,
                "capacitance_text": "C23",
                "inductance_expr": None,
                "inductance_text": None,
                "l_inverse_expr": None,
                "is_ground": False,
                "ground_offset_x": 0.0,
                "ground_offset_y": 0.0,
            },
            {
                "identifier": 12,
                "nodes": [1, 3],
                "capacitance_expr": c13,
                "capacitance_text": "C13",
                "inductance_expr": None,
                "inductance_text": None,
                "l_inverse_expr": None,
                "is_ground": False,
                "ground_offset_x": 0.0,
                "ground_offset_y": 0.0,
            },
        ],
        "selected_nodes": [1, 2],
        "focus_node": 1,
        "selected_node": None,
        "mode": None,
    }

    merged_snapshot, summary = CircuitGraphApp._merge_nodes_in_snapshot(
        snapshot, 1, {1, 2}
    )

    non_ground_edges = [
        edge for edge in merged_snapshot["edges"] if not edge.get("is_ground")
    ]
    assert len(non_ground_edges) == 2
    assert sorted(edge["identifier"] for edge in non_ground_edges) == [11, 12]
    assert sorted(tuple(edge["nodes"]) for edge in non_ground_edges) == [(1, 3), (1, 3)]
    assert summary["combined_ground_edges"] == 0

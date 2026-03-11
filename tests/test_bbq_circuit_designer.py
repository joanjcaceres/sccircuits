import numpy as np
import sympy as sp

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


def test_matrix_function_snippet_uses_triplets_and_materializes_correctly():
    app = _make_app()
    size, c_entries, _ = app._compute_matrix_entries()

    lines, params = app._matrix_function_snippet(
        "C_matrix_triplets", "C_matrix_func", size, c_entries
    )

    assert params == ["C1", "C2", "Cg"]
    assert any("def C_matrix_triplets" in line for line in lines)
    assert any("np.add.at(matrix, (rows, cols), data)" in line for line in lines)
    assert all("return np.array([" not in line for line in lines)

    namespace: dict[str, object] = {}
    exec("import math\nimport numpy as np\n\n" + "\n".join(lines), namespace)

    kwargs = {"C1": 1.0, "C2": 2.0, "Cg": 3.0}
    rows, cols, data, shape = namespace["C_matrix_triplets"](**kwargs)
    matrix = namespace["C_matrix_func"](**kwargs)

    assert shape == (3, 3)
    assert np.array_equal(rows, np.array([0, 0, 1, 1, 2]))
    assert np.array_equal(cols, np.array([0, 1, 0, 1, 2]))
    assert np.allclose(data, np.array([3.0, -3.0, -3.0, 3.0, 3.0]))
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

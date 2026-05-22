"""Black-box quantization from circuit capacitance and inductance matrices."""

from __future__ import annotations

from collections.abc import Iterable
from functools import reduce
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.constants import e, hbar  # type: ignore[import-untyped]
from scipy.linalg import cosm, eigh  # type: ignore[import-untyped]
from scipy.sparse import diags  # type: ignore[import-untyped]


FloatArray = NDArray[np.float64]

_CAPACITANCE_RELATIVE_TOLERANCE = 1e-12
_MODE_RELATIVE_TOLERANCE = 1e-12


class BBQ:
    """
    Black-box quantization for superconducting circuits.

    ``BBQ`` starts from the capacitance matrix ``C_matrix`` and inverse
    inductance matrix ``L_inv_matrix`` of a linearized circuit in node-flux
    coordinates. The normal modes are computed from the generalized eigenvalue
    problem:

        L_inv_matrix @ v_k = omega_k**2 * C_matrix @ v_k

    The columns of ``mode_vectors`` form the matrix ``U`` and are normalized
    as:

        U.T @ C_matrix @ U = I

    With ``non_linear_nodes=(node_a, node_b)``, the nonlinear branch phase uses
    the direction ``Phi_b - Phi_a``. Reversing the node order reverses the sign
    of ``phase_zpf_list``.

    For the full derivation, rendered equations, units, and branch convention,
    see ``docs/theory/circuit-matrix-quantization.md`` and
    ``docs/api/bbq.md`` in the project documentation.

    Parameters
    ----------
    C_matrix
        Symmetric capacitance matrix in Farads with shape ``(n, n)``.
    L_inv_matrix
        Symmetric inverse inductance matrix in inverse Henries with the same
        shape as ``C_matrix``.
    non_linear_nodes
        Node indices defining the nonlinear branch. The common two-node form
        ``(node_a, node_b)`` uses branch phase ``Phi_b - Phi_a``. A single node
        ``(node,)`` means phase from ground to that node.

    Attributes
    ----------
    linear_modes
        Positive angular frequencies ``omega_k`` in rad/s.
    linear_modes_GHz
        Frequencies ``omega_k / (2*pi)`` in GHz.
    mode_vectors
        C-normalized normal-mode vectors as columns.
    phase_zpf_list
        Dimensionless phase zero-point fluctuations for the nonlinear branch.

    Examples
    --------
    >>> import numpy as np
    >>> from sccircuits import BBQ
    >>> C = np.array([[40e-15, -32.9e-15], [-32.9e-15, 40e-15]])
    >>> L_inv = np.array([[1 / 1.23e-9, 0.0], [0.0, 1 / 1.23e-9]])
    >>> bbq = BBQ(C, L_inv, non_linear_nodes=(0, 1))
    >>> bbq.linear_modes_GHz.shape == bbq.phase_zpf_list.shape
    True
    """

    def __init__(
        self,
        C_matrix: np.ndarray,
        L_inv_matrix: np.ndarray,
        non_linear_nodes: tuple = (-1, 0),
    ) -> None:
        self.C_matrix = self._as_valid_matrix(C_matrix, "C_matrix")
        self.L_inv_matrix = self._as_valid_matrix(L_inv_matrix, "L_inv_matrix")

        if self.C_matrix.shape != self.L_inv_matrix.shape:
            raise ValueError(
                "C_matrix and L_inv_matrix must have the same shape."
            )

        self.circuit_dimensions = self.C_matrix.shape[0]
        self.non_linear_nodes = self._validate_nodes(non_linear_nodes)

        (
            self._capacitance_basis,
            self._capacitance_eigenvalues,
        ) = self._capacitance_physical_subspace()

        (
            self.mode_eigenvalues,
            self.mode_vectors,
            self._reduced_mode_vectors,
        ) = self._solve_generalized_modes()

        self.linear_modes = self._linear_modes()
        self.phase_zpf_list = self._phase_zpf_list()
        self.linear_modes_GHz = self._linear_modes_GHz()

    @staticmethod
    def _as_valid_matrix(matrix: np.ndarray, name: str) -> FloatArray:
        """Return a finite symmetric square matrix as an array."""
        array = np.asarray(matrix, dtype=float)

        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(f"{name} must be a square 2D matrix.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        if not np.allclose(array, array.T, rtol=1e-10, atol=1e-30):
            raise ValueError(f"{name} must be symmetric.")

        return array

    def _validate_nodes(self, nodes: tuple) -> tuple[int, ...]:
        """Validate nonlinear branch nodes and normalize negative indices."""
        if not isinstance(nodes, tuple):
            raise ValueError(
                "non_linear_nodes must be a tuple of node indices."
            )
        if len(nodes) not in (1, 2):
            raise ValueError(
                "non_linear_nodes must contain one or two node indices."
            )

        normalized_nodes = []
        for node in nodes:
            if not isinstance(node, (int, np.integer)):
                raise ValueError(
                    "non_linear_nodes must contain integer indices."
                )

            normalized = int(node)
            if normalized < 0:
                normalized += self.circuit_dimensions
            if normalized < 0 or normalized >= self.circuit_dimensions:
                raise ValueError(
                    "non_linear_nodes contains an index outside the circuit."
                )
            normalized_nodes.append(normalized)

        return tuple(normalized_nodes)

    def _capacitance_physical_subspace(
        self,
    ) -> tuple[FloatArray, FloatArray]:
        """
        Calculate the physical capacitance subspace.

        Tiny or null capacitance directions are projected out before solving
        the generalized eigenproblem. Negative capacitance directions beyond
        the numerical tolerance are rejected because they do not define a
        physical kinetic energy.
        """
        eigvals_C, eigvecs_C = eigh(self.C_matrix)
        scale = float(np.max(np.abs(eigvals_C)))
        tolerance = (
            _CAPACITANCE_RELATIVE_TOLERANCE * scale
            if scale > 0.0
            else _CAPACITANCE_RELATIVE_TOLERANCE
        )

        if np.any(eigvals_C < -tolerance):
            raise ValueError("C_matrix must be positive semidefinite.")

        physical_idx = eigvals_C > tolerance
        if not np.any(physical_idx):
            raise ValueError(
                "C_matrix has no positive physical capacitance subspace."
            )

        physical_eigvals = eigvals_C[physical_idx]
        physical_basis = eigvecs_C[:, physical_idx]

        return physical_basis, physical_eigvals

    @property
    def C_inv_sqrt(self) -> FloatArray:
        """
        Compatibility inverse square root of the capacitance matrix.

        This quantity is not needed by the primary generalized-eigenproblem
        calculation. It is built lazily for existing code that inspects the
        legacy mass-weighted representation.
        """
        if not hasattr(self, "_C_inv_sqrt"):
            inverse_sqrt_capacitance = np.diag(
                1.0 / np.sqrt(self._capacitance_eigenvalues)
            )
            self._C_inv_sqrt = (
                self._capacitance_basis
                @ inverse_sqrt_capacitance
                @ self._capacitance_basis.T
            )

        return self._C_inv_sqrt

    @property
    def dynamical_matrix(self) -> FloatArray:
        """
        Legacy mass-weighted matrix kept for compatibility and diagnostics.

        The scientifically primary calculation solves
        ``L_inv v = omega**2 C v`` directly. This matrix is computed only when
        requested by older analysis code.
        """
        if not hasattr(self, "_dynamical_matrix_cache"):
            self._dynamical_matrix_cache = (
                self.C_inv_sqrt @ self.L_inv_matrix @ self.C_inv_sqrt
            )

        return self._dynamical_matrix_cache

    @property
    def eigensys_dynamical_matrix(self) -> tuple[FloatArray, FloatArray]:
        """
        Legacy eigenvalues and vectors of :attr:`dynamical_matrix`.

        The eigenvalues match ``mode_eigenvalues``. The vectors are the
        mass-weighted representation corresponding to ``mode_vectors``.
        """
        return self.mode_eigenvalues, self._mass_weighted_mode_vectors()

    def _mass_weighted_mode_vectors(self) -> FloatArray:
        """Return mode vectors in the legacy mass-weighted representation."""
        sqrt_capacitance = np.diag(np.sqrt(self._capacitance_eigenvalues))
        return self._capacitance_basis @ (
            sqrt_capacitance @ self._reduced_mode_vectors
        )

    def _solve_generalized_modes(
        self,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Solve ``L_inv v = omega**2 C v`` on the physical capacitance subspace.

        Returns
        -------
        tuple of ndarray
            ``(omega_squared, mode_vectors, reduced_mode_vectors)``. The full
            mode vectors are C-normalized node-flux mode shapes, and the
            reduced vectors are the same modes expressed in the positive
            capacitance eigenbasis.
        """
        reduced_l_inv = (
            self._capacitance_basis.T
            @ self.L_inv_matrix
            @ self._capacitance_basis
        )
        reduced_capacitance = np.diag(self._capacitance_eigenvalues)

        eigenvalues, reduced_vectors = eigh(
            reduced_l_inv,
            reduced_capacitance,
        )
        scale = float(np.max(np.abs(eigenvalues)))
        tolerance = (
            _MODE_RELATIVE_TOLERANCE * scale
            if scale > 0.0
            else _MODE_RELATIVE_TOLERANCE
        )

        if np.any(eigenvalues < -tolerance):
            raise ValueError(
                "L_inv_matrix produces negative omega^2 modes on the physical "
                "capacitance subspace."
            )

        positive_idx = np.isfinite(eigenvalues) & (eigenvalues > tolerance)
        if not np.any(positive_idx):
            raise ValueError("No positive finite circuit modes were found.")

        omega_squared = eigenvalues[positive_idx]
        reduced_vectors = reduced_vectors[:, positive_idx]
        mode_vectors = self._capacitance_basis @ reduced_vectors

        return omega_squared, mode_vectors, reduced_vectors

    def _linear_modes(self) -> FloatArray:
        """
        Calculate angular normal-mode frequencies in rad/s.

        Returns
        -------
        ndarray
            Positive angular frequencies ``omega_k``.
        """
        return np.sqrt(self.mode_eigenvalues)

    def _linear_modes_GHz(self) -> FloatArray:
        """
        Convert normal-mode frequencies from angular rad/s to GHz.

        Returns
        -------
        ndarray
            Frequencies ``omega_k / (2*pi)`` in GHz.
        """
        return self.linear_modes / (2.0 * np.pi * 1e9)

    def _phase_zpf_list(self) -> FloatArray:
        """
        Calculate branch phase zero-point fluctuations.

        Returns
        -------
        ndarray
            Dimensionless phase fluctuations across ``non_linear_nodes``.
            Reversing the branch direction flips the sign of every entry.
        """
        phi_0 = hbar / (2.0 * e)
        zpf_flux = np.sqrt(hbar / (2.0 * self.linear_modes))

        if len(self.non_linear_nodes) == 2:
            node_a, node_b = self.non_linear_nodes
            branch_vector = (
                self.mode_vectors[node_b, :] - self.mode_vectors[node_a, :]
            )
        else:
            (node,) = self.non_linear_nodes
            branch_vector = self.mode_vectors[node, :]

        return np.asarray(branch_vector * zpf_flux / phi_0, dtype=float)

    def plot_linear_modes(
        self,
        which: Union[list[int], int] = 0,
    ) -> None:
        """
        Plot selected C-normalized linear mode shapes.

        Parameters
        ----------
        which
            Mode index or list of mode indices to plot.
        """
        if isinstance(which, int):
            which = [which]

        for mode_index in which:
            self._validate_mode_index(mode_index)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            self.mode_vectors[:, which],
            linestyle="-",
            label=[
                (
                    rf"$f_{i} = {self.linear_modes_GHz[i]:.2f}$ GHz, "
                    rf"$\varphi_{{zpf}} = {self.phase_zpf_list[i]:.1e}$"
                )
                for i in which
            ],
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Node index")
        ax.set_ylabel("C-normalized mode amplitude")
        ax.set_title("Linear Modes of the Circuit")
        ax.legend()
        plt.show()

    @property
    def selected_modes(self) -> list[int]:
        """Mode indices retained for Hamiltonian construction."""
        self._require_selected_modes()
        return self._selected_modes

    @selected_modes.setter
    def selected_modes(self, modes: Iterable[int] | int) -> None:
        raw_modes: tuple[int, ...]
        if isinstance(modes, (int, np.integer)):
            raw_modes = (int(modes),)
        elif (
            isinstance(modes, (str, bytes))
            or not isinstance(modes, Iterable)
        ):
            raise ValueError(
                "selected_modes must be an integer or a non-empty sequence "
                "of indices."
            )
        else:
            raw_modes = tuple(modes)

        if len(raw_modes) == 0:
            raise ValueError(
                "selected_modes must contain at least one mode index."
            )

        selected = []
        for mode in raw_modes:
            selected.append(self._validate_mode_index(mode))

        self._selected_modes = selected

    @property
    def dimensions(self) -> tuple[int, ...]:
        """Hilbert-space dimensions for ``selected_modes``."""
        if not hasattr(self, "_dimensions"):
            raise ValueError("Set dimensions before reading dimensions.")
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dim: tuple[int, ...] | list[int] | int) -> None:
        self._require_selected_modes()

        if isinstance(dim, (int, np.integer)):
            dim = (int(dim),)
        else:
            dim = tuple(int(value) for value in dim)

        if len(self.selected_modes) != len(dim):
            raise ValueError(
                "The amount of dimensions must match the number of "
                "selected modes."
            )
        if any(value <= 0 for value in dim):
            raise ValueError("All dimensions must be positive integers.")

        self._dimensions = dim
        self._total_dimension = int(np.prod(dim))

    def _validate_mode_index(self, mode_index: int) -> int:
        """Return a valid normalized mode index or raise ``ValueError``."""
        if not isinstance(mode_index, (int, np.integer)):
            raise ValueError("Mode indices must be integers.")

        normalized = int(mode_index)
        if normalized < 0:
            normalized += len(self.linear_modes)
        if normalized < 0 or normalized >= len(self.linear_modes):
            raise ValueError(
                "Mode index is outside the available linear modes."
            )

        return normalized

    def _require_selected_modes(self) -> None:
        if not hasattr(self, "_selected_modes"):
            raise ValueError(
                "Set selected_modes before configuring dimensions."
            )

    def _require_hamiltonian_basis(self) -> None:
        self._require_selected_modes()
        if not hasattr(self, "_dimensions"):
            raise ValueError("Set dimensions before building a Hamiltonian.")

    def hamiltonian_0(self) -> FloatArray:
        """
        Calculate the harmonic Hamiltonian in GHz.

        The diagonal contains ``f_k * (n_k + 1/2)`` for each selected mode,
        with ``f_k = omega_k / (2*pi)`` in GHz. The zero-point offset is
        included; transition-frequency calculations can subtract the ground
        state energy when only spacings matter.

        Returns
        -------
        ndarray
            Dense harmonic Hamiltonian on the selected tensor-product basis.
        """
        self._require_hamiltonian_basis()
        hamiltonian_0 = np.zeros(
            (self._total_dimension, self._total_dimension)
        )

        for idx, dimension in enumerate(self.dimensions):
            mode_index = self.selected_modes[idx]
            energy_GHz = self.linear_modes_GHz[mode_index]
            diagonal = energy_GHz * (np.arange(dimension) + 0.5)
            hamiltonian_0_subspace = diags([diagonal], [0]).toarray()

            factors = [
                hamiltonian_0_subspace if i == idx else np.eye(dim)
                for i, dim in enumerate(self.dimensions)
            ]

            hamiltonian_0 += self._kron_all(factors)

        return hamiltonian_0

    @property
    def Ej_suppression_factor(self) -> float:
        """
        Josephson-energy renormalization from modes omitted from the basis.

        Returns
        -------
        float
            ``exp(-0.5 * sum(phi_zpf_unselected**2))``.
        """
        self._require_selected_modes()
        all_modes_indices = np.arange(len(self.linear_modes))
        unselected_modes_indices = np.setdiff1d(
            all_modes_indices,
            self.selected_modes,
        )
        unselected_zpf = self.phase_zpf_list[unselected_modes_indices]
        return float(np.exp(-0.5 * np.sum(unselected_zpf**2)))

    @property
    def Ej_supression_factor(self) -> float:
        """
        Compatibility alias for :attr:`Ej_suppression_factor`.

        This misspelled name existed in earlier releases and remains available
        so existing analysis notebooks continue to run.
        """
        return self.Ej_suppression_factor

    @property
    def phase_operator_nl(self) -> list[FloatArray]:
        """
        Phase operators for selected modes on their truncated Fock spaces.

        Returns
        -------
        list of ndarray
            One dense phase operator per selected mode.
        """
        self._require_hamiltonian_basis()
        phase_operator_nl_list = []
        for idx, dimension in enumerate(self.dimensions):
            mode_index = self.selected_modes[idx]
            data = np.sqrt(np.arange(1, dimension))
            phase_operator = (
                self.phase_zpf_list[mode_index]
                * diags([data, data], [1, -1]).toarray()
            )
            phase_operator_nl_list.append(phase_operator)
        return phase_operator_nl_list

    @staticmethod
    def _kron_all(factors: list[FloatArray]) -> FloatArray:
        """Return the Kronecker product of all factors in order."""
        return reduce(
            lambda left, right: np.asarray(np.kron(left, right), dtype=float),
            factors,
        )

    def hamiltonian_nl(self, Ej: float, phi_ext: float) -> FloatArray:
        """
        Calculate the nonlinear Josephson Hamiltonian in GHz.

        Parameters
        ----------
        Ej
            Josephson energy in GHz.
        phi_ext
            External phase bias in radians.

        Returns
        -------
        ndarray
            Dense nonlinear Hamiltonian
            ``-Ej * (suppression*cos(phi + phi_ext) + phi**2/2)``.
        """
        self._require_hamiltonian_basis()
        phi_nl_total = np.zeros(
            (self._total_dimension, self._total_dimension)
        )
        phase_operators = self.phase_operator_nl

        for idx, phase_operator in enumerate(phase_operators):
            factors = [
                phase_operator if i == idx else np.eye(dim)
                for i, dim in enumerate(self.dimensions)
            ]
            phi_nl_total += self._kron_all(factors)

        identity = np.eye(self._total_dimension)
        cos_term = np.asarray(
            cosm(phi_nl_total + phi_ext * identity),
            dtype=float,
        )
        return np.asarray(
            -Ej
            * (
                self.Ej_suppression_factor * cos_term
                + 0.5 * phi_nl_total @ phi_nl_total
            ),
            dtype=float,
        )


if __name__ == "__main__":
    # Example usage
    from .utilities import El_to_L

    N = 120
    Cjb = 40e-15
    Cj = 32.9e-15
    Cg = 0

    Ejb = 7.5e9
    Lj = 1.23e-9
    Ljb = El_to_L(Ejb)

    c_diagonal = (Cg + 2 * Cj) * np.ones(N + 1)
    c_diagonal[0] = c_diagonal[-1] = Cjb + Cj + Cg
    c_off_diagonal = -Cj * np.ones(N)
    C_matrix = diags(
        [c_diagonal, c_off_diagonal, c_off_diagonal],
        [0, 1, -1],
    ).toarray()
    C_matrix[-1, 0] = C_matrix[0, -1] = -Cjb

    L_inv_diagonal = (2 / Lj) * np.ones(N + 1)
    L_inv_diagonal[0] = L_inv_diagonal[-1] = 1 / Lj + 1 / Ljb
    L_inv_off_diagonal = (-1 / Lj) * np.ones(N)
    L_inv_matrix = diags(
        [L_inv_diagonal, L_inv_off_diagonal, L_inv_off_diagonal], [0, 1, -1]
    ).toarray()
    L_inv_matrix[-1, 0] = L_inv_matrix[0, -1] = -1 / Ljb

    circuit = BBQ(
        C_matrix=C_matrix,
        L_inv_matrix=L_inv_matrix,
        non_linear_nodes=(-1, 0),
    )

    print(circuit.linear_modes_GHz[:3])

    circuit.plot_linear_modes(which=[0])

    circuit.selected_modes = [0]
    circuit.dimensions = 100

    hamiltonian_0 = circuit.hamiltonian_0()

    phi_ext_array = np.linspace(0, np.pi, 100)
    evals_array = []

    for phi_ext in phi_ext_array:
        hamiltonian_nl = circuit.hamiltonian_nl(Ej=Ejb / 1e9, phi_ext=phi_ext)
        hamiltonian = hamiltonian_0 + hamiltonian_nl
        evals = eigh(hamiltonian, eigvals_only=True)
        evals_array.append(evals)

    evals_matrix = np.array(evals_array)

    transition_energies = (
        evals_matrix[:, :6] - evals_matrix[:, 0][:, np.newaxis]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(phi_ext_array, transition_energies)
    ax.set_xlabel("External flux (rad)")
    ax.set_ylabel("Transition energy from ground state (GHz)")
    ax.set_title("Energy Levels vs External Flux")
    plt.show()

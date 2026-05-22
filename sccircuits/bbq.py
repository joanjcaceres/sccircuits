"""Black-box quantization from circuit capacitance and inductance matrices."""

from __future__ import annotations

from collections.abc import Iterable
from functools import reduce

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
    the direction ``Phi_b - Phi_a``. A list of such tuples represents multiple
    nonlinear branches. Reversing a branch tuple reverses the sign of that
    branch's zero-point phase fluctuations.

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
        Node indices defining one or more nonlinear branches. The common
        two-node form ``(node_a, node_b)`` uses branch phase ``Phi_b - Phi_a``.
        A single node ``(node,)`` means phase from ground to that node. A list
        such as ``[(0, 1), (2, 3)]`` defines multiple nonlinear branches.

    Attributes
    ----------
    linear_modes
        Positive angular frequencies ``omega_k`` in rad/s.
    linear_modes_GHz
        Frequencies ``omega_k / (2*pi)`` in GHz.
    mode_vectors
        C-normalized normal-mode vectors as columns.
    phase_zpf_list
        Dimensionless phase zero-point fluctuations. This is a vector for one
        nonlinear branch and a branch-by-mode matrix for multiple branches.
    phase_zpf_matrix
        Branch-by-mode matrix of dimensionless phase zero-point fluctuations.

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
        non_linear_nodes: tuple | Iterable[tuple] = (-1, 0),
    ) -> None:
        self.C_matrix = self._as_valid_matrix(C_matrix, "C_matrix")
        self.L_inv_matrix = self._as_valid_matrix(L_inv_matrix, "L_inv_matrix")

        if self.C_matrix.shape != self.L_inv_matrix.shape:
            raise ValueError(
                "C_matrix and L_inv_matrix must have the same shape."
            )

        self.circuit_dimensions = self.C_matrix.shape[0]
        self.nonlinear_branches = self._validate_nonlinear_branches(
            non_linear_nodes
        )
        self.non_linear_nodes = (
            self.nonlinear_branches[0]
            if len(self.nonlinear_branches) == 1
            else self.nonlinear_branches
        )
        self.branch_matrix = self._branch_matrix()

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
        self.phase_zpf_matrix = self._phase_zpf_matrix()
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

    def _validate_nonlinear_branches(
        self,
        nodes: tuple | Iterable[tuple],
    ) -> tuple[tuple[int, ...], ...]:
        """
        Validate nonlinear branch specifications.

        ``non_linear_nodes`` may be a single branch tuple or an iterable of
        branch tuples. A branch has either one node, meaning node-to-ground, or
        two nodes, meaning ``Phi_b - Phi_a``.
        """
        if isinstance(nodes, tuple) and all(
            isinstance(node, (int, np.integer)) for node in nodes
        ):
            return (self._validate_branch_nodes(nodes),)

        if isinstance(nodes, (str, bytes)) or not isinstance(nodes, Iterable):
            raise ValueError(
                "non_linear_nodes must be a branch tuple or an iterable of "
                "branch tuples."
            )

        branches = []
        for branch in nodes:
            if not isinstance(branch, tuple):
                raise ValueError(
                    "Each nonlinear branch must be a tuple of node indices."
                )
            branches.append(self._validate_branch_nodes(branch))

        if len(branches) == 0:
            raise ValueError(
                "non_linear_nodes must contain at least one branch."
            )

        return tuple(branches)

    def _validate_branch_nodes(self, nodes: tuple) -> tuple[int, ...]:
        """Validate one nonlinear branch and normalize negative indices."""
        if len(nodes) not in (1, 2):
            raise ValueError(
                "Each nonlinear branch must contain one or two node indices."
            )

        normalized_nodes = []
        for node in nodes:
            if not isinstance(node, (int, np.integer)):
                raise ValueError(
                    "Each nonlinear branch must contain integer indices."
                )

            normalized = int(node)
            if normalized < 0:
                normalized += self.circuit_dimensions
            if normalized < 0 or normalized >= self.circuit_dimensions:
                raise ValueError(
                    "A nonlinear branch contains an index outside the circuit."
                )
            normalized_nodes.append(normalized)

        return tuple(normalized_nodes)

    def _branch_matrix(self) -> FloatArray:
        """
        Return branch-incidence matrix B matching the documented convention.

        Each row corresponds to one nonlinear branch. For branch
        ``(node_a, node_b)``, ``B[row, node_a] = -1`` and
        ``B[row, node_b] = 1``. For branch ``(node,)``, ``B[row, node] = 1``.
        """
        B = np.zeros(
            (len(self.nonlinear_branches), self.circuit_dimensions),
            dtype=float,
        )
        for branch_index, branch in enumerate(self.nonlinear_branches):
            if len(branch) == 2:
                node_a, node_b = branch
                B[branch_index, node_a] = -1.0
                B[branch_index, node_b] = 1.0
            else:
                (node,) = branch
                B[branch_index, node] = 1.0

        return B

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
        L_inv_reduced = (
            self._capacitance_basis.T
            @ self.L_inv_matrix
            @ self._capacitance_basis
        )
        C_reduced = np.diag(self._capacitance_eigenvalues)

        omega_squared_all, U_reduced_all = eigh(
            L_inv_reduced,
            C_reduced,
        )
        scale = float(np.max(np.abs(omega_squared_all)))
        tolerance = (
            _MODE_RELATIVE_TOLERANCE * scale
            if scale > 0.0
            else _MODE_RELATIVE_TOLERANCE
        )

        if np.any(omega_squared_all < -tolerance):
            raise ValueError(
                "L_inv_matrix produces negative omega^2 modes on the physical "
                "capacitance subspace."
            )

        positive_idx = (
            np.isfinite(omega_squared_all)
            & (omega_squared_all > tolerance)
        )
        if not np.any(positive_idx):
            raise ValueError("No positive finite circuit modes were found.")

        omega_squared = omega_squared_all[positive_idx]
        U_reduced = U_reduced_all[:, positive_idx]
        U = self._capacitance_basis @ U_reduced

        return omega_squared, U, U_reduced

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

    def _phase_zpf_matrix(self) -> FloatArray:
        """
        Calculate branch phase zero-point fluctuations as a matrix.

        Returns
        -------
        ndarray
            Branch-by-mode matrix of dimensionless phase fluctuations. The
            rows follow ``nonlinear_branches`` and the columns follow
            ``linear_modes``.
        """
        phi_0 = hbar / (2.0 * e)
        zpf_flux = np.sqrt(hbar / (2.0 * self.linear_modes))
        branch_mode_amplitudes = self.branch_matrix @ self.mode_vectors

        return np.asarray(
            branch_mode_amplitudes * zpf_flux[np.newaxis, :] / phi_0,
            dtype=float,
        )

    def _phase_zpf_list(self) -> FloatArray:
        """
        Return phase ZPFs in the legacy single-branch-compatible shape.

        For one nonlinear branch this is a vector with one entry per mode. For
        multiple branches this is the same branch-by-mode matrix as
        ``phase_zpf_matrix``.
        """
        if len(self.nonlinear_branches) == 1:
            return self.phase_zpf_matrix[0, :]

        return self.phase_zpf_matrix

    def plot_linear_modes(
        self,
        which: Iterable[int] | int = 0,
    ) -> None:
        """
        Plot selected C-normalized linear mode shapes.

        Parameters
        ----------
        which
            Mode index or list of mode indices to plot.
        """
        if isinstance(which, (int, np.integer)):
            mode_indices = [self._validate_mode_index(which)]
        elif (
            isinstance(which, (str, bytes))
            or not isinstance(which, Iterable)
        ):
            raise ValueError(
                "which must be an integer or sequence of indices."
            )
        else:
            mode_indices = [
                self._validate_mode_index(mode_index)
                for mode_index in which
            ]

        if len(mode_indices) == 0:
            raise ValueError("which must contain at least one mode index.")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            self.mode_vectors[:, mode_indices],
            linestyle="-",
            label=[
                (
                    rf"$f_{i} = {self.linear_modes_GHz[i]:.2f}$ GHz, "
                    rf"$\varphi_{{zpf}} = {self._phase_zpf_label(i)}$"
                )
                for i in mode_indices
            ],
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Node index")
        ax.set_ylabel("C-normalized mode amplitude")
        ax.set_title("Linear Modes of the Circuit")
        ax.legend()
        plt.show()

    def _phase_zpf_label(self, mode_index: int) -> str:
        """Return a compact phase-ZPF label for plotting."""
        zpf_values = self.phase_zpf_matrix[:, mode_index]
        if len(zpf_values) == 1:
            return f"{zpf_values[0]:.1e}"

        return np.array2string(
            zpf_values,
            precision=1,
            separator=", ",
            suppress_small=False,
        )

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
                "Set selected_modes before using mode-dependent properties."
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

    def _Ej_suppression_factors(self) -> FloatArray:
        """
        Josephson-energy renormalization for each nonlinear branch.

        Returns
        -------
        ndarray
            One factor per nonlinear branch:
            ``exp(-0.5 * sum(phi_zpf_unselected**2))``.
        """
        self._require_selected_modes()
        all_modes_indices = np.arange(len(self.linear_modes))
        unselected_modes_indices = np.setdiff1d(
            all_modes_indices,
            self.selected_modes,
        )
        unselected_zpf = self.phase_zpf_matrix[:, unselected_modes_indices]
        return np.asarray(
            np.exp(-0.5 * np.sum(unselected_zpf**2, axis=1)),
            dtype=float,
        )

    @property
    def Ej_suppression_factor(self) -> float | FloatArray:
        """
        Josephson-energy renormalization from modes omitted from the basis.

        Returns
        -------
        float or ndarray
            A scalar for one nonlinear branch, or one factor per branch when
            multiple nonlinear branches are configured.
        """
        factors = self._Ej_suppression_factors()
        if len(factors) == 1:
            return float(factors[0])

        return factors

    @property
    def Ej_supression_factor(self) -> float | FloatArray:
        """
        Compatibility alias for :attr:`Ej_suppression_factor`.

        This misspelled name existed in earlier releases and remains available
        so existing analysis notebooks continue to run.
        """
        return self.Ej_suppression_factor

    @property
    def phase_operator_nl(self) -> list[FloatArray] | list[list[FloatArray]]:
        """
        Phase operators for selected modes on their truncated Fock spaces.

        Returns
        -------
        list
            For one nonlinear branch, one dense phase operator per selected
            mode. For multiple branches, one such list per branch.
        """
        self._require_hamiltonian_basis()
        branch_phase_operators = []
        for branch_index in range(len(self.nonlinear_branches)):
            phase_operator_nl_list = []
            for idx, dimension in enumerate(self.dimensions):
                mode_index = self.selected_modes[idx]
                data = np.sqrt(np.arange(1, dimension))
                phase_operator = (
                    self.phase_zpf_matrix[branch_index, mode_index]
                    * diags([data, data], [1, -1]).toarray()
                )
                phase_operator_nl_list.append(phase_operator)
            branch_phase_operators.append(phase_operator_nl_list)

        if len(branch_phase_operators) == 1:
            return branch_phase_operators[0]

        return branch_phase_operators

    def _branch_phase_operator(self, branch_index: int) -> FloatArray:
        """Return the full Hilbert-space phase operator for one branch."""
        branch_index = self._validate_branch_index(branch_index)
        phi_branch = np.zeros(
            (self._total_dimension, self._total_dimension)
        )

        for idx, dimension in enumerate(self.dimensions):
            mode_index = self.selected_modes[idx]
            data = np.sqrt(np.arange(1, dimension))
            phase_operator = (
                self.phase_zpf_matrix[branch_index, mode_index]
                * diags([data, data], [1, -1]).toarray()
            )
            factors = [
                phase_operator if i == idx else np.eye(dim)
                for i, dim in enumerate(self.dimensions)
            ]
            phi_branch += self._kron_all(factors)

        return phi_branch

    def _validate_branch_index(self, branch_index: int) -> int:
        """Return a valid nonlinear-branch index or raise ``ValueError``."""
        if not isinstance(branch_index, (int, np.integer)):
            raise ValueError("Branch indices must be integers.")

        normalized = int(branch_index)
        if normalized < 0:
            normalized += len(self.nonlinear_branches)
        if normalized < 0 or normalized >= len(self.nonlinear_branches):
            raise ValueError(
                "Branch index is outside the available nonlinear branches."
            )

        return normalized

    def _branch_parameter_array(
        self,
        values: float | Iterable[float],
        name: str,
    ) -> FloatArray:
        """Return one scalar parameter value per nonlinear branch."""
        if isinstance(values, (int, float, np.integer, np.floating)):
            branch_values = np.full(
                len(self.nonlinear_branches),
                float(values),
            )
        elif (
            isinstance(values, (str, bytes))
            or not isinstance(values, Iterable)
        ):
            raise ValueError(
                f"{name} must be a scalar or one value per nonlinear branch."
            )
        else:
            branch_values = np.asarray(tuple(values), dtype=float)

        if branch_values.shape != (len(self.nonlinear_branches),):
            raise ValueError(
                f"{name} must contain one value per nonlinear branch."
            )
        if not np.all(np.isfinite(branch_values)):
            raise ValueError(f"{name} must contain only finite values.")

        return branch_values

    @staticmethod
    def _kron_all(factors: list[FloatArray]) -> FloatArray:
        """Return the Kronecker product of all factors in order."""
        return reduce(
            lambda left, right: np.asarray(np.kron(left, right), dtype=float),
            factors,
        )

    def hamiltonian_nl(
        self,
        Ej: float | Iterable[float],
        phi_ext: float | Iterable[float],
    ) -> FloatArray:
        """
        Calculate the nonlinear Josephson Hamiltonian in GHz.

        Parameters
        ----------
        Ej
            Josephson energy in GHz. Use a scalar for one nonlinear branch, or
            one value per branch when multiple branches are configured.
        phi_ext
            External phase bias in radians. Use a scalar for one nonlinear
            branch, or one value per branch when multiple branches are
            configured.

        Returns
        -------
        ndarray
            Dense nonlinear Hamiltonian
            ``sum_b -Ej_b * (suppression_b*cos(phi_b + phi_ext_b)
            + phi_b**2/2)``.
        """
        self._require_hamiltonian_basis()
        Ej_values = self._branch_parameter_array(Ej, "Ej")
        phi_ext_values = self._branch_parameter_array(phi_ext, "phi_ext")
        suppression_factors = self._Ej_suppression_factors()
        hamiltonian_nl = np.zeros(
            (self._total_dimension, self._total_dimension)
        )
        identity = np.eye(self._total_dimension)

        for branch_index, Ej_value in enumerate(Ej_values):
            phi_branch = self._branch_phase_operator(branch_index)
            cos_term = np.asarray(
                cosm(phi_branch + phi_ext_values[branch_index] * identity),
                dtype=float,
            )
            hamiltonian_nl += -Ej_value * (
                suppression_factors[branch_index] * cos_term
                + 0.5 * phi_branch @ phi_branch
            )

        return np.asarray(hamiltonian_nl, dtype=float)


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

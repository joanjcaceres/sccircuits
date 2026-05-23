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

    ``BBQ`` starts from the capacitance matrix ``capacitance_matrix`` and
    inverse inductance matrix ``inverse_inductance_matrix`` of a linearized
    circuit in node-flux coordinates. The normal modes are computed from the
    generalized eigenvalue problem:

        inverse_inductance_matrix @ v_k = omega_k**2 * capacitance_matrix @ v_k

    The columns of ``normal_mode_vectors`` form the matrix ``U`` and are
    normalized as:

        U.T @ capacitance_matrix @ U = I

    With ``nonlinear_branches=(node_a, node_b)``, the nonlinear branch phase
    uses the direction ``Phi_b - Phi_a``. A list of such tuples represents
    multiple nonlinear branches. Reversing a branch tuple reverses the sign of
    that branch's zero-point phase fluctuations.

    For the full derivation, rendered equations, units, and branch convention,
    see ``docs/theory/circuit-matrix-quantization.md`` and
    ``docs/api/bbq.md`` in the project documentation.

    Parameters
    ----------
    capacitance_matrix
        Symmetric capacitance matrix in Farads with shape ``(n, n)``.
    inverse_inductance_matrix
        Symmetric inverse inductance matrix in inverse Henries with the same
        shape as ``capacitance_matrix``.
    nonlinear_branches
        Node indices defining one or more nonlinear branches. The common
        two-node form ``(node_a, node_b)`` uses branch phase ``Phi_b - Phi_a``.
        A single node ``(node,)`` means phase from ground to that node. A list
        such as ``[(0, 1), (2, 3)]`` defines multiple nonlinear branches.

    Attributes
    ----------
    angular_frequencies
        Positive angular frequencies ``omega_k`` in rad/s.
    frequencies_ghz
        Frequencies ``omega_k / (2*pi)`` in GHz.
    normal_mode_vectors
        C-normalized normal-mode vectors as columns.
    branch_phase_zpfs
        Branch-by-mode matrix of dimensionless phase zero-point fluctuations.

    Examples
    --------
    >>> import numpy as np
    >>> from sccircuits import BBQ
    >>> C = np.array([[40e-15, -32.9e-15], [-32.9e-15, 40e-15]])
    >>> L_inv = np.array([[1 / 1.23e-9, 0.0], [0.0, 1 / 1.23e-9]])
    >>> bbq = BBQ(C, L_inv, nonlinear_branches=(0, 1))
    >>> bbq.frequencies_ghz.shape[0] == bbq.branch_phase_zpfs.shape[1]
    True
    """

    def __init__(
        self,
        capacitance_matrix: np.ndarray,
        inverse_inductance_matrix: np.ndarray,
        nonlinear_branches: tuple | Iterable[tuple] = (-1, 0),
    ) -> None:
        self.capacitance_matrix = self._as_valid_matrix(
            capacitance_matrix,
            "capacitance_matrix",
        )
        self.inverse_inductance_matrix = self._as_valid_matrix(
            inverse_inductance_matrix,
            "inverse_inductance_matrix",
        )

        if (
            self.capacitance_matrix.shape
            != self.inverse_inductance_matrix.shape
        ):
            raise ValueError(
                "capacitance_matrix and inverse_inductance_matrix must have "
                "the same shape."
            )

        self.node_count = self.capacitance_matrix.shape[0]
        self.nonlinear_branches = self._validate_nonlinear_branches(
            nonlinear_branches
        )
        self.branch_incidence_matrix = self._branch_incidence_matrix()

        (
            self._capacitance_basis,
            self._capacitance_eigenvalues,
        ) = self._capacitance_physical_subspace()

        (
            self.angular_frequencies_squared,
            self.normal_mode_vectors,
            self._reduced_normal_mode_vectors,
        ) = self._solve_generalized_modes()

        self.angular_frequencies = self._angular_frequencies()
        self.branch_phase_zpfs = self._branch_phase_zpfs()
        self.frequencies_ghz = self._frequencies_ghz()

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

        ``nonlinear_branches`` may be a single branch tuple or an iterable of
        branch tuples. A branch has either one node, meaning node-to-ground, or
        two nodes, meaning ``Phi_b - Phi_a``.
        """
        if isinstance(nodes, tuple) and all(
            isinstance(node, (int, np.integer)) for node in nodes
        ):
            return (self._validate_branch_nodes(nodes),)

        if isinstance(nodes, (str, bytes)) or not isinstance(nodes, Iterable):
            raise ValueError(
                "nonlinear_branches must be a branch tuple or an iterable of "
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
                "nonlinear_branches must contain at least one branch."
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
                normalized += self.node_count
            if normalized < 0 or normalized >= self.node_count:
                raise ValueError(
                    "A nonlinear branch contains an index outside the circuit."
                )
            normalized_nodes.append(normalized)

        return tuple(normalized_nodes)

    def _branch_incidence_matrix(self) -> FloatArray:
        """
        Return branch-incidence matrix B matching the documented convention.

        Each row corresponds to one nonlinear branch. For branch
        ``(node_a, node_b)``, ``B[row, node_a] = -1`` and
        ``B[row, node_b] = 1``. For branch ``(node,)``, ``B[row, node] = 1``.
        """
        B = np.zeros(
            (len(self.nonlinear_branches), self.node_count),
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
        eigvals_C, eigvecs_C = eigh(self.capacitance_matrix)
        scale = float(np.max(np.abs(eigvals_C)))
        tolerance = (
            _CAPACITANCE_RELATIVE_TOLERANCE * scale
            if scale > 0.0
            else _CAPACITANCE_RELATIVE_TOLERANCE
        )

        if np.any(eigvals_C < -tolerance):
            raise ValueError(
                "capacitance_matrix must be positive semidefinite."
            )

        physical_idx = eigvals_C > tolerance
        if not np.any(physical_idx):
            raise ValueError(
                "capacitance_matrix has no positive physical capacitance "
                "subspace."
            )

        physical_eigvals = eigvals_C[physical_idx]
        physical_basis = eigvecs_C[:, physical_idx]

        return physical_basis, physical_eigvals

    def _solve_generalized_modes(
        self,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Solve ``L_inv v = omega**2 C v`` on the physical capacitance subspace.

        Returns
        -------
        tuple of ndarray
            ``(omega_squared, normal_mode_vectors,
            reduced_normal_mode_vectors)``. The full mode vectors are
            C-normalized node-flux mode shapes, and the reduced vectors are the
            same modes expressed in the positive capacitance eigenbasis.
        """
        L_inv_reduced = (
            self._capacitance_basis.T
            @ self.inverse_inductance_matrix
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
                "inverse_inductance_matrix produces negative omega^2 modes on "
                "the physical capacitance subspace."
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

    def _angular_frequencies(self) -> FloatArray:
        """
        Calculate angular normal-mode frequencies in rad/s.

        Returns
        -------
        ndarray
            Positive angular frequencies ``omega_k``.
        """
        return np.sqrt(self.angular_frequencies_squared)

    def _frequencies_ghz(self) -> FloatArray:
        """
        Convert normal-mode frequencies from angular rad/s to GHz.

        Returns
        -------
        ndarray
            Frequencies ``omega_k / (2*pi)`` in GHz.
        """
        return self.angular_frequencies / (2.0 * np.pi * 1e9)

    def _branch_phase_zpfs(self) -> FloatArray:
        """
        Calculate branch phase zero-point fluctuations as a matrix.

        Returns
        -------
        ndarray
            Branch-by-mode matrix of dimensionless phase fluctuations. The
            rows follow ``nonlinear_branches`` and the columns follow
            ``angular_frequencies``.
        """
        phi_0 = hbar / (2.0 * e)
        zpf_flux = np.sqrt(hbar / (2.0 * self.angular_frequencies))
        branch_mode_amplitudes = (
            self.branch_incidence_matrix @ self.normal_mode_vectors
        )

        return np.asarray(
            branch_mode_amplitudes * zpf_flux[np.newaxis, :] / phi_0,
            dtype=float,
        )

    def plot_modes(
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
            self.normal_mode_vectors[:, mode_indices],
            linestyle="-",
            label=[
                (
                    rf"$f_{i} = {self.frequencies_ghz[i]:.2f}$ GHz, "
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
        zpf_values = self.branch_phase_zpfs[:, mode_index]
        if len(zpf_values) == 1:
            return f"{zpf_values[0]:.1e}"

        return np.array2string(
            zpf_values,
            precision=1,
            separator=", ",
            suppress_small=False,
        )

    @property
    def selected_mode_indices(self) -> list[int]:
        """Mode indices retained for Hamiltonian construction."""
        self._require_selected_mode_indices()
        return self._selected_mode_indices

    @selected_mode_indices.setter
    def selected_mode_indices(self, modes: Iterable[int] | int) -> None:
        raw_modes: tuple[int, ...]
        if isinstance(modes, (int, np.integer)):
            raw_modes = (int(modes),)
        elif (
            isinstance(modes, (str, bytes))
            or not isinstance(modes, Iterable)
        ):
            raise ValueError(
                "selected_mode_indices must be an integer or a non-empty "
                "sequence of indices."
            )
        else:
            raw_modes = tuple(modes)

        if len(raw_modes) == 0:
            raise ValueError(
                "selected_mode_indices must contain at least one mode index."
            )

        selected = []
        for mode in raw_modes:
            selected.append(self._validate_mode_index(mode))

        self._selected_mode_indices = selected

    @property
    def truncation_dimensions(self) -> tuple[int, ...]:
        """Hilbert-space dimensions for ``selected_mode_indices``."""
        if not hasattr(self, "_truncation_dimensions"):
            raise ValueError(
                "Set truncation_dimensions before reading them."
            )
        return self._truncation_dimensions

    @truncation_dimensions.setter
    def truncation_dimensions(
        self,
        dimensions: tuple[int, ...] | list[int] | int,
    ) -> None:
        self._require_selected_mode_indices()

        truncation_dimensions: tuple[int, ...]
        if isinstance(dimensions, (int, np.integer)):
            truncation_dimensions = (int(dimensions),)
        else:
            truncation_dimensions = tuple(int(value) for value in dimensions)

        if len(self.selected_mode_indices) != len(truncation_dimensions):
            raise ValueError(
                "The number of truncation_dimensions entries must match the "
                "number of selected_mode_indices."
            )
        if any(value <= 0 for value in truncation_dimensions):
            raise ValueError(
                "All truncation_dimensions values must be positive integers."
            )

        self._truncation_dimensions = truncation_dimensions
        self._total_truncation_dimension = int(np.prod(truncation_dimensions))

    def _validate_mode_index(self, mode_index: int) -> int:
        """Return a valid normalized mode index or raise ``ValueError``."""
        if not isinstance(mode_index, (int, np.integer)):
            raise ValueError("Mode indices must be integers.")

        normalized = int(mode_index)
        if normalized < 0:
            normalized += len(self.angular_frequencies)
        if normalized < 0 or normalized >= len(self.angular_frequencies):
            raise ValueError(
                "Mode index is outside the available linear modes."
            )

        return normalized

    def _require_selected_mode_indices(self) -> None:
        if not hasattr(self, "_selected_mode_indices"):
            raise ValueError(
                "Set selected_mode_indices before using mode-dependent "
                "properties."
            )

    def _require_hamiltonian_basis(self) -> None:
        self._require_selected_mode_indices()
        if not hasattr(self, "_truncation_dimensions"):
            raise ValueError(
                "Set truncation_dimensions before building a Hamiltonian."
            )

    def hamiltonian_linear(self) -> FloatArray:
        """
        Calculate the linear harmonic Hamiltonian in GHz.

        The diagonal contains ``f_k * (n_k + 1/2)`` for each selected mode,
        with ``f_k = omega_k / (2*pi)`` in GHz. The zero-point offset is
        included; transition-frequency calculations can subtract the ground
        state energy when only spacings matter.

        Returns
        -------
        ndarray
            Dense linear Hamiltonian on the selected tensor-product basis.
        """
        self._require_hamiltonian_basis()
        H_linear = np.zeros(
            (
                self._total_truncation_dimension,
                self._total_truncation_dimension,
            )
        )

        for idx, dimension in enumerate(self.truncation_dimensions):
            mode_index = self.selected_mode_indices[idx]
            energy_GHz = self.frequencies_ghz[mode_index]
            diagonal = energy_GHz * (np.arange(dimension) + 0.5)
            H_linear_subspace = diags([diagonal], [0]).toarray()

            factors = [
                H_linear_subspace if i == idx else np.eye(dim)
                for i, dim in enumerate(self.truncation_dimensions)
            ]

            H_linear += self._kron_all(factors)

        return H_linear

    def _josephson_suppression_factors(self) -> FloatArray:
        """
        Josephson-energy renormalization for each nonlinear branch.

        Returns
        -------
        ndarray
            One factor per nonlinear branch:
            ``exp(-0.5 * sum(phi_zpf_unselected**2))``.
        """
        self._require_selected_mode_indices()
        all_modes_indices = np.arange(len(self.angular_frequencies))
        unselected_mode_indices = np.setdiff1d(
            all_modes_indices,
            self.selected_mode_indices,
        )
        unselected_zpf = self.branch_phase_zpfs[:, unselected_mode_indices]
        return np.asarray(
            np.exp(-0.5 * np.sum(unselected_zpf**2, axis=1)),
            dtype=float,
        )

    @property
    def josephson_suppression_factors(self) -> FloatArray:
        """
        Josephson-energy renormalization from modes omitted from the basis.

        Returns
        -------
        ndarray
            One suppression factor per nonlinear branch.
        """
        return self._josephson_suppression_factors()

    @property
    def branch_phase_operators(self) -> list[list[FloatArray]]:
        """
        Phase operators for selected modes on their truncated Fock spaces.

        Returns
        -------
        list
            One list per nonlinear branch. Each branch list contains one dense
            phase operator per selected mode.
        """
        self._require_hamiltonian_basis()
        branch_phase_operators = []
        for branch_index in range(len(self.nonlinear_branches)):
            branch_phase_operators_list = []
            for idx, dimension in enumerate(self.truncation_dimensions):
                mode_index = self.selected_mode_indices[idx]
                data = np.sqrt(np.arange(1, dimension))
                phase_operator = (
                    self.branch_phase_zpfs[branch_index, mode_index]
                    * diags([data, data], [1, -1]).toarray()
                )
                branch_phase_operators_list.append(phase_operator)
            branch_phase_operators.append(branch_phase_operators_list)

        return branch_phase_operators

    def _branch_phase_operator(self, branch_index: int) -> FloatArray:
        """Return the full Hilbert-space phase operator for one branch."""
        branch_index = self._validate_branch_index(branch_index)
        phi_branch = np.zeros(
            (
                self._total_truncation_dimension,
                self._total_truncation_dimension,
            )
        )

        for idx, dimension in enumerate(self.truncation_dimensions):
            mode_index = self.selected_mode_indices[idx]
            data = np.sqrt(np.arange(1, dimension))
            phase_operator = (
                self.branch_phase_zpfs[branch_index, mode_index]
                * diags([data, data], [1, -1]).toarray()
            )
            factors = [
                phase_operator if i == idx else np.eye(dim)
                for i, dim in enumerate(self.truncation_dimensions)
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

    def hamiltonian_nonlinear(
        self,
        josephson_energies: float | Iterable[float],
        external_phases: float | Iterable[float],
    ) -> FloatArray:
        """
        Calculate the nonlinear Josephson Hamiltonian in GHz.

        Parameters
        ----------
        josephson_energies
            Josephson energy in GHz. Use a scalar for one nonlinear branch, or
            one value per branch when multiple branches are configured.
        external_phases
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
        josephson_energy_values = self._branch_parameter_array(
            josephson_energies,
            "josephson_energies",
        )
        external_phase_values = self._branch_parameter_array(
            external_phases,
            "external_phases",
        )
        suppression_factors = self._josephson_suppression_factors()
        hamiltonian_nonlinear = np.zeros(
            (
                self._total_truncation_dimension,
                self._total_truncation_dimension,
            )
        )
        identity = np.eye(self._total_truncation_dimension)

        for branch_index, josephson_energy in enumerate(
            josephson_energy_values
        ):
            phi_branch = self._branch_phase_operator(branch_index)
            cos_term = np.asarray(
                cosm(
                    phi_branch
                    + external_phase_values[branch_index] * identity
                ),
                dtype=float,
            )
            hamiltonian_nonlinear += -josephson_energy * (
                suppression_factors[branch_index] * cos_term
                + 0.5 * phi_branch @ phi_branch
            )

        return np.asarray(hamiltonian_nonlinear, dtype=float)


if __name__ == "__main__":
    # Example usage
    from sccircuits.utilities import El_to_L

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
    capacitance_matrix = diags(
        [c_diagonal, c_off_diagonal, c_off_diagonal],
        [0, 1, -1],
    ).toarray()
    capacitance_matrix[-1, 0] = capacitance_matrix[0, -1] = -Cjb

    L_inv_diagonal = (2 / Lj) * np.ones(N + 1)
    L_inv_diagonal[0] = L_inv_diagonal[-1] = 1 / Lj + 1 / Ljb
    L_inv_off_diagonal = (-1 / Lj) * np.ones(N)
    inverse_inductance_matrix = diags(
        [L_inv_diagonal, L_inv_off_diagonal, L_inv_off_diagonal],
        [0, 1, -1],
    ).toarray()
    inverse_inductance_matrix[-1, 0] = (
        inverse_inductance_matrix[0, -1]
    ) = -1 / Ljb

    circuit = BBQ(
        capacitance_matrix=capacitance_matrix,
        inverse_inductance_matrix=inverse_inductance_matrix,
        nonlinear_branches=(-1, 0),
    )

    print(circuit.frequencies_ghz[:3])

    circuit.plot_modes(which=[0])

    circuit.selected_mode_indices = [0]
    circuit.truncation_dimensions = 100

    H_linear = circuit.hamiltonian_linear()

    phi_ext_array = np.linspace(0, np.pi, 100)
    evals_array = []

    for phi_ext in phi_ext_array:
        hamiltonian_nonlinear = circuit.hamiltonian_nonlinear(
            josephson_energies=Ejb / 1e9,
            external_phases=phi_ext,
        )
        hamiltonian = H_linear + hamiltonian_nonlinear
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

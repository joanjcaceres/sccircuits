"""Black-box quantization from circuit capacitance and inductance matrices."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.constants import e, hbar  # type: ignore[import-untyped]
from scipy.linalg import cosm, eigh, pinvh, solve  # type: ignore[import-untyped]
from scipy.sparse import diags  # type: ignore[import-untyped]


FloatArray = NDArray[np.float64]

_CAPACITANCE_RELATIVE_TOLERANCE = 1e-12
_MODE_RELATIVE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class _LinearModeSolution:
    """Internal result of the reduced linear circuit solve."""

    angular_frequencies_squared: FloatArray
    normal_mode_vectors: FloatArray
    reduced_normal_mode_vectors: FloatArray
    capacitance_basis: FloatArray
    capacitance_eigenvalues: FloatArray


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
    that branch's zero-point phase fluctuations. If neither
    ``nonlinear_branches`` nor ``junctions`` is provided, ``BBQ`` computes the
    linear modes without any nonlinear branch phase rows.

    ``BBQ`` remains a numerical object when cQEDraw junction records are used:
    web-specific identifiers are not retained. Row order is the contract:
    ``branch_phase_nodes[i]``, ``branch_phase_zpfs[i]``, and
    ``josephson_energies_ghz[i]`` correspond to ``junctions[i]``.

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
        such as ``[(0, 1), (2, 3)]`` defines multiple nonlinear branches. Omit
        this argument for a linear circuit with no nonlinear branches.
    junctions
        Optional cQEDraw-style Josephson junction records. Each record must
        include ``phase_positive_index`` and ``phase_negative_index``; ``None``
        means ground. When ``junctions`` is provided, do not also pass
        ``nonlinear_branches``. Records are consumed in input order.

    Attributes
    ----------
    angular_frequencies
        Positive angular frequencies ``omega_k`` in rad/s.
    frequencies_ghz
        Frequencies ``omega_k / (2*pi)`` in GHz.
    normal_mode_vectors
        C-normalized normal-mode vectors as columns.
    branch_phase_nodes
        One ``(positive_node, negative_node)`` pair per nonlinear branch. The
        branch phase is ``Phi_positive - Phi_negative`` and ``None`` means
        ground.
    branch_phase_zpfs
        Branch-by-mode matrix of dimensionless phase zero-point fluctuations.
        Its shape is ``(0, number_of_modes)`` when there are no nonlinear
        branches.
    josephson_energies_ghz
        Josephson energies in GHz when all junction records include
        ``E_j_GHz`` or ``L_j``; otherwise ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> from sccircuits import BBQ
    >>> C = np.array([[40e-15, -32.9e-15], [-32.9e-15, 40e-15]])
    >>> L_inv = np.array([[1 / 1.23e-9, 0.0], [0.0, 1 / 1.23e-9]])
    >>> bbq = BBQ(C, L_inv)
    >>> bbq.branch_phase_zpfs.shape[0]
    0
    """

    def __init__(
        self,
        capacitance_matrix: np.ndarray,
        inverse_inductance_matrix: np.ndarray,
        nonlinear_branches: tuple | Iterable[tuple] | None = None,
        *,
        junctions: Iterable[Mapping[str, object]] | None = None,
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
        if junctions is not None and nonlinear_branches is not None:
            raise ValueError(
                "Pass either junctions or nonlinear_branches, not both."
            )

        josephson_energies_ghz: FloatArray | None = None
        if junctions is not None:
            (
                self.nonlinear_branches,
                self.branch_phase_nodes,
                self.branch_incidence_matrix,
                josephson_energies_ghz,
            ) = self._junction_data_from_records(junctions, self.node_count)
        else:
            if nonlinear_branches is None:
                self.nonlinear_branches = ()
                self.branch_phase_nodes = ()
                self.branch_incidence_matrix = np.zeros(
                    (0, self.node_count),
                    dtype=float,
                )
            else:
                self.nonlinear_branches = self._validate_nonlinear_branches(
                    nonlinear_branches
                )
                self.branch_phase_nodes = (
                    self._branch_phase_nodes_from_nonlinear_branches()
                )
                self.branch_incidence_matrix = self._branch_incidence_matrix()

        linear_modes = self._linear_mode_solution(
            self.capacitance_matrix,
            self.inverse_inductance_matrix,
        )
        self._capacitance_basis = linear_modes.capacitance_basis
        self._capacitance_eigenvalues = linear_modes.capacitance_eigenvalues
        self.angular_frequencies_squared = (
            linear_modes.angular_frequencies_squared
        )
        self.normal_mode_vectors = linear_modes.normal_mode_vectors
        self._reduced_normal_mode_vectors = (
            linear_modes.reduced_normal_mode_vectors
        )

        self.angular_frequencies = self._angular_frequencies()
        self.branch_phase_zpfs = self._branch_phase_zpfs()
        self.frequencies_ghz = self._frequencies_ghz()
        self.josephson_energies_ghz = josephson_energies_ghz

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

    @classmethod
    def _junction_data_from_records(
        cls,
        junctions: Iterable[Mapping[str, object]],
        node_count: int,
    ) -> tuple[
        tuple[tuple[int, ...], ...],
        tuple[tuple[int | None, int | None], ...],
        FloatArray,
        FloatArray | None,
    ]:
        """Validate junction records and build branch tuples plus incidence."""
        if isinstance(junctions, (str, bytes)) or not isinstance(
            junctions, Iterable
        ):
            raise ValueError(
                "junctions must be an iterable of Josephson junction records."
            )

        records = tuple(junctions)
        if len(records) == 0:
            raise ValueError(
                "junctions must contain at least one Josephson junction "
                "record."
            )

        nonlinear_branches = []
        branch_phase_nodes: list[tuple[int | None, int | None]] = []
        josephson_energies: list[float] = []
        has_any_josephson_energy = False
        has_missing_josephson_energy = False
        B = np.zeros((len(records), node_count), dtype=float)

        for branch_index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(
                    "Each Josephson junction record must be a mapping."
                )

            phase_positive_index = cls._record_optional_node_index(
                record,
                "phase_positive_index",
                branch_index,
                node_count,
            )
            phase_negative_index = cls._record_optional_node_index(
                record,
                "phase_negative_index",
                branch_index,
                node_count,
            )
            matrix_nodes = cls._record_matrix_nodes(
                record,
                branch_index,
                node_count,
            )
            cls._validate_record_phase_nodes(
                matrix_nodes,
                phase_positive_index,
                phase_negative_index,
                branch_index,
            )

            if phase_negative_index is not None:
                B[branch_index, phase_negative_index] -= 1.0
            if phase_positive_index is not None:
                B[branch_index, phase_positive_index] += 1.0

            nonlinear_branches.append(
                cls._record_branch_tuple(
                    phase_positive_index,
                    phase_negative_index,
                )
            )
            branch_phase_nodes.append(
                (phase_positive_index, phase_negative_index)
            )

            josephson_energy = cls._record_josephson_energy_ghz(
                record,
                branch_index,
            )
            if josephson_energy is not None:
                has_any_josephson_energy = True
                josephson_energies.append(josephson_energy)
            else:
                has_missing_josephson_energy = True

        if has_any_josephson_energy and has_missing_josephson_energy:
            raise ValueError(
                "Either every Josephson junction record must include E_j_GHz "
                "or L_j, or none of them should."
            )
        josephson_energy_array = (
            np.asarray(josephson_energies, dtype=float)
            if has_any_josephson_energy
            else None
        )
        return (
            tuple(nonlinear_branches),
            tuple(branch_phase_nodes),
            B,
            josephson_energy_array,
        )

    @classmethod
    def _record_matrix_nodes(
        cls,
        record: Mapping[str, object],
        branch_index: int,
        node_count: int,
    ) -> tuple[int | None, int | None] | None:
        if "matrix_nodes" not in record:
            return None

        value = record["matrix_nodes"]
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 2
        ):
            raise ValueError(
                f"Josephson junction record {branch_index} matrix_nodes must "
                "contain two entries."
            )

        return (
            cls._optional_node_index_value(
                value[0],
                "matrix_nodes",
                branch_index,
                node_count,
            ),
            cls._optional_node_index_value(
                value[1],
                "matrix_nodes",
                branch_index,
                node_count,
            ),
        )

    @classmethod
    def _record_optional_node_index(
        cls,
        record: Mapping[str, object],
        key: str,
        branch_index: int,
        node_count: int,
    ) -> int | None:
        value = cls._required_record_key(record, key, branch_index)
        return cls._optional_node_index_value(
            value,
            key,
            branch_index,
            node_count,
        )

    @staticmethod
    def _required_record_key(
        record: Mapping[str, object],
        key: str,
        branch_index: int,
    ) -> object:
        if key not in record:
            raise ValueError(
                f"Josephson junction record {branch_index} is missing required "
                f"value {key!r}."
            )

        return record[key]

    @staticmethod
    def _optional_node_index_value(
        value: object,
        key: str,
        branch_index: int,
        node_count: int,
    ) -> int | None:
        if value is None:
            return None
        if not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"Josephson junction record {branch_index} {key} must contain "
                "integer matrix indices or None."
            )

        node_index = int(value)
        if node_index < 0 or node_index >= node_count:
            raise ValueError(
                f"Josephson junction record {branch_index} {key} contains an "
                "index outside the circuit."
            )

        return node_index

    @staticmethod
    def _validate_record_phase_nodes(
        matrix_nodes: tuple[int | None, int | None] | None,
        phase_positive_index: int | None,
        phase_negative_index: int | None,
        branch_index: int,
    ) -> None:
        if phase_positive_index is None and phase_negative_index is None:
            raise ValueError(
                f"Josephson junction record {branch_index} must define a "
                "positive or negative phase node, with at most one grounded "
                "side."
            )
        if (
            phase_positive_index is not None
            and phase_positive_index == phase_negative_index
        ):
            raise ValueError(
                f"Josephson junction record {branch_index} phase nodes must be "
                "different."
            )

        if matrix_nodes is None:
            return

        matrix_node_set = {node for node in matrix_nodes if node is not None}
        phase_node_set = {
            node
            for node in (phase_positive_index, phase_negative_index)
            if node is not None
        }
        phase_uses_ground = (
            phase_positive_index is None or phase_negative_index is None
        )
        matrix_uses_ground = any(node is None for node in matrix_nodes)
        if phase_uses_ground != matrix_uses_ground:
            raise ValueError(
                f"Josephson junction record {branch_index} must use None for "
                "the grounded side in both matrix_nodes and phase indices."
            )

        if not phase_node_set.issubset(matrix_node_set):
            raise ValueError(
                f"Josephson junction record {branch_index} phase indices must "
                "refer to its matrix_nodes."
            )

    @staticmethod
    def _record_positive_float(
        record: Mapping[str, object],
        key: str,
        branch_index: int,
    ) -> float:
        value = BBQ._required_record_key(record, key, branch_index)
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(
                f"Josephson junction record {branch_index} {key} must be a "
                "finite positive number."
            )

        float_value = float(value)
        if not np.isfinite(float_value) or float_value <= 0.0:
            raise ValueError(
                f"Josephson junction record {branch_index} {key} must be a "
                "finite positive number."
            )

        return float_value

    @classmethod
    def _record_josephson_energy_ghz(
        cls,
        record: Mapping[str, object],
        branch_index: int,
    ) -> float | None:
        if "E_j_GHz" in record and record["E_j_GHz"] is not None:
            return cls._record_positive_float(
                record,
                "E_j_GHz",
                branch_index,
            )
        if "L_j" in record and record["L_j"] is not None:
            josephson_inductance = cls._record_positive_float(
                record,
                "L_j",
                branch_index,
            )
            return cls._josephson_energy_ghz_from_inductance(
                josephson_inductance
            )

        return None

    @staticmethod
    def _josephson_energy_ghz_from_inductance(
        josephson_inductance: float,
    ) -> float:
        reduced_flux_quantum = hbar / (2.0 * e)
        planck_constant = 2.0 * np.pi * hbar
        return float(
            reduced_flux_quantum**2
            / (josephson_inductance * planck_constant * 1e9)
        )

    @staticmethod
    def _record_branch_tuple(
        phase_positive_index: int | None,
        phase_negative_index: int | None,
    ) -> tuple[int, ...]:
        if phase_positive_index is None:
            if phase_negative_index is None:
                raise ValueError(
                    "A Josephson junction record must contain a phase node."
                )
            return (phase_negative_index,)
        if phase_negative_index is None:
            return (phase_positive_index,)

        return (phase_negative_index, phase_positive_index)

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
        if isinstance(nodes, tuple) and len(nodes) == 0:
            return ()
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

        Each row corresponds to one nonlinear branch and follows
        ``branch_phase_nodes``. For ``(positive, negative)``,
        ``B[row, positive] = 1`` and ``B[row, negative] = -1``. ``None`` means
        ground.
        """
        B = np.zeros(
            (len(self.nonlinear_branches), self.node_count),
            dtype=float,
        )
        for branch_index, (positive, negative) in enumerate(
            self.branch_phase_nodes
        ):
            if negative is not None:
                B[branch_index, negative] -= 1.0
            if positive is not None:
                B[branch_index, positive] += 1.0

        return B

    def _branch_phase_nodes_from_nonlinear_branches(
        self,
    ) -> tuple[tuple[int | None, int | None], ...]:
        """Return phase nodes for the manual ``nonlinear_branches`` API."""
        phase_nodes: list[tuple[int | None, int | None]] = []
        for branch in self.nonlinear_branches:
            if len(branch) == 2:
                node_a, node_b = branch
                phase_nodes.append((node_b, node_a))
            else:
                (node,) = branch
                phase_nodes.append((node, None))

        return tuple(phase_nodes)

    @classmethod
    def _linear_mode_solution(
        cls,
        capacitance_matrix: FloatArray,
        inverse_inductance_matrix: FloatArray,
    ) -> _LinearModeSolution:
        """Solve the linear circuit after numerical reductions."""
        (
            reduced_capacitance,
            reduced_inverse_inductance,
            reduced_to_original,
        ) = cls._eliminate_frozen_coordinates(
            capacitance_matrix,
            inverse_inductance_matrix,
        )
        (
            capacitance_basis,
            capacitance_eigenvalues,
        ) = cls._capacitance_physical_subspace(reduced_capacitance)

        capacitance_reduced = np.diag(capacitance_eigenvalues)
        inverse_inductance_reduced = cls._symmetrize(
            capacitance_basis.T
            @ reduced_inverse_inductance
            @ capacitance_basis
        )
        capacitance_basis_original = reduced_to_original @ capacitance_basis

        (
            angular_frequencies_squared,
            reduced_normal_mode_vectors,
        ) = cls._solve_oscillator_modes(
            capacitance_reduced,
            inverse_inductance_reduced,
        )
        normal_mode_vectors = (
            capacitance_basis_original @ reduced_normal_mode_vectors
        )

        return _LinearModeSolution(
            angular_frequencies_squared=angular_frequencies_squared,
            normal_mode_vectors=normal_mode_vectors,
            reduced_normal_mode_vectors=reduced_normal_mode_vectors,
            capacitance_basis=capacitance_basis_original,
            capacitance_eigenvalues=capacitance_eigenvalues,
        )

    @classmethod
    def _eliminate_frozen_coordinates(
        cls,
        capacitance_matrix: FloatArray,
        inverse_inductance_matrix: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Eliminate coordinates with no kinetic energy by minimizing potential.

        A coordinate is classified as frozen only when its entire row and column
        in the capacitance matrix are numerically zero. Its reconstructed value
        is retained in ``reduced_to_original`` so branch phases on eliminated
        coordinates still use the original nodal basis.
        """
        node_count = capacitance_matrix.shape[0]
        capacitance_threshold = cls._matrix_relative_tolerance(
            capacitance_matrix,
            _CAPACITANCE_RELATIVE_TOLERANCE,
        )
        row_norms = np.max(np.abs(capacitance_matrix), axis=1)
        column_norms = np.max(np.abs(capacitance_matrix), axis=0)
        frozen_mask = (
            (row_norms <= capacitance_threshold)
            & (column_norms <= capacitance_threshold)
        )

        if not np.any(frozen_mask):
            return (
                capacitance_matrix.copy(),
                inverse_inductance_matrix.copy(),
                np.eye(node_count),
            )

        dynamic_mask = ~frozen_mask
        dynamic_indices = np.flatnonzero(dynamic_mask)
        frozen_indices = np.flatnonzero(frozen_mask)
        if dynamic_indices.size == 0:
            raise ValueError(
                "capacitance_matrix has no positive physical capacitance "
                "subspace."
            )

        capacitance_reduced = capacitance_matrix[
            np.ix_(dynamic_indices, dynamic_indices)
        ]
        l_dd = inverse_inductance_matrix[np.ix_(dynamic_indices, dynamic_indices)]
        l_df = inverse_inductance_matrix[np.ix_(dynamic_indices, frozen_indices)]
        l_fd = inverse_inductance_matrix[np.ix_(frozen_indices, dynamic_indices)]
        l_ff = inverse_inductance_matrix[np.ix_(frozen_indices, frozen_indices)]

        cls._validate_positive_semidefinite_matrix(
            l_ff,
            "the frozen inverse-inductance block",
        )
        l_ff_pinv = pinvh(l_ff, rtol=_MODE_RELATIVE_TOLERANCE)
        frozen_from_dynamic = l_ff_pinv @ l_fd
        residual = l_ff @ frozen_from_dynamic - l_fd
        if not np.allclose(
            residual,
            0.0,
            rtol=1e-10,
            atol=cls._matrix_relative_tolerance(
                inverse_inductance_matrix,
                _MODE_RELATIVE_TOLERANCE,
            ),
        ):
            raise ValueError(
                "Frozen coordinates are not constrained by the "
                "inverse_inductance_matrix potential block."
            )

        inverse_inductance_reduced = l_dd - l_df @ frozen_from_dynamic
        reduced_to_original = np.zeros(
            (node_count, dynamic_indices.size),
            dtype=float,
        )
        reduced_to_original[dynamic_indices, :] = np.eye(dynamic_indices.size)
        reduced_to_original[frozen_indices, :] = -frozen_from_dynamic

        return (
            cls._symmetrize(capacitance_reduced),
            cls._symmetrize(inverse_inductance_reduced),
            reduced_to_original,
        )

    @classmethod
    def _capacitance_physical_subspace(
        cls,
        capacitance_matrix: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """
        Calculate the physical capacitance subspace.

        Tiny or null capacitance directions are projected out after any frozen
        coordinate reduction. Negative capacitance directions beyond the
        numerical tolerance are rejected because they do not define a physical
        kinetic energy.
        """
        eigvals_C, eigvecs_C = eigh(capacitance_matrix)
        tolerance = cls._values_relative_tolerance(
            eigvals_C,
            _CAPACITANCE_RELATIVE_TOLERANCE,
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

    @classmethod
    def _solve_oscillator_modes(
        cls,
        capacitance_matrix: FloatArray,
        inverse_inductance_matrix: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """
        Isolate zero-potential modes and solve positive oscillator modes.

        The returned mode vectors live in the positive capacitance subspace and
        include the reconstruction induced by zero-potential coordinates.
        """
        stiffness_eigvals, stiffness_eigvecs = eigh(inverse_inductance_matrix)
        stiffness_tolerance = cls._values_relative_tolerance(
            stiffness_eigvals,
            _MODE_RELATIVE_TOLERANCE,
        )
        if np.any(stiffness_eigvals < -stiffness_tolerance):
            raise ValueError(
                "inverse_inductance_matrix produces negative potential modes "
                "on the physical capacitance subspace."
            )

        oscillator_idx = stiffness_eigvals > stiffness_tolerance
        if not np.any(oscillator_idx):
            raise ValueError("No positive finite circuit modes were found.")

        zero_idx = ~oscillator_idx
        oscillator_basis = stiffness_eigvecs[:, oscillator_idx]
        oscillator_stiffness = np.diag(stiffness_eigvals[oscillator_idx])
        oscillator_reconstruction = oscillator_basis
        oscillator_capacitance = cls._symmetrize(
            oscillator_basis.T @ capacitance_matrix @ oscillator_basis
        )

        if np.any(zero_idx):
            zero_basis = stiffness_eigvecs[:, zero_idx]
            c_oo = oscillator_capacitance
            c_oz = oscillator_basis.T @ capacitance_matrix @ zero_basis
            c_zo = c_oz.T
            c_zz = cls._symmetrize(
                zero_basis.T @ capacitance_matrix @ zero_basis
            )
            try:
                zero_from_oscillator = solve(
                    c_zz,
                    c_zo,
                    assume_a="pos",
                    check_finite=True,
                )
            except Exception as exc:
                raise ValueError(
                    "Zero-potential modes must have a positive capacitance "
                    "block to be eliminated."
                ) from exc

            oscillator_capacitance = cls._symmetrize(
                c_oo - c_oz @ zero_from_oscillator
            )
            oscillator_reconstruction = (
                oscillator_basis - zero_basis @ zero_from_oscillator
            )

        try:
            omega_squared_all, oscillator_modes_all = eigh(
                oscillator_stiffness,
                oscillator_capacitance,
            )
        except Exception as exc:
            raise ValueError(
                "The reduced capacitance matrix is not positive definite on "
                "the oscillator subspace."
            ) from exc

        tolerance = cls._values_relative_tolerance(
            omega_squared_all,
            _MODE_RELATIVE_TOLERANCE,
        )

        if np.any(omega_squared_all < -tolerance):
            raise ValueError(
                "inverse_inductance_matrix produces negative omega^2 modes "
                "on the oscillator subspace."
            )

        positive_idx = (
            np.isfinite(omega_squared_all)
            & (omega_squared_all > tolerance)
        )
        if not np.any(positive_idx):
            raise ValueError("No positive finite circuit modes were found.")

        omega_squared = omega_squared_all[positive_idx]
        oscillator_modes = oscillator_modes_all[:, positive_idx]
        reduced_modes = oscillator_reconstruction @ oscillator_modes

        return omega_squared, reduced_modes

    @staticmethod
    def _symmetrize(matrix: FloatArray) -> FloatArray:
        """Return a symmetrized copy of a floating-point matrix."""
        return np.asarray(0.5 * (matrix + matrix.T), dtype=float)

    @staticmethod
    def _matrix_relative_tolerance(matrix: FloatArray, rtol: float) -> float:
        scale = float(np.max(np.abs(matrix))) if matrix.size else 0.0
        return rtol * scale if scale > 0.0 else 0.0

    @staticmethod
    def _values_relative_tolerance(values: FloatArray, rtol: float) -> float:
        scale = float(np.max(np.abs(values))) if values.size else 0.0
        return rtol * scale if scale > 0.0 else 0.0

    @classmethod
    def _validate_positive_semidefinite_matrix(
        cls,
        matrix: FloatArray,
        name: str,
    ) -> None:
        eigvals = eigh(matrix, eigvals_only=True)
        tolerance = cls._values_relative_tolerance(
            eigvals,
            _MODE_RELATIVE_TOLERANCE,
        )
        if np.any(eigvals < -tolerance):
            raise ValueError(f"{name} must be positive semidefinite.")

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
            rows follow ``branch_phase_nodes`` and the columns follow
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

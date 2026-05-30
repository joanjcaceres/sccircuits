"""Matrix reductions used before black-box quantization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve


FloatArray = NDArray[np.float64]

_DEFAULT_RELATIVE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CoordinateReduction:
    """Result of eliminating coordinates from circuit matrices.

    Attributes
    ----------
    capacitance_matrix
        Reduced capacitance matrix.
    inverse_inductance_matrix
        Reduced inverse-inductance matrix.
    kept_indices
        Original coordinate indices kept in the reduced matrices.
    eliminated_indices
        Original coordinate indices eliminated from the reduced matrices.
    kept_mask
        Boolean mask selecting retained coordinates in the original basis.
    eliminated_mask
        Boolean mask selecting eliminated coordinates in the original basis.
    original_to_reduced
        Mapping from original coordinate index to reduced coordinate index for
        retained coordinates.
    reduced_to_original
        Linear transform from reduced coordinates to original coordinates. Use
        ``branch_incidence @ reduced_to_original`` to transform branch phase
        rows into the reduced coordinate basis.
    """

    capacitance_matrix: FloatArray
    inverse_inductance_matrix: FloatArray
    kept_indices: NDArray[np.int64]
    eliminated_indices: NDArray[np.int64]
    kept_mask: NDArray[np.bool_]
    eliminated_mask: NDArray[np.bool_]
    original_to_reduced: dict[int, int]
    reduced_to_original: FloatArray

    def reduced_index(self, original_index: int | None) -> int | None:
        """Return the reduced coordinate index for an original index.

        ``None`` is preserved for ground. A ``ValueError`` is raised if the
        requested coordinate was eliminated.
        """

        if original_index is None:
            return None

        try:
            return self.original_to_reduced[int(original_index)]
        except KeyError as exc:
            raise ValueError(
                f"Original coordinate {original_index} was eliminated and "
                "cannot be used as a reduced branch endpoint."
            ) from exc

    def transform_branch_incidence(
        self, branch_incidence_matrix: FloatArray
    ) -> FloatArray:
        """Transform original branch-incidence rows to the reduced basis."""

        branch_incidence = np.asarray(branch_incidence_matrix, dtype=float)
        if branch_incidence.ndim != 2:
            raise ValueError("branch_incidence_matrix must be two-dimensional.")
        if branch_incidence.shape[1] != self.reduced_to_original.shape[0]:
            raise ValueError(
                "branch_incidence_matrix must have one column per original coordinate."
            )

        return branch_incidence @ self.reduced_to_original


def _as_valid_matrix(matrix: FloatArray, name: str) -> FloatArray:
    array = np.asarray(matrix, dtype=float)

    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(array, array.T, rtol=1e-10, atol=1e-30):
        raise ValueError(f"{name} must be symmetric.")

    return array


def _matrix_zero_threshold(
    matrix: FloatArray,
    *,
    rtol: float,
    atol: float,
) -> float:
    scale = max(1.0, float(np.max(np.abs(matrix))))
    return atol + rtol * scale


def dynamic_coordinates_from_capacitance(
    capacitance_matrix: FloatArray,
    *,
    rtol: float = _DEFAULT_RELATIVE_TOLERANCE,
    atol: float = 0.0,
) -> NDArray[np.bool_]:
    """Infer coordinates that participate in the kinetic energy.

    Coordinates with zero rows and columns in ``capacitance_matrix`` are
    classified as frozen/constrained. This is a matrix-level convenience for
    simple cases. A future graph layer should pass an explicit mask when it has
    more structural information.
    """

    capacitance_matrix = _as_valid_matrix(
        capacitance_matrix,
        "capacitance_matrix",
    )
    threshold = _matrix_zero_threshold(
        capacitance_matrix,
        rtol=rtol,
        atol=atol,
    )

    row_norm = np.max(np.abs(capacitance_matrix), axis=1)
    col_norm = np.max(np.abs(capacitance_matrix), axis=0)
    frozen_mask = (row_norm <= threshold) & (col_norm <= threshold)
    return ~frozen_mask


def _validate_frozen_capacitance_block(
    capacitance_matrix: FloatArray,
    frozen_indices: NDArray[np.int64],
    *,
    rtol: float,
    atol: float,
) -> None:
    if frozen_indices.size == 0:
        return

    threshold = _matrix_zero_threshold(
        capacitance_matrix,
        rtol=rtol,
        atol=atol,
    )
    frozen_rows = capacitance_matrix[frozen_indices, :]
    frozen_cols = capacitance_matrix[:, frozen_indices]

    if (
        np.max(np.abs(frozen_rows)) > threshold
        or np.max(np.abs(frozen_cols)) > threshold
    ):
        raise ValueError(
            "Coordinates classified as frozen must have zero capacitance rows "
            "and columns in capacitance_matrix."
        )


def _solve_positive_potential_block(
    block: FloatArray,
    rhs: FloatArray,
) -> FloatArray:
    try:
        factor = cho_factor(block, lower=True, check_finite=True)
        return cho_solve(factor, rhs, check_finite=True)
    except Exception as exc:
        raise ValueError(
            "The eliminated inverse-inductance block must be positive "
            "definite for frozen-coordinate Schur reduction."
        ) from exc


def reduce_frozen_coordinates(
    capacitance_matrix: FloatArray,
    inverse_inductance_matrix: FloatArray,
    dynamic_mask: NDArray[np.bool_] | None = None,
    *,
    rtol: float = _DEFAULT_RELATIVE_TOLERANCE,
    atol: float = 0.0,
) -> CoordinateReduction:
    """Eliminate coordinates with potential energy but no kinetic energy.

    For coordinates split into dynamic coordinates ``d`` and frozen coordinates
    ``f``, this minimizes the quadratic potential with respect to ``f`` and
    computes:

    ``C_eff = C_dd``

    ``L_inv_eff = L_dd - L_df L_ff^{-1} L_fd``

    The eliminated coordinates must have zero rows and columns in
    ``capacitance_matrix`` and the eliminated potential block ``L_ff`` must be
    positive definite.

    This structured reduction is distinct from ``BBQ``'s numerical projection
    of null capacitance directions. It should be applied before constructing a
    ``BBQ`` object when the circuit graph identifies true frozen coordinates.
    """

    capacitance_matrix = _as_valid_matrix(capacitance_matrix, "capacitance_matrix")
    inverse_inductance_matrix = _as_valid_matrix(
        inverse_inductance_matrix,
        "inverse_inductance_matrix",
    )

    if capacitance_matrix.shape != inverse_inductance_matrix.shape:
        raise ValueError(
            "capacitance_matrix and inverse_inductance_matrix must have the same shape."
        )

    size = capacitance_matrix.shape[0]
    if dynamic_mask is None:
        dynamic_mask = dynamic_coordinates_from_capacitance(
            capacitance_matrix,
            rtol=rtol,
            atol=atol,
        )
    else:
        dynamic_mask = np.asarray(dynamic_mask, dtype=bool)

    if dynamic_mask.shape != (size,):
        raise ValueError(
            "dynamic_mask must have shape (N,), where N is the matrix size."
        )

    frozen_mask = ~dynamic_mask
    dynamic_indices = np.flatnonzero(dynamic_mask).astype(np.int64)
    frozen_indices = np.flatnonzero(frozen_mask).astype(np.int64)

    if dynamic_indices.size == 0:
        raise ValueError("No dynamic coordinates remain after reduction.")

    _validate_frozen_capacitance_block(
        capacitance_matrix,
        frozen_indices,
        rtol=rtol,
        atol=atol,
    )

    if frozen_indices.size == 0:
        reduced_capacitance = capacitance_matrix.copy()
        reduced_inverse_inductance = inverse_inductance_matrix.copy()
        reduced_to_original = np.eye(size)
    else:
        c_dd = capacitance_matrix[np.ix_(dynamic_indices, dynamic_indices)]
        l_dd = inverse_inductance_matrix[np.ix_(dynamic_indices, dynamic_indices)]
        l_df = inverse_inductance_matrix[np.ix_(dynamic_indices, frozen_indices)]
        l_fd = inverse_inductance_matrix[np.ix_(frozen_indices, dynamic_indices)]
        l_ff = inverse_inductance_matrix[np.ix_(frozen_indices, frozen_indices)]

        x = _solve_positive_potential_block(l_ff, l_fd)
        reduced_capacitance = c_dd
        reduced_inverse_inductance = l_dd - l_df @ x
        reduced_to_original = np.zeros((size, dynamic_indices.size), dtype=float)
        reduced_to_original[dynamic_indices, :] = np.eye(dynamic_indices.size)
        reduced_to_original[frozen_indices, :] = -x

    original_to_reduced = {
        int(original): int(reduced) for reduced, original in enumerate(dynamic_indices)
    }

    return CoordinateReduction(
        capacitance_matrix=0.5 * (reduced_capacitance + reduced_capacitance.T),
        inverse_inductance_matrix=0.5
        * (reduced_inverse_inductance + reduced_inverse_inductance.T),
        kept_indices=dynamic_indices,
        eliminated_indices=frozen_indices,
        kept_mask=dynamic_mask.copy(),
        eliminated_mask=frozen_mask.copy(),
        original_to_reduced=original_to_reduced,
        reduced_to_original=reduced_to_original,
    )

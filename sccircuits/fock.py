"""Analytical operators projected onto truncated Fock bases."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.special import eval_genlaguerre, gammaln  # type: ignore[import-untyped]


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _validate_dimension(dimension: int) -> int:
    """Return a positive integer Fock-space dimension."""
    if (
        isinstance(dimension, (bool, np.bool_))
        or not isinstance(dimension, (int, np.integer))
        or dimension <= 0
    ):
        raise ValueError("dimension must be a positive integer.")
    return int(dimension)


@lru_cache(maxsize=None)
def _fock_upper_triangle(
    dimension: int,
) -> tuple[
    NDArray[np.intp],
    NDArray[np.intp],
    NDArray[np.intp],
    FloatArray,
]:
    """Return cached dimension-dependent data for upper-triangular elements."""
    m, n = np.triu_indices(dimension)
    k = n - m
    log_factorials = gammaln(np.arange(dimension, dtype=np.float64) + 1.0)
    half_log_factorial_ratio = 0.5 * (log_factorials[m] - log_factorials[n])

    for array in (m, n, k, half_log_factorial_ratio):
        array.setflags(write=False)

    return m, n, k, half_log_factorial_ratio


def _upper_triangle_amplitudes(
    dimension: int,
    phase_coefficient: float,
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp], FloatArray]:
    """Return magnitudes of projected displacement-operator elements."""
    m, n, k, half_log_factorial_ratio = _fock_upper_triangle(dimension)
    coefficient_magnitude = abs(phase_coefficient)
    coefficient_squared = coefficient_magnitude**2
    amplitudes = np.exp(
        -0.5 * coefficient_squared
        + k * np.log(coefficient_magnitude)
        + half_log_factorial_ratio
    ) * eval_genlaguerre(m, k, coefficient_squared)
    return m, n, k, np.asarray(amplitudes, dtype=np.float64)


def _signed_cosine_fock_matrix(
    dimension: int,
    phase_coefficient: float,
    phase_offset: float,
) -> FloatArray:
    """Return a projected cosine for a possibly signed mode coefficient."""
    if phase_coefficient == 0.0:
        return np.asarray(
            np.eye(dimension, dtype=np.float64) * np.cos(phase_offset),
            dtype=np.float64,
        )

    m, n, k, amplitudes = _upper_triangle_amplitudes(
        dimension,
        phase_coefficient,
    )
    coefficient_sign = np.sign(phase_coefficient)
    cos_offset = np.cos(phase_offset)
    sin_offset = np.sin(phase_offset)
    phase_factors = np.array(
        [
            cos_offset,
            -coefficient_sign * sin_offset,
            -cos_offset,
            coefficient_sign * sin_offset,
        ],
        dtype=np.float64,
    )
    values = amplitudes * phase_factors[k % 4]

    operator = np.empty((dimension, dimension), dtype=np.float64)
    operator[m, n] = values
    off_diagonal = m != n
    operator[n[off_diagonal], m[off_diagonal]] = values[off_diagonal]
    return operator


def _displacement_fock_matrix(
    dimension: int,
    phase_coefficient: float,
) -> ComplexArray:
    r"""Return ``P exp(i * coefficient * (a + a.dag)) P``."""
    if phase_coefficient == 0.0:
        return np.eye(dimension, dtype=np.complex128)

    m, n, k, amplitudes = _upper_triangle_amplitudes(
        dimension,
        phase_coefficient,
    )
    coefficient_sign = np.sign(phase_coefficient)
    phase_factors = np.array(
        [1.0, 1.0j * coefficient_sign, -1.0, -1.0j * coefficient_sign],
        dtype=np.complex128,
    )
    values = amplitudes * phase_factors[k % 4]

    operator = np.empty((dimension, dimension), dtype=np.complex128)
    operator[m, n] = values
    off_diagonal = m != n
    operator[n[off_diagonal], m[off_diagonal]] = values[off_diagonal]
    return operator


def cosine_fock_matrix(
    dimension: int,
    phase_zpf: float,
    phase_offset: float = 0.0,
) -> FloatArray:
    r"""Return the exact projection of a cosine onto one Fock basis.

    The returned matrix contains

    .. math::

        \langle m | \cos[\phi_\mathrm{zpf}(a + a^\dagger)
        + \phi_\mathrm{offset}] | n \rangle

    for ``m, n < dimension``.  These matrix elements are evaluated from the
    analytical displacement-operator formula with generalized Laguerre
    polynomials.  This differs at finite truncation from applying a matrix
    cosine to the already truncated phase operator.

    Parameters
    ----------
    dimension
        Number of retained Fock states.
    phase_zpf
        Non-negative zero-point phase fluctuation amplitude.
    phase_offset
        Scalar phase added inside the cosine, in radians.
    """
    dimension = _validate_dimension(dimension)
    phase_zpf = float(phase_zpf)
    phase_offset = float(phase_offset)
    if not np.isfinite(phase_zpf):
        raise ValueError("phase_zpf must be finite.")
    if phase_zpf < 0.0:
        raise ValueError("phase_zpf must be non-negative.")
    if not np.isfinite(phase_offset):
        raise ValueError("phase_offset must be finite.")

    return _signed_cosine_fock_matrix(dimension, phase_zpf, phase_offset)


def cosine_fock_matrix_derivative(
    dimension: int,
    phase_zpf: float,
    phase_offset: float = 0.0,
) -> FloatArray:
    r"""Return the derivative of :func:`cosine_fock_matrix` by ``phase_zpf``.

    The derivative is evaluated before projecting onto the requested Fock
    space.  An extra Fock state is used internally to retain the boundary
    contribution of ``(a + a.dag) exp(i * phase_zpf * (a + a.dag))``.
    """
    dimension = _validate_dimension(dimension)
    phase_zpf = float(phase_zpf)
    phase_offset = float(phase_offset)
    if not np.isfinite(phase_zpf):
        raise ValueError("phase_zpf must be finite.")
    if phase_zpf < 0.0:
        raise ValueError("phase_zpf must be non-negative.")
    if not np.isfinite(phase_offset):
        raise ValueError("phase_offset must be finite.")

    exponential = _displacement_fock_matrix(dimension + 1, phase_zpf)
    quadrature_times_exponential = np.zeros(
        (dimension, dimension),
        dtype=np.complex128,
    )
    if dimension > 1:
        quadrature_times_exponential[1:, :] += (
            np.sqrt(np.arange(1, dimension))[:, np.newaxis]
            * exponential[: dimension - 1, :dimension]
        )
    quadrature_times_exponential += (
        np.sqrt(np.arange(1, dimension + 1))[:, np.newaxis]
        * exponential[1 : dimension + 1, :dimension]
    )

    derivative = np.real(
        np.exp(1.0j * phase_offset) * 1.0j * quadrature_times_exponential
    )
    return np.asarray(0.5 * (derivative + derivative.T), dtype=np.float64)


def cosine_fock_product_matrix(
    dimensions: Sequence[int],
    phase_coefficients: Sequence[float] | FloatArray,
    phase_offset: float = 0.0,
) -> FloatArray:
    r"""Return a cosine projected onto a tensor product of Fock bases.

    This evaluates the exact projection of

    .. math::

        \cos\left[\sum_j \lambda_j(a_j + a_j^\dagger)
        + \phi_\mathrm{offset}\right]

    by factorizing its displacement operator over the modes.  Unlike
    :func:`cosine_fock_matrix`, the coefficients may be signed because normal
    mode participation factors carry an orientation.
    """
    dimensions_tuple = tuple(_validate_dimension(value) for value in dimensions)
    coefficients = np.asarray(tuple(phase_coefficients), dtype=np.float64)
    phase_offset = float(phase_offset)

    if not dimensions_tuple:
        raise ValueError("dimensions must contain at least one Fock space.")
    if coefficients.shape != (len(dimensions_tuple),):
        raise ValueError(
            "phase_coefficients must contain one value per Fock-space dimension."
        )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("phase_coefficients must contain only finite values.")
    if not np.isfinite(phase_offset):
        raise ValueError("phase_offset must be finite.")

    if len(dimensions_tuple) == 1:
        return _signed_cosine_fock_matrix(
            dimensions_tuple[0],
            float(coefficients[0]),
            phase_offset,
        )

    exponential = np.array(
        [[np.exp(1.0j * phase_offset)]],
        dtype=np.complex128,
    )
    for dimension, coefficient in zip(dimensions_tuple, coefficients, strict=True):
        exponential = np.asarray(
            np.kron(
                exponential,
                _displacement_fock_matrix(dimension, float(coefficient)),
            ),
            dtype=np.complex128,
        )

    return np.asarray(exponential.real, dtype=np.float64)

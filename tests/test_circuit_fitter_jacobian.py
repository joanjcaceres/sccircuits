"""Numerical sanity checks for CircuitFitter's analytic Jacobian (Bogoliubov off)."""

from __future__ import annotations

import numpy as np

from sccircuits import CircuitFitter


def _finite_difference_jacobian(
    fitter: CircuitFitter,
    phi_ext: float,
    params: np.ndarray,
    epsilon: float = 1e-6,
):
    """Simple central finite difference Jacobian for eigenvalues."""

    base = fitter.eigenvalues_function(phi_ext, params)
    jac = np.zeros((base.size, params.size), dtype=float)

    for idx in range(params.size):
        dp = np.zeros_like(params)
        dp[idx] = epsilon
        f_plus = fitter.eigenvalues_function(phi_ext, params + dp)
        f_minus = fitter.eigenvalues_function(phi_ext, params - dp)
        jac[:, idx] = (f_plus - f_minus) / (2.0 * epsilon)

    return jac


def test_circuit_fitter_analytic_jacobian_matches_finite_difference():
    transitions = {
        (0, 1): [(0.0, 4.9), (0.25, 4.92)],
        (1, 2): [(0.0, 9.8), (0.25, 9.75)],
    }

    
    fitter = CircuitFitter(
        Ej_initial=29.89307839,
        non_linear_frequency_initial=4.2135773,
        non_linear_phase_zpf_initial=2.83521394,
        dimensions=[100, 10, 3, 3, 3],
        linear_frequencies_initial=[9.63835601,  6.77437803,  6.1852762, 10.33167439],
        linear_couplings_initial=[.02473813,  0.58687328,  1.75021775,  1.03461861],
        transitions=transitions,
        use_bogoliubov=False,
        truncation=[20, 40, 40, 40, 60],
        fit_Ej_second=False,
        enable_jacobian=True,
    )

    params = fitter.params_initial
    phi_ext = 0.1

    analytic = fitter.eigenvalues_jacobian(phi_ext, params)
    numeric = _finite_difference_jacobian(fitter, phi_ext, params)

    # Only the lowest few eigenstates are relevant for the supplied transitions.
    num_states = 12

    np.testing.assert_allclose(
        analytic[:num_states],
        numeric[:num_states],
        rtol=1e-4,
        atol=1e-6,
    )

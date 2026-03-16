#!/usr/bin/env python3
"""Example usage for the current SCCircuits public API."""

import numpy as np

from sccircuits import BBQ, Circuit, FitAnalysis, TransitionFitter, get_info


def run_circuit_example() -> None:
    """Construct a Circuit from harmonic-mode data and inspect it."""
    frequencies = [5.0, 6.2, 7.8]
    phase_zpf = [0.12, 0.08, 0.03]
    dimensions = [18, 10, 6]

    circuit = Circuit.from_harmonic_modes(
        frequencies=frequencies,
        phase_zpf=phase_zpf,
        dimensions=dimensions,
        Ej=0.95,
    )
    evals, _ = circuit.eigensystem(truncation=[12, 8, 6])
    star = circuit.star_representation()

    print("1. Circuit from harmonic modes")
    print(f"   nonlinear frequency: {circuit.non_linear_frequency:.4f} GHz")
    print(f"   nonlinear phase zpf: {circuit.non_linear_phase_zpf:.4f}")
    print(f"   lowest transition: {evals[1] - evals[0]:.4f} GHz")
    print(f"   star frequencies: {np.round(star['linear_frequencies'], 4)}")


def run_bbq_example() -> None:
    """Build a minimal BBQ model from circuit matrices."""
    c_matrix = np.array([[40.0e-15]], dtype=float)
    l_inv_matrix = np.array([[1.0 / 1.23e-9]], dtype=float)

    bbq = BBQ(c_matrix, l_inv_matrix, non_linear_nodes=(0,))
    bbq.selected_modes = [0]
    bbq.dimensions = (8,)
    h0 = bbq.hamiltonian_0()

    print("2. Black Box Quantization")
    print(f"   linear modes (GHz): {np.round(bbq.linear_modes_GHz, 4)}")
    print(f"   phase zpf: {np.round(bbq.phase_zpf_list, 6)}")
    print(f"   H0 shape: {h0.shape}")


def run_transition_fit_example() -> None:
    """Fit a tiny synthetic spectroscopy dataset."""

    def model(phi_ext: float, params: np.ndarray) -> np.ndarray:
        omega0, amplitude = params
        return np.array([0.0, omega0 + amplitude * np.cos(phi_ext)], dtype=float)

    data = {
        (0, 1): [
            (0.0, 5.25, 0.02),
            (np.pi / 2.0, 5.00, 0.02),
            (np.pi, 4.75, 0.02),
        ]
    }

    fitter = TransitionFitter(
        model_func=model,
        data=data,
        returns_eigenvalues=True,
    )
    result = fitter.fit(
        params_initial=[4.9, 0.1],
        bounds=([4.0, 0.0], [6.0, 1.0]),
        verbose=0,
    )
    analysis = FitAnalysis(result, parameter_labels=["omega0", "amplitude"])

    print("3. Transition fitting")
    print(f"   fitted parameters: {np.round(result.x, 6)}")
    print(f"   reduced chi^2: {analysis.chi2_reduced:.6f}")


def main() -> None:
    """Run the example demonstrating SCCircuits functionality."""
    print("=" * 60)
    print("SCCircuits - Example Usage")
    print("=" * 60)
    info = get_info()
    print(f"Package: {info['name']} v{info['version']}")
    print(f"Author: {info['author']}")
    print(f"Description: {info['description']}")
    print()

    run_circuit_example()
    print("\n" + "=" * 60)
    run_bbq_example()
    print("\n" + "=" * 60)
    run_transition_fit_example()
    print("\n" + "=" * 60)
    print("\nExample completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

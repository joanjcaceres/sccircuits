import numpy as np
from typing import Optional, Union

# --- SciPy imports (dense & sparse) ---
from scipy.sparse import diags
from scipy.linalg import cosm
from utilities import lanczos_krylov
from iterative_diagonalizer import IterativeHamiltonianDiagonalizer


class Circuit:
    def __init__(
        self,
        frequencies: Union[np.ndarray, list[float]],
        phase_zpf: Union[np.ndarray, list[float]],
        dimensions: list[int],
        Ej: float,
        phase_ext: float = 0,
        use_bogoliubov: bool = True,
    ):
        """
        Initializes a BBQ (Black Box Quantization) object.
        Parameters
        ----------
        frequencies : Union[np.ndarray, list[float]]
            Frequencies in GHz.
        phase_zpf : Union[np.ndarray, list[float]]
            Zero-point fluctuations in radians.
        dimensions : list[int]
            Dimensions of the BBQ system.
        Ej : float
            Josephson energy in GHz.
        phase_ext : float, optional
            External phase in radians, default is 0.
        use_bogoliubov : bool, optional
            If True, applies Bogoliubov transformation to the collective mode.
            If False, uses the original frequencies and phase_zpf without transformation.
            Default is True.
        Raises
        ------
        ValueError
            If the lengths of frequencies, phase_zpf, and dimensions do not match.
        """

        # Validate that frequencies, phase_zpf, and dimensions are of the same length
        if len(frequencies) != len(phase_zpf) or len(frequencies) != len(dimensions):
            raise ValueError(
                "Frequencies, phase_zpf, and dimensions must have the same length."
                f"But got {len(frequencies)}, {len(phase_zpf)}, and {len(dimensions)}."
            )

        self.frequencies = np.array(frequencies)
        self.phase_zpf = np.array(phase_zpf)

        if np.any(self.frequencies <= 0):
            raise ValueError("All frequencies must be positive.")
        if np.any(self.phase_zpf <= 0):
            raise ValueError("All phase_zpf must be positive.")

        self.dimensions = dimensions
        self.Ej = Ej
        self.phase_ext = phase_ext
        self.use_bogoliubov = use_bogoliubov
        self.modes = len(self.dimensions)

        self.non_linear_phase_zpf = np.linalg.norm(self.phase_zpf)
        normalized_phase_zpf = self.phase_zpf / self.non_linear_phase_zpf

        harmonic_hamiltonian = np.diag(self.frequencies)
        self.collective_frequency = (
            normalized_phase_zpf @ harmonic_hamiltonian @ normalized_phase_zpf
        )
        if use_bogoliubov:
            harmonic_hamiltonian[0, 0] = self.collective_frequency
            # Check stability condition: 2*Ej*phi_zpf_rms^2 < collective_frequency
            mu = 2 * self.Ej * self.non_linear_phase_zpf**2 / self.collective_frequency
            if mu >= 1:
                Ej_max = self.collective_frequency / (2 * self.non_linear_phase_zpf**2)
                raise ValueError(
                    f"EJ = {self.Ej:.3f} GHz exceeds the stability limit of {Ej_max:.3f} GHz "
                    "(requires 2*Ej*phi_zpf_rms^2 < collective_frequency)."
                )

        Q, self.T, alpha, beta, _ = lanczos_krylov(
            H=harmonic_hamiltonian, v=self.phase_zpf
        )
        self.non_linear_frequency = alpha[0]
        self.linear_frequencies = alpha[1:]
        self.linear_coupling = beta

        self.Ec = self.collective_frequency * self.non_linear_phase_zpf**2 / 4
        self.El = self.collective_frequency / 2 / self.non_linear_phase_zpf**2

        # self.linear_frequencies, self.linear_coupling = self._non_collective_eigsystem()

        # if self.use_bogoliubov:
        #     self.non_linear_frequency = self._non_linear_frequency()
        #     self.non_linear_phase_zpf = self._non_linear_phase_zpf()

    def hamiltonian_0(self, phase_ext: Optional[float] = None) -> np.ndarray:
        dimension = self.dimensions[0]
        if phase_ext is None:
            phase_ext = self.phase_ext
        freq_0 = self.non_linear_frequency
        phi_zpf_0 = self.non_linear_phase_zpf

        diagonal = freq_0 * (np.arange(dimension) + 1 / 2)
        hamiltonian = diags(diagonal, 0)

        data = np.sqrt(np.arange(1, dimension))
        phi_op = phi_zpf_0 * diags([data, data], [1, -1])

        hamiltonian -= self.Ej * cosm(phi_op + phase_ext * np.eye(dimension))

        return hamiltonian

    def eigensystem(self, truncation: int, phase_ext: Optional[float] = None):
        """
        Calculate the eigenvalues and eigenvectors of the total BBQ Hamiltonian
        using sequential coupling logic.
        """
        if self.use_bogoliubov:
            r = self.r_bogoliubov()
            data = np.sqrt(np.arange(1, self.dimensions[0]))

            collective_creation_operator = diags(
                [np.sinh(r) * data, np.cosh(r) * data], [1, -1], dtype=np.float64
            )
        else:
            # Without Bogoliubov transformation, use standard ladder operators
            data = np.sqrt(np.arange(1, self.dimensions[0]))
            collective_creation_operator = diags([data], [-1], dtype=np.float64)

        iterator = IterativeHamiltonianDiagonalizer(truncation)
        
        # Add initial mode (mode 0) with its coupling operator for the next mode
        if self.modes > 1:
            # Mode 0 couples to mode 1, so we need a coupling operator
            iterator.add_initial_mode(
                self.hamiltonian_0(phase_ext),
                collective_creation_operator,  # This will couple to mode 1
            )
        else:
            # Only one mode, no coupling needed
            iterator.add_initial_mode(
                self.hamiltonian_0(phase_ext),
                None,  # No coupling for single mode
            )

        # Add subsequent modes with sequential coupling
        for idx in range(self.modes - 1):
            frequency_k = self.linear_frequencies[idx]
            diag_k = frequency_k * (np.arange(self.dimensions[idx + 1]) + 1 / 2)
            hamiltonian_k = diags(diag_k, 0)

            # Current mode's coupling operator (couples to previous mode)
            data = np.sqrt(np.arange(1, self.dimensions[idx + 1]))
            linear_destroy_op_k = diags([data], [1], dtype=np.float64)
            
            # Coupling operator for next mode (if this is not the last mode)
            if idx < self.modes - 2:  # Not the last mode
                # For simplicity, use the same operator for next coupling
                # In more complex cases, this could be different
                coupling_operator_next = linear_destroy_op_k.T.copy()
            else:
                # This is the last mode, no next coupling
                coupling_operator_next = None
            
            # Add mode with sequential coupling
            iterator.add_mode(
                hamiltonian_k, 
                linear_destroy_op_k,      # Couples to previous mode
                coupling_operator_next,   # Will couple to next mode (if any)
                self.linear_coupling[idx] # Coupling strength
            )

        return iterator.energies, iterator.basis_vectors

    def r_bogoliubov(self) -> float:
        """
        Calculate the Bogoliubov transformation parameter r.
        
        Returns:
            float: The Bogoliubov parameter r
        """
        # Calculate the collective frequency from the transformed frequencies
        collective_frequency = self.non_linear_frequency  
        phi_zpf_rms = self.non_linear_phase_zpf
        
        r = (
            -1
            / 4
            * np.log(1 - 2 * self.Ej * phi_zpf_rms**2 / collective_frequency)
        )
        return r

    # def _non_collective_eigsystem(self) -> np.ndarray:
    #     """
    #     Calculate the eigenvalues and eigenvectors of the non-collective system.
    #     """
    #     normalized_phi_zpf = self.phase_zpf / np.linalg.norm(self.phase_zpf)
    #     provisional_linear_mode_amplitudes = null_space(
    #         normalized_phi_zpf.reshape(1, -1)
    #     )
    #     W = np.diag(self.frequencies)
    #     linear_mode_subblock = provisional_linear_mode_amplitudes.conj().T @ (
    #         W @ provisional_linear_mode_amplitudes
    #     )
    #     linear_frequencies, evecs = eigh(linear_mode_subblock)

    #     beta = provisional_linear_mode_amplitudes @ evecs
    #     linear_coupling = (self.frequencies * normalized_phi_zpf) @ beta

    #     return linear_frequencies, linear_coupling

    # def _non_linear_frequency(self) -> np.ndarray:
    #     non_linear_frequency = np.sqrt(
    #         self.collective_frequency
    #         * (self.collective_frequency - 2 * self.Ej * self.phi_zpf_rms**2)
    #     )
    #     return non_linear_frequency

    # def _non_linear_phase_zpf(self) -> np.ndarray:
    #     non_linear_phase_zpf = (
    #         self.phi_zpf_rms
    #         * (1 - (2 * self.Ej * self.phi_zpf_rms**2) / self.collective_frequency)
    #         ** -0.25
    #     )
    #     return non_linear_phase_zpf
    
if __name__ == "__main__":
    # Create a BBQ object for testing
    frequencies = np.array([5.0, 6.0, 7.8])
    phase_zpf = np.array([0.1, 0.2, 0.01])
    dimensions = [50, 10, 3]
    Ej = 1.0
    phase_ext = 0.0

    # Test with Bogoliubov transformation (default)
    bbq_with_bogoliubov = Circuit(
        frequencies, phase_zpf, dimensions, Ej, phase_ext, use_bogoliubov=True
    )
    print("With Bogoliubov transformation:")
    print(f"Non-linear frequency: {bbq_with_bogoliubov.non_linear_frequency:.3f} GHz")
    print(f"Non-linear phase ZPF: {bbq_with_bogoliubov.non_linear_phase_zpf:.3f}")

    # Test without Bogoliubov transformation
    bbq_without_bogoliubov = Circuit(
        frequencies, phase_zpf, dimensions, Ej, phase_ext, use_bogoliubov=False
    )
    print("\nWithout Bogoliubov transformation:")
    print(
        f"Non-linear frequency: {bbq_without_bogoliubov.non_linear_frequency:.3f} GHz"
    )
    print(f"Non-linear phase ZPF: {bbq_without_bogoliubov.non_linear_phase_zpf:.3f}")

    # Test the methods
    print("\nZero-order Hamiltonian (with Bogoliubov):")
    print(bbq_with_bogoliubov.hamiltonian_0())


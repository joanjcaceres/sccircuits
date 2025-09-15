import numpy as np
from typing import Optional, Union

# --- SciPy imports (dense & sparse) ---
from scipy.sparse import diags
from scipy.linalg import cosm
from sccircuits.utilities import lanczos_krylov
from sccircuits.iterative_diagonalizer import IterativeHamiltonianDiagonalizer


def _to_dense_if_sparse(matrix):
    """Convert a sparse matrix to dense if necessary."""
    return matrix.toarray() if hasattr(matrix, 'toarray') else matrix

class Circuit:
    """
    Main class for superconducting quantum circuit analysis.
    
    This class implements a comprehensive framework for analyzing superconducting
    quantum circuits, including support for multi-mode systems, Bogoliubov 
    transformations, and eigensystem calculations with truncation.
    
    The Circuit class can handle both single and multi-mode superconducting 
    circuits with arbitrary coupling between modes. It supports both dense and
    sparse matrix operations for efficient computation of large systems.
    """
    
    def __init__(
        self,
        frequencies: Union[np.ndarray, list[float]],
        phase_zpf: Union[np.ndarray, list[float]],
        dimensions: list[int],
        Ej: float,
        Gamma: Optional[float] = None,
        epsilon_r: Optional[float] = None,
        phase_ext: Optional[float] = 0,
        use_bogoliubov: Optional[bool] = False,
    ):
        """
        Initializes a Circuit object for superconducting circuit analysis.
        
        Parameters
        ----------
        frequencies : Union[np.ndarray, list[float]]
            Linear mode frequencies in GHz.
        phase_zpf : Union[np.ndarray, list[float]]
            Phase zero-point fluctuations in radians for each mode.
        dimensions : list[int]
            Dimensions of the circuit system.
        Ej : float
            Josephson energy in GHz.
        Gamma : Optional[float], optional
            Fermionic coupling strength in GHz. If provided along with epsilon_r, 
            includes a fermionic mode coupled to the primary bosonic mode.
            Default is None (no fermionic coupling).
        epsilon_r : Optional[float], optional
            Fermionic energy level spacing in GHz. If provided along with Gamma,
            includes a fermionic mode with Hamiltonian 2*epsilon_r*c†c.
            Default is None (no fermionic mode).
        phase_ext : float, optional
            External flux phase in radians, default is 0.
        use_bogoliubov : bool, optional
            If True, applies Bogoliubov transformation to the collective mode.
            If False, uses original parameters without transformation. Default is False.
            
        Raises
        ------
        ValueError
            If frequencies, phase_zpf, and dimensions have different lengths,
            or if any frequency or phase_zpf is non-positive, or if Ej exceeds
            the stability limit when use_bogoliubov=True.
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
        self.Gamma = Gamma
        self.epsilon_r = epsilon_r
        self.phase_ext = phase_ext
        self.use_bogoliubov = use_bogoliubov
        self.modes = len(self.dimensions)
        
        # Validate fermionic coupling parameters
        self.has_fermionic_coupling = (Gamma is not None and epsilon_r is not None)
        if (Gamma is None and epsilon_r is not None) or (Gamma is not None and epsilon_r is None):
            raise ValueError("Both Gamma and epsilon_r must be provided together for fermionic coupling, or both should be None.")

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

    def hamiltonian_nl(self, phase_ext: Optional[float] = None, return_coupling_ops: bool = False):
        """
        Calculate the primary bosonic Hamiltonian and optionally return coupling operators.
        
        Args:
            phase_ext: External phase offset
            return_coupling_ops: If True, also return pre-calculated coupling operators
            
        Returns:
            If return_coupling_ops is False: np.ndarray (Hamiltonian)
            If return_coupling_ops is True: tuple (Hamiltonian, cos_half_op, collective_creation_operator)
        """
        dimension_bosonic = self.dimensions[0]
        if phase_ext is None:
            phase_ext = self.phase_ext
        freq_0 = self.non_linear_frequency
        phi_zpf_0 = self.non_linear_phase_zpf

        # Construct the bosonic Hamiltonian using a diagonal matrix for efficiency (avoids dense matrix operations).
        diagonal = freq_0 * (np.arange(dimension_bosonic) + 1 / 2)
        hamiltonian = diags(diagonal, 0)

        data = np.sqrt(np.arange(1, dimension_bosonic))
        phi_op = phi_zpf_0 * diags([data, data], [1, -1])
        phi_op = _to_dense_if_sparse(phi_op)
        
        # Calculate the full cosine operator for the hamiltonian
        cos_full_op = cosm(phi_op + phase_ext * np.eye(dimension_bosonic))
        hamiltonian -= self.Ej * cos_full_op

        if return_coupling_ops:
            # Calculate coupling operators only when needed
            cos_half_op = cosm(phi_op / 2)  # For fermionic coupling (no phase_ext)
            
            # Calculate collective creation operator here to avoid duplication
            if self.use_bogoliubov:
                r = self.r_bogoliubov()
                # Cache sinh(r) and cosh(r) for efficiency
                if not hasattr(self, "_sinh_r") or not hasattr(self, "_cosh_r"):
                    self._sinh_r = np.sinh(r)
                    self._cosh_r = np.cosh(r)
                collective_creation_operator = diags(
                    [self._sinh_r * data, self._cosh_r * data], [1, -1], dtype=np.float64
                )
            else:
                collective_creation_operator = diags([data], [-1], dtype=np.float64)
            
            return hamiltonian, cos_half_op, collective_creation_operator
        else:
            return hamiltonian

    def _fermionic_hamiltonian(self) -> np.ndarray:
        """
        Create the fermionic Hamiltonian: 2 * epsilon_r * c†c
        
        Returns:
            np.ndarray: 2x2 fermionic Hamiltonian matrix
        """
        if not self.has_fermionic_coupling:
            raise ValueError("Fermionic coupling not enabled - both Gamma and epsilon_r must be specified")
        
        # c†c = |1⟩⟨1| (only occupied state has energy)
        return 2 * self.epsilon_r * np.diag([0, 1])
    
    
    def eigensystem(self, truncation: int, phase_ext: Optional[float] = None):
        """
        Calculate the eigenvalues and eigenvectors of the total Circuit Hamiltonian
        using sequential coupling logic.
        
        The truncation parameter is handled adaptively by the IterativeHamiltonianDiagonalizer:
        - It automatically adapts to the actual system dimensions at each step
        - Works correctly with fermionic modes (2×2) and bosonic modes of any size
        """
        iterator = IterativeHamiltonianDiagonalizer(truncation)
        
        if self.has_fermionic_coupling:
            
            # Step 1: Add initial fermionic mode
            H_fermion = self._fermionic_hamiltonian()  # 2x2 fermionic Hamiltonian
            
            # Fermionic coupling operator: c† for coupling to bosonic mode
            fermionic_creation_op = np.array([[0, 0], [1, 0]], dtype=float)
            
            iterator.add_initial_mode(
                H_fermion,
                fermionic_creation_op,  # This will couple to the bosonic mode
            )
            
            # Step 2: Add bosonic non-linear mode (coupled to fermion)
            # Get hamiltonian and coupling operators efficiently in one call
            if self.modes > 1:
                hamiltonian_nl, cos_half_op, collective_creation_operator = self.hamiltonian_nl(phase_ext, return_coupling_ops=True)
                next_coupling_op = collective_creation_operator  # For coupling to next bosonic mode
            else:
                hamiltonian_nl = self.hamiltonian_nl(phase_ext)
                # Still need cos_half_op for coupling to fermion
                dimension_bosonic = self.dimensions[0]
                data = np.sqrt(np.arange(1, dimension_bosonic))
                phi_op = self.non_linear_phase_zpf * diags([data, data], [1, -1])
                phi_op = _to_dense_if_sparse(phi_op)
                cos_half_op = cosm(phi_op / 2)
                next_coupling_op = None
            
            iterator.add_mode(
                hamiltonian_nl,
                cos_half_op,           # Couples to fermionic c†  
                next_coupling_op,      # For next bosonic coupling
                self.Gamma             # Coupling strength
            )
            
            # Step 3: Add remaining bosonic modes
            remaining_bosonic_modes = self.modes - 1
            
            for idx in range(0, remaining_bosonic_modes):
                frequency_k = self.linear_frequencies[idx]
                diag_k = frequency_k * (np.arange(self.dimensions[idx + 1]) + 1 / 2)
                hamiltonian_k = diags(diag_k, 0)
                
                # Convert to dense for compatibility
                hamiltonian_k = _to_dense_if_sparse(hamiltonian_k)

                # Current mode's coupling operator (couples to previous mode)
                data = np.sqrt(np.arange(1, self.dimensions[idx + 1]))
                linear_destroy_op_k = diags([data], [1], dtype=np.float64)
                
                # Convert to dense for compatibility
                linear_destroy_op_k = _to_dense_if_sparse(linear_destroy_op_k)
                
                # Coupling operator for next mode (if this is not the last mode)
                if idx < remaining_bosonic_modes - 1:  # Not the last bosonic mode
                    coupling_operator_next = linear_destroy_op_k.T.copy()
                else:
                    coupling_operator_next = None
                
                # Add mode with sequential coupling
                iterator.add_mode(
                    hamiltonian_k, 
                    linear_destroy_op_k,      # Couples to previous mode
                    coupling_operator_next,   # Will couple to next mode (if any)
                    self.linear_coupling[idx] # Coupling strength
                )
        
        else:
            # ORIGINAL ARCHITECTURE: Start with bosonic mode (no fermion)
            
            # Calculate hamiltonian and coupling operators
            if self.modes > 1:
                hamiltonian_0, cos_half_op, collective_creation_operator = self.hamiltonian_nl(phase_ext, return_coupling_ops=True)
                initial_coupling_op = collective_creation_operator  # Use creation operator for bosonic coupling
            else:
                hamiltonian_0 = self.hamiltonian_nl(phase_ext)
                initial_coupling_op = None
                
            iterator.add_initial_mode(
                hamiltonian_0,
                initial_coupling_op,
            )
            
            # Add remaining bosonic modes (standard logic)
            remaining_bosonic_modes = self.modes - 1
            
            for idx in range(0, remaining_bosonic_modes):
                frequency_k = self.linear_frequencies[idx]
                diag_k = frequency_k * (np.arange(self.dimensions[idx + 1]) + 1 / 2)
                hamiltonian_k = diags(diag_k, 0)
                
                # Convert to dense for compatibility
                hamiltonian_k = _to_dense_if_sparse(hamiltonian_k)

                # Current mode's coupling operator (couples to previous mode)
                data = np.sqrt(np.arange(1, self.dimensions[idx + 1]))
                linear_destroy_op_k = diags([data], [1], dtype=np.float64)
                
                # Convert to dense for compatibility
                linear_destroy_op_k = _to_dense_if_sparse(linear_destroy_op_k)
                
                # Coupling operator for next mode (if this is not the last mode)
                if idx < remaining_bosonic_modes - 1:  # Not the last bosonic mode
                    coupling_operator_next = linear_destroy_op_k.T.copy()
                else:
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
    # Example usage of the Circuit class
    frequencies = np.array([5.0, 6.0, 7.8])
    phase_zpf = np.array([0.1, 0.2, 0.01])
    dimensions = [50, 10, 3]
    Ej = 1.0
    phase_ext = 0.0

    # Test basic functionality
    circuit = Circuit(
        frequencies, phase_zpf, dimensions, Ej, 
        phase_ext=phase_ext, use_bogoliubov=True
    )
    print("Basic Circuit:")
    print(f"Non-linear frequency: {circuit.non_linear_frequency:.3f} GHz")
    print(f"Non-linear phase ZPF: {circuit.non_linear_phase_zpf:.3f}")

    # Test with fermionic coupling - NEW FEATURE
    circuit_with_fermion = Circuit(
        frequencies, phase_zpf, dimensions, Ej, 
        Gamma=0.5, epsilon_r=0.2,  # These enable fermionic coupling
        phase_ext=phase_ext, use_bogoliubov=True
    )
    print("\nCircuit with fermionic coupling:")
    print(f"Gamma (fermion-boson coupling): {circuit_with_fermion.Gamma} GHz")
    print(f"epsilon_r (fermion energy): {circuit_with_fermion.epsilon_r} GHz")
    
    # Compare eigenspectra
    energies_basic, _ = circuit.eigensystem(truncation=10)
    energies_with_fermion, _ = circuit_with_fermion.eigensystem(truncation=10)
    
    print(f"\nLowest 5 energies (basic): {energies_basic[:5]}")
    print(f"Lowest 5 energies (with fermion): {energies_with_fermion[:5]}")
    print("\n✅ Implementation successful!")


import numpy as np
from typing import Optional, Sequence, Union

# --- SciPy imports (dense & sparse) ---
from scipy.sparse import diags
from scipy.linalg import cosm, sinm
from sccircuits.utilities import lanczos_krylov
from sccircuits.iterative_diagonalizer import IterativeHamiltonianDiagonalizer

def harmonic_modes_to_physical(
    frequencies: Union[np.ndarray, Sequence[float]],
    phase_zpf: Union[np.ndarray, Sequence[float]],
) -> dict[str, Union[float, np.ndarray, dict]]:
    """Map harmonic mode data to physical parameters via Lanczos tridiagonalization."""
    frequencies = np.asarray(frequencies, dtype=float)
    phase_zpf = np.asarray(phase_zpf, dtype=float)

    if frequencies.ndim != 1 or phase_zpf.ndim != 1:
        raise ValueError("frequencies and phase_zpf must be one-dimensional sequences.")
    if frequencies.size != phase_zpf.size:
        raise ValueError(
            "frequencies and phase_zpf must have the same length for Lanczos transformation."
        )
    if frequencies.size == 0:
        raise ValueError("At least one mode is required for Lanczos transformation.")
    if np.any(frequencies <= 0):
        raise ValueError("All input frequencies must be positive for Lanczos transformation.")

    phi_norm = np.linalg.norm(phase_zpf)
    if phi_norm <= 0:
        raise ValueError("phase_zpf must contain at least one non-zero element.")

    normalized_phase_zpf = phase_zpf / phi_norm
    harmonic_hamiltonian = np.diag(frequencies)
    collective_frequency = normalized_phase_zpf @ harmonic_hamiltonian @ normalized_phase_zpf

    Q, T, alpha, beta, status = lanczos_krylov(H=harmonic_hamiltonian, v=phase_zpf)

    return {
        "non_linear_frequency": float(alpha[0]),
        "non_linear_phase_zpf": float(phi_norm),
        "linear_frequencies": alpha[1:],
        "linear_couplings": beta,
        "collective_frequency": float(collective_frequency),
        "basis": Q,
        "tridiagonal": T,
        "status": status,
    }


def bogoliubov_transform(
    Ej: float,
    non_linear_phase_zpf: float,
    mode_frequency: float,
) -> dict[str, float]:
    """Compute Bogoliubov squeezing parameters for the nonlinear mode."""
    if mode_frequency <= 0:
        raise ValueError("mode_frequency must be positive for the Bogoliubov transform.")
    if non_linear_phase_zpf <= 0:
        raise ValueError("non_linear_phase_zpf must be positive for the Bogoliubov transform.")

    mu = 2 * Ej * non_linear_phase_zpf**2 / mode_frequency
    if mu >= 1:
        Ej_max = mode_frequency / (2 * non_linear_phase_zpf**2)
        raise ValueError(
            f"EJ = {Ej:.3f} GHz exceeds the stability limit of {Ej_max:.3f} GHz "
            "(requires 2*Ej*phi_zpf_rms^2 < mode_frequency)."
        )

    r = -0.25 * np.log(1 - mu)
    return {
        "r": r,
        "sinh_r": np.sinh(r),
        "cosh_r": np.cosh(r),
        "mu": mu,
    }


class Circuit:
    """
    Main class for superconducting quantum circuit analysis.
    
    This class implements a comprehensive framework for analyzing superconducting
    quantum circuits, including support for multi-mode systems,
    and eigensystem calculations with truncation. The nonlinear Josephson
    potential can optionally include a second harmonic term -EJ2 cos(2*phi).
    
    The Circuit class can handle both single and multi-mode superconducting 
    circuits with arbitrary coupling between modes. It supports both dense and
    sparse matrix operations for efficient computation of large systems.
    """
    
    def __init__(
        self,
        non_linear_frequency: float,
        non_linear_phase_zpf: float,
        dimensions: Sequence[int],
        Ej: float,
        linear_frequencies: Optional[Sequence[float]] = None,
        linear_couplings: Optional[Sequence[float]] = None,
        Ej_second: float = 0.0,
        Gamma: Optional[float] = None,
        epsilon_r: Optional[float] = None,
        phase_ext: float = 0.0,
    ):
        """
        Initializes a Circuit object for superconducting circuit analysis.
        
        Parameters
        ----------
        non_linear_frequency : float
            Frequency of the nonlinear collective mode in GHz.
        non_linear_phase_zpf : float
            Zero-point phase fluctuation of the nonlinear mode (radians).
        linear_frequencies : Sequence[float] or None, optional
            Frequencies of the remaining linear modes (GHz). Provide a sequence of
            length ``len(dimensions) - 1`` when multiple modes are present; defaults
            to ``None`` for single-mode circuits.
        linear_couplings : Sequence[float] or None, optional
            Sequential coupling strengths between modes (GHz). Provide a sequence of
            length ``len(dimensions) - 1`` when multiple modes are present; defaults
            to ``None`` for single-mode circuits.
        dimensions : Sequence[int]
            Hilbert-space dimensions for each mode.
        Ej : float
            Josephson energy in GHz.
        Ej_second : float, optional
            Coefficient for the -EJ2*cos(2*phi) second harmonic Josephson term in GHz.
            Defaults to 0 to recover the standard single cosine potential.
        Gamma : Optional[float], optional
            Fermionic coupling strength in GHz. Provide together with epsilon_r.
        epsilon_r : Optional[float], optional
            Fermionic energy level spacing in GHz. Provide together with Gamma.
        phase_ext : float, optional
            External flux phase in radians, default is 0.
        Raises
        ------
        ValueError
            If dimensions are invalid, frequencies are non-positive, or fermionic
            parameters are supplied inconsistently.
        """

        self.non_linear_frequency = float(non_linear_frequency)
        self.non_linear_phase_zpf = float(non_linear_phase_zpf)

        if self.non_linear_frequency <= 0:
            raise ValueError("non_linear_frequency must be positive.")
        if self.non_linear_phase_zpf <= 0:
            raise ValueError("non_linear_phase_zpf must be positive.")

        self.dimensions = [int(d) for d in dimensions]
        if not self.dimensions:
            raise ValueError("At least one mode dimension must be specified.")
        if any(d <= 0 for d in self.dimensions):
            raise ValueError("All mode dimensions must be positive integers.")

        self.modes = len(self.dimensions)
        self.linear_mode_count = self.modes - 1
        expected_linear_modes = self.linear_mode_count

        if expected_linear_modes == 0:
            linear_frequencies_array = np.array([], dtype=float)
            linear_couplings_array = np.array([], dtype=float)
        else:
            if linear_frequencies is None or linear_couplings is None:
                raise ValueError(
                    "linear_frequencies and linear_couplings must be provided for multi-mode circuits."
                )
            if len(linear_frequencies) != expected_linear_modes or len(linear_couplings) != expected_linear_modes:
                raise ValueError(
                    "linear_frequencies and linear_couplings must each have length len(dimensions) - 1."
                )
            linear_frequencies_array = np.asarray(linear_frequencies, dtype=float)
            linear_couplings_array = np.asarray(linear_couplings, dtype=float)

        if np.any(linear_frequencies_array <= 0):
            raise ValueError("All linear_frequencies must be positive.")
        if np.any(linear_couplings_array < 0):
            raise ValueError("All linear_couplings must be non-negative.")

        self.linear_frequencies = linear_frequencies_array
        self.linear_coupling = linear_couplings_array

        self.Ej = Ej
        self.Ej_second = float(Ej_second)
        self.Gamma = Gamma
        self.epsilon_r = epsilon_r
        self.phase_ext = phase_ext

        self.has_fermionic_coupling = (Gamma is not None and epsilon_r is not None)
        if (Gamma is None and epsilon_r is not None) or (Gamma is not None and epsilon_r is None):
            raise ValueError("Both Gamma and epsilon_r must be provided together for fermionic coupling, or both should be None.")

        self._lanczos_basis: Optional[np.ndarray] = None
        self._lanczos_tridiagonal: Optional[np.ndarray] = None
        self._lanczos_status: Optional[dict] = None
        self._harmonic_modes_store: Optional[dict[str, np.ndarray]] = None

    @classmethod
    def from_harmonic_modes(
        cls,
        *,
        frequencies: Sequence[float],
        phase_zpf: Sequence[float],
        dimensions: Sequence[int],
        Ej: float,
        Ej_second: float = 0.0,
        Gamma: Optional[float] = None,
        epsilon_r: Optional[float] = None,
        phase_ext: float = 0.0,
    ) -> "Circuit":
        """
        Construct a Circuit by first converting harmonic inputs with Lanczos.
        """
        if len(frequencies) != len(dimensions) or len(phase_zpf) != len(dimensions):
            raise ValueError("frequencies, phase_zpf, and dimensions must share the same length.")

        params = harmonic_modes_to_physical(frequencies, phase_zpf)

        circuit = cls(
            non_linear_frequency=params["non_linear_frequency"],
            non_linear_phase_zpf=params["non_linear_phase_zpf"],
            linear_frequencies=params["linear_frequencies"],
            linear_couplings=params["linear_couplings"],
            dimensions=dimensions,
            Ej=Ej,
            Ej_second=Ej_second,
            Gamma=Gamma,
            epsilon_r=epsilon_r,
            phase_ext=phase_ext,
        )

        circuit._lanczos_basis = params["basis"]
        circuit._lanczos_tridiagonal = params["tridiagonal"]
        circuit._lanczos_status = params["status"]
        circuit._store_harmonic_modes(frequencies, phase_zpf)

        return circuit

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
        phi_op = phi_zpf_0 * diags([data, data], [1, -1]).toarray()
        gauge_invariant_phase_op = phi_op + phase_ext * np.eye(dimension_bosonic)
        # Calculate the nonlinear potential contributions. Keep phi_shift handy because
        # it is reused by multiple harmonic terms when present.
        cos_phi = cosm(gauge_invariant_phase_op)
        hamiltonian -= self.Ej * cos_phi

        if self.Ej_second != 0.0:
            cos_2phi = cosm(2.0 * gauge_invariant_phase_op)
            hamiltonian -= self.Ej_second * cos_2phi

        if return_coupling_ops:
            if self.has_fermionic_coupling:
                cos_half_op = cosm(gauge_invariant_phase_op / 2)  # For fermionic coupling (includes phase_ext)
            else:
                cos_half_op = None

            collective_creation_operator = diags([data], [-1], dtype=np.float64)

            return hamiltonian, cos_half_op, collective_creation_operator

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
    
    
    def eigensystem(
        self,
        truncation: int | Sequence[int],
        phase_ext: Optional[float] = None,
        *,
        track_operators: bool = False,
        store_basis: bool = False,
    ):
        """
        Calculate the eigenvalues and eigenvectors of the total Circuit Hamiltonian
        using sequential coupling logic.

        The ``truncation`` argument can be either a single integer applied to every
        diagonalization step or a sequence specifying the number of states to keep
        after each mode is incorporated (additional entries are ignored once all
        modes have been processed).

        The truncation parameter is handled adaptively by the
        :class:`IterativeHamiltonianDiagonalizer`:
        - It automatically adapts to the actual system dimensions at each step
        - Works correctly with fermionic modes (2×2) and bosonic modes of any size
        """
        iterator = IterativeHamiltonianDiagonalizer(truncation)
        
        if self.has_fermionic_coupling:
            
            # Step 1: Add initial fermionic mode
            H_fermion = self._fermionic_hamiltonian()  # 2x2 fermionic Hamiltonian
            
            # Fermionic operators: c† couples to bosonic mode, track c
            fermionic_creation_op = np.array([[0, 0], [1, 0]], dtype=float)
            fermionic_annihilation_op = fermionic_creation_op.T.conj()
            
            tracked_fermion = {"a_fermion": fermionic_annihilation_op} if track_operators else None
            iterator.add_initial_mode(
                H_fermion,
                fermionic_creation_op,  # Couples to bosonic mode
                tracked_operators=tracked_fermion,
                store_basis=store_basis,
            )
            
            # Step 2: Add bosonic non-linear mode (coupled to fermion)
            hamiltonian_nl, cos_half_op, collective_creation_operator = self.hamiltonian_nl(
                phase_ext, return_coupling_ops=True
            )
            collective_annihilation_operator = collective_creation_operator.conj().T
            next_coupling_op = (
                collective_creation_operator if self.modes > 1 else None
            )
            
            tracked_mode0 = {"a_mode0": collective_annihilation_operator} if track_operators else None
            iterator.add_mode(
                hamiltonian_nl,
                cos_half_op,           # Couples to fermionic c†  
                next_coupling_op,      # For next bosonic coupling
                self.Gamma,            # Coupling strength
                tracked_operators=tracked_mode0,
                store_basis=store_basis,
            )
            
            # Step 3: Add remaining bosonic modes
            remaining_bosonic_modes = self.modes - 1
            
            for idx in range(0, remaining_bosonic_modes):
                frequency_k = self.linear_frequencies[idx]
                diag_k = frequency_k * (np.arange(self.dimensions[idx + 1]) + 1 / 2)
                hamiltonian_k = diags(diag_k, 0)
                
                # Current mode's coupling operator (couples to previous mode)
                data = np.sqrt(np.arange(1, self.dimensions[idx + 1]))
                linear_destroy_op_k = diags([data], [1], dtype=np.float64)
                # Coupling operator for next mode (if this is not the last mode)
                if idx < remaining_bosonic_modes - 1:  # Not the last bosonic mode
                    coupling_operator_next = linear_destroy_op_k.T.copy()
                else:
                    coupling_operator_next = None
                
                # Add mode with sequential coupling
                tracked_linear = {f"a_mode{idx + 1}": linear_destroy_op_k} if track_operators else None
                iterator.add_mode(
                    hamiltonian_k, 
                    linear_destroy_op_k,      # Couples to previous mode
                    coupling_operator_next,   # Will couple to next mode (if any)
                    self.linear_coupling[idx], # Coupling strength
                    tracked_operators=tracked_linear,
                    store_basis=store_basis,
                )

        else:
            # ORIGINAL ARCHITECTURE: Start with bosonic mode (no fermion)
            
            # Calculate hamiltonian and coupling operators
            hamiltonian_0, cos_half_op, collective_creation_operator = self.hamiltonian_nl(
                phase_ext, return_coupling_ops=True
            )
            initial_coupling_op = (
                collective_creation_operator if self.modes > 1 else None
            )
            collective_annihilation_operator = collective_creation_operator.conj().T

            tracked_mode0 = {"a_mode0": collective_annihilation_operator} if track_operators else None
            iterator.add_initial_mode(
                hamiltonian_0,
                initial_coupling_op,
                tracked_operators=tracked_mode0,
                store_basis=store_basis,
            )
            
            # Add remaining bosonic modes (standard logic)
            remaining_bosonic_modes = self.modes - 1
            
            for idx in range(0, remaining_bosonic_modes):
                frequency_k = self.linear_frequencies[idx]
                diag_k = frequency_k * (np.arange(self.dimensions[idx + 1]) + 1 / 2)
                hamiltonian_k = diags(diag_k, 0)

                # Current mode's coupling operator (couples to previous mode)
                data = np.sqrt(np.arange(1, self.dimensions[idx + 1]))
                linear_destroy_op_k = diags([data], [1], dtype=np.float64)
                # Coupling operator for next mode (if this is not the last mode)
                if idx < remaining_bosonic_modes - 1:  # Not the last bosonic mode
                    coupling_operator_next = linear_destroy_op_k.T.copy()
                else:
                    coupling_operator_next = None
                
                # Add mode with sequential coupling
                tracked_linear = {f"a_mode{idx + 1}": linear_destroy_op_k} if track_operators else None
                iterator.add_mode(
                    hamiltonian_k, 
                    linear_destroy_op_k,      # Couples to previous mode
                    coupling_operator_next,   # Will couple to next mode (if any)
                    self.linear_coupling[idx], # Coupling strength
                    tracked_operators=tracked_linear,
                    store_basis=store_basis,
                )

        # Store the diagonalizer for access to basis transformations
        self._last_diagonalizer = iterator
        
        return iterator.energies, iterator.basis_vectors

    def get_basis_transformations(self) -> dict[int, np.ndarray]:
        """
        Get all basis transformation matrices from the last eigensystem calculation.
        
        Returns:
            dict[int, np.ndarray]: Dictionary mapping mode indices to their transformation matrices.
                                  Each matrix is an independent copy.
                                  
        Raises:
            AttributeError: If eigensystem() hasn't been called yet.
        """
        if not hasattr(self, '_last_diagonalizer') or self._last_diagonalizer is None:
            raise AttributeError("No eigensystem calculation found. Call eigensystem() first.")
        
        return self._last_diagonalizer.get_basis_transformations()

    def get_tracked_operators(self) -> dict[str, np.ndarray]:
        """
        Retrieve all operators tracked during the last eigensystem call.

        Returns:
            dict[str, np.ndarray]: Mapping from operator names to matrices
            expressed in the truncated eigenbasis.
        """
        if not hasattr(self, '_last_diagonalizer') or self._last_diagonalizer is None:
            raise AttributeError("No eigensystem calculation found. Call eigensystem() first.")

        return self._last_diagonalizer.get_tracked_operators()

    def get_annihilation_operator(self, name: str) -> np.ndarray:
        """
        Convenience accessor for a specific tracked annihilation operator.

        Args:
            name (str): Operator identifier (e.g., 'a_mode0', 'a_fermion').

        Returns:
            np.ndarray: Operator matrix in the truncated eigenbasis.
        """
        operators = self.get_tracked_operators()
        if name not in operators:
            available = ', '.join(sorted(operators.keys())) or 'none'
            raise KeyError(
                f"Operator '{name}' is not available. Available operators: {available}."
            )
        return operators[name]

    def optimize_dimensions_and_truncations(
        self,
        N: int,
        tolerance: float,
        initial_truncations: Sequence[int],
        tolerance_type: str = 'max',
        max_iterations: int = 100,
        dimension_increment: int = 1,
        truncation_increment: int = 1,
        verbose: bool = False,
    ) -> dict[str, Union[list[int], np.ndarray, int]]:
        """
        Optimize dimensions and truncations to achieve convergence of the first N eigenenergies.

        This method iteratively increases the Hilbert-space dimensions and truncation levels
        starting from the current instance's dimensions and the provided initial truncations,
        until the deviation between consecutive iterations for the first N states is below
        the specified tolerance.

        Parameters
        ----------
        N : int
            Number of eigenenergies to check for convergence.
        tolerance : float
            Tolerance for convergence. Deviation is computed as the maximum absolute difference
            ('max') or the sum of absolute differences ('absolute_sum') between consecutive
            iterations' energies.
        initial_truncations : Sequence[int]
            Initial truncation levels for each diagonalization step. Length must match the
            number of modes (including fermionic if present).
        tolerance_type : str, optional
            Type of deviation calculation: 'max' (default) or 'absolute_sum'.
        max_iterations : int, optional
            Maximum number of iterations to perform. Default is 100.
        dimension_increment : int, optional
            Increment for dimensions per iteration. Default is 1.
        truncation_increment : int, optional
            Increment for truncations per iteration. Default is 1.
        verbose : bool, optional
            If True, print iteration details. Default is False.

        Returns
        -------
        dict
            Dictionary containing:
            - 'dimensions': list of optimized dimensions
            - 'truncations': list of optimized truncations
            - 'energies': np.ndarray of the first N converged energies
            - 'iterations': int, number of iterations performed

        Raises
        ------
        ValueError
            If inputs are invalid or convergence is not reached within max_iterations.
        """
        if N <= 0:
            raise ValueError("N must be a positive integer.")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if tolerance_type not in ['max', 'absolute_sum']:
            raise ValueError("tolerance_type must be 'max' or 'absolute_sum'.")

        expected_trunc_len = self.modes + (1 if self.has_fermionic_coupling else 0)
        if len(initial_truncations) != expected_trunc_len:
            raise ValueError(
                f"initial_truncations must have length {expected_trunc_len} "
                f"(modes: {self.modes}, fermionic: {self.has_fermionic_coupling})."
            )

        current_dims = list(self.dimensions)
        current_truncs = list(initial_truncations)
        prev_energies = None
        converged = False
        converged_energies = None
        convergence_iteration = None

        for iteration in range(max_iterations):
            # Create a new Circuit instance with current dimensions
            new_circuit = Circuit(
                non_linear_frequency=self.non_linear_frequency,
                non_linear_phase_zpf=self.non_linear_phase_zpf,
                dimensions=current_dims,
                Ej=self.Ej,
                linear_frequencies=self.linear_frequencies,
                linear_couplings=self.linear_coupling,
                Ej_second=self.Ej_second,
                Gamma=self.Gamma,
                epsilon_r=self.epsilon_r,
                phase_ext=self.phase_ext,
            )

            # Copy harmonic modes store if present
            if self._harmonic_modes_store is not None:
                new_circuit._store_harmonic_modes(
                    self._harmonic_modes_store["frequencies"],
                    self._harmonic_modes_store["phase_zpf"]
                )

            energies, _ = new_circuit.eigensystem(truncation=current_truncs)
            energies_N = energies[:N]

            if prev_energies is not None:
                diff = np.abs(energies_N - prev_energies)
                if tolerance_type == 'max':
                    deviation = np.max(diff)
                else:
                    deviation = np.sum(diff)

                if verbose:
                    print(f"Iteration {iteration}: deviation = {deviation:.2e}")

                if deviation < tolerance:
                    converged = True
                    converged_energies = energies_N.copy()
                    convergence_iteration = iteration
                    if verbose:
                        print(f"Converged at iteration {iteration}")
                    break

            prev_energies = energies_N.copy()

            # Increase dimensions and truncations
            current_dims = [d + dimension_increment for d in current_dims]
            current_truncs = [t + truncation_increment for t in current_truncs]

        if not converged:
            if verbose:
                print(f"Did not converge within {max_iterations} iterations")
            raise ValueError(f"Did not converge within {max_iterations} iterations")

        # Minimize truncations by reducing each mode step-by-step
        for i in range(len(current_truncs)):
            min_trunc = 2 if i == 0 and self.has_fermionic_coupling else 1
            while current_truncs[i] > min_trunc:
                temp_truncs = current_truncs.copy()
                temp_truncs[i] -= truncation_increment
                if temp_truncs[i] < min_trunc:
                    break

                # Create circuit with reduced truncation
                new_circuit = Circuit(
                    non_linear_frequency=self.non_linear_frequency,
                    non_linear_phase_zpf=self.non_linear_phase_zpf,
                    dimensions=current_dims,
                    Ej=self.Ej,
                    linear_frequencies=self.linear_frequencies,
                    linear_couplings=self.linear_coupling,
                    Ej_second=self.Ej_second,
                    Gamma=self.Gamma,
                    epsilon_r=self.epsilon_r,
                    phase_ext=self.phase_ext,
                )
                if self._harmonic_modes_store is not None:
                    new_circuit._store_harmonic_modes(
                        self._harmonic_modes_store["frequencies"],
                        self._harmonic_modes_store["phase_zpf"]
                    )

                energies, _ = new_circuit.eigensystem(truncation=temp_truncs)
                if len(energies) < N:
                    break  # Cannot reduce further, not enough states
                energies_N = energies[:N]
                diff = np.abs(energies_N - converged_energies)
                if tolerance_type == 'max':
                    deviation = np.max(diff)
                else:
                    deviation = np.sum(diff)

                if deviation < tolerance:
                    current_truncs = temp_truncs
                    if verbose:
                        print(f"Reduced truncation {i} to {current_truncs[i]}")
                else:
                    break  # Cannot reduce further for this mode

        return {
            'dimensions': current_dims,
            'truncations': current_truncs,
            'energies': converged_energies,
            'iterations': convergence_iteration,
        }

    def parameter_names(self) -> list[str]:
        """Return the ordered list of physical parameters used by the circuit."""
        names = ["non_linear_frequency", "non_linear_phase_zpf", "Ej", "Ej_second"]
        names.extend(
            [f"linear_frequency_{idx}" for idx in range(len(self.linear_frequencies))]
        )
        names.extend(
            [f"linear_coupling_{idx}" for idx in range(len(self.linear_coupling))]
        )
        return names

    def parameter_vector(self) -> np.ndarray:
        """Return the current parameter values following :meth:`parameter_names`."""
        parts: list[np.ndarray] = [
            np.array([self.non_linear_frequency], dtype=float),
            np.array([self.non_linear_phase_zpf], dtype=float),
            np.array([self.Ej], dtype=float),
            np.array([self.Ej_second], dtype=float),
            np.array(self.linear_frequencies, dtype=float),
            np.array(self.linear_coupling, dtype=float),
        ]
        for idx in range(4, 6):
            if parts[idx].size == 0:
                parts[idx] = np.array([], dtype=float)
        return np.concatenate(parts)

    def eigensystem_with_gradients(
        self,
        truncation: int | Sequence[int],
        phase_ext: Optional[float] = None,
        *,
        track_operators: bool = True,
        store_basis: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Return eigenenergies, eigenvectors and Hellmann–Feynman gradients."""
        energies, vectors = self.eigensystem(
            truncation=truncation,
            phase_ext=phase_ext,
            track_operators=track_operators,
            store_basis=store_basis,
        )
        gradient_matrix, names = self._energy_derivative_matrix(
            phase_ext if phase_ext is not None else self.phase_ext
        )
        return energies, vectors, gradient_matrix, names

    # ------------------------------------------------------------------
    # Internal helpers for parameter derivatives
    # ------------------------------------------------------------------
    def _get_mode_annihilation(self, mode_index: int) -> np.ndarray:
        """Fetch annihilation operator for a mode in the eigenbasis."""
        name = f"a_mode{mode_index}"
        op = self.get_annihilation_operator(name)
        return op

    def _energy_derivative_matrix(self, phase_ext: float) -> tuple[np.ndarray, list[str]]:
        """Compute dE/dparam for all eigenstates via Hellmann–Feynman."""
        if not hasattr(self, "_last_diagonalizer") or self._last_diagonalizer is None:
            raise AttributeError("No eigensystem available. Call eigensystem() first.")

        tracked_ops = self.get_tracked_operators()
        num_states = self._last_diagonalizer.energies.shape[0]
        identity = np.eye(num_states, dtype=np.complex128)

        columns: list[np.ndarray] = []
        names: list[str] = []

        # Non-linear mode operators
        a0 = tracked_ops["a_mode0"]
        adag0 = a0.conj().T
        number0 = adag0 @ a0

        names.append("non_linear_frequency")
        columns.append(np.real(np.diag(number0 + 0.5 * identity)))

        phi_op = self.non_linear_phase_zpf * (a0 + adag0)
        phi_shift = phi_op + phase_ext * identity
        cos_phi = cosm(phi_shift)
        sin_phi = sinm(phi_shift)
        cos_2phi = cosm(2.0 * phi_shift)
        sin_2phi = sinm(2.0 * phi_shift)
        ann_sum = a0 + adag0

        names.append("non_linear_phase_zpf")
        dH_dphi = self.Ej * (sin_phi @ ann_sum)
        dH_dphi += 2.0 * self.Ej_second * (sin_2phi @ ann_sum)
        columns.append(np.real(np.diag(dH_dphi)))

        names.append("Ej")
        columns.append(np.real(np.diag(-cos_phi)))

        names.append("Ej_second")
        columns.append(np.real(np.diag(-cos_2phi)))

        # Linear mode frequencies
        for idx in range(len(self.linear_frequencies)):
            mode = idx + 1
            a_k = tracked_ops[f"a_mode{mode}"]
            adag_k = a_k.conj().T
            number_k = adag_k @ a_k
            names.append(f"linear_frequency_{idx}")
            columns.append(np.real(np.diag(number_k + 0.5 * identity)))

        # Linear mode couplings
        for idx in range(len(self.linear_coupling)):
            a_prev = tracked_ops[f"a_mode{idx}"]
            a_next = tracked_ops[f"a_mode{idx + 1}"]
            adag_prev = a_prev.conj().T
            adag_next = a_next.conj().T
            coupling_op = a_prev @ adag_next + adag_prev @ a_next
            names.append(f"linear_coupling_{idx}")
            columns.append(np.real(np.diag(coupling_op)))

        gradient_matrix = (
            np.column_stack(columns) if columns else np.zeros((num_states, 0))
        )
        return gradient_matrix, names

    @property
    def collective_frequency(self) -> float:
        """Return the collective mode frequency (currently equal to the nonlinear frequency)."""
        return self.non_linear_frequency

    @property
    def Ec(self) -> float:
        """Compute the charging energy in GHz for convenience."""
        return self.collective_frequency * self.non_linear_phase_zpf**2 / 4

    @property
    def El(self) -> float:
        """Compute the inductive energy in GHz for convenience."""
        return self.collective_frequency / (2 * self.non_linear_phase_zpf**2)

    @property
    def bare_modes(self) -> dict[str, np.ndarray]:
        """Convenience wrapper for :meth:`compute_bare_modes` using the circuit instance."""
        tridiagonal = self._build_lanczos_tridiagonal()
        return self.compute_bare_modes(
            tridiagonal=tridiagonal,
            non_linear_phase_zpf=self.non_linear_phase_zpf,
        )

    def _build_lanczos_tridiagonal(self) -> np.ndarray:
        """Construct the Lanczos tridiagonal matrix from current physical parameters."""
        diagonal = np.concatenate(
            (
                np.array([self.non_linear_frequency], dtype=float),
                np.asarray(self.linear_frequencies, dtype=float),
            )
        )
        tridiagonal = np.diag(diagonal)
        if self.linear_frequencies.size > 0:
            off_diag = (
                np.diag(self.linear_coupling, k=1) + np.diag(self.linear_coupling, k=-1)
            )
            tridiagonal = tridiagonal + off_diag
        return tridiagonal

    @staticmethod
    def compute_bare_modes(
        *,
        tridiagonal: np.ndarray,
        non_linear_phase_zpf: float,
    ) -> dict[str, np.ndarray]:
        """
        Recover harmonic frequencies and phase zero-point fluctuations from a Lanczos tridiagonal matrix.

        Parameters
        ----------
        tridiagonal : np.ndarray
            Symmetric tridiagonal matrix produced by the Lanczos transform of the harmonic sector.
        non_linear_phase_zpf : float
            Zero-point phase fluctuation amplitude of the collective nonlinear mode.

        Returns
        -------
        dict
            Dictionary with keys ``'frequencies'`` and ``'phase_zpf'`` (both ``np.ndarray``).
        """
        if tridiagonal.ndim != 2 or tridiagonal.shape[0] != tridiagonal.shape[1]:
            raise ValueError("tridiagonal must be a square matrix.")
        if non_linear_phase_zpf <= 0:
            raise ValueError("non_linear_phase_zpf must be positive.")

        bare_frequencies, eigenvectors = np.linalg.eigh(tridiagonal)

        # First mode corresponds to the collective nonlinear mode (lowest frequency)
        collective_eigenvector = eigenvectors[:, 0]
        if collective_eigenvector[0] < 0:
            collective_eigenvector *= -1

        bare_phase_zpf = non_linear_phase_zpf * collective_eigenvector

        return {
            "frequencies": bare_frequencies,
            "phase_zpf": bare_phase_zpf,
        }

    def harmonic_modes(self) -> dict[str, np.ndarray]:
        """Return cached harmonic data if the circuit was created from harmonic modes."""
        if self._harmonic_modes_store is None:
            raise AttributeError(
                "Harmonic mode data not available for this circuit instance."
            )
        return {
            "frequencies": self._harmonic_modes_store["frequencies"].copy(),
            "phase_zpf": self._harmonic_modes_store["phase_zpf"].copy(),
        }

    @property
    def frequencies(self) -> np.ndarray:
        """Return harmonic frequencies, deriving them if not explicitly stored."""
        if self._harmonic_modes_store is not None:
            return self._harmonic_modes_store["frequencies"].copy()
        return self.bare_modes["frequencies"].copy()

    @property
    def phase_zpf(self) -> np.ndarray:
        """Return harmonic phase ZPF amplitudes."""
        if self._harmonic_modes_store is not None:
            return self._harmonic_modes_store["phase_zpf"].copy()
        return self.bare_modes["phase_zpf"].copy()

    def _store_harmonic_modes(
        self,
        frequencies: Sequence[float],
        phase_zpf: Sequence[float],
    ) -> None:
        freq_array = np.asarray(frequencies, dtype=float)
        phase_array = np.asarray(phase_zpf, dtype=float)
        if freq_array.shape != phase_array.shape:
            raise ValueError("frequencies and phase_zpf must have matching shapes.")
        if freq_array.size != self.modes:
            raise ValueError(
                "Number of harmonic modes must equal the circuit mode count."
            )
        self._harmonic_modes_store = {
            "frequencies": freq_array.copy(),
            "phase_zpf": phase_array.copy(),
        }

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
    frequencies = [5.0, 6.0, 7.8]
    phase_zpf = [0.1, 0.2, 0.01]
    dimensions = [50, 10, 3]
    Ej = 1.0

    circuit = Circuit.from_harmonic_modes(
        frequencies=frequencies,
        phase_zpf=phase_zpf,
        dimensions=dimensions,
        Ej=Ej,
        phase_ext=0.0,
    )

    print("Basic Circuit:")
    print(f"Non-linear frequency: {circuit.non_linear_frequency:.3f} GHz")
    print(f"Non-linear phase ZPF: {circuit.non_linear_phase_zpf:.3f}")

    circuit_with_fermion = Circuit.from_harmonic_modes(
        frequencies=frequencies,
        phase_zpf=phase_zpf,
        dimensions=dimensions,
        Ej=Ej,
        Gamma=0.5,
        epsilon_r=0.2,
        phase_ext=0.0,
    )

    print("\nCircuit with fermionic coupling:")
    print(f"Gamma (fermion-boson coupling): {circuit_with_fermion.Gamma} GHz")
    print(f"epsilon_r (fermion energy): {circuit_with_fermion.epsilon_r} GHz")

    energies_basic, _ = circuit.eigensystem(truncation=10)
    energies_with_fermion, _ = circuit_with_fermion.eigensystem(truncation=10)

    print(f"\nLowest 5 energies (basic): {energies_basic[:5]}")
    print(f"Lowest 5 energies (with fermion): {energies_with_fermion[:5]}")

import numpy as np
from typing import Dict, Optional
from scipy import sparse
from scipy.linalg import eigh

class IterativeHamiltonianDiagonalizer:
    """
    Iteratively diagonalize a multi-mode Hamiltonian by adding one mode at a time,
    truncating to the lowest-energy subspace, and using sequential coupling.

    SEQUENTIAL COUPLING LOGIC:
    - Each mode couples only to the next mode in the sequence
    - Mode 0 has coupling operator X₀ that couples to mode 1
    - Mode 1 has coupling operator X₁ that couples to mode 2
    - And so on...
    - The coupling Hamiltonian between modes k and k+1 is: g_k * (X_k^eff ⊗ X_{k+1} + h.c.)
    - This is different from the previous logic where X₀ was propagated to all modes

    USAGE PATTERN:
    1. diag.add_initial_mode(H₀, X₀)  # X₀ will couple to mode 1
    2. diag.add_mode(H₁, X₁_current, X₁_next, g₁)  # X₁_current couples to X₀, X₁_next couples to mode 2
    3. diag.add_mode(H₂, X₂_current, X₂_next, g₂)  # X₂_current couples to X₁_next, X₂_next couples to mode 3
    4. etc.

    Attributes:
        num_keep                 (int): Number of low-energy states to retain at each step.
        energies                 (np.ndarray): Current truncated eigenvalues, shape (num_keep,).
        basis_vectors            (np.ndarray): Columns are the truncated basis vectors,
                                              shape (prev_dim, num_keep).
        effective_coupling       (np.ndarray): Effective coupling operator for the current mode,
                                              shape (num_keep, num_keep).
        tracked_operators        (dict): Dictionary storing operators to track through diagonalization.
        _mode_coupling_operators (dict): Internal storage for each mode's coupling operator.
        _mode_basis_transformations (dict): Maps mode index to its basis transformation matrix (evecs).
                                           Keys: mode indices (0, 1, 2, ...)
                                           Values: transformation matrices from full to truncated basis.
    """

    def __init__(self, num_keep: int):
        """
        Initialize the iterative diagonalizer.

        Args:
            num_keep (int): Number of eigenstates to keep after each diagonalization.
        """
        self.num_keep = num_keep
        self.energies: np.ndarray = None
        self.basis_vectors: np.ndarray = None
        self.effective_coupling: np.ndarray = None
        self.tracked_operators: dict = {}
        self._current_mode = -1  # Track which mode we're currently on
        self._mode_coupling_operators: dict = (
            {}
        )  # Store coupling operators for each mode
        self._mode_basis_transformations: dict = (
            {}
        )  # Store basis transformation matrices (evecs) for each mode


    def add_initial_mode(
        self,
        hamiltonian: np.ndarray,
        coupling_operator: np.ndarray = None,
        tracked_operators: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Diagonalize the first-mode Hamiltonian and optionally store its coupling operator.

        Args:
            hamiltonian               (np.ndarray): Hamiltonian of the first mode, shape (d0, d0).
            coupling_operator (np.ndarray, optional): Operator coupling this mode to the next mode,
                                                     shape (d0, d0). Can be None if this is the last mode.
            tracked_operators (dict[str, np.ndarray], optional): Operators defined on this mode that
                                                                 should be tracked in the truncated basis.
        """
        self._current_mode = 0
        
        # Adapt truncation to actual system size
        h0_size = hamiltonian.shape[0]
        effective_truncation = min(self.num_keep, h0_size)
        evals, evecs = eigh(hamiltonian, subset_by_index=(0, effective_truncation - 1))
        self.energies = evals
        self.basis_vectors = evecs

        # Store the basis transformation matrix for this mode
        self._mode_basis_transformations[self._current_mode] = evecs.copy()

        # Store the coupling operator for this mode (if provided)
        if coupling_operator is not None:
            # Compute effective coupling in truncated basis: V† X0 V
            self.effective_coupling = evecs.conj().T @ coupling_operator @ evecs
            self._mode_coupling_operators[0] = self.effective_coupling
        else:
            self.effective_coupling = None

        if tracked_operators:
            for name, operator in tracked_operators.items():
                operator_dense = self._ensure_dense(operator)
                if operator_dense.shape != hamiltonian.shape:
                    raise ValueError(
                        f"Tracked operator '{name}' has shape {operator_dense.shape},"
                        f" expected {hamiltonian.shape}."
                    )
                transformed = evecs.conj().T @ operator_dense @ evecs
                self.tracked_operators[name] = transformed.copy()

    def add_mode(
        self,
        hamiltonian: np.ndarray,
        current_coupling_operator: np.ndarray,
        coupling_operator_next: np.ndarray = None,
        coupling_strength: float = 1.0,
        tracked_operators: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Add a new mode with sequential coupling: each mode couples only to the next mode.

        Args:
            hamiltonian               (np.ndarray): Hamiltonian of the new mode, shape (dk, dk).
            current_coupling_operator (np.ndarray): Operator for the current mode that couples to the previous mode,
                                                   shape (dk, dk).
            coupling_operator_next     (np.ndarray, optional): Operator for this mode that will couple to the next mode,
                                                   shape (dk, dk). Can be None if this is the last mode.
            coupling_strength            (float): Strength g_k of the coupling term between previous and current mode.
            tracked_operators (dict[str, np.ndarray], optional): Operators defined on the new mode to be tracked.
        """
        self._current_mode += 1
        previous_dimension = len(self.energies)  # Use actual effective dimension from previous step
        new_dimension = hamiltonian.shape[0]

        # Get the coupling operator from the PREVIOUS mode (not the current one)
        prev_mode_index = self._current_mode - 1
        if prev_mode_index in self._mode_coupling_operators:
            prev_coupling = self._mode_coupling_operators[prev_mode_index]
        else:
            raise ValueError(
                f"No coupling operator found for previous mode {prev_mode_index}. "
                "Make sure to provide coupling_operator when adding the previous mode."
            )

        # Build the total Hamiltonian using both coupling operators
        H_total = self._build_total_hamiltonian_sequential(
            hamiltonian,
            prev_coupling,
            current_coupling_operator,
            coupling_strength,
            previous_dimension,
            new_dimension,
        )

        # Adapt truncation to actual system size
        total_size = H_total.shape[0]
        effective_truncation = min(self.num_keep, total_size)

        # Diagonalize
        evals, evecs = eigh(H_total, subset_by_index=(0, effective_truncation - 1))
        self.energies = evals
        self.basis_vectors = evecs

        # Store the basis transformation matrix for this mode
        self._mode_basis_transformations[self._current_mode] = evecs.copy()

        # Store the coupling operator for the NEXT iteration (if provided)
        if coupling_operator_next is not None:
            # Transform the coupling operator to the current truncated basis
            # coupling_operator_next acts on the current mode's space (dimension d_new)
            # But we need it in the truncated basis for use in the next iteration
            
            # First extend it to the full current space: I_prev ⊗ coupling_operator_next
            prev_identity = np.eye(previous_dimension)
            
            # Convert sparse matrices to dense for kron product
            if hasattr(coupling_operator_next, 'toarray'):
                coupling_operator_next = coupling_operator_next.toarray()
            
            # Extend: this creates an operator in the full (d_prev * d_new) space
            coupling_extended = np.kron(prev_identity, coupling_operator_next)
            
            # Transform to truncated basis: V† (I_prev ⊗ coupling_operator_next) V
            effective_coupling = evecs.conj().T @ coupling_extended @ evecs
            
            self._mode_coupling_operators[self._current_mode] = effective_coupling
            self.effective_coupling = effective_coupling
        else:
            self.effective_coupling = None

        # Update already tracked operators (they act on previous truncated space)
        if self.tracked_operators:
            identity_new = np.eye(new_dimension)
            updated_operators = {}
            for name, operator in self.tracked_operators.items():
                operator_extended = np.kron(operator, identity_new)
                transformed = evecs.conj().T @ operator_extended @ evecs
                updated_operators[name] = transformed.copy()
            self.tracked_operators = updated_operators

        # Register new operators provided for this mode
        if tracked_operators:
            identity_prev = np.eye(previous_dimension)
            for name, operator in tracked_operators.items():
                if name in self.tracked_operators:
                    raise ValueError(
                        f"Tracked operator '{name}' is already registered."
                    )
                operator_dense = self._ensure_dense(operator)
                if operator_dense.shape != hamiltonian.shape:
                    raise ValueError(
                        f"Tracked operator '{name}' has shape {operator_dense.shape},"
                        f" expected {hamiltonian.shape}."
                    )
                operator_extended = np.kron(identity_prev, operator_dense)
                transformed = evecs.conj().T @ operator_extended @ evecs
                self.tracked_operators[name] = transformed.copy()

    def _build_total_hamiltonian_sequential(
        self,
        new_hamiltonian: np.ndarray,
        prev_coupling: np.ndarray,
        current_coupling: np.ndarray,
        coupling_strength: float,
        d_prev: int,
        d_new: int,
    ) -> np.ndarray:
        """
        Build the total Hamiltonian for sequential coupling (each mode couples only to the next).

        Args:
            new_hamiltonian (np.ndarray): Hamiltonian of the new mode.
            prev_coupling (np.ndarray): Effective coupling operator from the previous mode.
            current_coupling (np.ndarray): Coupling operator for the current mode.
            coupling_strength (float): Coupling strength.
            d_prev (int): Dimension of previous truncated space.
            d_new (int): Dimension of new mode.

        Returns:
            np.ndarray: Total Hamiltonian matrix.
        """
        H_prev_sp = sparse.diags(self.energies)
        I_prev_sp = sparse.identity(d_prev)
        I_new_sp = sparse.identity(d_new)

        # For sequential coupling: g * (X_prev_eff ⊗ X_current + X_prev_eff† ⊗ X_current†)
        # This properly couples the previous mode to the current mode
        coupling_term = coupling_strength * (
            sparse.kron(prev_coupling, current_coupling)
            + sparse.kron(prev_coupling.conj().T, current_coupling.conj().T)
        )

        # Build total Hamiltonian: H_prev⊗I_new + I_prev⊗hk + g * (X_prev_eff⊗X_current + h.c.)
        H_total = (
            sparse.kron(H_prev_sp, I_new_sp)
            + sparse.kron(I_prev_sp, new_hamiltonian)
            + coupling_term
        )

        # Convert sparse to dense for diagonalization
        return H_total.toarray() if sparse.issparse(H_total) else H_total
    
    def get_basis_transformations(self) -> dict[int, np.ndarray]:
        """
        Get all basis transformation matrices for all processed modes.
        
        Returns:
            dict[int, np.ndarray]: Dictionary mapping mode indices to their transformation matrices.
                                  Each matrix is an independent copy, not a reference.
        """
        return {mode_idx: matrix.copy() for mode_idx, matrix in self._mode_basis_transformations.items()}

    def get_tracked_operators(self) -> dict[str, np.ndarray]:
        """
        Return copies of all tracked operators in the final truncated eigenbasis.

        Returns:
            dict[str, np.ndarray]: Mapping from operator names to their matrices.
        """
        return {name: operator.copy() for name, operator in self.tracked_operators.items()}

    @staticmethod
    def _ensure_dense(operator: np.ndarray) -> np.ndarray:
        """Convert sparse operators to dense arrays when necessary."""
        if hasattr(operator, "toarray"):
            return operator.toarray()
        return np.asarray(operator)

if __name__ == "__main__":
    # Example usage with sequential coupling (replace with your own operator constructors)

    def destroy(dimension: int) -> np.ndarray:
        indices = np.arange(1, dimension)
        data = np.sqrt(indices)
        return np.diag(data, k=1)


    def creation(dimension: int) -> np.ndarray:
        return destroy(dimension).T.conj()

    # Settings
    num_keep = 20
    cutoffs = [100, 80, 80, 80]
    coupling_strengths = [None, 0.1, 0.05, 0.02]
    lambda_val = 2.5

    # Initialize
    diag = IterativeHamiltonianDiagonalizer(num_keep)

    # First mode (depends on external parameter λ)
    a0, adag0 = creation(cutoffs[0]), destroy(cutoffs[0])
    H0 = lambda_val * (adag0 @ a0)
    X0 = a0 + adag0  # This mode's coupling operator (couples to mode 1)

    # Example: register cos(phi_zpf * (a† + a)) operator - Method 1 (manual)
    phi_zpf = 0.1  # example value
    x_operator = a0 + adag0  # position-like operator
    cos_operator = np.cos(phi_zpf * x_operator)  # compute cos BEFORE transformation

    # Add initial mode with its coupling operator
    diag.add_initial_mode(H0, X0)


    # Subsequent modes with sequential coupling
    for k in range(1, len(cutoffs)):
        ak, adagk = creation(cutoffs[k]), destroy(cutoffs[k])
        Hk = 1.0 * (adagk @ ak)
        Xk_current = (
            ak + adagk
        )  # This mode's operator that couples to the previous mode
        Xk_next = (
            ak + adagk if k < len(cutoffs) - 1 else None
        )  # This mode's operator for next coupling (None for last mode)

        # Add mode with sequential coupling
        diag.add_mode(Hk, Xk_current, Xk_next, coupling_strengths[k])

    print("\nTruncated energies:", diag.energies)
    print("\nFinal effective operators available:")
    for name in diag.tracked_operators:
        op_info = diag.tracked_operators[name]
        if op_info["effective_operator"] is not None:
            op = op_info["effective_operator"]
            print(f"  {name}: shape {op.shape}, max element {np.max(np.abs(op)):.6f}")
        else:
            print(f"  {name}: not yet available")

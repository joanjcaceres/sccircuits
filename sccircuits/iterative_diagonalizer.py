import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from typing import Tuple


def diagonalize_and_truncate(
    hamiltonian: np.ndarray, num_states: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize a Hermitian matrix (N ≲ 400) and keep only the lowest-energy eigenpairs.

    Args:
        hamiltonian (np.ndarray): Hermitian matrix of shape (N, N).
        num_states    (int): Number of lowest eigenstates to retain.

    Returns:
        eigenvalues  (np.ndarray): Sorted lowest `num_states` eigenvalues, shape (num_states,).
        eigenvectors (np.ndarray): Corresponding eigenvectors as columns, shape (N, num_states).
    """
    # Full diagonalization with LAPACK
    all_evals, all_evecs = eigh(hamiltonian, subset_by_index=(0, num_states - 1))
    # Pick the lowest num_states
    # idx = np.argsort(all_evals)[:num_states]
    return all_evals, all_evecs


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

    def register_operator(
        self, name: str, operator: np.ndarray, mode_index: int = 0
    ) -> None:
        """
        Register an operator to be tracked through the diagonalization process.

        Args:
            name (str): Name identifier for the operator.
            operator (np.ndarray): The operator matrix to track.
            mode_index (int): Index of the mode this operator belongs to (0 for first mode).
        """
        self.tracked_operators[name] = {
            "operator": operator,
            "mode_index": mode_index,
            "effective_operator": None,
        }

    def get_effective_operator(self, name: str) -> np.ndarray:
        """
        Get the effective representation of a tracked operator in the current truncated basis.

        Args:
            name (str): Name of the operator to retrieve.

        Returns:
            np.ndarray: Effective operator in the truncated basis, shape (num_keep, num_keep).
        """
        if name not in self.tracked_operators:
            raise KeyError(
                f"Operator '{name}' not registered. Use register_operator() first."
            )

        effective_op = self.tracked_operators[name]["effective_operator"]
        if effective_op is None:
            raise ValueError(
                f"Operator '{name}' not yet available. Add the corresponding mode first."
            )

        return effective_op

    def register_nonlinear_operator(
        self, name: str, function, base_operators: dict, mode_index: int = 0
    ) -> None:
        """
        Register a nonlinear function of operators (e.g., cos, exp, etc.) to be tracked.

        Args:
            name (str): Name identifier for the nonlinear operator.
            function: Function to apply to the base operators (e.g., np.cos, np.exp).
            base_operators (dict): Dictionary of base operators, e.g., {'x': a + adag, 'p': 1j*(adag - a)}.
            mode_index (int): Index of the mode this operator belongs to.

        Example:
            # For cos(phi * x) where x = a + a†
            diag.register_nonlinear_operator(
                "cos_phi_x",
                lambda x: np.cos(phi_zpf * x),
                {'x': a + adag},
                mode_index=0
            )
        """
        # Construct the argument for the nonlinear function
        if len(base_operators) == 1:
            arg_name, arg_op = next(iter(base_operators.items()))
            result_operator = function(arg_op)
        else:
            # For multivariate functions, pass as keyword arguments
            result_operator = function(**base_operators)

        self.register_operator(name, result_operator, mode_index)

    def register_coupling_operator(
        self, name: str, op1_name: str, op2_name: str, target_mode: int
    ) -> None:
        """
        Register a coupling operator like a₀ aₖ for Jacobian calculations.
        The operator will be constructed when the target mode is added.

        Args:
            name (str): Name for the coupling operator.
            op1_name (str): Name of the first operator (should be already registered).
            op2_name (str): Name of the second operator (will be registered when target_mode is added).
            target_mode (int): Mode index when this coupling becomes active.

        Example:
            # Register a₀ a₂ coupling (for gₖ a₀ aₖ terms)
            diag.register_coupling_operator("a0_a2_coupling", "a0", "a2", target_mode=2)
        """
        self.tracked_operators[name] = {
            "type": "coupling",
            "op1_name": op1_name,
            "op2_name": op2_name,
            "target_mode": target_mode,
            "effective_operator": None,
        }

    def add_initial_mode(
        self, h0: np.ndarray, coupling_operator: np.ndarray = None
    ) -> None:
        """
        Diagonalize the first-mode Hamiltonian and optionally store its coupling operator.

        Args:
            h0                (np.ndarray): Hamiltonian of the first mode, shape (d0, d0).
            coupling_operator (np.ndarray, optional): Operator coupling this mode to the next mode,
                                                     shape (d0, d0). Can be None if this is the last mode.
        """
        self._current_mode = 0
        evals, evecs = diagonalize_and_truncate(h0, self.num_keep)
        self.energies = evals
        self.basis_vectors = evecs

        # Store the coupling operator for this mode (if provided)
        if coupling_operator is not None:
            # Compute effective coupling in truncated basis: V† X0 V
            self.effective_coupling = evecs.conj().T @ coupling_operator @ evecs
            self._mode_coupling_operators[0] = self.effective_coupling
        else:
            self.effective_coupling = None

        # Update effective operators for mode 0
        self._update_operators_for_mode(0, evecs)

    def _update_operators_for_mode(self, mode_index: int, evecs: np.ndarray) -> None:
        """
        Update effective operators for a specific mode after diagonalization.

        Args:
            mode_index (int): Index of the mode being processed.
            evecs (np.ndarray): Eigenvectors from diagonalization.
        """
        for name, op_info in self.tracked_operators.items():
            # Skip coupling operators (they don't have mode_index)
            if op_info.get("type") == "coupling":
                continue

            if op_info.get("mode_index") == mode_index:
                # Transform operator to truncated basis: V† Op V
                op_info["effective_operator"] = (
                    evecs.conj().T @ op_info["operator"] @ evecs
                )

    def add_mode(
        self,
        hk: np.ndarray,
        current_coupling_operator: np.ndarray,
        coupling_operator_next: np.ndarray = None,
        coupling_strength: float = 1.0,
    ) -> None:
        """
        Add a new mode with sequential coupling: each mode couples only to the next mode.

        Args:
            hk                       (np.ndarray): Hamiltonian of the new mode, shape (dk, dk).
            current_coupling_operator (np.ndarray): Operator for the current mode that couples to the previous mode,
                                                   shape (dk, dk).
            coupling_operator_next   (np.ndarray, optional): Operator for this mode that will couple to the next mode,
                                                   shape (dk, dk). Can be None if this is the last mode.
            coupling_strength        (float): Strength g_k of the coupling term between previous and current mode.
        """
        self._current_mode += 1
        d_prev = self.num_keep
        d_new = hk.shape[0]

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
            hk,
            prev_coupling,
            current_coupling_operator,
            coupling_strength,
            d_prev,
            d_new,
        )

        # Diagonalize
        evals, evecs = diagonalize_and_truncate(H_total, self.num_keep)
        self.energies = evals
        self.basis_vectors = evecs

        # Store the coupling operator for the NEXT iteration (if provided)
        if coupling_operator_next is not None:
            # The coupling operator acts only on the new mode space.
            # We need to extend it to the full current space: I_prev ⊗ coupling_operator_next
            prev_identity = np.eye(d_prev)
            
            # Convert sparse matrices to dense for kron product
            if hasattr(coupling_operator_next, 'toarray'):
                coupling_operator_next = coupling_operator_next.toarray()
            
            # Correct extension: I_prev ⊗ coupling_operator_next
            coupling_extended = np.kron(prev_identity, coupling_operator_next)
            
            # Transform to the truncated basis
            current_effective_coupling = evecs.conj().T @ coupling_extended @ evecs
            self._mode_coupling_operators[self._current_mode] = (
                current_effective_coupling
            )
            self.effective_coupling = current_effective_coupling
        else:
            self.effective_coupling = None

        # Update all tracked operators
        self._update_all_operators(evecs, d_prev, d_new)

    def _build_total_hamiltonian_sequential(
        self,
        hk: np.ndarray,
        prev_coupling: np.ndarray,
        current_coupling: np.ndarray,
        coupling_strength: float,
        d_prev: int,
        d_new: int,
    ) -> np.ndarray:
        """
        Build the total Hamiltonian for sequential coupling (each mode couples only to the next).

        Args:
            hk (np.ndarray): Hamiltonian of the new mode.
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
            + sparse.kron(I_prev_sp, hk)
            + coupling_term
        )

        # Convert sparse to dense for diagonalization
        return H_total.toarray() if sparse.issparse(H_total) else H_total

    def _update_all_operators(self, evecs: np.ndarray, d_prev: int, d_new: int) -> None:
        """
        Update all tracked operators after adding a new mode.

        Args:
            evecs (np.ndarray): New eigenvectors from diagonalization.
            d_prev (int): Dimension of previous truncated space.
            d_new (int): Dimension of new mode.
        """
        for name, op_info in self.tracked_operators.items():
            if op_info.get("type") == "coupling":
                # Handle coupling operators like a₀ aₖ
                self._update_coupling_operator_type(name, op_info, evecs, d_prev, d_new)
            else:
                # Handle regular single-mode operators
                if op_info["effective_operator"] is not None:
                    # Extend existing effective operator to include new mode
                    if op_info["mode_index"] <= self._current_mode:
                        prev_op_extended = np.kron(
                            op_info["effective_operator"], np.eye(d_new)
                        )
                        # Transform to new truncated basis
                        op_info["effective_operator"] = (
                            evecs.conj().T @ prev_op_extended @ evecs
                        )
                elif op_info["mode_index"] == self._current_mode:
                    # This is a new operator for the current mode
                    prev_identity = np.eye(d_prev)
                    op_extended = np.kron(prev_identity, op_info["operator"])
                    op_info["effective_operator"] = evecs.conj().T @ op_extended @ evecs

    def _update_coupling_operator_type(
        self, name: str, op_info: dict, evecs: np.ndarray, d_prev: int, d_new: int
    ) -> None:
        """
        Update coupling operators like a₀ aₖ when the target mode becomes available.

        Args:
            name (str): Name of the coupling operator.
            op_info (dict): Operator info dictionary.
            evecs (np.ndarray): Current eigenvectors.
            d_prev (int): Previous dimension.
            d_new (int): New mode dimension.
        """
        if self._current_mode == op_info["target_mode"]:
            # Now we can construct the coupling operator
            op1_eff = None
            op2_original = None

            # Find the first operator (should already be effective)
            for other_name, other_info in self.tracked_operators.items():
                if (
                    other_name == op_info["op1_name"]
                    and other_info.get("type") != "coupling"
                    and other_info["effective_operator"] is not None
                ):
                    op1_eff = other_info["effective_operator"]
                    break

            # Find the second operator (should be newly added)
            for other_name, other_info in self.tracked_operators.items():
                if (
                    other_name == op_info["op2_name"]
                    and other_info.get("type") != "coupling"
                    and other_info["mode_index"] == self._current_mode
                ):
                    op2_original = other_info["operator"]
                    break

            if op1_eff is not None and op2_original is not None:
                # Construct the coupling operator: op1_eff ⊗ op2_original
                coupling_full = np.kron(op1_eff, op2_original)
                # Transform to truncated basis
                op_info["effective_operator"] = evecs.conj().T @ coupling_full @ evecs

        elif (
            op_info["effective_operator"] is not None
            and self._current_mode > op_info["target_mode"]
        ):
            # Update existing coupling operator for new modes
            prev_op_extended = np.kron(op_info["effective_operator"], np.eye(d_new))
            op_info["effective_operator"] = evecs.conj().T @ prev_op_extended @ evecs

    def add_mode_sequential(
        self,
        hk: np.ndarray,
        current_coupling_operator: np.ndarray,
        coupling_operator_next: np.ndarray = None,
        coupling_strength: float = 1.0,
    ) -> None:
        """
        Convenience method for adding a mode with sequential coupling (same as add_mode).

        This method makes it explicit that we're using sequential coupling where:
        - Mode k-1 couples to mode k using mode k-1's coupling operator and mode k's current_coupling_operator
        - Mode k will couple to mode k+1 using coupling_operator_next (if provided)

        Args:
            hk                       (np.ndarray): Hamiltonian of the new mode, shape (dk, dk).
            current_coupling_operator (np.ndarray): Operator for the current mode that couples to the previous mode.
            coupling_operator_next   (np.ndarray, optional): Operator for this mode that will couple to the next mode.
            coupling_strength        (float): Strength g_k of the coupling between previous and current mode.
        """
        self.add_mode(
            hk, current_coupling_operator, coupling_operator_next, coupling_strength
        )

    def get_mode_coupling_operator(self, mode_index: int) -> np.ndarray:
        """
        Get the effective coupling operator for a specific mode.

        Args:
            mode_index (int): Index of the mode.

        Returns:
            np.ndarray: Effective coupling operator for the mode.
        """
        if mode_index in self._mode_coupling_operators:
            return self._mode_coupling_operators[mode_index]
        else:
            raise ValueError(f"No coupling operator found for mode {mode_index}")

    def has_coupling_operator(self, mode_index: int) -> bool:
        """
        Check if a mode has a coupling operator stored.

        Args:
            mode_index (int): Index of the mode.

        Returns:
            bool: True if the mode has a coupling operator.
        """
        return mode_index in self._mode_coupling_operators


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

    # Register operators to track (before adding the mode)
    diag.register_operator("a0", a0, mode_index=0)
    diag.register_operator("adag0", adag0, mode_index=0)
    diag.register_operator("n0", adag0 @ a0, mode_index=0)  # number operator

    # Register coupling operators for Jacobian calculations
    # These will be constructed when the target modes are added
    diag.register_coupling_operator("a0_a1_coupling", "a0", "a1", target_mode=1)
    diag.register_coupling_operator("a0_a2_coupling", "a0", "a2", target_mode=2)
    diag.register_coupling_operator("a0_a3_coupling", "a0", "a3", target_mode=3)

    # Example: register cos(phi_zpf * (a† + a)) operator - Method 1 (manual)
    phi_zpf = 0.1  # example value
    x_operator = a0 + adag0  # position-like operator
    cos_operator = np.cos(phi_zpf * x_operator)  # compute cos BEFORE transformation
    diag.register_operator("cos_phi_x", cos_operator, mode_index=0)

    # Example: register cos(phi_zpf * (a† + a)) operator - Method 2 (convenience method)
    diag.register_nonlinear_operator(
        "cos_phi_x_v2", lambda x: np.cos(phi_zpf * x), {"x": a0 + adag0}, mode_index=0
    )

    # Add initial mode with its coupling operator
    diag.add_initial_mode(H0, X0)

    print("After initial mode:")
    print("a0 effective shape:", diag.get_effective_operator("a0").shape)

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

        # Register operators for this mode before adding it
        diag.register_operator(f"a{k}", ak, mode_index=k)
        diag.register_operator(f"adag{k}", adagk, mode_index=k)

        # Add mode with sequential coupling
        diag.add_mode(Hk, Xk_current, Xk_next, coupling_strengths[k])

        print(f"After adding mode {k}:")
        print(f"a0 effective shape: {diag.get_effective_operator('a0').shape}")
        print(f"a{k} effective shape: {diag.get_effective_operator(f'a{k}').shape}")

    print("\nTruncated energies:", diag.energies)
    print("\nFinal effective operators available:")
    for name in diag.tracked_operators:
        op_info = diag.tracked_operators[name]
        if op_info["effective_operator"] is not None:
            op = op_info["effective_operator"]
            print(f"  {name}: shape {op.shape}, max element {np.max(np.abs(op)):.6f}")
        else:
            print(f"  {name}: not yet available")

    # Example: Calculate matrix element <0|a0|1> in the effective basis
    a0_eff = diag.get_effective_operator("a0")
    matrix_element_01 = a0_eff[0, 1]  # <state_0|a0|state_1>
    print(f"\nExample matrix element <0|a0|1> = {matrix_element_01:.6f}")

    # Example: Calculate matrix element <0|cos(phi*x)|0> in the effective basis
    cos_op_eff = diag.get_effective_operator("cos_phi_x")
    cos_matrix_element_00 = cos_op_eff[0, 0]  # <state_0|cos(phi*x)|state_0>
    print(f"Example matrix element <0|cos(φx)|0> = {cos_matrix_element_00:.6f}")

    # ===== COUPLING OPERATORS FOR JACOBIANS =====
    print("\n" + "=" * 60)
    print("COUPLING OPERATORS FOR JACOBIAN CALCULATIONS")
    print("=" * 60)

    coupling_operators = ["a0_a1_coupling", "a0_a2_coupling", "a0_a3_coupling"]

    for coupling_name in coupling_operators:
        try:
            coupling_eff = diag.get_effective_operator(coupling_name)
            print(f"\n{coupling_name}:")
            print(f"  Shape: {coupling_eff.shape}")
            print(f"  Diagonal (first 3): {np.diag(coupling_eff)[:3]}")
            print(f"  Max absolute element: {np.max(np.abs(coupling_eff)):.6f}")

            # Show some off-diagonal elements (these should be non-zero for a†a operators)
            print(
                f"  Off-diagonal elements [0,1], [1,0], [1,2]: {coupling_eff[0,1]:.6f}, {coupling_eff[1,0]:.6f}, {coupling_eff[1,2]:.6f}"
            )

            # Example Jacobian calculation: ∂E_i/∂g_k = ⟨ψ_i|a_0 a_k|ψ_i⟩
            print("  Jacobian elements ⟨ψ_i|a0_ak|ψ_i⟩ for derivatives ∂E_i/∂g_k:")
            for i in range(min(3, coupling_eff.shape[0])):
                jacobian_element = coupling_eff[i, i].real  # diagonal elements
                print(f"    ∂E_{i}/∂g_k = {jacobian_element:.6f}")

        except KeyError:
            print(f"\n{coupling_name}: not available (target mode not reached)")

    print("\nNote: These coupling operators are crucial for computing derivatives")
    print("of eigenvalues with respect to coupling parameters in Hamiltonians")
    print("of the form H = H0 + H1 + ... + Σ g_k a_0 a_k.")

    # Demonstration: Why doing cos() after transformation is WRONG
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Why cos(effective_operator) is WRONG")
    print("=" * 60)

    # Correct way: cos() computed before transformation
    x_eff_correct = diag.get_effective_operator("cos_phi_x")
    correct_result = x_eff_correct[0, 0]

    # Wrong way: cos() computed after transformation
    x_eff = diag.get_effective_operator("a0") + diag.get_effective_operator("adag0")
    wrong_cos_op = np.cos(phi_zpf * x_eff)  # This is MATHEMATICALLY INCORRECT
    wrong_result = wrong_cos_op[0, 0]

    print(
        f"Correct method (cos before transform):  <0|cos(φx)|0> = {correct_result:.6f}"
    )
    print(
        f"Wrong method (cos after transform):    <0|cos(φx_eff)|0> = {wrong_result:.6f}"
    )
    print(
        f"Relative error: {abs(wrong_result - correct_result)/abs(correct_result)*100:.2f}%"
    )

    print("\nConclusion: You MUST compute cos(φ(a† + a)) before any transformations!")
    print("The nonlinear function cos() does NOT commute with unitary transformations.")

    print("\n" + "=" * 80)
    print("SUMMARY: How to use this class effectively")
    print("=" * 80)
    print(
        "1. Register individual operators: diag.register_operator('a0', a0, mode_index=0)"
    )
    print(
        "2. Register nonlinear operators: diag.register_nonlinear_operator('cos_x', np.cos, {'x': x_op})"
    )
    print("3. Add modes iteratively: diag.add_initial_mode() then diag.add_mode()")
    print("4. Get effective operators: diag.get_effective_operator('operator_name')")
    print("5. For tensor products: Use np.kron(eff_op1, eff_op2) after all modes added")
    print("=" * 80)

    # Final demonstration of tensor product
    print("\nFinal example - Tensor product after all modes:")
    try:
        a0_eff = diag.get_effective_operator("a0")
        a1_eff = diag.get_effective_operator("a1")
        kron_a0_a1 = np.kron(a0_eff, a1_eff)
        print(f"np.kron(a0_eff, a1_eff) shape: {kron_a0_a1.shape}")
        print(f"Matrix element [0,0]: {kron_a0_a1[0,0]:.6f}")
        print("This is the efficient way to get tensor products!")
    except Exception as e:
        print(f"Error computing tensor product: {e}")

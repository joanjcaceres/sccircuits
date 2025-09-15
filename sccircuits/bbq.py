from scipy.constants import hbar, e
from scipy.linalg import cosm, eigh
import numpy as np
from functools import reduce
from scipy.sparse import diags
from typing import Union
import matplotlib.pyplot as plt


class BBQ:
    """
    Black Box Quantization (BBQ) for superconducting circuits.
    
    This class implements the Black Box Quantization method for analyzing
    superconducting quantum circuits from their classical circuit parameters
    (capacitance and inductance matrices). It calculates linear modes, 
    frequencies, and zero-point fluctuations needed for circuit quantization.
    
    The BBQ method is particularly useful when you have the classical circuit
    description and want to extract the quantum parameters without detailed
    knowledge of the circuit topology.
    
    Attributes
    ----------
    C_matrix : np.ndarray
        Capacitance matrix of the circuit
    L_inv_matrix : np.ndarray  
        Inverse inductance matrix of the circuit
    non_linear_nodes : tuple
        Indices of nodes containing nonlinear elements (Josephson junctions)
    """
    
    def __init__(
        self,
        C_matrix: np.ndarray,
        L_inv_matrix: np.ndarray,
        non_linear_nodes: tuple = (-1, 0),
    ):
        """
        Initialize BBQ analysis from circuit matrices.
        
        Parameters
        ----------
        C_matrix : np.ndarray
            Capacitance matrix of the circuit in Farads. Should be symmetric
            and positive definite.
        L_inv_matrix : np.ndarray
            Inverse inductance matrix (1/L) of the circuit in 1/Henries.
            Should have the same shape as C_matrix.
        non_linear_nodes : tuple, optional
            Indices specifying the nodes containing nonlinear elements
            (typically Josephson junctions). Default is (-1, 0).
            
        Raises
        ------
        ValueError
            If C_matrix and L_inv_matrix don't have the same shape.
        """
        self.C_matrix = C_matrix
        self.L_inv_matrix = L_inv_matrix
        self.non_linear_nodes = non_linear_nodes

        # Validate matrices have the same shape
        if C_matrix.shape != L_inv_matrix.shape:
            raise ValueError("C_matrix and L_inv_matrix must have the same shape.")

        self.circuit_dimensions = C_matrix.shape[0]  # Assuming square matrices

        self.C_inv_sqrt = self._C_inv_sqrt()
        self.dynamical_matrix = self._dynamical_matrix()
        self.eigensys_dynamical_matrix = self._eigensys_dynamical_matrix()
        self.linear_modes = self._linear_modes()
        self.phase_zpf_list = self._phase_zpf_list()
        self.linear_modes_GHz = self._linear_modes_GHz()

    def _C_inv_sqrt(self) -> np.ndarray:
        """
        Calculate the square root of the inverse capacitance matrix.

        Returns:
            np.ndarray: The square root of the inverse capacitance matrix.
        """
        eigvals_C, eigvecs_C = eigh(self.C_matrix)
        tolerance = 1e-20
        nonzero_idx = eigvals_C > tolerance

        eigvals_C_nz = eigvals_C[nonzero_idx]
        eigvecs_C_nz = eigvecs_C[:, nonzero_idx]

        C_inv_sqrt = eigvecs_C_nz @ np.diag(1 / np.sqrt(eigvals_C_nz)) @ eigvecs_C_nz.T
        return C_inv_sqrt

    def _dynamical_matrix(self) -> np.ndarray:
        """
        Calculate the dynamical matrix.

        Returns:
            np.ndarray: The dynamical matrix.
        """
        C_inv_sqrt = self.C_inv_sqrt
        dynamical_matrix = C_inv_sqrt @ self.L_inv_matrix @ C_inv_sqrt
        return dynamical_matrix

    def _eigensys_dynamical_matrix(self) -> tuple:
        """
        Calculate the eigenvalues and eigenvectors of the dynamical matrix.

        Returns:
            tuple: Eigenvalues and eigenvectors of the dynamical matrix.
        """
        dynamical_matrix_evals, dynamical_matrix_evecs = eigh(self.dynamical_matrix)

        # Remove non stationary modes
        if self.circuit_dimensions > 1:
            crosses_zero = np.any(
                dynamical_matrix_evecs[:-1, :] * dynamical_matrix_evecs[1:, :] <= 0,
                axis=0,
            )
            dynamical_matrix_evals = dynamical_matrix_evals[crosses_zero]
            dynamical_matrix_evecs = dynamical_matrix_evecs[:, crosses_zero]

        return dynamical_matrix_evals, dynamical_matrix_evecs

    def _linear_modes(self) -> np.ndarray:
        """
        Calculate the linear modes of the circuit.

        Returns:
            np.ndarray: The linear modes of the circuit.
        """
        dynamical_matrix_evals, _ = self.eigensys_dynamical_matrix
        return np.sqrt(dynamical_matrix_evals)

    def _linear_modes_GHz(self) -> np.ndarray:
        """
        Calculate the linear modes in GHz.

        Returns:
            np.ndarray: The linear modes in GHz.
        """
        return self.linear_modes / 2 / np.pi / 1e9

    def _phase_zpf_list(self) -> list:
        """
        Calculate the zero-point fluctuation phase for the selected modes.
        Returns:
            list: The zero-point fluctuation phase for each selected mode.
        """
        phi_0 = hbar / 2 / e
        _, dynamical_matrix_evecs = self.eigensys_dynamical_matrix

        vec = self.C_inv_sqrt @ dynamical_matrix_evecs

        if len(self.non_linear_nodes) > 1:
            phase_zpf_list = (
                np.sqrt(hbar / 2 / self.linear_modes)
                / phi_0
                * (vec[self.non_linear_nodes[-1]] - vec[self.non_linear_nodes[0]])
            )
        else:
            phase_zpf_list = np.sqrt(hbar / 2 / self.linear_modes) / phi_0 * vec[0]
        return phase_zpf_list

    def plot_linear_modes(self, which: Union[list, int] = 0) -> None:
        """
        Plot the linear modes of the circuit.

        Args:
            which (Union[list, int]): The index or list of indices of the modes to plot.
        """
        if isinstance(which, int):
            which = [which]

        _, dynamical_matrix_evecs = self.eigensys_dynamical_matrix

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            dynamical_matrix_evecs[:, which],
            linestyle="-",
            label=[
                rf"$f_{i} = {self.linear_modes_GHz[i]:.2f}$, $\phi_{{zpf}} = {self.phase_zpf_list[i]:.1e}$"
                for i in which
            ],
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Node Index")
        ax.set_ylabel("Mode Amplitude")
        ax.set_title("Linear Modes of the Circuit")
        ax.legend()
        plt.show()

    @property
    def selected_modes(self) -> list:
        return self._selected_modes

    @selected_modes.setter
    def selected_modes(self, modes: list) -> None:
        self._selected_modes = modes

    @property
    def dimensions(self) -> tuple:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dim: tuple) -> None:
        if isinstance(dim, int):
            dim = (dim,)
        if len(self.selected_modes) != len(dim):
            raise ValueError(
                "The amount of dimensions must match the number of selected modes."
            )
        self._dimensions = dim
        self._total_dimension = np.prod(dim)

    def hamiltonian_0(self) -> np.ndarray:
        """
        Calculate the zero-order Hamiltonian.

        Returns:
            np.ndarray: The zero-order Hamiltonian.
        """
        hamiltonian_0 = np.zeros((self._total_dimension, self._total_dimension))

        for idx, dimension in enumerate(self.dimensions):
            mode_index = self.selected_modes[idx]
            energy_GHz = self.linear_modes[mode_index] / 2 / np.pi / 1e9
            diagonal = energy_GHz * (np.arange(dimension) + 1 / 2)
            hamiltonian_0_subspace = diags([diagonal], [0]).toarray()

            factors = [
                hamiltonian_0_subspace if i == idx else np.eye(dim)
                for i, dim in enumerate(self.dimensions)
            ]

            H_emb = reduce(lambda x, y: np.kron(x, y), factors)
            hamiltonian_0 += H_emb

        return hamiltonian_0

    @property
    def Ej_supression_factor(self) -> float:
        all_modes_indices = np.arange(len(self.linear_modes))
        unselected_modes_indices = np.setdiff1d(all_modes_indices, self.selected_modes)
        unselected_zpf = self.phase_zpf_list[unselected_modes_indices]
        return np.exp(-1 / 2 * np.sum(unselected_zpf**2))

    @property
    def phase_operator_nl(self) -> np.ndarray:
        phase_operator_nl_list = []
        for idx, dimension in enumerate(self.dimensions):
            mode_index = self.selected_modes[idx]
            data = np.sqrt(np.arange(1, dimension))
            phase_operator_nl = (
                self.phase_zpf_list[mode_index] * diags([data, data], [1, -1]).toarray()
            )
            phase_operator_nl_list.append(phase_operator_nl)
        return phase_operator_nl_list

    def hamiltonian_nl(self, Ej: float, phi_ext: float) -> np.ndarray:
        """
        Calculate the non-linear Hamiltonian for the circuit.

        Args:
            Ej (float): Josephson energy in GHz.
            phi_ext (float): External flux in rads.

        Returns:
            np.ndarray: The non-linear Hamiltonian.
        """
        phi_nl_total = np.zeros((self._total_dimension, self._total_dimension))

        for idx, _ in enumerate(self.dimensions):
            phase_operator_nl = self.phase_operator_nl[idx]

            factors = [
                phase_operator_nl if i == idx else np.eye(dim)
                for i, dim in enumerate(self.dimensions)
            ]

            phase_operator_nl_term = reduce(lambda x, y: np.kron(x, y), factors)

            phi_nl_total += phase_operator_nl_term

        hamiltonian_nl = -Ej * (
            self.Ej_supression_factor
            * cosm(phi_nl_total + phi_ext * np.eye(self._total_dimension))
            + 1 / 2 * phi_nl_total @ phi_nl_total
        )

        return hamiltonian_nl


if __name__ == "__main__":
    # Example usage
    from .utilities import El_to_L

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
    C_matrix = diags([c_diagonal, c_off_diagonal, c_off_diagonal], [0, 1, -1]).toarray()
    C_matrix[-1, 0] = C_matrix[0, -1] = -Cjb

    L_inv_diagonal = (2 / Lj) * np.ones(N + 1)
    L_inv_diagonal[0] = L_inv_diagonal[-1] = 1 / Lj + 1 / Ljb
    L_inv_off_diagonal = (-1 / Lj) * np.ones(N)
    L_inv_matrix = diags(
        [L_inv_diagonal, L_inv_off_diagonal, L_inv_off_diagonal], [0, 1, -1]
    ).toarray()
    L_inv_matrix[-1, 0] = L_inv_matrix[0, -1] = -1 / Ljb

    circuit = BBQ(
        C_matrix=C_matrix,
        L_inv_matrix=L_inv_matrix,
        non_linear_nodes=(-1, 0),  # Non-linear nodes
    )

    print(circuit.linear_modes_GHz[:3])

    # Example of plotting linear modes
    circuit.plot_linear_modes(which=[0])

    # This one selected afterward seen the linear_modes_GHz and the plots
    circuit.selected_modes = [0]
    circuit.dimensions = 100

    hamiltonian_0 = circuit.hamiltonian_0()

    phi_ext_array = np.linspace(0, np.pi, 100)
    evals_array = []

    for phi_ext in phi_ext_array:
        hamiltonian_nl = circuit.hamiltonian_nl(Ej=Ejb / 1e9, phi_ext=phi_ext)
        hamiltonian = hamiltonian_0 + hamiltonian_nl
        evals = eigh(hamiltonian, eigvals_only=True)
        evals_array.append(evals)

    evals_array = np.array(evals_array)

    transition_energies = evals_array[:, :6] - evals_array[:, 0][:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(phi_ext_array, transition_energies)
    ax.set_xlabel("External Flux (rad)")
    ax.set_ylabel("Transition energy from ground state (GHz)")
    ax.set_title("Energy Levels vs External Flux")
    plt.show()

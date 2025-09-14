"""
Basic tests for Circuit class.
"""

import pytest
import numpy as np

from sccircuits import Circuit


class TestCircuit:
    """Test cases for the Circuit class."""
    
    def test_circuit_initialization(self):
        """Test basic circuit initialization."""
        frequencies = [5.0, 6.0]
        phase_zpf = [0.1, 0.2]
        dimensions = [50, 10]
        Ej = 1.0
        
        circuit = Circuit(
            frequencies=frequencies,
            phase_zpf=phase_zpf,
            dimensions=dimensions,
            Ej=Ej
        )
        
        assert circuit.modes == 2
        assert circuit.Ej == Ej
        assert np.allclose(circuit.frequencies, frequencies)
        assert np.allclose(circuit.phase_zpf, phase_zpf)
    
    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        # Mismatched lengths
        with pytest.raises(ValueError, match="must have the same length"):
            Circuit([5.0], [0.1, 0.2], [50], 1.0)
        
        # Negative frequency
        with pytest.raises(ValueError, match="must be positive"):
            Circuit([-1.0], [0.1], [50], 1.0)
        
        # Negative phase_zpf  
        with pytest.raises(ValueError, match="must be positive"):
            Circuit([5.0], [-0.1], [50], 1.0)
        # Negative phase_zpf
    def test_hamiltonian_construction(self):
        """Test Hamiltonian matrix construction."""
        circuit = Circuit([5.0], [0.1], [10], 1.0)
        H = circuit.hamiltonian_0()
        
        # Check that Hamiltonian is square matrix
        assert H.shape == (10, 10)
        
        # Convert to dense array for Hermitian check
        if hasattr(H, 'toarray'):
            H_dense = H.toarray()
        elif hasattr(H, 'A'):
            H_dense = H.A
        else:
            H_dense = np.array(H)
            
        # Check that it's approximately Hermitian (within numerical precision)
        assert np.allclose(H_dense, H_dense.T.conj())
    
    def test_eigensystem_calculation(self):
        """Test eigensystem calculation."""
        circuit = Circuit([5.0], [0.1], [20], 1.0)
        evals, evecs = circuit.eigensystem(truncation=10)
        
        # Check output shapes
        assert len(evals) == 10
        assert evecs.shape == (20, 10)
        
        # Check that eigenvalues are in ascending order
        assert np.all(evals[:-1] <= evals[1:])
        
        # Check that eigenvectors are normalized
        for i in range(evecs.shape[1]):
            assert np.isclose(np.linalg.norm(evecs[:, i]), 1.0, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])
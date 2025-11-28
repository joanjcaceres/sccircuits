# SCCircuits - Superconducting Circuit Analysis Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17751124.svg)](https://doi.org/10.5281/zenodo.17751124)

A comprehensive Python package for analyzing superconducting quantum circuits, including Black Box Quantization (BBQ) and circuit parameter fitting capabilities.

## Features

- **Circuit Analysis**: Complete superconducting circuit analysis with support for multi-mode systems
- **Black Box Quantization (BBQ)**: Circuit quantization from capacitance and inductance matrices
- **Parameter Fitting**: Advanced fitting capabilities for experimental transition frequency data
- **Interactive Tools**: Point picking tools for data analysis and visualization
- **Numerical Utilities**: Specialized algorithms for quantum circuit Hamiltonians

## Installation

### From source:
```bash
git clone https://github.com/joanjcaceres/sccircuits.git
cd sccircuits
pip install -e .
```

### Dependencies

The package requires:
- `numpy` - Numerical computations
- `scipy` - Scientific computing and optimization
- `matplotlib` - Plotting and visualization
- `ipywidgets` (optional) - Interactive Jupyter widgets

## Quick Start

### Basic Circuit Analysis

```python
import numpy as np
from sccircuits import Circuit

# Define circuit parameters
frequencies = [5.0, 6.0, 7.8]  # GHz
phase_zpf = [0.1, 0.2, 0.01]   # radians
dimensions = [50, 10, 3]       # Hilbert space dimensions
Ej = 1.0                       # GHz
phase_ext = 0.0                # external phase

# Create circuit
circuit = Circuit(
    frequencies=frequencies,
    phase_zpf=phase_zpf,
    dimensions=dimensions,
    Ej=Ej,
    phase_ext=phase_ext
)

# Get eigenenergies and eigenstates
evals, evecs = circuit.eigensystem(truncation=40)
print(f"Ground state energy: {evals[0]:.3f} GHz")
```

### Parameter Fitting

```python
from sccircuits import CircuitFitter

# Experimental transition data
transitions = {
    (0, 1): [(0.0, 4.9), (np.pi/2, 4.7), (np.pi, 4.8)],  # (phase_ext, frequency)
    (0, 2): [(0.0, 9.8), (np.pi/2, 9.4), (np.pi, 9.6)]
}

# Create fitter
fitter = CircuitFitter(
    Ej_initial=1.0,
    non_linear_frequency_initial=5.0,
    non_linear_phase_zpf_initial=0.1,
    dimensions=[50],
    transitions=transitions,
)

# Fit parameters
result = fitter.fit(verbose=1)
stats = fitter.get_fit_statistics()
print(f"Reduced chi²: {stats['reduced_chi_square']:.3f}")
```

### Black Box Quantization

```python
from sccircuits import BBQ
import numpy as np

# Define circuit matrices (example for a transmon)
C_matrix = np.array([[40e-15, -32.9e-15], 
                     [-32.9e-15, 32.9e-15]])  # Capacitance matrix
L_inv_matrix = np.array([[0, 0], 
                         [0, 1/1.23e-9]])      # Inverse inductance matrix

# Create BBQ object
bbq = BBQ(C_matrix, L_inv_matrix, non_linear_nodes=(-1, 0))

# Analyze linear modes
print("Linear mode frequencies (GHz):", bbq.linear_modes_GHz)
print("Phase ZPF:", bbq.phase_zpf_list)
```

### Interactive Point Picking

```python
from sccircuits import PointPicker
import matplotlib.pyplot as plt

# Create some data to pick points from
fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x) + 0.1*np.random.randn(100)
ax.plot(x, y)

# Start interactive point picker
picker = PointPicker(ax)
picker.start()

# After picking points, access them
points = picker.get_points()
print(f"Picked {len(points)} points")
```

## Package Structure

```
sccircuits/
├── __init__.py           # Package initialization and public API
├── circuit.py            # Main Circuit class for superconducting circuits
├── CircuitFitter.py      # Parameter fitting for circuit models
├── bbq.py               # Black Box Quantization implementation
├── transition_fitter.py  # General transition frequency fitting
├── pointpicker.py       # Interactive point selection tool
├── utilities.py         # Numerical utilities (Lanczos algorithm, etc.)
└── iterative_diagonalizer.py  # Multi-mode Hamiltonian diagonalization
```

## Core Classes

### Circuit
Main class for superconducting quantum circuit analysis with support for:
- Multi-mode systems with arbitrary coupling
- Bogoliubov transformations for enhanced stability
- Eigensystem calculation with truncation
- Both dense and sparse matrix operations

### CircuitFitter
Advanced parameter fitting class featuring:
- Multiple optimization algorithms (least squares, differential evolution)
- Statistical analysis and uncertainty quantification
- Multistart optimization for global minima
- Comprehensive fit quality assessment

### BBQ (Black Box Quantization)
Implements circuit quantization from classical circuit parameters:
- Capacitance and inductance matrix analysis
- Linear mode calculation and visualization
- Phase zero-point fluctuation determination
- Support for arbitrary circuit topologies

### TransitionFitter
General-purpose fitting tool for experimental data:
- Flexible model function interface
- Weighted fitting with outlier detection
- Convergence and residual analysis
- Multiple optimization backends

## Examples

The package includes comprehensive examples in the source code. Each major class includes example usage in its `if __name__ == "__main__":` block.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

<!-- ## Citation

If you use this package in your research, please cite:

```bibtex
@software{sccircuits,
  author = {Joan Caceres},
  title = {SCCircuits: Superconducting Circuit Analysis Package},
  url = {https://github.com/joanjcaceres/sccircuits},
  version = {0.1.0},
  year = {2024}
}
``` -->

## Contact

Joan Caceres - contact@joanjcaceres.com

Project Link: [https://github.com/joanjcaceres/sccircuits](https://github.com/joanjcaceres/sccircuits)

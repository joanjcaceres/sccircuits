"""
SCCircuits - Superconducting Circuit Analysis Package

A comprehensive Python package for analyzing superconducting quantum circuits,
including Black Box Quantization (BBQ) and circuit fitting capabilities.

Main Classes:
    Circuit: Main circuit class for superconducting quantum circuits
    CircuitFitter: Parameter fitting for circuit models
    BBQ: Black Box Quantization analysis
    TransitionFitter: General transition frequency fitting
    PointPicker: Interactive point selection tool for data analysis

Utilities:
    lanczos_krylov: Lanczos algorithm for Hermitian matrices
    IterativeHamiltonianDiagonalizer: Multi-mode Hamiltonian diagonalization
"""

__version__ = "0.1.0"
__author__ = "Joan Caceres"
__email__ = "contact@joancaceres.com"

# Core circuit analysis classes
from .circuit import Circuit
from .CircuitFitter import CircuitFitter
from .bbq import BBQ

# Fitting and analysis tools
from .transition_fitter import TransitionFitter
from .pointpicker import PointPicker

# Numerical utilities
from .iterative_diagonalizer import IterativeHamiltonianDiagonalizer
from .utilities import lanczos_krylov

# Public API - what gets imported with "from sccircuits import *"
__all__ = [
    # Core classes
    "Circuit",
    "CircuitFitter", 
    "BBQ",
    
    # Analysis tools
    "TransitionFitter",
    "PointPicker",
    
    # Utilities
    "IterativeHamiltonianDiagonalizer",
    "lanczos_krylov",
    
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
]

# Package information
def get_version():
    """Return the version of the sccircuits package."""
    return __version__

def get_info():
    """Return basic information about the sccircuits package."""
    return {
        "name": "sccircuits", 
        "version": __version__,
        "author": __author__,
        "description": "Superconducting Circuit Analysis Package",
        "main_classes": ["Circuit", "CircuitFitter", "BBQ", "TransitionFitter"],
    }

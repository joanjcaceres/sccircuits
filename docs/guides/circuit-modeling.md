# Circuit Modeling

`Circuit` represents a nonlinear collective mode coupled to zero or more linear
chain modes. The class supports two main entry points depending on what data you
already have.

## 1. Start from Chain Parameters

Use the direct constructor when you already know the collective nonlinear mode,
the chain frequencies, and the nearest-neighbor couplings:

```python
from sccircuits import Circuit

circuit = Circuit(
    non_linear_frequency=4.9,
    non_linear_phase_zpf=0.18,
    dimensions=[16, 8, 6],
    Ej=0.95,
    linear_frequencies=[6.4, 8.1],
    linear_couplings=[0.22, 0.08],
)
```

This is the internal representation used by `eigensystem()`.

## 2. Start from Harmonic Modes

If your inputs come from an EPR calculation, finite-element simulation, or a
linearized circuit analysis, `Circuit.from_harmonic_modes()` is usually the
better entry point:

```python
circuit = Circuit.from_harmonic_modes(
    frequencies=[5.0, 6.2, 7.8],
    phase_zpf=[0.12, 0.08, 0.03],
    dimensions=[18, 10, 6],
    Ej=0.95,
)
```

Internally, SCCircuits uses a Lanczos transform to turn those harmonic inputs
into a chain Hamiltonian. You can recover both descriptions:

```python
harmonic = circuit.harmonic_modes()
star = circuit.star_representation()
params = circuit.parameter_dict()
```

## Solving the Spectrum

Use `eigensystem()` to iteratively build the coupled Hamiltonian and keep the
lowest-energy states:

```python
evals, evecs = circuit.eigensystem(truncation=[20, 10, 6])
```

The truncation can be:

- a single integer, reused at every step
- a list with one truncation per diagonalization stage

When you need operator tracking for matrix elements in the truncated basis, use:

```python
evals, evecs = circuit.eigensystem(
    truncation=[20, 10, 6],
    track_operators=True,
    store_basis=True,
)
operators = circuit.get_tracked_operators()
```

## Star vs Chain View

The chain basis is efficient for iterative diagonalization. The star basis is
usually easier to interpret physically because each linear mode couples directly
to the nonlinear core.

```python
star = circuit.star_representation()
print(star["linear_frequencies"])
print(star["linear_couplings"])
```

The theory section explains how those two views are related.

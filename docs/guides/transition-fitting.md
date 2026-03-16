# Transition Fitting

`TransitionFitter` fits experimental transition data against any model function
that returns either:

- a vector of eigenvalues, or
- a Hamiltonian matrix that should be diagonalized internally

## Data Format

The fitter expects a dictionary keyed by transition indices:

```python
data = {
    (0, 1): [(0.0, 5.25, 0.02), (np.pi, 4.75, 0.02)],
    (1, 2): [(0.0, 9.70, 0.03)],
}
```

Each point is `(phi_ext, value)` or `(phi_ext, value, sigma)`.

## Model Function

For package-specific workflows, the model function often builds a `Circuit`
instance from the current parameter vector and returns its eigenvalues:

```python
def model(phi_ext, params):
    Ej, omega_nl, phi_zpf = params
    circuit = Circuit(
        non_linear_frequency=omega_nl,
        non_linear_phase_zpf=phi_zpf,
        dimensions=[18],
        Ej=Ej,
        phase_ext=phi_ext,
    )
    evals, _ = circuit.eigensystem(truncation=4)
    return evals
```

Then fit it:

```python
fitter = TransitionFitter(model_func=model, data=data, returns_eigenvalues=True)
result = fitter.fit(
    params_initial=[0.9, 4.8, 0.15],
    bounds=([0.0, 4.0, 0.01], [2.0, 6.0, 0.5]),
    verbose=0,
)
```

## Diagnostics with `FitAnalysis`

`TransitionFitter` focuses on optimization and transition bookkeeping. For
post-fit diagnostics, pass the SciPy result to `FitAnalysis`:

```python
from sccircuits import FitAnalysis

analysis = FitAnalysis(result, parameter_labels=["Ej", "omega_nl", "phi_zpf"])
print(analysis.chi2_reduced)
print(analysis.parameter_errors)
```

`FitAnalysis` also includes plotting helpers for correlation matrices, singular
values, and residual summaries.

## YAML Input from `PointPicker`

If you tagged spectroscopy points with `PointPicker`, you can load them
directly:

```python
data = TransitionFitter.load_from_yaml("spectroscopy_points.yaml", x_scale=2.0 * np.pi)
```

That gives you a `data` dictionary in the same shape used by the fitter.

## Current Scope

SCCircuits no longer ships a dedicated `CircuitFitter` wrapper. The supported
fitting surface is the generic pair:

- `TransitionFitter` for optimization
- `FitAnalysis` for diagnostics

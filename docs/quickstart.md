# Quickstart

This page gives one small example for the circuit model and one for the fitting
stack. For the full workflow, move on to the user guides.

## Build a Circuit from Harmonic-Mode Data

`Circuit.from_harmonic_modes()` is the most direct entry point when you already
know the harmonic frequencies and phase zero-point fluctuations of the linearized
system.

```{doctest}
>>> from sccircuits import Circuit
>>> circuit = Circuit.from_harmonic_modes(
...     frequencies=[5.0, 6.2, 7.8],
...     phase_zpf=[0.12, 0.08, 0.03],
...     dimensions=[10, 6, 4],
...     Ej=0.95,
... )
>>> evals, _ = circuit.eigensystem(truncation=[8, 6, 4])
>>> len(evals)
4
>>> star = circuit.star_representation()
>>> star["linear_frequencies"].shape
(2,)
```

What this gives you:

- a chain representation used internally by `Circuit.eigensystem()`
- access to the original harmonic data through `circuit.harmonic_modes()`
- a derived star representation through `circuit.star_representation()`

## Fit Synthetic Transition Data

`TransitionFitter` is model-agnostic. You provide a function that returns either
eigenvalues or a Hamiltonian, and the fitter handles the transition bookkeeping.

```{doctest}
>>> import numpy as np
>>> from sccircuits import FitAnalysis, TransitionFitter
>>> def model(phi_ext, params):
...     omega0, amplitude = params
...     return np.array([0.0, omega0 + amplitude * np.cos(phi_ext)])
>>> data = {
...     (0, 1): [
...         (0.0, 5.25, 0.02),
...         (np.pi / 2.0, 5.00, 0.02),
...         (np.pi, 4.75, 0.02),
...     ]
... }
>>> fitter = TransitionFitter(model_func=model, data=data, returns_eigenvalues=True)
>>> result = fitter.fit(
...     params_initial=[4.9, 0.1],
...     bounds=([4.0, 0.0], [6.0, 1.0]),
...     verbose=0,
... )
>>> result.x.shape
(2,)
>>> analysis = FitAnalysis(result, parameter_labels=["omega0", "amplitude"])
>>> analysis.n_params
2
```

## Next Steps

- [Circuit Modeling](guides/circuit-modeling.md)
- [BBQ Workflow](guides/bbq-workflow.md)
- [Transition Fitting](guides/transition-fitting.md)
- [Theory Background](theory/background.md)

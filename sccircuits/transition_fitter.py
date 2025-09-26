"""Tools for fitting transition frequencies versus an external control.

The :class:`TransitionFitter` orchestrates nonlinear regression of multiple
spectroscopic transitions simultaneously.  End users provide a *model function*
that predicts eigenvalues (or a Hamiltonian to be diagonalised internally), a
dictionary of experimental transition data, and an initial guess for the model
parameters.  The fitter takes care of normalising transitions, caching repeated
diagonalisations, running the chosen optimiser, and producing extensive
diagnostic statistics.

Typical workflow::

    >>> def model(phi_ext, params):
    ...     # return eigenvalues for the system at external flux phi_ext
    ...     return np.array([...])
    >>> data = {
    ...     (0, 1): [(0.0, 5.1), (0.5, 5.3)],
    ...     (1, 2): [(0.0, 4.9), (0.5, 4.7)],
    ... }
    >>> fitter = TransitionFitter(model_func=model, data=data,
    ...                           returns_eigenvalues=True)
    >>> result = fitter.fit(params_initial=[5.0, 0.1], verbose=1)
    >>> stats = fitter.get_fit_statistics()

This module depends on :mod:`numpy` and :mod:`scipy` (≥1.10).
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from IPython.display import clear_output
from scipy import stats
from scipy.linalg import eigh
from scipy.optimize import least_squares, differential_evolution

Transition = Tuple[int, int]  # (i, j) with j > i


@dataclass
class DataPoint:
    """Single experimental measurement used by :class:`TransitionFitter`.

    Parameters
    ----------
    phi_ext : float
        External control value (for example, reduced flux in units of ``Φ₀``).
    value : float
        Measured transition frequency/energy corresponding to ``phi_ext``.
    sigma : float, optional
        One-sigma uncertainty for ``value``.  When left ``None`` the fitter
        falls back to :attr:`TransitionFitter.DEFAULT_SIGMA`.
    """

    phi_ext: float
    value: float  # experimental frequency or energy
    sigma: Optional[float] = None  # measurement uncertainty (standard deviation)

    def __post_init__(self) -> None:
        self.phi_ext = float(self.phi_ext)
        self.value = float(self.value)
        if self.sigma is not None:
            self.sigma = float(self.sigma)
            if self.sigma <= 0:
                raise ValueError(f"sigma must be positive when provided, got {self.sigma}")


class TransitionFitter:
    """Nonlinear fitter for multi-transition spectroscopy datasets.

    Features
    --------
    * simultaneous fitting of several transitions with optional per-point
      uncertainties;
    * automatic normalisation of transition indices so ``(i, j)`` and ``(j, i)``
      refer to the same dataset;
    * caching of Hamiltonian diagonalisations per external parameter value;
    * streamlined integration with SciPy's :func:`least_squares` and :func:`differential_evolution` optimisers;
    * rich post-fit diagnostics (statistics, residual analysis, history).
    """

    DEFAULT_SIGMA = 1.0

    def __init__(
        self,
        model_func: Callable[[float, Sequence[float]], Union[np.ndarray, np.ndarray]],
        data: Dict[Transition, Sequence[Union[DataPoint, Tuple[float, float]]]],
        returns_eigenvalues: bool = False,
        jacobian_func: Optional[
            Callable[[float, Sequence[float]], np.ndarray]
        ] = None,
    ) -> None:
        """Create a fitter instance.

        Parameters
        ----------
        model_func : callable
            Callable receiving ``(phi_ext, params)`` and returning either a
            Hermitian matrix (when ``returns_eigenvalues=False``) or the array of
            eigenvalues at that operating point (when ``True``).
        data : dict[(int, int), sequence]
            Experimental observations.  Keys specify the transition ``(i, j)``;
            values contain either :class:`DataPoint` instances or
            ``(phi_ext, value[, sigma])`` tuples.
        returns_eigenvalues : bool, default ``False``
            Flag describing the output of ``model_func``.
        jacobian_func : callable, optional
            Callable returning the eigenvalue Jacobian ``∂E/∂params`` when
            invoked as ``jacobian_func(phi_ext, params)``. If provided, the
            fitter supplies an analytic Jacobian to :func:`least_squares`.
        """
        self.model_func = model_func
        self.returns_eigenvalues = returns_eigenvalues
        self.jacobian_func = jacobian_func

        # Normalize and sort transition keys so (i,j) and (j,i) are treated the same
        self.data: Dict[Transition, List[DataPoint]] = {}
        for transition, data_points in data.items():
            sorted_t = tuple(sorted(transition))
            points = [
                DataPoint(*dp) if not isinstance(dp, DataPoint) else dp
                for dp in data_points
            ]
            if sorted_t in self.data:
                self.data[sorted_t].extend(points)
            else:
                self.data[sorted_t] = points

        self.result = None  # Will be set after fitting

        # History of (params, residuals) recorded each iteration
        self.history: List[Tuple[np.ndarray, np.ndarray]] = []
        self._last_recorded_params: np.ndarray | None = None
        self._verbose: int = 0
        self._user_callback: Optional[Callable[[np.ndarray, np.ndarray], None]] = None
        self._last_evals_cache: Optional[Dict[float, np.ndarray]] = None
        self._last_jac_cache: Optional[Dict[float, np.ndarray]] = None
        self._method: Optional[str] = None

    @staticmethod
    def _transition_freq(evals: np.ndarray, i: int, j: int) -> float:
        """Return the absolute difference ``|E_j - E_i|`` for sorted eigenvalues."""
        return abs(evals[j] - evals[i])

    def _theory(
        self, phi_ext: float, params: np.ndarray, transition: Transition
    ) -> float:
        """Compute a single theoretical transition value for ``phi_ext``."""
        if self.returns_eigenvalues:
            evals = self.model_func(phi_ext, params)
        else:
            H = self.model_func(phi_ext, params)
            evals = eigh(H, eigvals_only=True)
        i, j = transition
        return self._transition_freq(evals, i, j)

    def residuals(self, params: np.ndarray) -> np.ndarray:
        """Return the weighted residual vector used by least-squares solvers.

        The routine caches eigenvalues for each unique ``phi_ext`` encountered
        during the residual evaluation so that costly diagonalisations are not
        repeated.
        """
        residual_list: List[float] = []
        evals_cache: Dict[float, np.ndarray] = {}  # phi_ext -> eigenvalues

        for transition, data_points in self.data.items():
            i, j = transition
            for data_point in data_points:
                # Cache eigenvalues for each unique phi_ext
                if data_point.phi_ext not in evals_cache:
                    if self.returns_eigenvalues:
                        evals_cache[data_point.phi_ext] = self.model_func(
                            data_point.phi_ext, params
                        )
                    else:
                        H = self.model_func(data_point.phi_ext, params)
                        evals_cache[data_point.phi_ext] = eigh(H, eigvals_only=True)
                evals = evals_cache[data_point.phi_ext]

                theoretical_value = self._transition_freq(evals, i, j)
                diff = theoretical_value - data_point.value
                if data_point.sigma is not None:
                    diff /= data_point.sigma
                residual_list.append(diff)

        res = np.asarray(residual_list)
        self._last_evals_cache = {k: v.copy() for k, v in evals_cache.items()}
        # Record unique parameter evaluations
        if self._last_recorded_params is None or not np.allclose(
            self._last_recorded_params, params
        ):
            self.history.append((params.copy(), res.copy()))
            self._last_recorded_params = params.copy()
            if self._verbose and self._method != 'differential_evolution':
                # clear the previous output to avoid flooding the notebook
                clear_output(wait=True)
                print(
                    f"Iter {len(self.history)}: params = {params}, residual_norm = {np.linalg.norm(res)}"
                )
            if self._user_callback:
                self._user_callback(params.copy(), res.copy())
        return res

    def fit(
        self,
        params_initial: Optional[Sequence[float]] = None,
        *,
        bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        verbose: int = 0,
        callback: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
        method: str = 'least_squares',
        **kwargs,
    ):
        """Perform a nonlinear least-squares fit for the supplied data.

        Parameters
        ----------
        params_initial : sequence of float, optional
            Starting point passed to :func:`scipy.optimize.least_squares`.  When
            omitted, the midpoint of ``bounds`` is used.
        bounds : tuple, optional
            Tuple ``(lower_bounds, upper_bounds)`` forwarded to
            :func:`least_squares`.
        verbose : int, default 0
            Verbosity level (0, 1, 2).
        callback : callable, optional
            Function invoked with ``(params, residuals)`` whenever a new
            parameter vector is recorded.
        method : str, default 'least_squares'
            Optimization method to use. Options: 'least_squares', 'differential_evolution'.
        **kwargs
            Additional keyword arguments forwarded directly to the chosen optimizer.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Optimisation result from the chosen optimizer.
        """
        # Reset history and callback state
        self.history = []
        self._last_recorded_params = None
        self._verbose = verbose
        self._user_callback = callback
        self._method = method

        if params_initial is not None:
            initial = np.asarray(params_initial, dtype=float)
        else:
            if bounds is None:
                raise ValueError(
                    "params_initial must be provided when bounds are omitted."
                )
            lower, upper = bounds
            lower = np.asarray(lower, dtype=float)
            upper = np.asarray(upper, dtype=float)
            initial = (lower + upper) / 2.0
            if verbose:
                print(
                    "Initial parameters not provided; using midpoint of bounds:"
                    f" {initial}"
                )

        if method == 'least_squares':
            jac_callable = self.jacobian if self.jacobian_func is not None else None
            result = self._fit_least_squares(initial, bounds, verbose, jac_callable, **kwargs)
        elif method == 'differential_evolution':
            if bounds is None:
                raise ValueError("bounds must be provided when using differential_evolution")
            result = self._fit_differential_evolution(initial, bounds, verbose, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        self.result = result
        return result

    def _fit_least_squares(
        self,
        initial: np.ndarray,
        bounds: Optional[Tuple[Sequence[float], Sequence[float]]],
        verbose: int,
        jac_callable: Optional[Callable[[np.ndarray], np.ndarray]],
        **ls_kwargs,
    ):
        """Internal helper wrapping :func:`scipy.optimize.least_squares`."""
        if bounds is not None:
            ls_kwargs["bounds"] = bounds
        if jac_callable is not None:
            ls_kwargs["jac"] = jac_callable
        return least_squares(self.residuals, initial, verbose=verbose, **ls_kwargs)

    def _fit_differential_evolution(
        self,
        initial: np.ndarray,
        bounds: Tuple[Sequence[float], Sequence[float]],
        verbose: int,
        **de_kwargs,
    ):
        def cost_func(params):
            res = self.residuals(params)
            return np.sum(res**2)

        # Convert bounds format for DE: from ((lower,), (upper,)) to [(l1,u1), (l2,u2), ...]
        lower, upper = bounds
        bounds_de = list(zip(lower, upper))

        # Callback for verbose output
        de_callback = None
        if verbose > 0:
            gen_count = 0
            def de_callback(x, convergence):
                nonlocal gen_count
                gen_count += 1
                cost = cost_func(x)
                residual_norm = np.sqrt(cost)
                print(f"DE Gen {gen_count}: params = {x}, residual_norm = {residual_norm}")

        # Set defaults for DE
        de_kwargs.setdefault('maxiter', 1000)
        de_kwargs.setdefault('popsize', 15)
        if verbose > 0:
            de_kwargs['disp'] = False  # Disable default disp, use our callback
            de_kwargs['callback'] = de_callback

        return differential_evolution(cost_func, bounds=bounds_de, **de_kwargs)

    def _evaluate_eigen_derivatives(
        self, phi_ext: float, params: np.ndarray
    ) -> np.ndarray:
        if self.jacobian_func is None:
            raise RuntimeError("No jacobian function was supplied.")
        derivatives = self.jacobian_func(phi_ext, params)
        derivatives = np.asarray(derivatives, dtype=float)
        return derivatives

    def jacobian(self, params: np.ndarray) -> np.ndarray:
        """Assemble the residual Jacobian for least-squares solvers."""
        if self.jacobian_func is None:
            raise RuntimeError("Jacobian requested but no jacobian_func provided.")

        jac_rows: List[np.ndarray] = []
        deriv_cache: Dict[float, np.ndarray] = {}

        for transition, data_points in self.data.items():
            i, j = transition
            for data_point in data_points:
                phi = data_point.phi_ext
                if phi not in deriv_cache:
                    deriv_cache[phi] = self._evaluate_eigen_derivatives(phi, params)
                derivs = deriv_cache[phi]
                row = derivs[j] - derivs[i]
                if data_point.sigma is not None:
                    row = row / data_point.sigma
                jac_rows.append(row)

        return np.vstack(jac_rows)


    def get_theoretical_curve(
        self,
        transition: Transition,
        phi_ext_values: Optional[Sequence[float]] = None,
        params: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """
        Generate theoretical curve for a transition.

        - If phi_ext_values is None, use experimental phi_ext points.
        - If params is None, use fitted parameters (or initial if fit() was not called).
        """
        # Ensure transition key is sorted
        transition = tuple(sorted(transition))

        # Select parameters
        if params is None:
            if self.result is None:
                raise RuntimeError(
                    "No fitted parameters available. Provide 'params' explicitly."
                )
            params = self.result.x
        params = np.asarray(params, dtype=float)

        # Select phi_ext values
        if phi_ext_values is None:
            phi_ext_values = [dp.phi_ext for dp in self.data[transition]]
        phi_ext_values = np.asarray(phi_ext_values, dtype=float)

        # Compute theoretical values
        return np.array(
            [self._theory(phi, params, transition) for phi in phi_ext_values]
        )

    def _get_base_data_arrays(self):
        """Helper method to get base data arrays to avoid recomputation.

        Returns:
            tuple: (y_exp, y_theo, sigma_array, residuals)
        """
        if self.result is None:
            raise RuntimeError("No fit has been run yet. Call fit() first.")

        # Use cached computation if available
        if hasattr(self, "_cached_base_data") and hasattr(self, "_cached_params"):
            if np.array_equal(self._cached_params, self.result.x):
                return self._cached_base_data

        y_exp: List[float] = []
        y_theo: List[float] = []
        sigma_list: List[float] = []

        for transition, data_points in self.data.items():
            phi_values = [dp.phi_ext for dp in data_points]
            theo_values = self.get_theoretical_curve(transition, phi_values)

            for dp, tv in zip(data_points, theo_values):
                y_exp.append(dp.value)
                y_theo.append(tv)
                sigma_list.append(dp.sigma if dp.sigma is not None else self.DEFAULT_SIGMA)

        y_exp_array = np.array(y_exp)
        y_theo_array = np.array(y_theo)
        sigma_array = np.array(sigma_list, dtype=float)
        residuals = self.residuals(self.result.x)

        # Cache the results
        self._cached_base_data = (y_exp_array, y_theo_array, sigma_array, residuals)
        self._cached_params = self.result.x.copy()

        return self._cached_base_data

    def save_results_csv(self, filepath: str):
        """
        Save basic fit results to a CSV file.

        NOTE: This method saves only basic information. For complete analysis
        capabilities, consider using pickle serialization to preserve the
        full result object with Jacobian, convergence history, etc.

        Columns: transition_i, transition_j, phi_ext, experimental, theoretical, residual, sigma
        """
        if self.result is None:
            raise RuntimeError("No fit has been run yet. Call fit() first.")
        import csv

        with open(filepath, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write fit summary (parameters and cost)
            writer.writerow(["# TransitionFitter Results (Basic)"])
            writer.writerow(["# For complete analysis, use pickle serialization"])
            writer.writerow(["params"] + [str(p) for p in self.result.x.tolist()])
            writer.writerow(["cost", str(self.result.cost)])
            # Blank line before data
            writer.writerow([])
            # Header row
            writer.writerow(
                [
                    "transition_i",
                    "transition_j",
                    "phi_ext",
                    "experimental",
                    "theoretical",
                    "residual",
                    "sigma",
                ]
            )
            # Data rows
            for transition, dps in self.data.items():
                i, j = transition
                phi_values = [dp.phi_ext for dp in dps]
                theo_values = self.get_theoretical_curve(transition, phi_values)
                for dp, tv in zip(dps, theo_values):
                    writer.writerow(
                        [
                            i,
                            j,
                            dp.phi_ext,
                            dp.value,
                            float(tv),
                            float(tv - dp.value),
                            dp.sigma if dp.sigma is not None else self.DEFAULT_SIGMA,
                        ]
                    )
        print(f"Basic results saved to CSV file: {filepath}")
        print(
            "Note: For complete analysis (uncertainties, convergence, etc.), use pickle serialization."
        )

    def save_complete_result(self, filepath: str):
        """
        Save complete TransitionFitter state including full result object.

        This preserves all information including Jacobian, convergence history,
        and enables full analysis capabilities when loaded.

        Args:
            filepath (str): Path where to save the pickle file
        """
        if self.result is None:
            raise RuntimeError("No fit has been run yet. Call fit() first.")

        import pickle
        from pathlib import Path

        pickle_path = Path(filepath).with_suffix(".pkl")

        # Save complete state
        complete_state = {
            "transition_fitter": self,
            "timestamp": str(np.datetime64("now")),
            "version": "1.0",
        }

        with open(pickle_path, "wb") as f:
            pickle.dump(complete_state, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Complete TransitionFitter state saved to: {pickle_path}")
        print("All analysis methods will be available when loaded.")

    @classmethod
    def load_complete_result(cls, filepath: str):
        """
        Load complete TransitionFitter state from pickle file.

        Args:
            filepath (str): Path to the pickle file

        Returns:
            TransitionFitter: Loaded instance with complete state
        """
        import pickle
        from pathlib import Path

        pickle_path = Path(filepath).with_suffix(".pkl")

        with open(pickle_path, "rb") as f:
            complete_state = pickle.load(f)

        fitter = complete_state["transition_fitter"]

        print(f"Complete TransitionFitter loaded from: {pickle_path}")
        print(f"Saved on: {complete_state.get('timestamp', 'Unknown')}")
        if fitter.result is not None:
            print("All analysis methods are available.")
        else:
            print("Warning: No fit result found in loaded data.")

        return fitter


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    import math

    def example_hamiltonian(phi: float, p):
        # Simple two-level system
        Delta, E_c = p
        return np.array(
            [[0.5 * E_c, Delta * math.cos(phi)], [Delta * math.cos(phi), -0.5 * E_c]]
        )

    # Example experimental data for transition (0,1)
    data_example = {
        (0, 1): [
            (0.0, 5.1),
            (0.5 * math.pi, 4.8),
            (math.pi, 5.2),
            (1.5 * math.pi, 4.9),
        ],
    }

    print("=== TRANSITIONFITTER ANALYSIS DEMO ===\n")

    # Example 1: Basic fit with comprehensive analysis
    print("1. Basic fit with analysis:")
    fitter = TransitionFitter(
        model_func=example_hamiltonian,
        data=data_example,
    )
    result = fitter.fit(
        params_initial=[1.0, 10.0], bounds=([0.0, 5.0], [5.0, 15.0]), verbose=1
    )

    # Print formatted summary
    fitter.print_fit_summary()

    print("\n" + "=" * 60)

    # Example 1b: Fit with differential_evolution
    print("1b. Fit with differential_evolution:")
    fitter_de = TransitionFitter(
        model_func=example_hamiltonian,
        data=data_example,
    )
    result_de = fitter_de.fit(
        bounds=([0.0, 5.0], [5.0, 15.0]), method='differential_evolution', verbose=1
    )

    # Print formatted summary
    fitter_de.print_fit_summary()

    print("\n" + "=" * 60)

    # Example 2: Individual analysis methods
    print("2. Individual analysis methods:")

    # Fit statistics
    stats = fitter.get_fit_statistics()
    print(f"R² = {stats['r_squared']:.4f}")
    print(f"RMSE = {stats['rmse']:.6g}")
    print(f"Reduced χ² = {stats['reduced_chi_squared']:.4f}")

    # Residual analysis
    residuals = fitter.get_residual_analysis()
    print(f"Residual std = {residuals['std']:.6g}")
    print(f"Outliers detected = {residuals['n_outliers']}")

    # Parameter uncertainty
    uncertainty = fitter.get_parameter_uncertainty()
    if uncertainty["standard_errors"] is not None:
        print("Parameter uncertainties:")
        for i, (val, err) in enumerate(zip(result.x, uncertainty["standard_errors"])):
            print(f"  p[{i}] = {val:.6g} ± {err:.3g}")

    print("\n" + "=" * 60)

    # Example 3: Comprehensive report
    print("3. Comprehensive report:")
    report = fitter.get_comprehensive_report()
    print(f"Optimizer: {report['fit_info']['optimizer']}")
    print(f"Success: {report['fit_info']['success']}")
    print(f"Function evaluations: {report['fit_info']['n_function_evaluations']}")

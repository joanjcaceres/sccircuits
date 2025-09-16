"""
transition_fitter.py

TransitionFitter class to fit experimental transition data (i -> j) dependent on an external parameter (phi_ext).

Basic usage:
    1. Define a function model_func(phi_ext, params) returning the system's Hermitian matrix (default) or directly eigenvalues if returns_eigenvalues=True.
    2. Prepare a `data` dictionary with keys (i, j) and lists of (phi_ext, experimental value) pairs.
    3. Instantiate TransitionFitter and call .fit().

Dependencies: numpy, scipy ≥1.10
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from IPython.display import clear_output
from scipy import stats
from scipy.linalg import eigh
from scipy.optimize import differential_evolution, least_squares

Transition = Tuple[int, int]  # (i, j) with j > i


@dataclass
class DataPoint:
    """Holds a single experimental data point (phi_ext, value, optional sigma)."""

    phi_ext: float
    value: float  # experimental frequency or energy
    sigma: Optional[float] = None  # measurement uncertainty (standard deviation)

    def __post_init__(self) -> None:
        self.phi_ext = float(self.phi_ext)
        self.value = float(self.value)
        if self.sigma is not None:
            self.sigma = float(self.sigma)
            if self.sigma <= 0:
                raise ValueError("sigma must be positive when provided")


class TransitionFitter:
    """Fits multiple transitions simultaneously with per-point uncertainties.
    Transitions (i, j) and (j, i) are treated equivalently by normalizing keys.
    Supports multiple optimization methods: least_squares, differential_evolution, or custom optimizers.
    """

    def __init__(
        self,
        model_func: Callable[[float, Sequence[float]], Union[np.ndarray, np.ndarray]],
        data: Dict[Transition, Sequence[Union[DataPoint, Tuple[float, float]]]],
        params_initial: Optional[Sequence[float]] = None,
        returns_eigenvalues: bool = False,
        optimizer: str = "least_squares",
    ) -> None:
        self.model_func = model_func
        self.optimizer = optimizer
        if params_initial is None:
            self.params_initial = None
        else:
            self.params_initial = np.asarray(params_initial, dtype=float)
        self.returns_eigenvalues = returns_eigenvalues

        # Validate optimizer
        valid_optimizers = ["least_squares", "differential_evolution", "custom"]
        if optimizer not in valid_optimizers:
            raise ValueError(
                f"optimizer must be one of {valid_optimizers}, got '{optimizer}'"
            )

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
        self.de_history = []  # For differential evolution specific history
        self._last_recorded_params: np.ndarray | None = None
        self._verbose: int = 0
        self._user_callback: Optional[Callable[[np.ndarray, np.ndarray], None]] = None

    @staticmethod
    def _transition_freq(evals: np.ndarray, i: int, j: int) -> float:
        """Compute |Ej - Ei| assuming eigenvalues are sorted."""
        return abs(evals[j] - evals[i])

    def _theory(
        self, phi_ext: float, params: np.ndarray, transition: Transition
    ) -> float:
        """Compute theoretical transition for given phi_ext and parameters."""
        if self.returns_eigenvalues:
            evals = self.model_func(phi_ext, params)
        else:
            H = self.model_func(phi_ext, params)
            evals = eigh(H, eigvals_only=True)
        i, j = transition
        return self._transition_freq(evals, i, j)

    def residuals(self, params: np.ndarray) -> np.ndarray:
        """Weighted residuals vector for least-squares minimization.

        Uses a cache so the Hamiltonian diagonalisation for a given phi_ext
        is executed only once per residual evaluation.
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
        # Record unique parameter evaluations
        if self._last_recorded_params is None or not np.allclose(
            self._last_recorded_params, params
        ):
            self.history.append((params.copy(), res.copy()))
            self._last_recorded_params = params.copy()
            if self._verbose:
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
        bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        verbose: int = 0,
        callback: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
        optimizer: Optional[str] = None,
        **kwargs,
    ):
        """Perform nonlinear fitting using the specified optimization method.

        Parameters
        ----------
        bounds : tuple, optional
            Tuple (lower_bounds, upper_bounds) for parameter bounds.
        verbose : int, default 0
            Verbosity level (0, 1, 2).
        callback : callable, optional
            Function called with (params, residuals) when `residuals`
            encounters a new parameter vector.
        optimizer : str, optional
            Override the default optimizer for this fit. If None, uses self.optimizer.
        **kwargs
            Additional keyword arguments passed to the specific optimizer.

        Returns
        -------
        scipy.optimize.OptimizeResult
            The optimization result object.
        """
        # Use provided optimizer or fall back to instance default
        opt_method = optimizer if optimizer is not None else self.optimizer

        # Reset history and callback state
        self.history = []
        self.de_history = []
        self._last_recorded_params = None
        self._verbose = verbose
        self._user_callback = callback

        # Infer initial parameters if not provided, using midpoint of bounds
        if self.params_initial is None:
            if bounds is None:
                raise ValueError(
                    "No initial parameters and no bounds provided to infer initial guess."
                )
            lower, upper = bounds
            lower = np.asarray(lower, dtype=float)
            upper = np.asarray(upper, dtype=float)
            self.params_initial = (lower + upper) / 2.0
            if verbose:
                print(
                    f"Initial parameters not provided; using midpoint of bounds: {self.params_initial}"
                )

        # Dispatch to appropriate fitting method
        if opt_method == "least_squares":
            self.result = self._fit_least_squares(bounds, verbose, **kwargs)
        elif opt_method == "differential_evolution":
            if bounds is None:
                raise ValueError("bounds must be provided for differential_evolution")
            self.result = self._fit_differential_evolution(bounds, verbose, **kwargs)
        elif opt_method == "custom":
            # For custom optimizers, user must provide 'optimizer_func' in kwargs
            if "optimizer_func" not in kwargs:
                raise ValueError(
                    "For 'custom' optimizer, must provide 'optimizer_func' in kwargs"
                )
            self.result = self._fit_custom(bounds, verbose, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {opt_method}")

        return self.result

    def _fit_least_squares(self, bounds, verbose, **ls_kwargs):
        """Internal method for least squares fitting."""
        if bounds is not None:
            ls_kwargs["bounds"] = bounds
        return least_squares(
            self.residuals, self.params_initial, verbose=verbose, **ls_kwargs
        )

    def _fit_differential_evolution(self, bounds, verbose, **de_kwargs):
        """Internal method for differential evolution fitting."""
        lower, upper = bounds
        if len(lower) != len(upper):
            raise ValueError("Lower and upper bounds must have the same length.")
        de_bounds = list(zip(lower, upper))

        # Extract specific DE parameters
        de_polish = de_kwargs.pop("de_polish", False)
        ls_kwargs = de_kwargs.pop("ls_kwargs", {})

        # Ensure 'polish' and 'disp' are not duplicated
        de_kwargs.pop("polish", None)
        de_kwargs.pop("disp", None)

        # Setup DE callback for verbose output
        if verbose and "callback" not in de_kwargs:
            _de_gen = {"count": 0}

            def _de_callback(xk, convergence):
                _de_gen["count"] += 1
                cost = float(np.sum(self.residuals(xk) ** 2))
                self.de_history.append((xk.copy(), cost))
                print(
                    f"[DE] generation {_de_gen['count']}: cost={cost:.6g}, params={xk}"
                )

            de_kwargs["callback"] = _de_callback

        # Objective for DE: sum of squared residuals (scalar)
        def _objective(p):
            p = np.asarray(p, dtype=float)
            return float(np.sum(self.residuals(p) ** 2))

        # Stage 1: global search (DE)
        de_result = differential_evolution(
            _objective,
            bounds=de_bounds,
            polish=de_polish,
            disp=bool(verbose),
            **de_kwargs,
        )

        # Stage 2: local refinement with least squares
        original_init = self.params_initial
        self.params_initial = de_result.x
        try:
            ls_result = self._fit_least_squares(bounds, verbose, **ls_kwargs)
        finally:
            self.params_initial = original_init

        # Attach DE output for transparency
        ls_result.de_result = de_result
        return ls_result

    def _fit_custom(self, bounds, verbose, **kwargs):
        """Internal method for custom optimizer fitting."""
        optimizer_func = kwargs.pop("optimizer_func")

        # The custom optimizer should accept:
        # - objective function (returns scalar cost)
        # - initial parameters
        # - bounds (optional)
        # - any additional kwargs
        def _objective(p):
            p = np.asarray(p, dtype=float)
            return float(np.sum(self.residuals(p) ** 2))

        return optimizer_func(
            _objective, self.params_initial, bounds=bounds, verbose=verbose, **kwargs
        )

    def fit_with_de(
        self,
        bounds: Tuple[Sequence[float], Sequence[float]],
        de_kwargs: Optional[dict] = None,
        ls_kwargs: Optional[dict] = None,
        de_polish: bool = False,
        verbose: int = 0,
    ):
        """
        Convenience method for two-stage fit using differential evolution
        followed by least-squares refinement. This is equivalent to calling
        fit(optimizer="differential_evolution", ...).

        Parameters
        ----------
        bounds : tuple
            Tuple ``(lower_bounds, upper_bounds)`` for parameter bounds.
        de_kwargs : dict, optional
            Extra keyword arguments for differential_evolution.
        ls_kwargs : dict, optional
            Extra keyword arguments for the subsequent least_squares stage.
        de_polish : bool, default ``False``
            Whether to let DE perform its own internal L‑BFGS‑B "polish" step.
        verbose : int, default ``0``
            Verbosity level.

        Returns
        -------
        scipy.optimize.OptimizeResult
            The result object with an extra attribute ``de_result``.
        """
        # Combine parameters into unified kwargs
        kwargs = de_kwargs.copy() if de_kwargs else {}
        kwargs["de_polish"] = de_polish
        kwargs["ls_kwargs"] = ls_kwargs or {}

        return self.fit(
            bounds=bounds, verbose=verbose, optimizer="differential_evolution", **kwargs
        )

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
            params = self.result.x if self.result is not None else self.params_initial
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
                sigma_list.append(dp.sigma if dp.sigma is not None else 1.0)

        y_exp_array = np.array(y_exp)
        y_theo_array = np.array(y_theo)
        sigma_array = np.array(sigma_list, dtype=float)
        residuals = self.residuals(self.result.x)

        # Cache the results
        self._cached_base_data = (y_exp_array, y_theo_array, sigma_array, residuals)
        self._cached_params = self.result.x.copy()

        return self._cached_base_data

    def get_fit_statistics(self) -> dict:
        """
        Calculate comprehensive fit quality statistics.

        Returns
        -------
        dict
            Dictionary containing R², adjusted R², chi-squared, reduced chi-squared,
            AIC, BIC, RMSE, and degrees of freedom.
        """
        if self.result is None:
            raise RuntimeError("No fit has been run yet. Call fit() first.")

        # Get base data (with caching)
        y_exp, y_theo, sigma_array, residuals = self._get_base_data_arrays()

        n_data_points = len(residuals)
        n_params = len(self.result.x)
        dof = n_data_points - n_params  # degrees of freedom

        # Unweighted residuals for some statistics
        unweighted_residuals = y_theo - y_exp

        # Weighted residuals using sigma (chi-squared style)
        weighted_residuals = unweighted_residuals / sigma_array

        # Calculate statistics (using weighted residuals for proper fit statistics)
        ss_res_weighted = np.sum(
            weighted_residuals**2
        )  # Weighted sum of squared residuals
        ss_res = np.sum(unweighted_residuals**2)  # Unweighted for R²

        # For R², we use weighted mean if weights are not uniform
        weights = 1.0 / (sigma_array**2)
        if np.allclose(weights, weights[0]):
            # Uniform weights - use simple mean
            y_mean = np.mean(y_exp)
        else:
            # Non-uniform weights - use weighted mean
            y_mean = np.average(y_exp, weights=weights)

        ss_tot = np.sum(
            weights * (y_exp - y_mean) ** 2
        )  # Weighted total sum of squares

        # R-squared and adjusted R-squared
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        adj_r_squared = (
            1 - (ss_res / dof) / (ss_tot / (n_data_points - 1))
            if dof > 0 and n_data_points > 1
            else 0
        )

        # Chi-squared (using weighted residuals - this is the proper fit cost)
        chi_squared = ss_res_weighted  # This matches self.result.cost
        reduced_chi_squared = chi_squared / dof if dof > 0 else np.inf

        # RMSE (unweighted for interpretability)
        rmse = np.sqrt(ss_res / n_data_points)

        # Information criteria
        log_likelihood = -0.5 * chi_squared
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_data_points) - 2 * log_likelihood

        return {
            "r_squared": r_squared,
            "adjusted_r_squared": adj_r_squared,
            "chi_squared": chi_squared,
            "reduced_chi_squared": reduced_chi_squared,
            "rmse": rmse,
            "aic": aic,
            "bic": bic,
            "degrees_of_freedom": dof,
            "n_data_points": n_data_points,
            "n_parameters": n_params,
            "cost": float(self.result.cost),
        }

    def get_residual_analysis(self) -> dict:
        """
        Perform comprehensive residual analysis.

        Returns
        -------
        dict
            Dictionary containing residual statistics, normality tests,
            outlier detection, and autocorrelation analysis.
        """
        if self.result is None:
            raise RuntimeError("No fit has been run yet. Call fit() first.")

        # Get base data arrays (with caching)
        y_exp, y_theo, sigma_array, residuals = self._get_base_data_arrays()

        # Create detailed residuals data for outlier analysis
        residuals_data = []
        idx = 0
        for transition, data_points in self.data.items():
            for dp in data_points:
                unweighted_residual = y_theo[idx] - y_exp[idx]
                sigma_val = sigma_array[idx]
                weighted_residual = unweighted_residual / sigma_val
                residuals_data.append(
                    {
                        "transition": transition,
                        "phi_ext": dp.phi_ext,
                        "unweighted_residual": unweighted_residual,
                        "weighted_residual": weighted_residual,
                        "experimental": dp.value,
                        "theoretical": y_theo[idx],
                        "sigma": dp.sigma,
                    }
                )
                idx += 1

        unweighted_res = np.array([r["unweighted_residual"] for r in residuals_data])
        weighted_res = np.array([r["weighted_residual"] for r in residuals_data])

        # Basic statistics (both weighted and unweighted)
        mean_res = np.mean(unweighted_res)
        std_res = np.std(unweighted_res, ddof=1)
        mean_weighted_res = np.mean(weighted_res)
        std_weighted_res = np.std(weighted_res, ddof=1)

        # Normality tests
        try:
            shapiro_stat, shapiro_p = stats.shapiro(unweighted_res)
        except Exception:
            shapiro_stat, shapiro_p = np.nan, np.nan

        try:
            ks_stat, ks_p = stats.kstest(
                unweighted_res, "norm", args=(mean_res, std_res)
            )
        except Exception:
            ks_stat, ks_p = np.nan, np.nan

        # Outlier detection (using IQR method)
        q1, q3 = np.percentile(unweighted_res, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = []
        for i, res_data in enumerate(residuals_data):
            if (
                res_data["unweighted_residual"] < lower_bound
                or res_data["unweighted_residual"] > upper_bound
            ):
                outliers.append(
                    {
                        "index": i,
                        "transition": res_data["transition"],
                        "phi_ext": res_data["phi_ext"],
                        "residual": res_data["unweighted_residual"],
                        "experimental": res_data["experimental"],
                        "theoretical": res_data["theoretical"],
                    }
                )

        # Autocorrelation (if residuals are ordered by phi_ext)
        try:
            # Sort by phi_ext for autocorrelation analysis
            sorted_indices = np.argsort([r["phi_ext"] for r in residuals_data])
            sorted_residuals = unweighted_res[sorted_indices]

            if len(sorted_residuals) > 1:
                autocorr_1 = np.corrcoef(sorted_residuals[:-1], sorted_residuals[1:])[
                    0, 1
                ]
            else:
                autocorr_1 = np.nan
        except Exception:
            autocorr_1 = np.nan

        return {
            "mean": mean_res,
            "std": std_res,
            "weighted_mean": mean_weighted_res,
            "weighted_std": std_weighted_res,
            "min": np.min(unweighted_res),
            "max": np.max(unweighted_res),
            "median": np.median(unweighted_res),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "normality_tests": {
                "shapiro_wilk": {"statistic": shapiro_stat, "p_value": shapiro_p},
                "kolmogorov_smirnov": {"statistic": ks_stat, "p_value": ks_p},
            },
            "outliers": outliers,
            "n_outliers": len(outliers),
            "autocorrelation_lag1": autocorr_1,
            "residuals_data": residuals_data,
        }

    def get_parameter_uncertainty(self, confidence_level: float = 0.95) -> dict:
        """
        Estimate parameter uncertainties using the Jacobian at the optimum.

        Parameters
        ----------
        confidence_level : float, default 0.95
            Confidence level for parameter confidence intervals.

        Returns
        -------
        dict
            Dictionary containing covariance matrix, standard errors,
            confidence intervals, and correlation matrix.
        """
        if self.result is None:
            raise RuntimeError("No fit has been run yet. Call fit() first.")

        if not hasattr(self.result, "jac") or self.result.jac is None:
            return {
                "covariance_matrix": None,
                "standard_errors": None,
                "confidence_intervals": None,
                "correlation_matrix": None,
                "message": "Jacobian not available. Try fitting with method that provides Jacobian.",
            }

        try:
            # Calculate covariance matrix from Jacobian
            J = self.result.jac
            # For weighted least squares: C = (J^T J)^(-1) * sigma^2
            JTJ = J.T @ J

            # Estimate variance from residuals
            residuals = self.residuals(self.result.x)
            dof = len(residuals) - len(self.result.x)
            if dof > 0:
                sigma_squared = np.sum(residuals**2) / dof
            else:
                sigma_squared = 1.0

            # Covariance matrix
            try:
                cov_matrix = np.linalg.inv(JTJ) * sigma_squared
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudo-inverse
                cov_matrix = np.linalg.pinv(JTJ) * sigma_squared

            # Standard errors (diagonal elements)
            std_errors = np.sqrt(np.diag(cov_matrix))

            # Confidence intervals
            from scipy.stats import t

            t_value = t.ppf((1 + confidence_level) / 2, dof)
            confidence_intervals = []

            for i, (param, std_err) in enumerate(zip(self.result.x, std_errors)):
                margin = t_value * std_err
                ci_lower = param - margin
                ci_upper = param + margin
                confidence_intervals.append(
                    {
                        "parameter_index": i,
                        "parameter_value": param,
                        "std_error": std_err,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "margin": margin,
                    }
                )

            # Correlation matrix
            correlation_matrix = np.zeros_like(cov_matrix)
            for i in range(len(std_errors)):
                for j in range(len(std_errors)):
                    if std_errors[i] > 0 and std_errors[j] > 0:
                        correlation_matrix[i, j] = cov_matrix[i, j] / (
                            std_errors[i] * std_errors[j]
                        )

            return {
                "covariance_matrix": cov_matrix,
                "standard_errors": std_errors,
                "confidence_intervals": confidence_intervals,
                "correlation_matrix": correlation_matrix,
                "confidence_level": confidence_level,
                "degrees_of_freedom": dof,
            }

        except Exception as e:
            return {
                "covariance_matrix": None,
                "standard_errors": None,
                "confidence_intervals": None,
                "correlation_matrix": None,
                "error": f"Error calculating uncertainties: {str(e)}",
            }

    def get_convergence_analysis(self) -> dict:
        """
        Analyze the convergence history of the fit.

        Returns
        -------
        dict
            Dictionary containing convergence metrics and iteration history.
        """
        if self.result is None:
            raise RuntimeError("No fit has been run yet. Call fit() first.")

        if not self.history:
            return {
                "message": "No convergence history available. Set verbose > 0 during fitting.",
                "n_iterations": 0,
            }

        # Extract cost evolution
        costs = [np.sum(residuals**2) for params, residuals in self.history]
        n_iterations = len(costs)

        # Calculate convergence metrics
        if n_iterations > 1:
            initial_cost = costs[0]
            final_cost = costs[-1]
            cost_reduction = initial_cost - final_cost
            cost_reduction_percent = (
                (cost_reduction / initial_cost) * 100 if initial_cost > 0 else 0
            )

            # Calculate convergence rate (last 10% of iterations)
            n_tail = max(1, n_iterations // 10)
            if n_tail > 1:
                tail_costs = costs[-n_tail:]
                cost_changes = np.diff(tail_costs)
                avg_change_rate = np.mean(cost_changes) if len(cost_changes) > 0 else 0
            else:
                avg_change_rate = 0

            # Parameter evolution (standard deviation across iterations)
            param_evolution = [params for params, residuals in self.history]
            param_std = (
                np.std(param_evolution, axis=0)
                if len(param_evolution) > 1
                else np.zeros(len(self.result.x))
            )
        else:
            initial_cost = final_cost = costs[0] if costs else 0
            cost_reduction = cost_reduction_percent = avg_change_rate = 0
            param_std = np.zeros(len(self.result.x))

        # Differential Evolution history if available
        de_analysis = {}
        if self.de_history:
            de_costs = [cost for params, cost in self.de_history]
            de_analysis = {
                "n_generations": len(de_costs),
                "initial_cost": de_costs[0] if de_costs else None,
                "final_cost": de_costs[-1] if de_costs else None,
                "cost_evolution": de_costs,
            }

        return {
            "n_iterations": n_iterations,
            "initial_cost": initial_cost,
            "final_cost": final_cost,
            "cost_reduction": cost_reduction,
            "cost_reduction_percent": cost_reduction_percent,
            "avg_change_rate_final": avg_change_rate,
            "parameter_stability": param_std.tolist(),
            "cost_evolution": costs,
            "differential_evolution": de_analysis if de_analysis else None,
        }

    def get_transition_analysis(self) -> dict:
        """
        Analyze fit quality for each transition individually.

        Returns
        -------
        dict
            Dictionary containing per-transition statistics.
        """
        if self.result is None:
            raise RuntimeError("No fit has been run yet. Call fit() first.")

        transition_stats = {}

        for transition, data_points in self.data.items():
            phi_values = [dp.phi_ext for dp in data_points]
            experimental = [dp.value for dp in data_points]
            theoretical = self.get_theoretical_curve(transition, phi_values)

            exp_array = np.array(experimental)
            theo_array = np.array(theoretical)
            residuals = theo_array - exp_array
            sigma_array = np.array(
                [dp.sigma if dp.sigma is not None else 1.0 for dp in data_points],
                dtype=float,
            )

            # Basic statistics
            n_points = len(data_points)
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            max_error = np.max(np.abs(residuals))

            # R-squared for this transition
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((exp_array - np.mean(exp_array)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Weighted cost contribution using sigma (chi-squared component)
            weighted_cost = float(np.sum((residuals / sigma_array) ** 2))

            transition_stats[transition] = {
                "n_points": n_points,
                "rmse": rmse,
                "mae": mae,
                "max_absolute_error": max_error,
                "r_squared": r_squared,
                "weighted_cost": weighted_cost,
                "residuals": residuals.tolist(),
                "experimental_values": experimental,
                "theoretical_values": theoretical.tolist(),
                "phi_ext_values": phi_values,
                "sigma_values": [dp.sigma for dp in data_points],
            }

        return transition_stats

    def get_comprehensive_report(self) -> dict:
        """
        Generate a comprehensive analysis report combining all analysis methods.

        Returns
        -------
        dict
            Complete fit analysis report.
        """
        if self.result is None:
            raise RuntimeError("No fit has been run yet. Call fit() first.")

        return {
            "fit_info": {
                "optimizer": self.optimizer,
                "success": self.result.success
                if hasattr(self.result, "success")
                else True,
                "message": getattr(self.result, "message", "No message available"),
                "n_function_evaluations": getattr(self.result, "nfev", "Not available"),
                "n_jacobian_evaluations": getattr(self.result, "njev", "Not available"),
            },
            "parameters": {
                "values": self.result.x.tolist(),
                "initial_values": self.params_initial.tolist()
                if self.params_initial is not None
                else None,
            },
            "fit_statistics": self.get_fit_statistics(),
            "residual_analysis": self.get_residual_analysis(),
            "parameter_uncertainty": self.get_parameter_uncertainty(),
            "convergence_analysis": self.get_convergence_analysis(),
            "transition_analysis": self.get_transition_analysis(),
        }

    def print_fit_summary(
        self, show_transitions: bool = True, show_outliers: bool = True
    ):
        """
        Print a formatted summary of the fit results.

        Parameters
        ----------
        show_transitions : bool, default True
            Whether to show individual transition analysis.
        show_outliers : bool, default True
            Whether to show outlier information.
        """
        if self.result is None:
            print("No fit has been run yet. Call fit() first.")
            return

        stats = self.get_fit_statistics()
        residual_stats = self.get_residual_analysis()
        convergence = self.get_convergence_analysis()
        uncertainty = self.get_parameter_uncertainty()

        print("=" * 60)
        print("FIT ANALYSIS SUMMARY")
        print("=" * 60)

        # Basic fit info
        print(f"Optimizer: {self.optimizer}")
        print(f"Success: {getattr(self.result, 'success', 'Unknown')}")
        print(f"Function evaluations: {getattr(self.result, 'nfev', 'N/A')}")

        # Parameters
        print(f"\nPARAMETERS ({len(self.result.x)} total):")
        for i, param in enumerate(self.result.x):
            std_err = uncertainty.get("standard_errors", [None] * len(self.result.x))[i]
            if std_err is not None:
                print(f"  p[{i}] = {param:.6g} ± {std_err:.3g}")
            else:
                print(f"  p[{i}] = {param:.6g}")

        # Fit quality
        print("\nFIT QUALITY:")
        print(f"  R² = {stats['r_squared']:.4f}")
        print(f"  Adjusted R² = {stats['adjusted_r_squared']:.4f}")
        print(f"  RMSE = {stats['rmse']:.6g}")
        print(f"  χ² = {stats['chi_squared']:.4f}")
        print(f"  Reduced χ² = {stats['reduced_chi_squared']:.4f}")
        print(f"  AIC = {stats['aic']:.2f}")
        print(f"  BIC = {stats['bic']:.2f}")

        # Residual analysis
        print("\nRESIDUAL ANALYSIS:")
        print(f"  Mean residual = {residual_stats['mean']:.6g}")
        print(f"  Std residual = {residual_stats['std']:.6g}")
        print(f"  Outliers detected = {residual_stats['n_outliers']}")

        # Normality tests
        shapiro = residual_stats["normality_tests"]["shapiro_wilk"]
        print(f"  Normality (Shapiro-Wilk): p = {shapiro['p_value']:.4f}")

        # Convergence
        if convergence["n_iterations"] > 0:
            print("\nCONVERGENCE:")
            print(f"  Iterations = {convergence['n_iterations']}")
            print(f"  Cost reduction = {convergence['cost_reduction_percent']:.2f}%")

        # Individual transitions
        if show_transitions:
            transitions = self.get_transition_analysis()
            print("\nTRANSITION ANALYSIS:")
            for transition, stats_t in transitions.items():
                print(
                    f"  {transition}: R² = {stats_t['r_squared']:.4f}, "
                    f"RMSE = {stats_t['rmse']:.6g}, "
                    f"Points = {stats_t['n_points']}"
                )

        # Outliers
        if show_outliers and residual_stats["n_outliers"] > 0:
            print(f"\nOUTLIERS ({residual_stats['n_outliers']} detected):")
            for outlier in residual_stats["outliers"][:5]:  # Show first 5
                print(
                    f"  Transition {outlier['transition']}, "
                    f"φ = {outlier['phi_ext']:.4f}, "
                    f"residual = {outlier['residual']:.6g}"
                )
            if residual_stats["n_outliers"] > 5:
                print(f"  ... and {residual_stats['n_outliers'] - 5} more")

        print("=" * 60)

    def fit_multistart(
        self,
        n_starts: int,
        bounds: Tuple[Sequence[float], Sequence[float]],
        perturb_scale: float = 0.1,
        optimizer: Optional[str] = None,
        verbose: int = 0,
        **kwargs,
    ):
        """
        Perform multiple fits with different starting points and return the best result.

        Parameters
        ----------
        n_starts : int
            Number of different starting points to try.
        bounds : tuple
            Tuple (lower_bounds, upper_bounds) for parameter bounds.
        perturb_scale : float, default 0.1
            Scale for random perturbations of initial parameters.
        optimizer : str, optional
            Override the default optimizer. If None, uses self.optimizer.
        verbose : int, default 0
            Verbosity level.
        **kwargs
            Additional arguments passed to the fit method.

        Returns
        -------
        scipy.optimize.OptimizeResult
            The best result from all starts.
        """
        if self.params_initial is None:
            if bounds is None:
                raise ValueError("No initial parameters and no bounds provided.")
            lower, upper = bounds
            lower = np.asarray(lower, dtype=float)
            upper = np.asarray(upper, dtype=float)
            self.params_initial = (lower + upper) / 2.0

        original_params = self.params_initial.copy()
        best_result = None
        best_cost = np.inf
        all_results = []

        lower, upper = bounds
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        for start in range(n_starts):
            if verbose:
                print(f"Multistart {start + 1}/{n_starts}")

            if start == 0:
                # First start uses original initial parameters
                self.params_initial = original_params.copy()
            else:
                # Generate random perturbation within bounds
                perturbation = np.random.normal(0, perturb_scale, len(original_params))
                perturbed_params = original_params + perturbation
                # Ensure within bounds
                perturbed_params = np.clip(perturbed_params, lower, upper)
                self.params_initial = perturbed_params

            try:
                result = self.fit(
                    bounds=bounds,
                    verbose=max(0, verbose - 1),
                    optimizer=optimizer,
                    **kwargs,
                )

                cost = (
                    result.cost
                    if hasattr(result, "cost")
                    else np.sum(self.residuals(result.x) ** 2)
                )
                all_results.append((result, cost))

                if cost < best_cost:
                    best_cost = cost
                    best_result = result

                if verbose:
                    print(f"  Cost: {cost:.6g}")

            except Exception as e:
                if verbose:
                    print(f"  Failed: {str(e)}")
                continue

        # Restore original parameters
        self.params_initial = original_params

        if best_result is not None:
            self.result = best_result
            if verbose:
                print(f"Best result: cost = {best_cost:.6g}")
        else:
            raise RuntimeError("All multistart attempts failed.")

        # Attach multistart information
        best_result.multistart_results = all_results
        best_result.n_starts = n_starts

        return best_result

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
            writer.writerow(["optimizer", self.optimizer])
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
                            dp.sigma if dp.sigma is not None else "",
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

    print("=== TRANSITIONFITTER ENHANCED ANALYSIS DEMO ===\n")

    # Example 1: Basic fit with comprehensive analysis
    print("1. Basic fit with analysis:")
    fitter = TransitionFitter(
        model_func=example_hamiltonian,
        params_initial=[1.0, 10.0],
        data=data_example,
        optimizer="least_squares",
    )
    result = fitter.fit(bounds=([0.0, 5.0], [5.0, 15.0]), verbose=1)

    # Print formatted summary
    fitter.print_fit_summary()

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

    # Example 3: Multistart optimization
    print("3. Multistart optimization:")
    best_result = fitter.fit_multistart(
        n_starts=3, bounds=([0.0, 5.0], [5.0, 15.0]), perturb_scale=0.2, verbose=1
    )
    print(f"Best cost from {best_result.n_starts} starts: {best_result.cost:.6g}")

    print("\n" + "=" * 60)

    # Example 4: Comprehensive report
    print("4. Comprehensive report:")
    report = fitter.get_comprehensive_report()
    print(f"Optimizer: {report['fit_info']['optimizer']}")
    print(f"Success: {report['fit_info']['success']}")
    print(f"Function evaluations: {report['fit_info']['n_function_evaluations']}")

    # Example 5: Using differential evolution with analysis
    print("\n5. Differential Evolution with analysis:")
    fitter_de = TransitionFitter(
        model_func=example_hamiltonian,
        params_initial=[1.0, 10.0],
        data=data_example,
        optimizer="differential_evolution",
    )
    result_de = fitter_de.fit_with_de(bounds=([0.0, 5.0], [5.0, 15.0]), verbose=1)

    # Convergence analysis
    convergence = fitter_de.get_convergence_analysis()
    if convergence["n_iterations"] > 0:
        print(f"Convergence: {convergence['n_iterations']} iterations")
        print(f"Cost reduction: {convergence['cost_reduction_percent']:.2f}%")

    # Transition-specific analysis
    transition_analysis = fitter_de.get_transition_analysis()
    for transition, stats_t in transition_analysis.items():
        print(
            f"Transition {transition}: R² = {stats_t['r_squared']:.4f}, "
            f"RMSE = {stats_t['rmse']:.6g}"
        )

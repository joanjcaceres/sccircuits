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
from scipy.optimize import least_squares

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
    * streamlined integration with SciPy's :func:`least_squares` optimiser;
    * rich post-fit diagnostics (statistics, residual analysis, history).
    """

    DEFAULT_SIGMA = 1.0

    def __init__(
        self,
        model_func: Callable[[float, Sequence[float]], Union[np.ndarray, np.ndarray]],
        data: Dict[Transition, Sequence[Union[DataPoint, Tuple[float, float]]]],
        returns_eigenvalues: bool = False,
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
        """
        self.model_func = model_func
        self.returns_eigenvalues = returns_eigenvalues

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
        params_initial: Optional[Sequence[float]] = None,
        *,
        bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        verbose: int = 0,
        callback: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
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
        **kwargs
            Additional keyword arguments forwarded directly to
            :func:`least_squares`.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Optimisation result from :func:`least_squares`.
        """
        # Reset history and callback state
        self.history = []
        self._last_recorded_params = None
        self._verbose = verbose
        self._user_callback = callback

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

        result = self._fit_least_squares(initial, bounds, verbose, **kwargs)
        self.result = result
        return result

    def _fit_least_squares(
        self,
        initial: np.ndarray,
        bounds: Optional[Tuple[Sequence[float], Sequence[float]]],
        verbose: int,
        **ls_kwargs,
    ):
        """Internal helper wrapping :func:`scipy.optimize.least_squares`."""
        if bounds is not None:
            ls_kwargs["bounds"] = bounds
        return least_squares(self.residuals, initial, verbose=verbose, **ls_kwargs)

    # differential-evolution-based helpers have been removed for simplicity.

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

        return {
            "n_iterations": n_iterations,
            "initial_cost": initial_cost,
            "final_cost": final_cost,
            "cost_reduction": cost_reduction,
            "cost_reduction_percent": cost_reduction_percent,
            "avg_change_rate_final": avg_change_rate,
            "parameter_stability": param_std.tolist(),
            "cost_evolution": costs,
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
                [dp.sigma if dp.sigma is not None else self.DEFAULT_SIGMA for dp in data_points],
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
                "optimizer": "least_squares",
                "success": self.result.success
                if hasattr(self.result, "success")
                else True,
                "message": getattr(self.result, "message", "No message available"),
                "n_function_evaluations": getattr(self.result, "nfev", "Not available"),
                "n_jacobian_evaluations": getattr(self.result, "njev", "Not available"),
            },
            "parameters": {
                "values": self.result.x.tolist(),
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
        print("Optimizer: least_squares")
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

    # Multi-start convenience wrappers were removed to keep the interface lean.

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

import matplotlib.pyplot as plt
import numpy as np


class FitAnalysis:
    """
    A class for comprehensive analysis of least squares fitting results from scipy.optimize.least_squares.

    This class computes chi-squared statistics, parameter uncertainties, correlations, and performs
    singular value decomposition (SVD) on the Jacobian matrix. It provides methods for visualizing
    the correlation matrix, singular values, and singular vectors.

    Parameters
    ----------
    res : scipy.optimize.OptimizeResult
        The result object returned by scipy.optimize.least_squares.
    parameter_labels : list of str, optional
        Labels for the parameters. If None, defaults to ['param0', 'param1', ...].

    Attributes
    ----------
    theta_hat : ndarray
        The fitted parameter values.
    residuals : ndarray
        The residuals (fun attribute from least_squares).
    jacobian : ndarray
        The Jacobian matrix (jac attribute from least_squares).
    n_data : int
        Number of data points.
    n_params : int
        Number of parameters.
    chi2 : float
        Chi-squared value (sum of squared residuals).
    chi2_reduced : float
        Reduced chi-squared value.
    covariance : ndarray
        Covariance matrix of the parameters.
    correlation : ndarray
        Correlation matrix of the parameters.
    U : ndarray
        Left singular vectors from SVD of Jacobian.
    S : ndarray
        Singular values from SVD of Jacobian.
    Vt : ndarray
        Transpose of right singular vectors from SVD.
    V : ndarray
        Right singular vectors from SVD (Vt.T).
    parameter_errors : ndarray
        Standard errors of the parameters (sqrt of diagonal covariance).
    parameter_labels : list of str
        Labels for the parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import least_squares
    >>> # Example fitting function and data
    >>> def model(params, x):
    ...     return params[0] * np.exp(-params[1] * x)
    >>> def residuals(params, x, y):
    ...     return model(params, x) - y
    >>> x = np.linspace(0, 1, 10)
    >>> y = 2 * np.exp(-3 * x) + 0.1 * np.random.randn(10)
    >>> res = least_squares(residuals, [1, 1], args=(x, y))
    >>> analysis = FitAnalysis(res)
    >>> print(f"Chi-squared reduced: {analysis.chi2_reduced:.3f}")
    >>> analysis.plot_correlation()  # heatmap with annotations
    >>> analysis.plot_correlation(annotate_values=False)  # heatmap without annotations
    """

    def __init__(self, res, parameter_labels=None):
        """
        Initialize the FitAnalysis object.

        Parameters
        ----------
        res : scipy.optimize.OptimizeResult
            The result from scipy.optimize.least_squares.
        parameter_labels : list of str, optional
            Labels for parameters. If None, uses default 'param{i}'.
        """
        self.theta_hat = res.x
        self.residuals = res.fun
        self.jacobian = res.jac
        self.n_data, self.n_params = self.jacobian.shape

        # Compute chi-squared
        self.chi2 = np.dot(self.residuals, self.residuals)
        self.dof = max(self.n_data - self.n_params, 1)
        self.chi2_reduced = self.chi2 / self.dof

        # SVD of Jacobian
        self.U, self.S, self.Vt = np.linalg.svd(self.jacobian, full_matrices=False)
        self.V = self.Vt.T

        # Covariance matrix
        H_inv = (self.Vt.T * (1.0 / (self.S**2))) @ self.Vt
        self.covariance = self.chi2_reduced * H_inv
        self.parameter_errors = np.sqrt(np.diag(self.covariance))

        # Correlation matrix
        self.correlation = self.covariance / np.outer(
            self.parameter_errors, self.parameter_errors
        )

        # Parameter labels
        if parameter_labels is None:
            self.parameter_labels = [f"param{i}" for i in range(self.n_params)]
        else:
            if len(parameter_labels) != self.n_params:
                raise ValueError(
                    f"Length of parameter_labels ({len(parameter_labels)}) must match number of parameters ({self.n_params})"
                )
            self.parameter_labels = parameter_labels

    def plot_correlation(
        self, fig=None, ax=None, mask_upper=True, cmap="RdBu", vmin=-1, vmax=1, annotate_values=True
    ):
        """
        Plot the correlation matrix as a heatmap with optional value annotations.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure to plot on. If None, creates a new figure.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new axes.
        mask_upper : bool, optional
            If True, masks the upper triangle (including diagonal) with NaN.
        cmap : str, optional
            Colormap for the heatmap.
        vmin, vmax : float, optional
            Colorbar limits.
        annotate_values : bool, optional
            If True, annotates each cell with its correlation value.

        Returns
        -------
        fig, ax : matplotlib objects
            The figure and axes used for plotting.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        corr = self.correlation.copy()
        if mask_upper:
            mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
            corr[mask] = np.nan

        cax = ax.matshow(corr, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(cax, ax=ax)
        ax.set_title("Parameter Correlation Matrix")
        ax.set_xticks(range(self.n_params))
        ax.set_yticks(range(self.n_params))
        ax.set_xticklabels(self.parameter_labels, rotation=45, ha="right")
        ax.set_yticklabels(self.parameter_labels)

        # Add value annotations if requested
        if annotate_values:
            for i in range(self.n_params):
                for j in range(self.n_params):
                    if not mask_upper or i >= j:  # Only annotate lower triangle + diagonal if masked
                        value = self.correlation[i, j]
                        # Choose text color based on background intensity
                        text_color = 'white' if abs(value) > 0.5 else 'black'
                        ax.text(j, i, f'{value:.2f}',
                               ha='center', va='center',
                               color=text_color, fontweight='bold', fontsize=8)

        return fig, ax

    def plot_singular_values(self, fig=None, ax=None):
        """
        Plot the singular values of the Jacobian on a semilog scale.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure to plot on. If None, creates a new figure.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new axes.

        Returns
        -------
        fig, ax : matplotlib objects
            The figure and axes used for plotting.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        ax.semilogy(range(len(self.S)), self.S, marker="o", linestyle="-")
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Singular Value")
        ax.set_title("Singular Values of Jacobian")
        ax.set_xticks(range(len(self.S)))
        ax.set_xticklabels([str(i + 1) for i in range(len(self.S))])
        ax.grid(True, alpha=0.3)

        return fig, ax

    def plot_singular_vectors(self, num_vectors=4, fig=None, ax=None):
        """
        Plot the absolute values of the first num_vectors singular vectors.

        Parameters
        ----------
        num_vectors : int, optional
            Number of singular vectors to plot.
        fig : matplotlib.figure.Figure, optional
            Figure to plot on. If None, creates a new figure.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new axes.

        Returns
        -------
        fig, ax : matplotlib objects
            The figure and axes used for plotting.
        """
        if num_vectors > self.n_params:
            num_vectors = self.n_params

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
        markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

        for i in range(num_vectors):
            ax.plot(
                np.abs(self.V[:, i]),
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linestyle="-",
                linewidth=2,
                markersize=6,
                markerfacecolor=colors[i % len(colors)],
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=f"{i + 1}st SV",
            )

        ax.set_xticks(range(self.n_params))
        ax.set_xticklabels(self.parameter_labels, rotation=45, ha="right")
        ax.set_xlabel("Parameters")
        ax.set_ylabel("|V| (Absolute value of singular vector components)")
        ax.set_title("Singular Vector Components (Parameter Sensitivities)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig, ax

    def plot_residuals(self, exp_vals, sigma, fig=None, ax=None):
        """
        Plot the residuals in absolute units and normalized by sigma.

        Parameters
        ----------
        exp_vals : array_like
            Experimental x values (e.g., phase values).
        sigma : array_like
            Uncertainties in the data.
        fig : matplotlib.figure.Figure, optional
            Figure to plot on. If None, creates a new figure with 2 subplots.
        ax : array of matplotlib.axes.Axes, optional
            Axes array [ax1, ax2] to plot on. If None, creates new subplots.

        Returns
        -------
        fig, ax : matplotlib objects
            The figure and axes used for plotting.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

        ax[0].axhline(0, color="k", lw=1)
        ax[0].scatter(exp_vals, self.residuals * sigma * 1e3, s=15)
        ax[0].set_ylabel("Residuals (MHz)")  # Assuming MHz units, adjust as needed

        ax[1].axhline(0, color="k", lw=1)
        ax[1].scatter(exp_vals, self.residuals, s=15)
        ax[1].set_ylabel(r"Residuals ($\sigma$)")
        ax[1].set_xlabel("Experimental Values")

        plt.tight_layout()

        return fig, ax

    def summary(self):
        """
        Print a summary of the fit analysis.

        Returns
        -------
        str
            Summary string.
        """
        summary_str = "Fit Analysis Summary:\n"
        summary_str += f"Number of data points: {self.n_data}\n"
        summary_str += f"Number of parameters: {self.n_params}\n"
        summary_str += f"Degrees of freedom: {self.dof}\n"
        summary_str += f"Chi-squared: {self.chi2:.3f}\n"
        summary_str += f"Reduced chi-squared: {self.chi2_reduced:.3f}\n"
        summary_str += "Parameter values and errors:\n"
        for _, (label, val, err) in enumerate(
            zip(self.parameter_labels, self.theta_hat, self.parameter_errors)
        ):
            summary_str += f"  {label}: {val:.5g} ± {err:.3g}\n"
        return summary_str

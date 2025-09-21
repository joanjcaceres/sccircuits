import csv

import numpy as np
from .circuit import Circuit

from .transition_fitter import TransitionFitter


class CircuitFitter:
    def __init__(
        self,
        Ej_initial: float,
        frequencies_initial: list[float],
        phase_zpf_initial: list[float],
        dimensions: list[int],
        transitions: dict[tuple[int, int], list[tuple[float, float]]],
        Ej_lower_bound: float = 0.0,
        frequencies_bounds: list[tuple[float, float]] = None,
        phase_zpf_bounds: list[tuple[float, float]] = None,
        epsilon_r_initial: float = None,
        Gamma_initial: float = None,
        epsilon_r_bounds: tuple[float, float] = None,
        Gamma_bounds: tuple[float, float] = None,
        truncation: int = 40,
        x_scale: float = 2 * np.pi,
        optimizer: str = "least_squares",
        use_bogoliubov: bool = True,
    ):
        if len(frequencies_initial) != len(phase_zpf_initial) or len(
            frequencies_initial
        ) != len(dimensions):
            raise ValueError(
                    f"Frequencies, phase_zpf, and dimensions must have the same length. "
                    f"But got {len(frequencies_initial)}, {len(phase_zpf_initial)}, and {len(dimensions)}."

            )
            
        # Validate fermionic coupling parameters
        self.has_fermionic_coupling = (epsilon_r_initial is not None and Gamma_initial is not None)
        if (epsilon_r_initial is None and Gamma_initial is not None) or (epsilon_r_initial is not None and Gamma_initial is None):
            raise ValueError("Both epsilon_r_initial and Gamma_initial must be provided together for fermionic coupling, or both should be None.")
        
        self.modes = len(frequencies_initial)
        self.Ej_initial = Ej_initial
        self.frequencies_initial = np.asarray(frequencies_initial)
        self.phase_zpf_initial = np.asarray(phase_zpf_initial)
        self.dimensions = dimensions
        self.truncation = truncation
        self.Ej_lower_bound = Ej_lower_bound
        self.frequencies_bounds = frequencies_bounds
        self.phase_zpf_bounds = phase_zpf_bounds
        
        # Store fermionic coupling parameters
        self.epsilon_r_initial = epsilon_r_initial
        self.Gamma_initial = Gamma_initial
        self.epsilon_r_bounds = epsilon_r_bounds
        self.Gamma_bounds = Gamma_bounds
        
        self.x_scale = x_scale
        self.use_bogoliubov = use_bogoliubov

        self.params_initial = self._params_initial()

        self.transition_fitter = TransitionFitter(
            model_func=self.eigenvalues_function,
            data=transitions,
            returns_eigenvalues=True,
        )

        self.bounds = self._bounds()

    def _bounds(self):
        if self.use_bogoliubov:
            # Use k parameterization with sigmoid mapping to Ej
            lower_bounds = [self._Ej_to_k(self.Ej_lower_bound)]
        else:
            # Use direct Ej parameterization
            lower_bounds = [self.Ej_lower_bound]

        upper_bounds = [np.inf]

        if self.frequencies_bounds is None:
            lower_bounds.extend([0.0] * self.modes)
            upper_bounds.extend([np.inf] * self.modes)
        else:
            lower_bounds.extend([b[0] for b in self.frequencies_bounds])
            upper_bounds.extend([b[1] for b in self.frequencies_bounds])
        if self.phase_zpf_bounds is None:
            lower_bounds.extend([0.0] * self.modes)
            upper_bounds.extend([np.inf] * self.modes)
        else:
            lower_bounds.extend([b[0] for b in self.phase_zpf_bounds])
            upper_bounds.extend([b[1] for b in self.phase_zpf_bounds])

        # Add fermionic coupling parameter bounds if enabled
        if self.has_fermionic_coupling:
            # epsilon_r bounds
            if self.epsilon_r_bounds is None:
                lower_bounds.append(0.0)  # epsilon_r should be positive
                upper_bounds.append(np.inf)
            else:
                lower_bounds.append(self.epsilon_r_bounds[0])
                upper_bounds.append(self.epsilon_r_bounds[1])
            
            # Gamma bounds  
            if self.Gamma_bounds is None:
                lower_bounds.append(0.0)  # Gamma should be positive
                upper_bounds.append(np.inf)
            else:
                lower_bounds.append(self.Gamma_bounds[0])
                upper_bounds.append(self.Gamma_bounds[1])

        return np.array([lower_bounds, upper_bounds])

    def fit(self, verbose=1):  # Adapt for DE.
        return self.transition_fitter.fit(
            params_initial=self.params_initial, bounds=self.bounds, verbose=verbose
        )

    def eigenvalues_function(self, phase_ext, params):
        phase_ext *= self.x_scale
        Ej, frequencies, phase_zpf, epsilon_r, Gamma = self._convert_params_to_lists(params)

        circuit = Circuit(
            frequencies=frequencies,
            phase_zpf=phase_zpf,
            dimensions=self.dimensions,
            Ej=Ej,
            Gamma=Gamma,
            epsilon_r=epsilon_r,
            phase_ext=phase_ext,
            use_bogoliubov=self.use_bogoliubov,
        )

        evals, _ = circuit.eigensystem(truncation=self.truncation, phase_ext=phase_ext)
        return evals

    def _params_initial(self):
        if self.use_bogoliubov:
            # Use k parameterization
            first_param = self._Ej_to_k(self.Ej_initial)
        else:
            # Use direct Ej parameterization
            first_param = self.Ej_initial

        params = [
            first_param,
            *self.frequencies_initial,
            *self.phase_zpf_initial,
        ]
        
        # Add fermionic coupling parameters if enabled
        if self.has_fermionic_coupling:
            params.extend([self.epsilon_r_initial, self.Gamma_initial])
        
        return params

    def _k_to_Ej(self, k, frequencies, phase_zpf):
        frequencies = np.array(frequencies)
        phase_zpf = np.array(phase_zpf)
        phi_zpf_rms = np.sqrt(np.sum(phase_zpf**2))
        collective_frequency = np.sum(frequencies * phase_zpf**2) / phi_zpf_rms**2
        Ej_max = collective_frequency / (2 * phi_zpf_rms**2)
        sigmoid = 1 / (1 + np.exp(-k))
        Ej = Ej_max * sigmoid
        return Ej

    def _convert_params_to_lists(self, params):
        frequencies = np.array(params[1 : 1 + self.modes])
        phase_zpf = np.array(params[1 + self.modes : 1 + 2 * self.modes])

        if self.use_bogoliubov:
            # Convert k to Ej using sigmoid mapping
            Ej = self._k_to_Ej(params[0], frequencies, phase_zpf)
        else:
            # Use direct Ej parameterization
            Ej = params[0]

        # Extract fermionic coupling parameters if enabled
        if self.has_fermionic_coupling:
            epsilon_r = params[1 + 2 * self.modes]
            Gamma = params[1 + 2 * self.modes + 1]
        else:
            epsilon_r = None
            Gamma = None

        return Ej, frequencies, phase_zpf, epsilon_r, Gamma

    def _Ej_to_k(self, Ej):
        phi_zpf_rms = np.sqrt(np.sum(self.phase_zpf_initial**2))
        collective_frequency = (
            np.sum(self.frequencies_initial * self.phase_zpf_initial**2)
            / phi_zpf_rms**2
        )
        Ej_max = collective_frequency / (2 * phi_zpf_rms**2)
        sigmoid = Ej / Ej_max
        k_value = -np.log((1 / sigmoid) - 1)
        return k_value

    def save_results_csv(self, filepath: str):
        """
        Save circuit fit results to a CSV file with both raw parameters and physical values.

        Args:
            filepath (str): Path where to save the CSV file
        """
        if self.transition_fitter.result is None:
            raise RuntimeError("No fit has been run yet. Call fit() first.")

        # Get fitted parameters
        fitted_params = self.transition_fitter.result.x
        Ej, frequencies, phase_zpf = self._convert_params_to_lists(fitted_params)

        circuit = Circuit(
            frequencies=frequencies,
            phase_zpf=phase_zpf,
            dimensions=self.dimensions,
            Ej=Ej,
            phase_ext=0,
            use_bogoliubov=self.use_bogoliubov,
        )

        with open(filepath, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write header with fit information
            writer.writerow(["# circuit Fit Results"])
            writer.writerow(["# Timestamp:", f"{np.datetime64('now')}"])
            writer.writerow(["# Cost:", str(self.transition_fitter.result.cost)])
            writer.writerow(["# Optimizer:", self.transition_fitter.optimizer])
            writer.writerow([])

            # Write raw parameters (k or Ej, frequencies, phase_zpf)
            writer.writerow(["# Raw Parameters"])
            writer.writerow(["parameter", "value"])
            if self.use_bogoliubov:
                writer.writerow(["k", fitted_params[0]])
            else:
                writer.writerow(["Ej", fitted_params[0]])
            writer.writerow(["# Frequencies"])
            for i in range(len(frequencies)):
                writer.writerow([frequencies[i]])
            writer.writerow(["# Phase ZPF"])
            for i in range(len(phase_zpf)):
                writer.writerow([phase_zpf[i]])
            writer.writerow([])

            # Write physical parameters

            writer.writerow(["# Physical Parameters"])
            writer.writerow(["parameter", "value", "unit"])
            writer.writerow(["Ej", Ej, "GHz"])
            writer.writerow(["collective_frequency", circuit.collective_frequency, "GHz"])
            writer.writerow(["non_linear_frequency", circuit.non_linear_frequency, "GHz"])
            writer.writerow(["non_linear_phase_zpf", circuit.non_linear_phase_zpf, "rad"])
            for i in range(circuit.modes - 1):
                writer.writerow(
                    [f"linear_frequency_{i}", circuit.linear_frequencies[i], "GHz"]
                )
            for i in range(circuit.modes - 1):
                writer.writerow([f"linear_coupling_{i}", circuit.linear_coupling[i], "GHz"])

            # Write configuration parameters
            writer.writerow(["# Configuration"])
            writer.writerow(["parameter", "value"])
            writer.writerow(["modes", self.modes])
            writer.writerow(["truncation", self.truncation])
            writer.writerow(["x_scale", self.x_scale])
            writer.writerow(["use_bogoliubov", self.use_bogoliubov])
            for i, dim in enumerate(self.dimensions):
                writer.writerow([f"dimension_{i}", dim])
            writer.writerow([])

            # Write experimental vs theoretical data
            writer.writerow(["# Data Points"])
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

            for transition, data_points in self.transition_fitter.data.items():
                phi_values = [dp.phi_ext for dp in data_points]
                theo_values = self.transition_fitter.get_theoretical_curve(
                    transition, phi_values
                )
                for dp, tv in zip(data_points, theo_values):
                    residual = tv - dp.value
                    writer.writerow(
                        [
                            transition[0],
                            transition[1],
                            dp.phi_ext,
                            dp.value,
                            tv,
                            residual,
                            dp.sigma if dp.sigma is not None else "",
                        ]
                    )

        print(f"circuit fit results saved to: {filepath}")

    def save_complete_result(self, filepath: str):
        """
        Save complete circuit fit results including the full optimization result object.

        Delegates to TransitionFitter and saves additional CSV for readability.

        Args:
            filepath (str): Path where to save the files (without extension)
        """
        if self.transition_fitter.result is None:
            raise RuntimeError("No fit has been run yet. Call fit() first.")

        # Use TransitionFitter's complete save functionality
        self.transition_fitter.save_complete_result(filepath)

        # Also save CSV for human readability
        from pathlib import Path

        csv_path = Path(filepath).with_suffix(".csv")
        self.save_results_csv(str(csv_path))

        print(f"Additional CSV file saved: {csv_path}")
        print("Use get_analysis_report() for circuit-specific parameters.")

    @classmethod
    def load_complete_result(cls, filepath: str):
        """
        Load complete circuit fit results from a pickle file.

        Note: This loads a TransitionFitter and wraps it in a Fit_circuit instance.
        The original circuit configuration must be reconstructed from the fit data.

        Args:
            filepath (str): Path to the pickle file

        Returns:
            Fit_circuit: New instance with the complete loaded result
        """
        # Load the TransitionFitter
        loaded_transition_fitter = TransitionFitter.load_complete_result(filepath)

        # We need to reconstruct circuit configuration from the fit parameters
        # This is a limitation - we don't have the original circuit config
        print("Warning: circuit configuration reconstructed from fit parameters.")
        print("Some original settings (bounds, truncation) may not be preserved.")

        # Extract basic info from the loaded fitter
        transitions = loaded_transition_fitter.data

        # Minimal reconstruction - user may need to adjust
        estimated_modes = 1  # Default assumption

        new_fitter = cls(
            Ej_initial=1.0,  # Default - user should verify
            frequencies_initial=[5.0] * estimated_modes,  # Default - user should verify
            phase_zpf_initial=[0.5] * estimated_modes,  # Default - user should verify
            dimensions=[10] * estimated_modes,  # Default - user should verify
            transitions=transitions,
            truncation=40,  # Default - user should verify
            optimizer=loaded_transition_fitter.optimizer,
        )

        # Replace with the loaded transition_fitter
        new_fitter.transition_fitter = loaded_transition_fitter

        print("circuit fitter loaded with reconstructed configuration.")
        print("Please verify initial parameters match your original setup.")

        return new_fitter

    def get_analysis_report(self) -> dict:
        """
        Get comprehensive analysis report using all TransitionFitter capabilities.

        This method leverages the complete result object to provide detailed
        analysis including parameter uncertainties, convergence analysis, etc.

        Returns:
            dict: Complete analysis report from TransitionFitter with circuit parameters
        """
        if self.transition_fitter.result is None:
            raise RuntimeError("No fit has been run yet. Call fit() first.")

        # Get comprehensive report from TransitionFitter
        report = self.transition_fitter.get_comprehensive_report()

        # Add circuit-specific information
        fitted_params = self.transition_fitter.result.x
        Ej, frequencies, phase_zpf = self._convert_params_to_lists(fitted_params)

        circuit = Circuit(
            frequencies=frequencies,
            phase_zpf=phase_zpf,
            dimensions=self.dimensions,
            Ej=Ej,
            phase_ext=0,
            use_bogoliubov=self.use_bogoliubov,
        )

        report["circuit_parameters"] = {
            "Ej": float(Ej),
            "collective_frequency": float(circuit.collective_frequency),
            "non_linear_frequency": float(circuit.non_linear_frequency),
            "non_linear_phase_zpf": float(circuit.non_linear_phase_zpf),
            "linear_frequencies": [float(f) for f in circuit.linear_frequencies]
            if circuit.modes > 1
            else [],
            "linear_coupling": [float(c) for c in circuit.linear_coupling]
            if circuit.modes > 1
            else [],
            "modes": self.modes,
            "truncation": self.truncation,
            "use_bogoliubov": self.use_bogoliubov,
        }

        return report

    def print_analysis_summary(
        self, show_transitions: bool = True, show_outliers: bool = True
    ):
        """
        Print formatted analysis summary including circuit-specific parameters.

        This delegates to TransitionFitter's print_fit_summary and adds circuit info.

        Args:
            show_transitions (bool): Whether to show individual transition analysis
            show_outliers (bool): Whether to show outlier information
        """
        if self.transition_fitter.result is None:
            raise RuntimeError("No fit has been run yet. Call fit() first.")

        # Print TransitionFitter summary
        self.transition_fitter.print_fit_summary(show_transitions, show_outliers)

        # Add circuit-specific information
        fitted_params = self.transition_fitter.result.x
        Ej, frequencies, phase_zpf = self._convert_params_to_lists(fitted_params)

        circuit = Circuit(
            frequencies=frequencies,
            phase_zpf=phase_zpf,
            dimensions=self.dimensions,
            Ej=Ej,
            phase_ext=0,
            use_bogoliubov=self.use_bogoliubov,
        )

        print("\ncircuit PARAMETERS:")
        print(f"  Ej = {Ej:.4f} GHz")
        print(f"  Collective frequency = {circuit.collective_frequency:.4f} GHz")
        print(f"  Non-linear frequency = {circuit.non_linear_frequency:.4f} GHz")
        print(f"  Non-linear phase ZPF = {circuit.non_linear_phase_zpf:.4f} rad")

        if circuit.modes > 1:
            for i, (freq, coupling) in enumerate(
                zip(circuit.linear_frequencies, circuit.linear_coupling)
            ):
                print(
                    f"  Linear mode {i}: {freq:.4f} GHz (coupling: {coupling:.4f} GHz)"
                )

        print(f"  Modes = {self.modes}")
        print(f"  Truncation = {self.truncation}")
        print(f"  Use Bogoliubov = {self.use_bogoliubov}")

    # Delegate common analysis methods to TransitionFitter
    def get_fit_statistics(self):
        """Get fit statistics. Delegates to TransitionFitter."""
        return self.transition_fitter.get_fit_statistics()

    def get_residual_analysis(self):
        """Get residual analysis. Delegates to TransitionFitter."""
        return self.transition_fitter.get_residual_analysis()

    def get_parameter_uncertainty(self, confidence_level: float = 0.95):
        """Get parameter uncertainties. Delegates to TransitionFitter."""
        return self.transition_fitter.get_parameter_uncertainty(confidence_level)

    def get_convergence_analysis(self):
        """Get convergence analysis. Delegates to TransitionFitter."""
        return self.transition_fitter.get_convergence_analysis()

    def get_transition_analysis(self):
        """Get per-transition analysis. Delegates to TransitionFitter."""
        return self.transition_fitter.get_transition_analysis()

    def get_comprehensive_report(self):
        """Get comprehensive report with circuit parameters. Same as get_analysis_report()."""
        return self.get_analysis_report()

    def print_fit_summary(
        self, show_transitions: bool = True, show_outliers: bool = True
    ):
        """Print fit summary with circuit parameters. Same as print_analysis_summary()."""
        return self.print_analysis_summary(show_transitions, show_outliers)

    def fit_multistart(self, *args, **kwargs):
        """Multi-start fitting. Delegates to TransitionFitter."""
        return self.transition_fitter.fit_multistart(*args, **kwargs)

    @classmethod
    def load_results_csv(cls, filepath: str):
        """
        Load circuit fit results from a CSV file and return a dictionary with all information.

        Args:
            filepath (str): Path to the CSV file

        Returns:
            dict: Dictionary containing fitted parameters, physical values, and configuration
        """
        results = {
            "raw_parameters": {},
            "physical_parameters": {},
            "configuration": {},
            "fit_info": {},
            "data_points": [],
        }

        with open(filepath, newline="") as csvfile:
            reader = csv.reader(csvfile)
            current_section = None

            for row in reader:
                if not row or row[0].startswith("#"):
                    if len(row) > 1 and "Cost:" in row[1]:
                        results["fit_info"]["cost"] = float(
                            row[1].split(":")[1].strip()
                        )
                    elif len(row) > 1 and "Optimizer:" in row[1]:
                        results["fit_info"]["optimizer"] = row[1].split(":")[1].strip()
                    elif len(row) > 1 and "Timestamp:" in row[1]:
                        results["fit_info"]["timestamp"] = row[1].split(":")[1].strip()
                    continue

                if row[0] == "parameter" and len(row) > 1:
                    if row[1] == "value":
                        if len(row) == 2:
                            current_section = "raw_parameters"
                        else:  # has unit column
                            current_section = "physical_parameters"
                    continue
                elif row[0] == "transition_i":
                    current_section = "data_points"
                    continue

                if current_section == "raw_parameters" and len(row) >= 2:
                    try:
                        results["raw_parameters"][row[0]] = float(row[1])
                    except ValueError:
                        results["configuration"][row[0]] = row[1]
                elif current_section == "physical_parameters" and len(row) >= 2:
                    try:
                        results["physical_parameters"][row[0]] = {
                            "value": float(row[1]),
                            "unit": row[2] if len(row) > 2 else "",
                        }
                    except ValueError:
                        pass
                elif current_section == "data_points" and len(row) >= 7:
                    try:
                        sigma_val = row[6].strip()
                        data_point = {
                            "transition": (int(row[0]), int(row[1])),
                            "phi_ext": float(row[2]),
                            "experimental": float(row[3]),
                            "theoretical": float(row[4]),
                            "residual": float(row[5]),
                            "sigma": float(sigma_val) if sigma_val else None,
                        }
                        results["data_points"].append(data_point)
                    except ValueError:
                        pass

        print(f"circuit fit results loaded from: {filepath}")
        return results

    def create_from_csv(self, filepath: str, transitions_data: dict = None):
        """
        Create a new Fit_circuit instance using parameters loaded from a CSV file.

        NOTE: This method loads only basic parameter information from CSV.
        For complete analysis capabilities (including parameter uncertainties,
        convergence analysis, etc.), use load_complete_result() instead.

        Args:
            filepath (str): Path to the CSV file with saved fit results
            transitions_data (dict, optional): New transition data. If None, uses data from CSV.

        Returns:
            Fit_circuit: New instance with loaded parameters as initial values
        """
        loaded_results = self.load_results_csv(filepath)

        # Extract parameters
        raw_params = loaded_results["raw_parameters"]
        config = loaded_results["configuration"]

        # Reconstruct initial parameters
        k = raw_params["k"]
        frequencies = [
            raw_params[f"frequency_{i}"]
            for i in range(int(config.get("modes", self.modes)))
        ]
        phase_zpf = [
            raw_params[f"phase_zpf_{i}"]
            for i in range(int(config.get("modes", self.modes)))
        ]
        dimensions = [
            int(config.get(f"dimension_{i}", 2))
            for i in range(int(config.get("modes", self.modes)))
        ]

        # Convert k back to Ej for initial value
        Ej_initial = self._k_to_Ej(k, frequencies, phase_zpf)

        # Prepare transitions data
        if transitions_data is None:
            transitions_data = {}
            for dp in loaded_results["data_points"]:
                transition = dp["transition"]
                if transition not in transitions_data:
                    transitions_data[transition] = []
                transitions_data[transition].append((dp["phi_ext"], dp["experimental"]))

        # Create new instance
        new_fitter = CircuitFitter(
            Ej_initial=Ej_initial,
            frequencies_initial=frequencies,
            phase_zpf_initial=phase_zpf,
            dimensions=dimensions,
            transitions=transitions_data,
            truncation=int(config.get("truncation", 40)),
            x_scale=float(config.get("x_scale", 2 * np.pi)),
            optimizer=loaded_results["fit_info"].get("optimizer", "least_squares"),
        )

        return new_fitter

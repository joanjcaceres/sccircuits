import numpy as np
from typing import Dict, Optional, Sequence, Tuple, Union

from .circuit import Circuit
from .transition_fitter import DataPoint, TransitionFitter

TransitionKey = Tuple[int, int]
TransitionDatum = Union[DataPoint, Tuple[float, float], Tuple[float, float, float]]
TransitionData = Dict[TransitionKey, Sequence[TransitionDatum]]


class CircuitFitter:
    """Fit superconducting circuit parameters against transition spectroscopy data."""

    def __init__(
        self,
        Ej_initial: float,
        non_linear_frequency_initial: float,
        non_linear_phase_zpf_initial: float,
        dimensions: Sequence[int],
        transitions: TransitionData,
        linear_frequencies_initial: Optional[Sequence[float]] = None,
        linear_couplings_initial: Optional[Sequence[float]] = None,
        Ej_second_initial: Optional[float] = None,
        Ej_lower_bound: float = 0.0,
        non_linear_frequency_bounds: Optional[Tuple[float, float]] = None,
        non_linear_phase_zpf_bounds: Optional[Tuple[float, float]] = None,
        linear_frequencies_bounds: Optional[Sequence[Tuple[float, float]]] = None,
        linear_couplings_bounds: Optional[Sequence[Tuple[float, float]]] = None,
        epsilon_r_initial: Optional[float] = None,
        Gamma_initial: Optional[float] = None,
        epsilon_r_bounds: Optional[Tuple[float, float]] = None,
        Gamma_bounds: Optional[Tuple[float, float]] = None,
        Ej_second_bounds: Optional[Tuple[float, float]] = None,
        truncation: "int | Sequence[int]" = 40,
        optimizer: str = "least_squares",
        use_bogoliubov: bool = True,
        fit_Ej_second: bool = False,
    ) -> None:
        self.dimensions = [int(dim) for dim in dimensions]
        if not self.dimensions:
            raise ValueError("At least one Hilbert-space dimension must be provided.")
        if any(dim <= 0 for dim in self.dimensions):
            raise ValueError("All Hilbert-space dimensions must be positive integers.")

        self.modes = len(self.dimensions)
        self.linear_mode_count = max(self.modes - 1, 0)

        self.truncation = truncation
        self.optimizer = optimizer
        self.use_bogoliubov = use_bogoliubov

        self.Ej_initial = float(Ej_initial)
        self.Ej_lower_bound = float(Ej_lower_bound)

        self.non_linear_frequency_initial = float(non_linear_frequency_initial)
        if self.non_linear_frequency_initial <= 0:
            raise ValueError("non_linear_frequency_initial must be positive.")

        self.non_linear_phase_zpf_initial = float(non_linear_phase_zpf_initial)
        if self.non_linear_phase_zpf_initial <= 0:
            raise ValueError("non_linear_phase_zpf_initial must be positive.")

        self.fit_Ej_second = bool(fit_Ej_second)
        if self.fit_Ej_second:
            if Ej_second_initial is None:
                raise ValueError(
                    "Ej_second_initial must be provided when fit_Ej_second=True."
                )
            self.Ej_second_initial = float(Ej_second_initial)
            self.Ej_second_value = self.Ej_second_initial
        else:
            if Ej_second_bounds is not None:
                raise ValueError(
                    "Ej_second_bounds cannot be provided when fit_Ej_second=False."
                )
            self.Ej_second_initial = None
            self.Ej_second_value = (
                float(Ej_second_initial) if Ej_second_initial is not None else 0.0
            )

        if linear_frequencies_initial is None:
            linear_frequencies_initial = []
        if linear_couplings_initial is None:
            linear_couplings_initial = []

        self.linear_frequencies_initial = np.asarray(linear_frequencies_initial, dtype=float)
        self.linear_couplings_initial = np.asarray(linear_couplings_initial, dtype=float)

        if self.linear_mode_count == 0:
            if self.linear_frequencies_initial.size or self.linear_couplings_initial.size:
                raise ValueError(
                    "No linear modes expected; do not supply linear_frequencies_initial or linear_couplings_initial."
                )
            self.linear_frequencies_initial = np.array([], dtype=float)
            self.linear_couplings_initial = np.array([], dtype=float)
        else:
            expected = self.linear_mode_count
            if (
                self.linear_frequencies_initial.size != expected
                or self.linear_couplings_initial.size != expected
            ):
                raise ValueError(
                    "linear_frequencies_initial and linear_couplings_initial must both have length len(dimensions) - 1."
                )
            if np.any(self.linear_frequencies_initial <= 0):
                raise ValueError("All linear_frequencies_initial must be positive.")
            if np.any(self.linear_couplings_initial < 0):
                raise ValueError("All linear_couplings_initial must be non-negative.")

        self.non_linear_frequency_bounds = (
            tuple(non_linear_frequency_bounds)
            if non_linear_frequency_bounds is not None
            else None
        )
        self.non_linear_phase_zpf_bounds = (
            tuple(non_linear_phase_zpf_bounds)
            if non_linear_phase_zpf_bounds is not None
            else None
        )

        if linear_frequencies_bounds is not None:
            if len(linear_frequencies_bounds) != self.linear_mode_count:
                raise ValueError(
                    "linear_frequencies_bounds must have length len(dimensions) - 1 when provided."
                )
            self.linear_frequencies_bounds = [tuple(b) for b in linear_frequencies_bounds]
        else:
            self.linear_frequencies_bounds = None

        if linear_couplings_bounds is not None:
            if len(linear_couplings_bounds) != self.linear_mode_count:
                raise ValueError(
                    "linear_couplings_bounds must have length len(dimensions) - 1 when provided."
                )
            self.linear_couplings_bounds = [tuple(b) for b in linear_couplings_bounds]
        else:
            self.linear_couplings_bounds = None

        self.epsilon_r_bounds = (
            tuple(epsilon_r_bounds) if epsilon_r_bounds is not None else None
        )
        self.Gamma_bounds = tuple(Gamma_bounds) if Gamma_bounds is not None else None
        self.Ej_second_bounds = (
            tuple(Ej_second_bounds) if Ej_second_bounds is not None else None
        ) if self.fit_Ej_second else None

        self.has_fermionic_coupling = (epsilon_r_initial is not None and Gamma_initial is not None)
        if (epsilon_r_initial is None) ^ (Gamma_initial is None):
            raise ValueError("Both epsilon_r_initial and Gamma_initial must be provided together, or neither.")

        self.epsilon_r_initial = None if epsilon_r_initial is None else float(epsilon_r_initial)
        self.Gamma_initial = None if Gamma_initial is None else float(Gamma_initial)

        self.params_initial = self._build_initial_parameter_vector()
        self.bounds = self._build_bounds()

        jacobian_callable = None #if self.has_fermionic_coupling else self.eigenvalues_jacobian

        self.transition_fitter = TransitionFitter(
            model_func=self.eigenvalues_function,
            data=transitions,
            returns_eigenvalues=True,
            jacobian_func=jacobian_callable,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, verbose: int = 1, **kwargs):
        if self.optimizer != "least_squares":
            raise NotImplementedError("Currently only 'least_squares' optimizer is supported.")
        return self.transition_fitter.fit(
            params_initial=self.params_initial,
            bounds=self.bounds,
            verbose=verbose,
            **kwargs,
        )

    def fit_multistart(self, *args, **kwargs):
        return self.transition_fitter.fit_multistart(*args, **kwargs)

    def eigenvalues_function(self, phase_ext: float, params: Sequence[float]) -> np.ndarray:
        circuit = self._circuit_from_params(phase_ext, params)
        eigenvalues, _ = circuit.eigensystem(truncation=self.truncation, phase_ext=phase_ext)
        return eigenvalues

    def eigenvalues_jacobian(self, phase_ext: float, params: Sequence[float]) -> np.ndarray:
        circuit = self._circuit_from_params(phase_ext, params)
        _, _, gradients, names = circuit.eigensystem_with_gradients(
            truncation=self.truncation,
            phase_ext=phase_ext,
        )

        name_to_index = {name: idx for idx, name in enumerate(names)}

        (
            Ej,
            non_linear_frequency,
            non_linear_phase_zpf,
            _Ej_second,
            _linear_frequencies,
            _linear_couplings,
            _epsilon_r,
            _Gamma,
        ) = self._unpack_parameter_vector(params)

        columns = []
        dEdEj = gradients[:, name_to_index["Ej"]]
        if self.use_bogoliubov:
            k = params[0]
            dEj_dk = self._dEj_dk(k, non_linear_frequency, non_linear_phase_zpf)
            columns.append(dEdEj * dEj_dk)
        else:
            columns.append(dEdEj)

        columns.append(gradients[:, name_to_index["non_linear_frequency"]])
        columns.append(gradients[:, name_to_index["non_linear_phase_zpf"]])

        if self.fit_Ej_second:
            ej_second_idx = name_to_index.get("Ej_second")
            if ej_second_idx is None:
                raise KeyError("Gradient for Ej_second not found in circuit Jacobian output.")
            columns.append(gradients[:, ej_second_idx])

        for idx in range(self.linear_mode_count):
            name = f"linear_frequency_{idx}"
            columns.append(gradients[:, name_to_index[name]])
        for idx in range(self.linear_mode_count):
            name = f"linear_coupling_{idx}"
            columns.append(gradients[:, name_to_index[name]])

        return np.column_stack(columns)

    def get_fit_statistics(self):
        return self.transition_fitter.get_fit_statistics()

    def get_residual_analysis(self):
        return self.transition_fitter.get_residual_analysis()

    def get_parameter_uncertainty(self, confidence_level: float = 0.95):
        return self.transition_fitter.get_parameter_uncertainty(confidence_level)

    def get_convergence_analysis(self):
        return self.transition_fitter.get_convergence_analysis()

    def get_transition_analysis(self):
        return self.transition_fitter.get_transition_analysis()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_initial_parameter_vector(self) -> np.ndarray:
        params = []
        Ej_initial = self.Ej_initial
        if self.use_bogoliubov:
            k_initial = self._Ej_to_k(
                Ej_initial,
                self.non_linear_frequency_initial,
                self.non_linear_phase_zpf_initial,
            )
            params.append(k_initial)
        else:
            params.append(Ej_initial)

        params.append(self.non_linear_frequency_initial)
        params.append(self.non_linear_phase_zpf_initial)
        if self.fit_Ej_second:
            params.append(self.Ej_second_initial)
        params.extend(self.linear_frequencies_initial.tolist())
        params.extend(self.linear_couplings_initial.tolist())

        if self.has_fermionic_coupling:
            params.append(self.epsilon_r_initial)
            params.append(self.Gamma_initial)

        return np.asarray(params, dtype=float)

    def _build_bounds(self) -> np.ndarray:
        lower: list[float] = []
        upper: list[float] = []

        if self.use_bogoliubov:
            Ej_max = self._Ej_max(
                self.non_linear_frequency_initial, self.non_linear_phase_zpf_initial
            )
            if self.Ej_lower_bound >= Ej_max:
                raise ValueError(
                    "Ej_lower_bound must be strictly smaller than the Bogoliubov stability limit Ej_max."
                )
            lower.append(
                self._Ej_to_k(
                    self.Ej_lower_bound,
                    self.non_linear_frequency_initial,
                    self.non_linear_phase_zpf_initial,
                )
            )
            upper.append(np.inf)
        else:
            lower.append(self.Ej_lower_bound)
            upper.append(np.inf)

        # non-linear frequency bound
        if self.non_linear_frequency_bounds is None:
            lower.append(0.0)
            upper.append(np.inf)
        else:
            if len(self.non_linear_frequency_bounds) != 2:
                raise ValueError(
                    "non_linear_frequency_bounds must be a 2-tuple (min, max)."
                )
            lower.append(self.non_linear_frequency_bounds[0])
            upper.append(self.non_linear_frequency_bounds[1])

        # non-linear phase zpf bound
        if self.non_linear_phase_zpf_bounds is None:
            lower.append(0.0)
            upper.append(np.inf)
        else:
            if len(self.non_linear_phase_zpf_bounds) != 2:
                raise ValueError(
                    "non_linear_phase_zpf_bounds must be a 2-tuple (min, max)."
                )
            lower.append(self.non_linear_phase_zpf_bounds[0])
            upper.append(self.non_linear_phase_zpf_bounds[1])

        # Ej_second bounds (only when fitting)
        if self.fit_Ej_second:
            if self.Ej_second_bounds is None:
                lower.append(-np.inf)
                upper.append(np.inf)
            else:
                if len(self.Ej_second_bounds) != 2:
                    raise ValueError("Ej_second_bounds must be a 2-tuple (min, max).")
                lower.append(self.Ej_second_bounds[0])
                upper.append(self.Ej_second_bounds[1])

        # linear frequencies bounds
        if self.linear_mode_count > 0:
            if self.linear_frequencies_bounds is not None:
                for bounds in self.linear_frequencies_bounds:
                    lower.append(bounds[0])
                    upper.append(bounds[1])
            else:
                lower.extend([0.0] * self.linear_mode_count)
                upper.extend([np.inf] * self.linear_mode_count)

            # linear couplings bounds (non-negative by default)
            if self.linear_couplings_bounds is not None:
                for bounds in self.linear_couplings_bounds:
                    lower.append(bounds[0])
                    upper.append(bounds[1])
            else:
                lower.extend([0.0] * self.linear_mode_count)
                upper.extend([np.inf] * self.linear_mode_count)

        if self.has_fermionic_coupling:
            if self.epsilon_r_bounds is None:
                lower.append(0.0)
                upper.append(np.inf)
            else:
                lower.append(self.epsilon_r_bounds[0])
                upper.append(self.epsilon_r_bounds[1])

            if self.Gamma_bounds is None:
                lower.append(0.0)
                upper.append(np.inf)
            else:
                lower.append(self.Gamma_bounds[0])
                upper.append(self.Gamma_bounds[1])

        return np.array([lower, upper], dtype=float)

    def _circuit_from_params(self, phase_ext: float, params: Sequence[float]) -> Circuit:
        (
            Ej,
            non_linear_frequency,
            non_linear_phase_zpf,
            Ej_second,
            linear_frequencies,
            linear_couplings,
            epsilon_r,
            Gamma,
        ) = self._unpack_parameter_vector(params)

        linear_frequencies_seq = None if linear_frequencies.size == 0 else linear_frequencies
        linear_couplings_seq = None if linear_couplings.size == 0 else linear_couplings

        return Circuit(
            non_linear_frequency=non_linear_frequency,
            non_linear_phase_zpf=non_linear_phase_zpf,
            linear_frequencies=linear_frequencies_seq,
            linear_couplings=linear_couplings_seq,
            dimensions=self.dimensions,
            Ej=Ej,
            Ej_second=Ej_second,
            Gamma=Gamma,
            epsilon_r=epsilon_r,
            phase_ext=phase_ext,
        )

    def _unpack_parameter_vector(
        self, params: Sequence[float]
    ) -> Tuple[
        float,
        float,
        float,
        float,
        np.ndarray,
        np.ndarray,
        Optional[float],
        Optional[float],
    ]:
        idx = 0
        params = np.asarray(params, dtype=float)

        k_or_Ej = params[idx]
        idx += 1

        non_linear_frequency = params[idx]
        idx += 1

        non_linear_phase_zpf = params[idx]
        idx += 1

        if self.fit_Ej_second:
            Ej_second = params[idx]
            idx += 1
        else:
            Ej_second = self.Ej_second_value

        linear_freq_slice = slice(idx, idx + self.linear_mode_count)
        linear_frequencies = params[linear_freq_slice]
        idx += self.linear_mode_count

        linear_coup_slice = slice(idx, idx + self.linear_mode_count)
        linear_couplings = params[linear_coup_slice]
        idx += self.linear_mode_count

        if self.use_bogoliubov:
            Ej = self._k_to_Ej(k_or_Ej, non_linear_frequency, non_linear_phase_zpf)
        else:
            Ej = k_or_Ej

        epsilon_r = None
        Gamma = None
        if self.has_fermionic_coupling:
            epsilon_r = params[idx]
            idx += 1
            Gamma = params[idx]

        return (
            float(Ej),
            float(non_linear_frequency),
            float(non_linear_phase_zpf),
            float(Ej_second),
            np.asarray(linear_frequencies, dtype=float),
            np.asarray(linear_couplings, dtype=float),
            None if epsilon_r is None else float(epsilon_r),
            None if Gamma is None else float(Gamma),
        )

    @staticmethod
    def _Ej_max(non_linear_frequency: float, non_linear_phase_zpf: float) -> float:
        if non_linear_phase_zpf <= 0:
            raise ValueError("non_linear_phase_zpf must be positive for Bogoliubov mapping.")
        return non_linear_frequency / (2.0 * non_linear_phase_zpf**2)

    def _Ej_to_k(
        self,
        Ej: float,
        non_linear_frequency: float,
        non_linear_phase_zpf: float,
    ) -> float:
        if Ej <= 0:
            return -np.inf
        Ej_max = self._Ej_max(non_linear_frequency, non_linear_phase_zpf)
        if Ej >= Ej_max:
            raise ValueError(
                "Ej must be strictly smaller than Ej_max = collective_frequency/(2*phi_zpf^2)"
            )
        sigmoid = Ej / Ej_max
        return np.log(sigmoid / (1.0 - sigmoid))

    def _k_to_Ej(
        self,
        k: float,
        non_linear_frequency: float,
        non_linear_phase_zpf: float,
    ) -> float:
        Ej_max = self._Ej_max(non_linear_frequency, non_linear_phase_zpf)
        sigmoid = 1.0 / (1.0 + np.exp(-k))
        return Ej_max * sigmoid

    def _dEj_dk(
        self,
        k: float,
        non_linear_frequency: float,
        non_linear_phase_zpf: float,
    ) -> float:
        Ej_max = self._Ej_max(non_linear_frequency, non_linear_phase_zpf)
        sigmoid = 1.0 / (1.0 + np.exp(-k))
        return Ej_max * sigmoid * (1.0 - sigmoid)

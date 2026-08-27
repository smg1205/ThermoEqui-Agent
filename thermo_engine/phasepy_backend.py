"""Optional Phasepy adapter behind the stable thermodynamic backend contract."""

from __future__ import annotations

import hashlib
import importlib
import json
from importlib.metadata import PackageNotFoundError, version
from math import exp, isfinite
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from thermo import ChemicalConstantsPackage
from thermo.interaction_parameters import IPDB

from schemas.domain import CalculationResult, EquilibriumPoint, FailureType, PhaseResult, TaskManifest
from thermo_engine.errors import ThermoEquiError
from thermo_engine.thermo_backend import ThermoPengRobinsonBackend


class PhasepyPengRobinsonBackend:
    """Phasepy Peng-Robinson VLE adapter using reviewed ChemSep ``kij`` values."""

    dependency = "phasepy==0.0.56"
    parameter_table = "ChemSep PR"
    model_name = "Phasepy/Peng-Robinson"

    def __init__(self) -> None:
        self._phasepy: Any | None = None
        self._equilibrium: Any | None = None
        self._parameter_set_id: str | None = None
        self._parameter_sources: list[dict[str, str]] = []

    @property
    def backend_version(self) -> str:
        try:
            return f"phasepy/{version('phasepy')}"
        except PackageNotFoundError:
            return "phasepy/unavailable"

    def _load(self) -> tuple[Any, Any]:
        if self._phasepy is not None and self._equilibrium is not None:
            return self._phasepy, self._equilibrium
        try:
            self._phasepy = importlib.import_module("phasepy")
            self._equilibrium = importlib.import_module("phasepy.equilibrium")
        except (ImportError, OSError) as exc:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                "The selected Phasepy backend is not installed in this Python runtime.",
                "Use Python 3.11 or 3.12 and install the optional phase-engines dependencies.",
                {"dependency": self.dependency, "cause": type(exc).__name__},
            ) from None
        return self._phasepy, self._equilibrium

    @staticmethod
    def _require_pressure(request: TaskManifest) -> float:
        if request.conditions.pressure_kPa is None:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "Pressure is required for this calculation.",
                "Provide pressure in kPa.",
            )
        return request.conditions.pressure_kPa

    @staticmethod
    def _require_temperature(request: TaskManifest) -> float:
        if request.conditions.temperature_K is None:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "Temperature is required for this calculation.",
                "Provide temperature in K.",
            )
        return request.conditions.temperature_K

    @staticmethod
    def _require_composition(request: TaskManifest, field: str) -> list[float]:
        values = getattr(request.conditions, field)
        if values is None:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                f"{field} is required for this calculation.",
                f"Provide {len(request.components)} normalized mole fractions.",
            )
        if len(values) != len(request.components):
            raise ThermoEquiError(
                FailureType.SEMANTIC_FAILURE,
                f"{field} length does not match the component count.",
                "Provide one mole fraction per component in component order.",
            )
        return cast(list[float], values)

    @staticmethod
    def _require_binary(request: TaskManifest) -> None:
        if len(request.components) != 2:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                "Phasepy curve and azeotrope endpoints currently support binary mixtures only.",
                "Provide two components or request a point/flash calculation.",
            )

    def _build_model(self, request: TaskManifest) -> tuple[Any, Any]:
        phasepy, _ = self._load()
        identifiers = [component.cas_number or component.component_id for component in request.components]
        try:
            constants, _ = ChemicalConstantsPackage.from_IDs(identifiers)
        except (ValueError, LookupError, TypeError):
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "The reviewed pure-property database could not resolve every Phasepy component.",
                "Provide canonical names or CAS numbers.",
                {"components": identifiers},
            ) from None
        inapplicable = ThermoPengRobinsonBackend.inapplicable_components(constants)
        if inapplicable:
            raise ThermoEquiError(
                FailureType.PARAMETER_OUT_OF_DOMAIN,
                "The Phasepy Peng-Robinson adapter is limited to hydrocarbons and reviewed light gases.",
                "Select a reviewed association or activity-coefficient model.",
                {"inapplicable_components": inapplicable},
            )
        if len(constants.CASs) < 2:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                "The Phasepy mixture adapter requires at least two components.",
                "Provide a molecular mixture.",
            )
        missing_pairs = [
            [constants.CASs[i], constants.CASs[j]]
            for i in range(len(constants.CASs))
            for j in range(i + 1, len(constants.CASs))
            if not IPDB.has_ip_specific(self.parameter_table, [constants.CASs[i], constants.CASs[j]], "kij")
        ]
        if missing_pairs:
            raise ThermoEquiError(
                FailureType.MISSING_PARAMETERS,
                "Phasepy Peng-Robinson binary interaction parameters are missing.",
                "Import reviewed kij values or select another applicable model.",
                {"parameter_table": self.parameter_table, "missing_pairs": missing_pairs},
            )
        kijs = IPDB.get_ip_asymmetric_matrix(self.parameter_table, constants.CASs, "kij")
        components = [
            phasepy.component(
                name=str(name),
                Tc=float(tc),
                Pc=float(pc) / 1e5,
                Zc=float(zc),
                Vc=float(vc) * 1e6,
                w=float(omega),
                Mw=float(mw),
            )
            for name, tc, pc, zc, vc, omega, mw in zip(
                constants.names,
                constants.Tcs,
                constants.Pcs,
                constants.Zcs,
                constants.Vcs,
                constants.omegas,
                constants.MWs,
                strict=True,
            )
        ]
        mixture = phasepy.mixture(components[0], components[1])
        for component in components[2:]:
            mixture.add_component(component)
        mixture.kij_cubic(np.asarray(kijs, dtype=float))
        model = phasepy.preos(mixture, mixrule="qmr")
        snapshot = {
            "component_order": list(constants.CASs),
            "kij": kijs,
            "parameter_form": "Peng-Robinson kij",
            "source_table": self.parameter_table,
            "phasepy_version": self.backend_version,
        }
        digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self._parameter_set_id = f"phasepy-chemsep-pr:{digest}"
        self._parameter_sources = [
            {
                "component": component.name,
                "property": "Pure-component critical constants",
                "source_title": "CalebBell/thermo",
                "source_identifier": "https://github.com/CalebBell/thermo",
            }
            for component in request.components
        ] + [
            {
                "component": " / ".join(component.name for component in request.components),
                "property": "Peng-Robinson binary interaction parameter kij",
                "source_title": self.parameter_table,
                "source_identifier": (
                    "https://github.com/CalebBell/thermo/tree/master/thermo/Interaction%20Parameters/ChemSep"
                ),
                "parameter_set_id": self._parameter_set_id,
                "parameter_values": json.dumps(kijs),
            }
        ]
        return model, constants

    def parameter_sources(self, request: TaskManifest) -> list[dict[str, str]]:
        if not self._parameter_sources:
            self._build_model(request)
        return self._parameter_sources

    @staticmethod
    def _wilson_terms(constants: Any, temperature: float) -> NDArray[np.float64]:
        return np.asarray(
            [
                float(pc) / 1e5 * exp(5.373 * (1.0 + float(omega)) * (1.0 - float(tc) / temperature))
                for pc, tc, omega in zip(constants.Pcs, constants.Tcs, constants.omegas, strict=True)
            ]
        )

    @classmethod
    def _temperature_guess(
        cls,
        constants: Any,
        composition: NDArray[np.float64],
        pressure_bar: float,
        dew: bool,
    ) -> float:
        def objective(temperature: float) -> float:
            terms = cls._wilson_terms(constants, temperature)
            estimate = 1.0 / float(np.sum(composition / terms)) if dew else float(np.sum(composition * terms))
            return estimate - pressure_bar

        lower = max(30.0, min(float(value) for value in constants.Tcs) * 0.25)
        upper = max(float(value) for value in constants.Tcs) * 1.5
        try:
            return float(brentq(objective, lower, upper))
        except ValueError:
            return float(np.dot(composition, np.asarray(constants.Tbs, dtype=float)))

    @staticmethod
    def _residual(
        model: Any,
        temperature: float,
        pressure_bar: float,
        liquid: NDArray[np.float64],
        vapor: NDArray[np.float64],
    ) -> float:
        log_phi_l = np.asarray(model.logfugef(liquid, temperature, pressure_bar, "L")[0], dtype=float)
        log_phi_v = np.asarray(model.logfugef(vapor, temperature, pressure_bar, "V")[0], dtype=float)
        active = (liquid > 1e-14) & (vapor > 1e-14)
        if not np.any(active):
            return 0.0
        values = np.log(liquid[active]) + log_phi_l[active] - np.log(vapor[active]) - log_phi_v[active]
        residual = float(np.max(np.abs(values)))
        if not isfinite(residual):
            raise ThermoEquiError(
                FailureType.PHYSICAL_VALIDATION_FAILURE,
                "Phasepy returned non-finite fugacity residuals.",
                "Do not use the result; review model applicability and conditions.",
            )
        return residual

    def _point(self, model: Any, output: Any) -> EquilibriumPoint:
        liquid = np.asarray(output.X, dtype=float)
        vapor = np.asarray(output.Y, dtype=float)
        return EquilibriumPoint(
            temperature_K=float(output.T),
            pressure_kPa=float(output.P) * 100.0,
            liquid_composition=liquid.tolist(),
            vapor_composition=vapor.tolist(),
            equilibrium_residual=self._residual(model, float(output.T), float(output.P), liquid, vapor),
        )

    def _boundary_result(self, request: TaskManifest, model: Any, output: Any) -> CalculationResult:
        point = self._point(model, output)
        residual = point.equilibrium_residual
        return CalculationResult(
            task_id=request.task_id,
            calculation_type=request.calculation_type,
            input_snapshot=request.model_dump(mode="json"),
            model_name=self.model_name,
            parameter_set_id=self._parameter_set_id,
            points=[point],
            temperature_K=point.temperature_K,
            pressure_kPa=point.pressure_kPa,
            phase_state="two_phase",
            converged=abs(float(output.error)) <= 1e-6 and residual <= 2e-6,
            residual=residual,
            iterations=int(output.iter),
            warnings=["ChemSep PR parameter applicability requires engineering review."],
            backend_version=self.backend_version,
            solver_name="phasepy.preos / phasepy.equilibrium",
        )

    def bubble_point(self, request: TaskManifest) -> CalculationResult:
        pressure_bar = self._require_pressure(request) / 100.0
        liquid = np.asarray(self._require_composition(request, "liquid_composition"), dtype=float)
        model, constants = self._build_model(request)
        _, equilibrium = self._load()
        temperature_guess = self._temperature_guess(constants, liquid, pressure_bar, dew=False)
        terms = self._wilson_terms(constants, temperature_guess)
        vapor_guess = liquid * terms / pressure_bar
        vapor_guess /= vapor_guess.sum()
        try:
            output = equilibrium.bubbleTy(
                vapor_guess,
                temperature_guess,
                liquid,
                pressure_bar,
                model,
                full_output=True,
            )
        except (ValueError, RuntimeError, ZeroDivisionError, OverflowError):
            raise self._nonconvergence("bubble point") from None
        return self._boundary_result(request, model, output)

    def dew_point(self, request: TaskManifest) -> CalculationResult:
        pressure_bar = self._require_pressure(request) / 100.0
        vapor = np.asarray(self._require_composition(request, "vapor_composition"), dtype=float)
        model, constants = self._build_model(request)
        _, equilibrium = self._load()
        temperature_guess = self._temperature_guess(constants, vapor, pressure_bar, dew=True)
        terms = self._wilson_terms(constants, temperature_guess)
        liquid_guess = vapor * pressure_bar / terms
        liquid_guess /= liquid_guess.sum()
        try:
            output = equilibrium.dewTx(
                liquid_guess,
                temperature_guess,
                vapor,
                pressure_bar,
                model,
                full_output=True,
            )
        except (ValueError, RuntimeError, ZeroDivisionError, OverflowError):
            raise self._nonconvergence("dew point") from None
        return self._boundary_result(request, model, output)

    def _bubble_at_temperature(
        self,
        request: TaskManifest,
        model: Any,
        constants: Any,
        liquid: NDArray[np.float64],
        temperature: float,
    ) -> Any:
        _, equilibrium = self._load()
        terms = self._wilson_terms(constants, temperature)
        pressure_guess = max(float(np.sum(liquid * terms)), 1e-8)
        vapor_guess = liquid * terms / pressure_guess
        vapor_guess /= vapor_guess.sum()
        try:
            return equilibrium.bubblePy(
                vapor_guess,
                pressure_guess,
                liquid,
                temperature,
                model,
                full_output=True,
            )
        except (ValueError, RuntimeError, ZeroDivisionError, OverflowError):
            raise self._nonconvergence("isothermal bubble point") from None

    @staticmethod
    def _nonconvergence(operation: str) -> ThermoEquiError:
        return ThermoEquiError(
            FailureType.NUMERICAL_NONCONVERGENCE,
            f"Phasepy did not converge for the requested {operation}.",
            "Review conditions, initial state, and Peng-Robinson applicability.",
        )

    def isobaric_vle(self, request: TaskManifest) -> CalculationResult:
        self._require_binary(request)
        pressure = self._require_pressure(request)
        points: list[EquilibriumPoint] = []
        iterations = 0
        for fraction in np.linspace(0.0, 1.0, request.points):
            point_request = request.model_copy(
                update={
                    "calculation_type": "bubble_point",
                    "conditions": request.conditions.model_copy(
                        update={"liquid_composition": [float(fraction), float(1.0 - fraction)]}
                    ),
                }
            )
            result = self.bubble_point(point_request)
            points.extend(result.points)
            iterations += result.iterations
        return self._curve_result(request, points, iterations, pressure_kpa=pressure)

    def isothermal_vle(self, request: TaskManifest) -> CalculationResult:
        self._require_binary(request)
        temperature = self._require_temperature(request)
        model, constants = self._build_model(request)
        points: list[EquilibriumPoint] = []
        iterations = 0
        for fraction in np.linspace(0.0, 1.0, request.points):
            output = self._bubble_at_temperature(
                request,
                model,
                constants,
                np.asarray([fraction, 1.0 - fraction], dtype=float),
                temperature,
            )
            points.append(self._point(model, output))
            iterations += int(output.iter)
        return self._curve_result(request, points, iterations, temperature=temperature)

    def _curve_result(
        self,
        request: TaskManifest,
        points: list[EquilibriumPoint],
        iterations: int,
        *,
        temperature: float | None = None,
        pressure_kpa: float | None = None,
    ) -> CalculationResult:
        residual = max(point.equilibrium_residual for point in points)
        return CalculationResult(
            task_id=request.task_id,
            calculation_type=request.calculation_type,
            input_snapshot=request.model_dump(mode="json"),
            model_name=self.model_name,
            parameter_set_id=self._parameter_set_id,
            points=points,
            temperature_K=temperature,
            pressure_kPa=pressure_kpa,
            phase_state="curve",
            converged=residual <= 2e-6,
            residual=residual,
            iterations=iterations,
            warnings=["ChemSep PR parameter applicability requires engineering review."],
            backend_version=self.backend_version,
            solver_name="phasepy.preos / phasepy.equilibrium",
        )

    def tp_flash(self, request: TaskManifest) -> CalculationResult:
        temperature = self._require_temperature(request)
        pressure_kpa = self._require_pressure(request)
        pressure_bar = pressure_kpa / 100.0
        feed = np.asarray(self._require_composition(request, "feed_composition"), dtype=float)
        model, constants = self._build_model(request)
        _, equilibrium = self._load()
        k_values = self._wilson_terms(constants, temperature) / pressure_bar
        f0 = float(np.sum(feed * (k_values - 1.0)))
        f1 = float(np.sum(feed * (k_values - 1.0) / k_values))
        if f0 <= 0.0 or f1 >= 0.0:
            raise ThermoEquiError(
                FailureType.PHASE_INSTABILITY,
                "Phasepy flash initialization indicates a single phase; "
                "this adapter does not infer it from Wilson K-values.",
                "Use the Clapeyron or CalebBell/thermo backend for a stability-aware single-phase flash.",
            )
        beta_guess = float(brentq(lambda beta: np.sum(feed * (k_values - 1.0) / (1 + beta * (k_values - 1))), 0, 1))
        liquid_guess = feed / (1.0 + beta_guess * (k_values - 1.0))
        vapor_guess = k_values * liquid_guess
        liquid_guess /= liquid_guess.sum()
        vapor_guess /= vapor_guess.sum()
        try:
            output = equilibrium.flash(
                liquid_guess,
                vapor_guess,
                "LV",
                feed,
                temperature,
                pressure_bar,
                model,
                K_tol=1e-12,
                full_output=True,
            )
        except (ValueError, RuntimeError, ZeroDivisionError, OverflowError):
            raise self._nonconvergence("TP flash") from None
        beta = float(output.beta)
        if not isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise self._nonconvergence("TP flash")
        liquid = np.asarray(output.X, dtype=float)
        vapor = np.asarray(output.Y, dtype=float)
        residual = self._residual(model, temperature, pressure_bar, liquid, vapor)
        return CalculationResult(
            task_id=request.task_id,
            calculation_type=request.calculation_type,
            input_snapshot=request.model_dump(mode="json"),
            model_name=self.model_name,
            parameter_set_id=self._parameter_set_id,
            phases=[
                PhaseResult(phase="liquid", fraction=1.0 - beta, composition=liquid.tolist()),
                PhaseResult(phase="vapor", fraction=beta, composition=vapor.tolist()),
            ],
            temperature_K=temperature,
            pressure_kPa=pressure_kpa,
            vapor_fraction=beta,
            phase_state="two_phase",
            converged=abs(float(output.error)) <= 1e-6 and residual <= 2e-6,
            residual=residual,
            iterations=int(output.iter),
            warnings=[
                "ChemSep PR parameter applicability requires engineering review.",
                "Phasepy TP flash does not expose an independent tangent-plane stability report.",
            ],
            backend_version=self.backend_version,
            solver_name="phasepy.equilibrium.flash / preos",
        )

    def phase_stability(self, request: TaskManifest) -> CalculationResult:
        raise ThermoEquiError(
            FailureType.UNSUPPORTED_MODEL,
            "The Phasepy adapter does not expose a validated phase-stability result in this release.",
            "Use a backend with an implemented stability contract.",
        )

    def azeotrope(self, request: TaskManifest) -> CalculationResult:
        curve = self.isobaric_vle(request)
        candidates = [
            point
            for point in curve.points[1:-1]
            if max(
                abs(liquid - vapor)
                for liquid, vapor in zip(point.liquid_composition, point.vapor_composition, strict=True)
            )
            <= 1e-3
        ]
        message = (
            "No internal azeotrope candidate met |x-y| <= 1e-3."
            if not candidates
            else "Candidate points require local refinement before engineering use."
        )
        return curve.model_copy(update={"points": candidates, "warnings": [*curve.warnings, message]})

    def lle(self, request: TaskManifest) -> CalculationResult:
        raise ThermoEquiError(
            FailureType.MISSING_PARAMETERS,
            "Phasepy LLE is not enabled without an evidence-bearing NRTL or UNIQUAC parameter set.",
            "Import reviewed activity-coefficient parameters before enabling LLE.",
        )

    def infinite_dilution_activity(self, request: TaskManifest) -> CalculationResult:
        raise ThermoEquiError(
            FailureType.UNSUPPORTED_MODEL,
            "Phasepy does not provide infinite-dilution activity coefficients through this adapter.",
            "Use the PGSSI backend for gamma-infinity predictions.",
        )

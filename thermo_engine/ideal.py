"""Tested internal Ideal/Raoult backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from schemas.domain import (
    CalculationResult,
    EquilibriumPoint,
    FailureType,
    PhaseResult,
    TaskManifest,
)
from thermo_engine.errors import ThermoEquiError
from thermo_engine.properties import PureComponent, component_sources, resolve_component


@dataclass(frozen=True)
class SolverTolerances:
    root: float = 1e-9
    composition: float = 1e-9
    equilibrium: float = 1e-7


class IdealRaoultBackend:
    """Deterministic ideal liquid/ideal gas backend using Raoult's law."""

    version = "internal-ideal-raoult/0.1.0"

    def __init__(self, tolerances: SolverTolerances | None = None) -> None:
        self.tolerances = tolerances or SolverTolerances()

    def parameter_sources(self, request: TaskManifest) -> list[dict[str, str]]:
        return component_sources(self._components(request))

    @staticmethod
    def _components(request: TaskManifest) -> list[PureComponent]:
        return [resolve_component(component) for component in request.components]

    @staticmethod
    def _require_pressure(request: TaskManifest) -> float:
        pressure = request.conditions.pressure_kPa
        if pressure is None:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "Pressure is required for this calculation.",
                "Provide a positive pressure in kPa.",
            )
        return pressure

    @staticmethod
    def _require_temperature(request: TaskManifest) -> float:
        temperature = request.conditions.temperature_K
        if temperature is None:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "Temperature is required for this calculation.",
                "Provide a positive temperature in K.",
            )
        return temperature

    @staticmethod
    def _require_composition(request: TaskManifest, field: str, expected: int) -> NDArray[np.float64]:
        values = getattr(request.conditions, field)
        if values is None:
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                f"{field} is required for this calculation.",
                f"Provide {expected} normalized mole fractions.",
            )
        if len(values) != expected:
            raise ThermoEquiError(
                FailureType.SEMANTIC_FAILURE,
                f"{field} length does not match the component count.",
                "Provide one mole fraction per component in component order.",
            )
        return np.asarray(values, dtype=float)

    @staticmethod
    def _temperature_bounds(components: list[PureComponent]) -> tuple[float, float]:
        minimum = min(cor.minimum_temperature_K for c in components for cor in c.correlations)
        maximum = max(cor.maximum_temperature_K for c in components for cor in c.correlations)
        return max(1.0, minimum - 80.0), maximum + 80.0

    @staticmethod
    def _pressures(components: list[PureComponent], temperature_K: float) -> tuple[NDArray[np.float64], list[str]]:
        values: list[float] = []
        warnings: list[str] = []
        for component in components:
            pressure, in_range, correlation = component.vapor_pressure_kPa(temperature_K)
            values.append(pressure)
            if not in_range:
                warnings.append(
                    f"{component.identity.name} Antoine correlation extrapolated at "
                    f"{temperature_K:.3f} K outside "
                    f"{correlation.minimum_temperature_K}-{correlation.maximum_temperature_K} K."
                )
        return np.asarray(values), warnings

    def _result(
        self,
        request: TaskManifest,
        *,
        points: list[EquilibriumPoint],
        temperature_K: float | None,
        pressure_kPa: float | None,
        residual: float,
        iterations: int,
        warnings: list[str],
        phases: list[PhaseResult] | None = None,
        vapor_fraction: float | None = None,
        phase_state: str = "curve",
    ) -> CalculationResult:
        return CalculationResult(
            task_id=request.task_id,
            calculation_type=request.calculation_type,
            input_snapshot=request.model_dump(mode="json"),
            model_name="Ideal/Raoult",
            parameter_set_id=None,
            points=points,
            phases=phases or [],
            temperature_K=temperature_K,
            pressure_kPa=pressure_kPa,
            vapor_fraction=vapor_fraction,
            phase_state=phase_state,
            converged=True,
            residual=float(residual),
            iterations=iterations,
            warnings=list(dict.fromkeys(warnings)),
            backend_version=self.version,
            solver_name="scipy.optimize.brentq / analytical Raoult / Rachford-Rice",
        )

    def bubble_point(self, request: TaskManifest) -> CalculationResult:
        components = self._components(request)
        pressure = self._require_pressure(request)
        liquid = self._require_composition(request, "liquid_composition", len(components))
        low, high = self._temperature_bounds(components)

        def objective(temperature: float) -> float:
            psat, _ = self._pressures(components, temperature)
            return float(np.dot(liquid, psat) - pressure)

        temperature, iterations = self._root(objective, low, high)
        psat, warnings = self._pressures(components, temperature)
        vapor = liquid * psat / pressure
        vapor /= vapor.sum()
        residual = float(np.max(np.abs(vapor * pressure - liquid * psat)) / pressure)
        point = EquilibriumPoint(
            temperature_K=temperature,
            pressure_kPa=pressure,
            liquid_composition=liquid.tolist(),
            vapor_composition=vapor.tolist(),
            equilibrium_residual=residual,
        )
        return self._result(
            request,
            points=[point],
            temperature_K=temperature,
            pressure_kPa=pressure,
            residual=residual,
            iterations=iterations,
            warnings=warnings,
            phase_state="two_phase",
        )

    def dew_point(self, request: TaskManifest) -> CalculationResult:
        components = self._components(request)
        pressure = self._require_pressure(request)
        vapor = self._require_composition(request, "vapor_composition", len(components))
        low, high = self._temperature_bounds(components)

        def objective(temperature: float) -> float:
            psat, _ = self._pressures(components, temperature)
            return float(np.sum(vapor * pressure / psat) - 1.0)

        temperature, iterations = self._root(objective, low, high)
        psat, warnings = self._pressures(components, temperature)
        liquid = vapor * pressure / psat
        liquid /= liquid.sum()
        residual = float(np.max(np.abs(vapor * pressure - liquid * psat)) / pressure)
        point = EquilibriumPoint(
            temperature_K=temperature,
            pressure_kPa=pressure,
            liquid_composition=liquid.tolist(),
            vapor_composition=vapor.tolist(),
            equilibrium_residual=residual,
        )
        return self._result(
            request,
            points=[point],
            temperature_K=temperature,
            pressure_kPa=pressure,
            residual=residual,
            iterations=iterations,
            warnings=warnings,
            phase_state="two_phase",
        )

    def isobaric_vle(self, request: TaskManifest) -> CalculationResult:
        if len(request.components) != 2:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                "The 0.1 isobaric curve endpoint supports binary mixtures only.",
                "Provide exactly two components.",
            )
        all_points: list[EquilibriumPoint] = []
        warnings: list[str] = []
        iterations = 0
        for fraction in np.linspace(0.0, 1.0, request.points):
            conditions = request.conditions.model_copy(
                update={"liquid_composition": [float(fraction), float(1.0 - fraction)]}
            )
            point_request = request.model_copy(update={"conditions": conditions})
            point_result = self.bubble_point(point_request)
            all_points.extend(point_result.points)
            warnings.extend(point_result.warnings)
            iterations += point_result.iterations
        residual = max(point.equilibrium_residual for point in all_points)
        return self._result(
            request,
            points=all_points,
            temperature_K=None,
            pressure_kPa=self._require_pressure(request),
            residual=residual,
            iterations=iterations,
            warnings=warnings,
        )

    def isothermal_vle(self, request: TaskManifest) -> CalculationResult:
        if len(request.components) != 2:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                "The 0.1 isothermal curve endpoint supports binary mixtures only.",
                "Provide exactly two components.",
            )
        components = self._components(request)
        temperature = self._require_temperature(request)
        psat, warnings = self._pressures(components, temperature)
        all_points: list[EquilibriumPoint] = []
        for fraction in np.linspace(0.0, 1.0, request.points):
            liquid = np.asarray([fraction, 1.0 - fraction], dtype=float)
            pressure = float(np.dot(liquid, psat))
            vapor = liquid * psat / pressure
            residual = float(np.max(np.abs(vapor * pressure - liquid * psat)) / pressure)
            all_points.append(
                EquilibriumPoint(
                    temperature_K=temperature,
                    pressure_kPa=pressure,
                    liquid_composition=liquid.tolist(),
                    vapor_composition=vapor.tolist(),
                    equilibrium_residual=residual,
                )
            )
        return self._result(
            request,
            points=all_points,
            temperature_K=temperature,
            pressure_kPa=None,
            residual=max(point.equilibrium_residual for point in all_points),
            iterations=0,
            warnings=warnings,
        )

    def tp_flash(self, request: TaskManifest) -> CalculationResult:
        components = self._components(request)
        temperature = self._require_temperature(request)
        pressure = self._require_pressure(request)
        feed = self._require_composition(request, "feed_composition", len(components))
        psat, warnings = self._pressures(components, temperature)
        k_values = psat / pressure

        def rachford_rice(beta: float) -> float:
            return float(np.sum(feed * (k_values - 1.0) / (1.0 + beta * (k_values - 1.0))))

        f0 = rachford_rice(0.0)
        f1 = rachford_rice(1.0)
        iterations = 0
        if f0 <= 0.0:
            beta = 0.0
            liquid = feed.copy()
            vapor = k_values * liquid
            vapor /= vapor.sum()
            state = "liquid"
        elif f1 >= 0.0:
            beta = 1.0
            vapor = feed.copy()
            liquid = vapor / k_values
            liquid /= liquid.sum()
            state = "vapor"
        else:
            beta, info = brentq(
                rachford_rice,
                0.0,
                1.0,
                xtol=self.tolerances.root,
                full_output=True,
            )
            iterations = info.iterations
            liquid = feed / (1.0 + beta * (k_values - 1.0))
            vapor = k_values * liquid
            liquid /= liquid.sum()
            vapor /= vapor.sum()
            state = "two_phase"
        material = (1.0 - beta) * liquid + beta * vapor
        residual = max(
            float(np.max(np.abs(material - feed))),
            abs(rachford_rice(beta)) if state == "two_phase" else 0.0,
        )
        phases = [
            PhaseResult(phase="liquid", fraction=1.0 - beta, composition=liquid.tolist()),
            PhaseResult(phase="vapor", fraction=beta, composition=vapor.tolist()),
        ]
        return self._result(
            request,
            points=[],
            phases=phases,
            temperature_K=temperature,
            pressure_kPa=pressure,
            vapor_fraction=float(beta),
            residual=residual,
            iterations=iterations,
            warnings=warnings,
            phase_state=state,
        )

    def phase_stability(self, request: TaskManifest) -> CalculationResult:
        result = self.tp_flash(request)
        message = (
            "Ideal K-value phase classification only; tangent-plane stability is unavailable "
            "for the 0.1 internal backend."
        )
        return result.model_copy(update={"warnings": [*result.warnings, message]})

    def azeotrope(self, request: TaskManifest) -> CalculationResult:
        curve = self.isobaric_vle(request)
        candidates = [
            point
            for point in curve.points[1:-1]
            if max(abs(a - b) for a, b in zip(point.liquid_composition, point.vapor_composition, strict=True)) <= 1e-3
        ]
        warning = (
            "No internal azeotrope candidate met |x-y| <= 1e-3."
            if not candidates
            else "Candidate points require refinement before engineering use."
        )
        return curve.model_copy(update={"points": candidates, "warnings": [*curve.warnings, warning]})

    def lle(self, request: TaskManifest) -> CalculationResult:
        raise ThermoEquiError(
            FailureType.UNSUPPORTED_MODEL,
            "Ideal/Raoult cannot represent liquid-liquid phase splitting.",
            "Use the LLE contract with an evidence-backed NRTL or UNIQUAC backend when available.",
        )

    def infinite_dilution_activity(self, request: TaskManifest) -> CalculationResult:
        raise ThermoEquiError(
            FailureType.UNSUPPORTED_MODEL,
            "Ideal/Raoult does not provide infinite-dilution activity coefficients.",
            "Use the PGSSI backend for gamma-infinity predictions.",
        )

    def _root(self, objective: Callable[[float], float], low: float, high: float) -> tuple[float, int]:
        try:
            root, info = brentq(objective, low, high, xtol=self.tolerances.root, full_output=True)
        except (ValueError, RuntimeError) as error:
            raise ThermoEquiError(
                FailureType.NUMERICAL_NONCONVERGENCE,
                "Could not bracket or converge the phase-equilibrium root.",
                "Check conditions, property ranges, and model applicability.",
                {"lower_K": low, "upper_K": high, "solver_error": str(error)},
            ) from error
        if not info.converged or not isfinite(root):
            raise ThermoEquiError(
                FailureType.NUMERICAL_NONCONVERGENCE,
                "The phase-equilibrium solver did not converge.",
                "Review the bracket and property correlations.",
            )
        return float(root), int(info.iterations)

"""Shared cubic equation-of-state backend base for the ``thermo`` adapters."""

from __future__ import annotations

from math import isfinite, log
from typing import Any, cast

import numpy as np
from thermo import ChemicalConstantsPackage

from schemas.domain import (
    CalculationResult,
    EquilibriumPoint,
    FailureType,
    PhaseResult,
    TaskManifest,
)
from thermo_engine.errors import ThermoEquiError

ALLOWLISTED_LIGHT_GAS_CAS = frozenset(
    {
        "124-38-9",  # carbon dioxide
        "630-08-0",  # carbon monoxide
        "1333-74-0",  # hydrogen
        "7727-37-9",  # nitrogen
        "7782-44-7",  # oxygen
        "7440-37-1",  # argon
    }
)


class CubicEosBackend:
    """Common PR/SRK behavior on top of ``thermo.FlashVL`` and ``CEOS*`` phases."""

    version = ""
    model_name = ""
    solver_name = ""
    eos_class: Any = None
    allowlisted_light_gas_cas = ALLOWLISTED_LIGHT_GAS_CAS

    def __init__(self) -> None:
        self._parameter_set_id: str | None = None
        self._parameter_sources: list[dict[str, str]] = []

    @classmethod
    def _inapplicable_components(cls, constants: Any) -> list[str]:
        inapplicable: list[str] = []
        for name, cas_number, atoms in zip(
            constants.names,
            constants.CASs,
            constants.atomss,
            strict=True,
        ):
            elements = set(atoms)
            is_hydrocarbon = "C" in elements and elements <= {"C", "H"}
            if not is_hydrocarbon and cas_number not in cls.allowlisted_light_gas_cas:
                inapplicable.append(str(name).title())
        return inapplicable

    @classmethod
    def inapplicable_components(cls, constants: Any) -> list[str]:
        """Return components outside the reviewed cubic EOS applicability domain."""
        return cls._inapplicable_components(constants)

    @classmethod
    def supports_system(cls, request: TaskManifest) -> bool:
        identifiers = [component.cas_number or component.component_id for component in request.components]
        try:
            constants, _ = ChemicalConstantsPackage.from_IDs(identifiers)
        except (ValueError, LookupError, TypeError):
            return False
        return not cls._inapplicable_components(constants)

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

    def _flasher(self, request: TaskManifest) -> Any:
        raise NotImplementedError

    def _phase_fugacities(self, phase: Any) -> list[float]:
        raise NotImplementedError

    def _result_warnings(self) -> list[str]:
        raise NotImplementedError

    def _curve_warnings(self) -> list[str]:
        return []

    @staticmethod
    def _convergence(solution: Any) -> tuple[float, int]:
        convergence = solution.flash_convergence
        if not isinstance(convergence, dict) or "err" not in convergence or "iterations" not in convergence:
            raise ThermoEquiError(
                FailureType.NUMERICAL_NONCONVERGENCE,
                "The thermo flash solver returned no convergence diagnostics.",
                "Do not use this result; review the solver specification and conditions.",
            )
        residual = abs(float(convergence["err"]))
        iterations = int(convergence["iterations"])
        if not isfinite(residual):
            raise ThermoEquiError(
                FailureType.NUMERICAL_NONCONVERGENCE,
                "The thermo flash solver returned a non-finite residual.",
                "Review conditions, component data, and model applicability.",
            )
        return residual, iterations

    def _equilibrium_residual(self, solution: Any) -> float:
        """Independently check component fugacity equality for a returned phase pair."""
        liquid_phase = getattr(solution, "liquid0", None)
        gas_phase = getattr(solution, "gas", None)
        if liquid_phase is None or gas_phase is None:
            return 0.0
        liquid_fugacities = self._phase_fugacities(liquid_phase)
        vapor_fugacities = self._phase_fugacities(gas_phase)
        if len(liquid_fugacities) != len(vapor_fugacities):
            raise ThermoEquiError(
                FailureType.PHYSICAL_VALIDATION_FAILURE,
                "The returned phase fugacity vectors have incompatible sizes.",
                "Do not use this result; review the solver output.",
            )
        residuals: list[float] = []
        for liquid, vapor in zip(liquid_fugacities, vapor_fugacities, strict=True):
            liquid_value = float(liquid)
            vapor_value = float(vapor)
            if abs(liquid_value) <= 1e-30 and abs(vapor_value) <= 1e-30:
                continue
            if liquid_value <= 0.0 or vapor_value <= 0.0 or not isfinite(liquid_value) or not isfinite(vapor_value):
                raise ThermoEquiError(
                    FailureType.PHYSICAL_VALIDATION_FAILURE,
                    "The returned phase fugacities are non-positive or non-finite.",
                    "Do not use this result; review conditions, properties, and model applicability.",
                )
            residuals.append(abs(log(liquid_value / vapor_value)))
        return max(residuals, default=0.0)

    def _flash(self, request: TaskManifest, **specifications: object) -> Any:
        try:
            return self._flasher(request).flash(**specifications)
        except ThermoEquiError:
            raise
        except (ValueError, RuntimeError, ZeroDivisionError, OverflowError):
            raise ThermoEquiError(
                FailureType.NUMERICAL_NONCONVERGENCE,
                f"The thermo {self.model_name} solver did not converge.",
                "Review the phase specification, conditions, and parameter applicability.",
            ) from None

    @staticmethod
    def _phases(solution: Any) -> tuple[list[PhaseResult], float]:
        vapor_fraction = float(solution.VF)
        phases: list[PhaseResult] = []
        liquid_phase = getattr(solution, "liquid0", None)
        gas_phase = getattr(solution, "gas", None)
        if liquid_phase is not None:
            phases.append(
                PhaseResult(
                    phase="liquid",
                    fraction=1.0 - vapor_fraction,
                    composition=[float(value) for value in liquid_phase.zs],
                )
            )
        if gas_phase is not None:
            phases.append(
                PhaseResult(
                    phase="vapor",
                    fraction=vapor_fraction,
                    composition=[float(value) for value in gas_phase.zs],
                )
            )
        return phases, vapor_fraction

    def _point(self, solution: Any) -> EquilibriumPoint:
        liquid_phase = getattr(solution, "liquid0", None)
        gas_phase = getattr(solution, "gas", None)
        if liquid_phase is None or gas_phase is None:
            raise ThermoEquiError(
                FailureType.PHYSICAL_VALIDATION_FAILURE,
                "A requested phase-boundary calculation did not return both phases.",
                "Review the phase specification and model applicability.",
            )
        self._convergence(solution)
        return EquilibriumPoint(
            temperature_K=float(solution.T),
            pressure_kPa=float(solution.P) / 1000.0,
            liquid_composition=[float(value) for value in liquid_phase.zs],
            vapor_composition=[float(value) for value in gas_phase.zs],
            equilibrium_residual=self._equilibrium_residual(solution),
        )

    def _result(
        self,
        request: TaskManifest,
        solution: Any,
        *,
        points: list[EquilibriumPoint] | None = None,
        warnings: list[str] | None = None,
    ) -> CalculationResult:
        solver_residual, iterations = self._convergence(solution)
        equilibrium_residual = self._equilibrium_residual(solution)
        phases, vapor_fraction = self._phases(solution)
        phase_state = "liquid" if vapor_fraction <= 1e-12 else "vapor" if vapor_fraction >= 1.0 - 1e-12 else "two_phase"
        return CalculationResult(
            task_id=request.task_id,
            calculation_type=request.calculation_type,
            input_snapshot=request.model_dump(mode="json"),
            model_name=self.model_name,
            parameter_set_id=self._parameter_set_id,
            points=points or [],
            phases=phases,
            temperature_K=float(solution.T),
            pressure_kPa=float(solution.P) / 1000.0,
            vapor_fraction=vapor_fraction,
            phase_state=phase_state,
            converged=solver_residual <= 1e-6,
            residual=equilibrium_residual,
            iterations=iterations,
            warnings=[*self._result_warnings(), *(warnings or [])],
            backend_version=self.version,
            solver_name=self.solver_name,
        )

    def _curve_result(
        self,
        request: TaskManifest,
        points: list[EquilibriumPoint],
        *,
        temperature_K: float | None = None,
        pressure_kPa: float | None = None,
        converged: bool = True,
        iterations: int = 0,
    ) -> CalculationResult:
        return CalculationResult(
            task_id=request.task_id,
            calculation_type=request.calculation_type,
            input_snapshot=request.model_dump(mode="json"),
            model_name=self.model_name,
            parameter_set_id=self._parameter_set_id,
            points=points,
            temperature_K=temperature_K,
            pressure_kPa=pressure_kPa,
            phase_state="curve",
            converged=converged,
            residual=max(point.equilibrium_residual for point in points),
            iterations=iterations,
            warnings=[*self._result_warnings(), *self._curve_warnings()],
            backend_version=self.version,
            solver_name=self.solver_name,
        )

    def _pure_endpoint_point(
        self,
        request: TaskManifest,
        pure_index: int,
        *,
        temperature_K: float | None = None,
        pressure_kPa: float | None = None,
    ) -> EquilibriumPoint:
        if (temperature_K is None) == (pressure_kPa is None):
            raise ValueError("Exactly one of temperature_K or pressure_kPa is required for a pure endpoint.")
        composition = [0.0] * len(request.components)
        composition[pure_index] = 1.0
        specifications: dict[str, object] = {"VF": 0.0, "zs": composition}
        if temperature_K is not None:
            specifications["T"] = temperature_K
        else:
            specifications["P"] = (pressure_kPa or 0.0) * 1000.0
        return self._point(self._flash(request, **specifications))

    def bubble_point(self, request: TaskManifest) -> CalculationResult:
        pressure = self._require_pressure(request)
        liquid = self._require_composition(request, "liquid_composition")
        solution = self._flash(request, P=pressure * 1000.0, VF=0.0, zs=liquid)
        return self._result(request, solution, points=[self._point(solution)])

    def dew_point(self, request: TaskManifest) -> CalculationResult:
        pressure = self._require_pressure(request)
        vapor = self._require_composition(request, "vapor_composition")
        solution = self._flash(request, P=pressure * 1000.0, VF=1.0, zs=vapor)
        return self._result(request, solution, points=[self._point(solution)])

    def isobaric_vle(self, request: TaskManifest) -> CalculationResult:
        if len(request.components) != 2:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                "The curve endpoint currently supports binary mixtures only.",
                "Provide exactly two components or use a point/flash calculation.",
            )
        pressure = self._require_pressure(request)
        points: list[EquilibriumPoint] = []
        iterations = 0
        converged = True
        for index, fraction in enumerate(np.linspace(0.0, 1.0, request.points)):
            if index == 0 or index == request.points - 1:
                points.append(
                    self._pure_endpoint_point(
                        request,
                        1 if index == 0 else 0,
                        pressure_kPa=pressure,
                    )
                )
                continue
            solution = self._flash(
                request,
                P=pressure * 1000.0,
                VF=0.0,
                zs=[float(fraction), float(1.0 - fraction)],
            )
            points.append(self._point(solution))
            solver_residual, point_iterations = self._convergence(solution)
            converged = converged and solver_residual <= 1e-6
            iterations += point_iterations
        return self._curve_result(
            request,
            points,
            pressure_kPa=pressure,
            converged=converged,
            iterations=iterations,
        )

    def isothermal_vle(self, request: TaskManifest) -> CalculationResult:
        if len(request.components) != 2:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                "The curve endpoint currently supports binary mixtures only.",
                "Provide exactly two components or use a point/flash calculation.",
            )
        temperature = self._require_temperature(request)
        points: list[EquilibriumPoint] = []
        iterations = 0
        converged = True
        for index, fraction in enumerate(np.linspace(0.0, 1.0, request.points)):
            if index == 0 or index == request.points - 1:
                points.append(
                    self._pure_endpoint_point(
                        request,
                        1 if index == 0 else 0,
                        temperature_K=temperature,
                    )
                )
                continue
            solution = self._flash(
                request,
                T=temperature,
                VF=0.0,
                zs=[float(fraction), float(1.0 - fraction)],
            )
            points.append(self._point(solution))
            solver_residual, point_iterations = self._convergence(solution)
            converged = converged and solver_residual <= 1e-6
            iterations += point_iterations
        return self._curve_result(
            request,
            points,
            temperature_K=temperature,
            converged=converged,
            iterations=iterations,
        )

    def tp_flash(self, request: TaskManifest) -> CalculationResult:
        temperature = self._require_temperature(request)
        pressure = self._require_pressure(request)
        feed = self._require_composition(request, "feed_composition")
        solution = self._flash(request, T=temperature, P=pressure * 1000.0, zs=feed)
        return self._result(request, solution)

    def phase_stability(self, request: TaskManifest) -> CalculationResult:
        result = self.tp_flash(request)
        return result.model_copy(
            update={
                "warnings": [
                    *result.warnings,
                    "Phase stability was evaluated by thermo.FlashVL during phase identification.",
                ]
            }
        )

    def azeotrope(self, request: TaskManifest) -> CalculationResult:
        curve = self.isobaric_vle(request)
        candidates = [
            point
            for point in curve.points[1:-1]
            if max(
                abs(liquid - vapor)
                for liquid, vapor in zip(
                    point.liquid_composition,
                    point.vapor_composition,
                    strict=True,
                )
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
            FailureType.UNSUPPORTED_MODEL,
            f"The {self.model_name} adapter is not enabled for LLE in this release.",
            "Use a validated activity-coefficient LLE backend with evidence-bearing parameters.",
        )

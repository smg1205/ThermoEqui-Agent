"""CalebBell/thermo Soave-Redlich-Kwong adapter for binary VLE and flash."""

from __future__ import annotations

import json
from importlib.metadata import version
from math import isfinite, log
from typing import Any

import numpy as np
from thermo import SRKMIX, CEOSGas, CEOSLiquid, ChemicalConstantsPackage, FlashVL
from thermo.eos import SRK as PureSRK

from schemas.domain import (
    CalculationResult,
    EquilibriumPoint,
    FailureType,
    ParameterSet,
    PhaseResult,
    TaskManifest,
)
from thermo_engine.errors import ThermoEquiError
from thermo_engine.parameters import is_srk_kij_parameter_set, matching_parameter_set
from thermo_engine.thermo_backend import ThermoPengRobinsonBackend


class ThermoSrkBackend(ThermoPengRobinsonBackend):
    """Soave-Redlich-Kwong VLE and flash adapter executed by ``thermo``.

    The pilot adapter is binary-only and requires an explicit reviewed or
    user-attested ``ParameterSet`` carrying an SRK ``kij``.
    """

    version = f"thermo/{version('thermo')}"

    def __init__(self) -> None:
        super().__init__()
        self._parameter_set: ParameterSet | None = None

    def _find_parameter_set(self, request: TaskManifest) -> ParameterSet:
        if len(request.components) != 2:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                "The SRK pilot adapter is currently binary-only.",
                "Provide exactly two components or wait for reviewed multicomponent kij support.",
            )
        parameter_set = matching_parameter_set(request.parameters, request.components, "SRK")
        if parameter_set is None:
            missing_pairs = [[component.cas_number or component.component_id for component in request.components]]
            raise ThermoEquiError(
                FailureType.MISSING_PARAMETERS,
                "SRK binary interaction parameters are missing for this component set.",
                "Import a reviewed SRK kij parameter set or select another model.",
                {"model": "SRK", "parameter_form": "SRK kij", "missing_pairs": missing_pairs},
            )
        if not is_srk_kij_parameter_set(parameter_set):
            raise ThermoEquiError(
                FailureType.MISSING_PARAMETERS,
                "The SRK parameter set does not contain a complete kij entry.",
                "Provide a binary SRK parameter set with parameter_form 'SRK kij' and a 'kij' value.",
                {"model": "SRK", "parameter_set_id": parameter_set.parameter_set_id},
            )
        self._parameter_set = parameter_set
        self._parameter_set_id = parameter_set.parameter_set_id
        return parameter_set

    def _build_sources(
        self,
        request: TaskManifest,
        parameter_set: ParameterSet,
    ) -> list[dict[str, str]]:
        component_names = " / ".join(component.name for component in request.components)
        return [
            {
                "component": component.name,
                "property": "Pure-component constants and property correlations",
                "source_title": "CalebBell/thermo",
                "source_identifier": "https://github.com/CalebBell/thermo",
                "temperature_range_K": "model-dependent",
            }
            for component in request.components
        ] + [
            {
                "component": component_names,
                "component_order": json.dumps(parameter_set.component_order),
                "property": "SRK binary interaction parameter kij",
                "parameter_form": parameter_set.parameter_form,
                "parameter_values": json.dumps(parameter_set.parameters),
                "parameter_units": json.dumps(parameter_set.units),
                "parameter_set_id": parameter_set.parameter_set_id,
                "quality_level": parameter_set.quality_level,
                "source_title": parameter_set.source_title or parameter_set.source_type.replace("_", " "),
                "source_identifier": parameter_set.source_identifier or "user-supplied",
                "source_type": parameter_set.source_type,
                "temperature_range_K": str(parameter_set.temperature_range_K),
                "pressure_range_kPa": str(parameter_set.pressure_range_kPa),
            }
        ]

    def parameter_sources(self, request: TaskManifest) -> list[dict[str, str]]:
        if self._parameter_sources:
            return self._parameter_sources
        parameter_set = self._find_parameter_set(request)
        self._parameter_sources = self._build_sources(request, parameter_set)
        return self._parameter_sources

    def _flasher(self, request: TaskManifest) -> Any:
        identifiers = [component.cas_number or component.component_id for component in request.components]
        try:
            constants, properties = ChemicalConstantsPackage.from_IDs(identifiers)
        except (ValueError, LookupError, TypeError):
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "The thermo property database could not resolve every SRK component.",
                "Provide canonical names or CAS numbers supported by the reviewed property source.",
                {"components": identifiers},
            ) from None

        inapplicable_components = self._inapplicable_components(constants)
        if inapplicable_components:
            raise ThermoEquiError(
                FailureType.PARAMETER_OUT_OF_DOMAIN,
                "The current SRK adapter is limited to hydrocarbons and reviewed light gases.",
                "Choose a validated association/activity-coefficient model for this system.",
                {"inapplicable_components": inapplicable_components},
            )

        parameter_set = self._find_parameter_set(request)
        kij = float(parameter_set.parameters["kij"])
        kijs = np.zeros((2, 2), dtype=float)
        kijs[0, 1] = kijs[1, 0] = kij
        self._parameter_sources = self._build_sources(request, parameter_set)

        eos_kwargs = {
            "Pcs": np.asarray(constants.Pcs, dtype=float),
            "Tcs": np.asarray(constants.Tcs, dtype=float),
            "omegas": np.asarray(constants.omegas, dtype=float),
            "kijs": kijs,
        }
        gas = CEOSGas(
            SRKMIX,
            eos_kwargs=eos_kwargs,
            HeatCapacityGases=properties.HeatCapacityGases,
        )
        liquid = CEOSLiquid(
            SRKMIX,
            eos_kwargs=eos_kwargs,
            HeatCapacityGases=properties.HeatCapacityGases,
        )
        flasher = FlashVL(constants, properties, liquid=liquid, gas=gas)
        flasher.DEW_BUBBLE_NEWTON_XTOL = 1e-10
        flasher.DEW_BUBBLE_QUASI_NEWTON_XTOL = 1e-10
        return flasher

    def _warnings(self) -> list[str]:
        warnings = [
            "SRK is a pilot adapter; reviewed kij coverage and benchmark closure are pending.",
            "SRK kij parameter applicability requires engineering review.",
        ]
        if self._parameter_set is not None and self._parameter_set.source_type == "test_fixture":
            warnings.append("Parameter set is test_fixture and is not engineering evidence.")
        return warnings

    def _pure_endpoint_point(
        self,
        request: TaskManifest,
        pure_index: int,
        *,
        temperature: float | None = None,
        pressure_kpa: float | None = None,
    ) -> EquilibriumPoint:
        if (temperature is None) == (pressure_kpa is None):
            raise ValueError("Exactly one of temperature_K or pressure_kPa is required for a pure endpoint.")
        self._find_parameter_set(request)
        identifiers = [component.cas_number or component.component_id for component in request.components]
        constants, _ = ChemicalConstantsPackage.from_IDs(identifiers)
        inapplicable_components = self._inapplicable_components(constants)
        if inapplicable_components:
            raise ThermoEquiError(
                FailureType.PARAMETER_OUT_OF_DOMAIN,
                "The current SRK adapter is limited to hydrocarbons and reviewed light gases.",
                "Choose a validated association/activity-coefficient model for this system.",
                {"inapplicable_components": inapplicable_components},
            )
        pure = PureSRK(
            Tc=float(constants.Tcs[pure_index]),
            Pc=float(constants.Pcs[pure_index]),
            omega=float(constants.omegas[pure_index]),
            T=temperature or 300.0,
            P=(pressure_kpa * 1000.0) if pressure_kpa is not None else 1e5,
        )
        composition = [0.0] * len(identifiers)
        composition[pure_index] = 1.0
        if temperature is not None:
            pressure_pa = float(pure.Psat(temperature))
            point_temperature = temperature
            point_pressure_kpa = pressure_pa / 1000.0
        else:
            point_temperature = float(pure.Tsat(pressure_kpa * 1000.0))
            point_pressure_kpa = float(pressure_kpa or 0.0)
        return EquilibriumPoint(
            temperature_K=point_temperature,
            pressure_kPa=point_pressure_kpa,
            liquid_composition=composition,
            vapor_composition=composition,
            equilibrium_residual=0.0,
        )

    @staticmethod
    def _equilibrium_residual(solution: Any) -> float:
        """Independently check component fugacity equality for a returned phase pair."""
        liquid_phase = getattr(solution, "liquid0", None)
        gas_phase = getattr(solution, "gas", None)
        if liquid_phase is None or gas_phase is None:
            return 0.0
        liquid_fugacities = liquid_phase.fugacities_at_zs(np.asarray(liquid_phase.zs, dtype=float))
        vapor_fugacities = gas_phase.fugacities_at_zs(np.asarray(gas_phase.zs, dtype=float))
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

    @staticmethod
    def _point(solution: Any) -> EquilibriumPoint:
        liquid_phase = getattr(solution, "liquid0", None)
        gas_phase = getattr(solution, "gas", None)
        if liquid_phase is None or gas_phase is None:
            raise ThermoEquiError(
                FailureType.PHYSICAL_VALIDATION_FAILURE,
                "A requested phase-boundary calculation did not return both phases.",
                "Review the phase specification and model applicability.",
            )
        ThermoSrkBackend._convergence(solution)
        return EquilibriumPoint(
            temperature_K=float(solution.T),
            pressure_kPa=float(solution.P) / 1000.0,
            liquid_composition=[float(value) for value in liquid_phase.zs],
            vapor_composition=[float(value) for value in gas_phase.zs],
            equilibrium_residual=ThermoSrkBackend._equilibrium_residual(solution),
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
        phases: list[PhaseResult] = []
        vapor_fraction = float(solution.VF)
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
        phase_state = "liquid" if vapor_fraction <= 1e-12 else "vapor" if vapor_fraction >= 1.0 - 1e-12 else "two_phase"
        return CalculationResult(
            task_id=request.task_id,
            calculation_type=request.calculation_type,
            input_snapshot=request.model_dump(mode="json"),
            model_name="SRK",
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
            warnings=[*self._warnings(), *(warnings or [])],
            backend_version=self.version,
            solver_name="thermo.FlashVL / SRKMIX",
        )

    def _flash(self, request: TaskManifest, **specifications: object) -> Any:
        try:
            return self._flasher(request).flash(**specifications)
        except ThermoEquiError:
            raise
        except (ValueError, RuntimeError, ZeroDivisionError, OverflowError):
            raise ThermoEquiError(
                FailureType.NUMERICAL_NONCONVERGENCE,
                "The thermo SRK solver did not converge.",
                "Review the phase specification, conditions, and parameter applicability.",
            ) from None

    def isobaric_vle(self, request: TaskManifest) -> CalculationResult:
        if len(request.components) != 2:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                "The SRK curve endpoint currently supports binary mixtures only.",
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
                        pressure_kpa=pressure,
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
            pressure_kpa=pressure,
            converged=converged,
            iterations=iterations,
        )

    def isothermal_vle(self, request: TaskManifest) -> CalculationResult:
        if len(request.components) != 2:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                "The SRK curve endpoint currently supports binary mixtures only.",
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
                        temperature=temperature,
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
            temperature=temperature,
            converged=converged,
            iterations=iterations,
        )

    def _curve_result(
        self,
        request: TaskManifest,
        points: list[EquilibriumPoint],
        *,
        temperature: float | None = None,
        pressure_kpa: float | None = None,
        converged: bool = True,
        iterations: int = 0,
    ) -> CalculationResult:
        return CalculationResult(
            task_id=request.task_id,
            calculation_type=request.calculation_type,
            input_snapshot=request.model_dump(mode="json"),
            model_name="SRK",
            parameter_set_id=self._parameter_set_id,
            points=points,
            temperature_K=temperature,
            pressure_kPa=pressure_kpa,
            phase_state="curve",
            converged=converged,
            residual=max(point.equilibrium_residual for point in points),
            iterations=iterations,
            warnings=[
                *self._warnings(),
                "Pure curve endpoints use thermo.eos.SRK saturation values.",
            ],
            backend_version=self.version,
            solver_name="thermo.FlashVL / SRKMIX",
        )

    def lle(self, request: TaskManifest) -> CalculationResult:
        raise ThermoEquiError(
            FailureType.UNSUPPORTED_MODEL,
            "The SRK pilot adapter is not enabled for LLE in this release.",
            "Use a validated activity-coefficient LLE backend with evidence-bearing parameters.",
        )

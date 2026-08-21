"""Shared pilot backend for explicit-kij cubic equation-of-state adapters."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from thermo import CEOSGas, CEOSLiquid, ChemicalConstantsPackage, FlashVL

from schemas.domain import EquilibriumPoint, FailureType, ParameterSet, TaskManifest
from thermo_engine.cubic_eos_backend import CubicEosBackend
from thermo_engine.errors import ThermoEquiError
from thermo_engine.parameters import matching_parameter_set


class PilotKijCubicEosBackend(CubicEosBackend):
    """Pilot cubic EOS adapter that requires an explicit binary ``kij``.

    Subclasses set the model name, thermo mixture/pure EOS classes, and the
    accepted parameter-form aliases; all calculation paths come from
    :class:`CubicEosBackend`.
    """

    model_name = ""
    solver_name = ""
    eos_class: Any = None
    pure_eos_class: Any = None
    parameter_form_aliases: frozenset[str] = frozenset({"kij", "binary kij"})
    parameter_property = "binary interaction parameter kij"

    def __init__(self) -> None:
        super().__init__()
        self._parameter_set: ParameterSet | None = None

    def _is_valid_parameter_set(self, parameter_set: ParameterSet) -> bool:
        return (
            parameter_set.model_name.casefold() == self.model_name.casefold()
            and len(parameter_set.component_order) == 2
            and parameter_set.parameter_form.casefold() in self.parameter_form_aliases
            and "kij" in parameter_set.parameters
            and "kij" in parameter_set.units
        )

    def _find_parameter_set(self, request: TaskManifest) -> ParameterSet:
        if len(request.components) != 2:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                f"The {self.model_name} pilot adapter is currently binary-only.",
                "Provide exactly two components or wait for reviewed multicomponent kij support.",
            )
        parameter_set = matching_parameter_set(request.parameters, request.components, self.model_name)
        if parameter_set is None:
            missing_pairs = [[component.cas_number or component.component_id for component in request.components]]
            raise ThermoEquiError(
                FailureType.MISSING_PARAMETERS,
                f"{self.model_name} binary interaction parameters are missing for this component set.",
                "Import a reviewed kij parameter set or select another model.",
                {"model": self.model_name, "parameter_form": f"{self.model_name} kij", "missing_pairs": missing_pairs},
            )
        if not self._is_valid_parameter_set(parameter_set):
            raise ThermoEquiError(
                FailureType.MISSING_PARAMETERS,
                f"The {self.model_name} parameter set does not contain a complete kij entry.",
                f"Provide a binary {self.model_name} parameter set with parameter_form '{self.model_name} kij' "
                "and a 'kij' value.",
                {"model": self.model_name, "parameter_set_id": parameter_set.parameter_set_id},
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
                "property": self.parameter_property,
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

    def _warnings(self) -> list[str]:
        warnings = [
            f"{self.model_name} is a pilot adapter; reviewed kij coverage and benchmark closure are pending.",
            f"{self.model_name} kij parameter applicability requires engineering review.",
        ]
        if self._parameter_set is not None and self._parameter_set.source_type == "test_fixture":
            warnings.append("Parameter set is test_fixture and is not engineering evidence.")
        return warnings

    def _result_warnings(self) -> list[str]:
        return self._warnings()

    def _curve_warnings(self) -> list[str]:
        return [f"Pure curve endpoints use thermo.eos.{self.pure_eos_class.__name__} saturation values."]

    def _phase_fugacities(self, phase: Any) -> list[float]:
        return list(phase.fugacities_at_zs(np.asarray(phase.zs, dtype=float)))

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
        self._find_parameter_set(request)
        identifiers = [component.cas_number or component.component_id for component in request.components]
        constants, _ = ChemicalConstantsPackage.from_IDs(identifiers)
        inapplicable_components = self._inapplicable_components(constants)
        if inapplicable_components:
            raise ThermoEquiError(
                FailureType.PARAMETER_OUT_OF_DOMAIN,
                f"The current {self.model_name} adapter is limited to hydrocarbons and reviewed light gases.",
                "Choose a validated association/activity-coefficient model for this system.",
                {"inapplicable_components": inapplicable_components},
            )
        pure = self.pure_eos_class(
            Tc=float(constants.Tcs[pure_index]),
            Pc=float(constants.Pcs[pure_index]),
            omega=float(constants.omegas[pure_index]),
            T=temperature_K or 300.0,
            P=(pressure_kPa * 1000.0) if pressure_kPa is not None else 1e5,
        )
        composition = [0.0] * len(identifiers)
        composition[pure_index] = 1.0
        if temperature_K is not None:
            pressure_pa = float(pure.Psat(temperature_K))
            point_temperature = temperature_K
            point_pressure_kpa = pressure_pa / 1000.0
        else:
            assert pressure_kPa is not None
            point_temperature = float(pure.Tsat(pressure_kPa * 1000.0))
            point_pressure_kpa = float(pressure_kPa or 0.0)
        return EquilibriumPoint(
            temperature_K=point_temperature,
            pressure_kPa=point_pressure_kpa,
            liquid_composition=composition,
            vapor_composition=composition,
            equilibrium_residual=0.0,
        )

    def _flasher(self, request: TaskManifest) -> Any:
        identifiers = [component.cas_number or component.component_id for component in request.components]
        try:
            constants, properties = ChemicalConstantsPackage.from_IDs(identifiers)
        except (ValueError, LookupError, TypeError):
            raise ThermoEquiError(
                FailureType.MISSING_DATA,
                "The thermo property database could not resolve every component.",
                "Provide canonical names or CAS numbers supported by the reviewed property source.",
                {"components": identifiers},
            ) from None

        inapplicable_components = self._inapplicable_components(constants)
        if inapplicable_components:
            raise ThermoEquiError(
                FailureType.PARAMETER_OUT_OF_DOMAIN,
                f"The current {self.model_name} adapter is limited to hydrocarbons and reviewed light gases.",
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
            self.eos_class,
            eos_kwargs=eos_kwargs,
            HeatCapacityGases=properties.HeatCapacityGases,
        )
        liquid = CEOSLiquid(
            self.eos_class,
            eos_kwargs=eos_kwargs,
            HeatCapacityGases=properties.HeatCapacityGases,
        )
        flasher = FlashVL(constants, properties, liquid=liquid, gas=gas)
        flasher.DEW_BUBBLE_NEWTON_XTOL = 1e-10
        flasher.DEW_BUBBLE_QUASI_NEWTON_XTOL = 1e-10
        return flasher

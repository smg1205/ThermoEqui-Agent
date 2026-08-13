"""CalebBell/thermo Peng-Robinson adapter built on the shared cubic EOS base."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from typing import Any

from thermo import PRMIX, CEOSGas, CEOSLiquid, ChemicalConstantsPackage, FlashVL
from thermo.interaction_parameters import IPDB

from schemas.domain import FailureType, TaskManifest
from thermo_engine.cubic_eos_backend import CubicEosBackend
from thermo_engine.errors import ThermoEquiError


class ThermoPengRobinsonBackend(CubicEosBackend):
    """Peng-Robinson VLE and flash calculations executed by ``thermo``."""

    version = f"thermo/{version('thermo')}"
    model_name = "Peng-Robinson"
    solver_name = "thermo.FlashVL / PRMIX"
    eos_class = PRMIX
    parameter_table = "ChemSep PR"

    def parameter_sources(self, request: TaskManifest) -> list[dict[str, str]]:
        if self._parameter_sources:
            return self._parameter_sources
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
                "property": "Peng-Robinson binary interaction parameter kij",
                "source_title": "ChemSep PR",
                "source_identifier": (
                    "https://github.com/CalebBell/thermo/tree/master/thermo/Interaction%20Parameters/ChemSep"
                ),
                "temperature_range_K": "temperature-independent",
            }
        ]

    def _result_warnings(self) -> list[str]:
        return ["ChemSep PR parameter applicability requires engineering review."]

    def _phase_fugacities(self, phase: Any) -> list[float]:
        return list(phase.fugacities())

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
                "The current Peng-Robinson adapter is limited to hydrocarbons and reviewed light gases.",
                "Choose a validated association/activity-coefficient model for this system.",
                {"inapplicable_components": inapplicable_components},
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
                "Peng-Robinson binary interaction parameters are missing for this component set.",
                "Import reviewed kij values or select another applicable model.",
                {"model": "Peng-Robinson", "parameter_table": self.parameter_table, "missing_pairs": missing_pairs},
            )
        kijs = IPDB.get_ip_asymmetric_matrix(self.parameter_table, constants.CASs, "kij")
        parameter_snapshot = {
            "component_order": constants.CASs,
            "matrix": kijs,
            "parameter_form": "Peng-Robinson kij",
            "source_table": self.parameter_table,
            "thermo_version": self.version,
            "units": "dimensionless",
        }
        snapshot_json = json.dumps(parameter_snapshot, sort_keys=True, separators=(",", ":"))
        self._parameter_set_id = f"chemsep-pr:{hashlib.sha256(snapshot_json.encode()).hexdigest()}"
        component_names = " / ".join(component.name for component in request.components)
        self._parameter_sources = [
            {
                "component": component.name,
                "property": "Pure-component constants and property correlations",
                "source_title": "CalebBell/thermo",
                "source_identifier": "https://github.com/CalebBell/thermo",
                "source_version": self.version,
                "temperature_range_K": "model-dependent",
            }
            for component in request.components
        ] + [
            {
                "component": component_names,
                "component_order": json.dumps(constants.CASs),
                "property": "Peng-Robinson binary interaction parameter kij",
                "parameter_form": "symmetric kij matrix",
                "parameter_values": json.dumps(kijs),
                "parameter_units": "dimensionless",
                "parameter_set_id": self._parameter_set_id,
                "quality_level": "upstream database snapshot; engineering review required",
                "source_title": self.parameter_table,
                "source_identifier": (
                    "https://github.com/CalebBell/thermo/tree/master/thermo/Interaction%20Parameters/ChemSep"
                ),
                "source_version": self.version,
                "temperature_range_K": "temperature-independent",
            }
        ]
        eos_kwargs = {
            "Pcs": constants.Pcs,
            "Tcs": constants.Tcs,
            "omegas": constants.omegas,
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

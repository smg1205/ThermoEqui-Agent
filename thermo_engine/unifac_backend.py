"""Original UNIFAC pilot backend built on CalebBell/thermo group contribution."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

import numpy as np
from thermo.unifac import UNIFAC_gammas, UNIFAC_group_assignment_DDBST

from schemas.domain import ComponentIdentity, FailureType, TaskManifest
from thermo_engine._actcoeff_base import ActCoeffBackend
from thermo_engine.errors import ThermoEquiError

GROUP_ASSIGNMENT_MODEL = "UNIFAC"


def has_unifac_group_assignments(components: list[ComponentIdentity]) -> bool:
    """Return True when every component has a non-empty DDBST UNIFAC assignment."""
    for component in components:
        if component.cas_number is None:
            return False
        try:
            groups = UNIFAC_group_assignment_DDBST(component.cas_number, GROUP_ASSIGNMENT_MODEL)
        except (ValueError, KeyError, TypeError):
            return False
        if not groups:
            return False
    return True


class UnifacBackend(ActCoeffBackend):
    """Predictive activity-coefficient backend using original UNIFAC.

    Group assignments and interaction parameters come from the DDBST tables
    shipped with CalebBell/thermo; no binary ParameterSet is required.
    """

    model_name = "UNIFAC"
    version = f"thermo/{version('thermo')}"
    solver_name = "UNIFAC activity coefficients (original)"

    def __init__(self) -> None:
        super().__init__()
        self._assignments: dict[str, dict[int, int]] = {}
        self._sources: list[dict[str, str]] = []

    def _assign_groups(self, component: ComponentIdentity) -> dict[int, int]:
        """Resolve and cache DDBST UNIFAC subgroup assignments for a component."""
        cas_number = component.cas_number
        if cas_number is None:
            raise ThermoEquiError(
                FailureType.MISSING_PARAMETERS,
                "UNIFAC requires a CAS number to resolve group assignments.",
                "Provide the canonical CAS number for every component.",
                {"model": "UNIFAC", "component": component.name},
            )
        if cas_number in self._assignments:
            return self._assignments[cas_number]
        try:
            groups = UNIFAC_group_assignment_DDBST(cas_number, GROUP_ASSIGNMENT_MODEL)
        except (ValueError, KeyError, TypeError):
            groups = {}
        if not groups:
            raise ThermoEquiError(
                FailureType.MISSING_PARAMETERS,
                "UNIFAC group assignment is unavailable for this component.",
                "Choose a reviewed binary-parameter model or verify the component identity.",
                {"model": "UNIFAC", "component": component.name, "cas_number": cas_number},
            )
        self._assignments[cas_number] = groups
        return {int(key): int(value) for key, value in groups.items()}

    def _params(self, request: TaskManifest) -> dict[str, Any]:
        return {
            "chemgroups": [self._assign_groups(component) for component in request.components],
        }

    def _gamma(
        self,
        xs: np.ndarray[Any, Any],
        T: float,
        names: list[str],
        params: dict[str, Any] | None,
    ) -> np.ndarray[Any, Any]:
        del names
        if params is None or "chemgroups" not in params:
            raise ThermoEquiError(
                FailureType.MISSING_PARAMETERS,
                "UNIFAC group assignments are missing from the backend parameters.",
                "Re-run through the deterministic UNIFAC backend.",
            )
        gammas = UNIFAC_gammas(
            T=float(T),
            xs=[float(value) for value in xs],
            chemgroups=params["chemgroups"],
        )
        values = np.asarray(gammas, dtype=float)
        if values.shape != xs.shape or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ThermoEquiError(
                FailureType.PHYSICAL_VALIDATION_FAILURE,
                "UNIFAC returned non-positive or non-finite activity coefficients.",
                "Do not use this result; review component identities and conditions.",
            )
        return values

    def _warnings(self) -> list[str]:
        return [
            "UNIFAC is a predictive pilot; benchmark closure and applicability review are pending.",
            "UNIFAC group assignments and interaction parameters require engineering review.",
        ]

    def parameter_sources(self, request: TaskManifest) -> list[dict[str, str]]:
        if self._sources:
            return self._sources
        self._params(request)
        component_sources = [
            {
                "component": component.name,
                "property": "UNIFAC subgroup assignment",
                "parameter_values": str(self._assign_groups(component)),
                "source_title": "CalebBell/thermo DDBST UNIFAC assignments",
                "source_identifier": "https://github.com/CalebBell/thermo",
                "source_type": "database",
                "quality_level": "database snapshot; engineering review required",
                "temperature_range_K": "model-dependent",
            }
            for component in request.components
        ]
        interaction_source = {
            "component": " / ".join(component.name for component in request.components),
            "property": "Original UNIFAC interaction parameters",
            "source_title": "UNIFAC original interaction parameters (DDBST, shipped with CalebBell/thermo)",
            "source_identifier": (
                "https://github.com/CalebBell/thermo/tree/main/thermo/Phase%20Change/"
                "UNIFAC%20original%20interaction%20parameters.tsv"
            ),
            "source_type": "database",
            "quality_level": "database snapshot; engineering review required",
            "temperature_range_K": "model-dependent",
        }
        self._sources = [*component_sources, interaction_source]
        return self._sources

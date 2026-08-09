"""Deterministic backend registry for phase-equilibrium model adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from schemas.domain import FailureType, TaskManifest
from thermo_engine.backend import ThermodynamicBackend
from thermo_engine.clapeyron_backend import ClapeyronPengRobinsonBackend
from thermo_engine.errors import ThermoEquiError
from thermo_engine.ideal import IdealRaoultBackend
from thermo_engine.phasepy_backend import PhasepyPengRobinsonBackend
from thermo_engine.properties import resolve_component
from thermo_engine.thermo_backend import ThermoPengRobinsonBackend
from thermo_engine.nrtl_backend import NrtlBackend
from thermo_engine.uniquac_backend import UniquacBackend
from thermo_engine.wilson_backend import WilsonBackend
from thermo_engine.actcoeff_params import lookup_nrtl, lookup_uniquac


@dataclass(frozen=True)
class BackendRegistration:
    canonical_name: str
    aliases: frozenset[str]
    supported_calculations: frozenset[str]
    factory: Callable[[], ThermodynamicBackend]

    def matches(self, model_name: str) -> bool:
        return model_name.casefold() in self.aliases


class ThermodynamicBackendRegistry:
    """Resolve a reviewed model name to one deterministic backend implementation."""

    def __init__(self, registrations: tuple[BackendRegistration, ...]) -> None:
        self.registrations = registrations

    @staticmethod
    def route_task(task: TaskManifest) -> TaskManifest:
        """Apply conservative model defaults only when the caller did not choose one."""
        if task.model_name is not None:
            return task
        if task.equilibrium_type == "LLE":
            raise ThermoEquiError(
                FailureType.MISSING_PARAMETERS,
                "Current production backends cannot represent liquid-liquid equilibrium; "
                "LLE requires an evidence-backed NRTL or UNIQUAC parameter set.",
                "Select NRTL or UNIQUAC and import reviewed binary parameters.",
            )
        pressure = task.conditions.pressure_kPa
        ideal_components_available = True
        for component in task.components:
            try:
                resolve_component(component)
            except ThermoEquiError:
                ideal_components_available = False
                break
        peng_robinson_applicable = ThermoPengRobinsonBackend.supports_system(task)
        if pressure is not None and pressure > 500.0:
            if not peng_robinson_applicable:
                raise ThermoEquiError(
                    FailureType.PARAMETER_OUT_OF_DOMAIN,
                    "No available production backend is applicable to this high-pressure system.",
                    "Choose a reviewed association/activity-coefficient/EOS model with evidence-bearing parameters.",
                )
            selected = "Peng-Robinson"
            reason = "Peng-Robinson was selected for an allowlisted high-pressure hydrocarbon/light-gas system."
        elif ideal_components_available:
            selected = "Ideal/Raoult"
            reason = "Ideal/Raoult was selected within its reviewed low-pressure pure-property registry."
        elif peng_robinson_applicable:
            selected = "Peng-Robinson"
            reason = "Peng-Robinson was selected because the components are outside the local Ideal registry."
        elif _has_activity_coeff_params(task):
            selected, reason = _route_activity_coeff_model(task)
        else:
            raise ThermoEquiError(
                FailureType.PARAMETER_OUT_OF_DOMAIN,
                "No available production backend is applicable to this component set.",
                "Select an implemented model with reviewed properties and interaction parameters.",
            )
        return task.model_copy(
            update={
                "model_name": selected,
                "assumptions": [*task.assumptions, reason],
            }
        )

    def resolve(self, task: TaskManifest) -> ThermodynamicBackend:
        requested = self.route_task(task).model_name or "Ideal/Raoult"
        registration = next((item for item in self.registrations if item.matches(requested)), None)
        if registration is None:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                f"Model {requested} has no resolved production parameter set/backend.",
                "Import an evidence-bearing parameter set or choose an available model.",
            )
        if task.calculation_type not in registration.supported_calculations:
            raise ThermoEquiError(
                FailureType.UNSUPPORTED_MODEL,
                f"Model {registration.canonical_name} does not implement {task.calculation_type}.",
                "Choose a calculation supported by the selected model.",
            )
        backend = registration.factory()
        backend.parameter_sources(task)
        return backend

def _has_activity_coeff_params(task: TaskManifest) -> bool:
    """Check if the component set exists in any activity-coefficient parameter database."""
    if any(
        parameter_set.model_name.casefold() in {"nrtl", "uniquac", "wilson"}
        and [
            name.casefold() for name in parameter_set.component_order
        ]
        == [component.name.casefold() for component in task.components]
        for parameter_set in task.parameters
    ):
        return True
    names = [c.name for c in task.components]
    return lookup_nrtl(names) is not None or lookup_uniquac(names) is not None


def _route_activity_coeff_model(task: TaskManifest) -> tuple[str, str]:
    """Select the best available activity-coefficient model for this system.
    Priority: NRTL (most general) > UNIQUAC > Wilson.
    """
    names = [c.name for c in task.components]
    requested = {
        parameter_set.model_name.casefold()
        for parameter_set in task.parameters
        if [
            name.casefold() for name in parameter_set.component_order
        ]
        == [component.name.casefold() for component in task.components]
    }
    if "nrtl" in requested:
        return ("NRTL", "NRTL auto-selected from the request parameter set.")
    if "uniquac" in requested:
        return ("UNIQUAC", "UNIQUAC auto-selected from the request parameter set.")
    if "wilson" in requested:
        return ("Wilson", "Wilson auto-selected from the request parameter set.")
    if lookup_nrtl(names) is not None:
        return ("NRTL", "NRTL auto-selected for low-pressure system with reviewed binary interaction parameters.")
    if lookup_uniquac(names) is not None:
        return ("UNIQUAC", "UNIQUAC auto-selected for low-pressure system with reviewed binary interaction parameters.")
    # Wilson is the last resort; if none match, the caller should not reach here
    return ("Wilson", "Wilson auto-selected for low-pressure system with reviewed binary interaction parameters.")


DEFAULT_BACKEND_REGISTRY = ThermodynamicBackendRegistry(
    (
        BackendRegistration(
            canonical_name="Ideal/Raoult",
            aliases=frozenset({"ideal", "raoult", "ideal/raoult"}),
            supported_calculations=frozenset(
                {
                    "bubble_point",
                    "dew_point",
                    "isobaric_vle",
                    "isothermal_vle",
                    "tp_flash",
                    "phase_stability",
                    "azeotrope",
                    "lle",
                }
            ),
            factory=IdealRaoultBackend,
        ),
        BackendRegistration(
            canonical_name="Peng-Robinson",
            aliases=frozenset({"peng-robinson", "peng robinson", "pr"}),
            supported_calculations=frozenset(
                {
                    "bubble_point",
                    "dew_point",
                    "isobaric_vle",
                    "isothermal_vle",
                    "tp_flash",
                    "phase_stability",
                    "azeotrope",
                }
            ),
            factory=ThermoPengRobinsonBackend,
        ),
        BackendRegistration(
            canonical_name="Phasepy/Peng-Robinson",
            aliases=frozenset(
                {
                    "phasepy",
                    "phasepy/pr",
                    "phasepy/peng-robinson",
                    "phasepy/peng robinson",
                }
            ),
            supported_calculations=frozenset(
                {
                    "bubble_point",
                    "dew_point",
                    "isobaric_vle",
                    "isothermal_vle",
                    "tp_flash",
                    "azeotrope",
                }
            ),
            factory=PhasepyPengRobinsonBackend,
        ),
        BackendRegistration(
            canonical_name="Clapeyron/Peng-Robinson",
            aliases=frozenset(
                {
                    "clapeyron",
                    "clapeyron/pr",
                    "clapeyron/peng-robinson",
                    "clapeyron/peng robinson",
                }
            ),
            supported_calculations=frozenset(
                {
                    "bubble_point",
                    "dew_point",
                    "isobaric_vle",
                    "isothermal_vle",
                    "tp_flash",
                    "azeotrope",
                }
            ),
            factory=ClapeyronPengRobinsonBackend,
        ),
        BackendRegistration(
            canonical_name="NRTL",
            aliases=frozenset({"nrtl"}),
            supported_calculations=frozenset(
                {
                    "bubble_point",
                    "dew_point",
                    "isobaric_vle",
                    "isothermal_vle",
                    "tp_flash",
                    "phase_stability",
                    "azeotrope",
                    "lle",
                }
            ),
            factory=NrtlBackend,
        ),
        BackendRegistration(
            canonical_name="UNIQUAC",
            aliases=frozenset({"uniquac"}),
            supported_calculations=frozenset(
                {
                    "bubble_point",
                    "dew_point",
                    "isobaric_vle",
                    "isothermal_vle",
                    "tp_flash",
                    "phase_stability",
                    "azeotrope",
                    "lle",
                }
            ),
            factory=UniquacBackend,
        ),
        BackendRegistration(
            canonical_name="Wilson",
            aliases=frozenset({"wilson"}),
            supported_calculations=frozenset(
                {
                    "bubble_point",
                    "dew_point",
                    "isobaric_vle",
                    "isothermal_vle",
                    "tp_flash",
                    "phase_stability",
                    "azeotrope",
                }
            ),
            factory=WilsonBackend,
        ),
    )
)

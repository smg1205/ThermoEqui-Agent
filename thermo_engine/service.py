"""Public Python API for deterministic calculation and validation."""

from __future__ import annotations

from schemas.domain import CalculationResult, FailureType, TaskManifest, ValidationReport
from thermo_engine.backend import ThermodynamicBackend
from thermo_engine.errors import ThermoEquiError
from thermo_engine.identity import is_electrolyte_identity, resolve_external_component
from thermo_engine.registry import DEFAULT_BACKEND_REGISTRY
from thermo_engine.validation import validate_result

UNSUPPORTED_SCOPE_MARKERS = (
    "electrolyte",
    "电解质",
    "polymer",
    "聚合物",
    "hydrate",
    "水合物",
    "pseudocomponent",
    "假组分",
    "polymorph",
    "多晶",
    "reactive",
    "反应相平衡",
    "solid-liquid",
    "sle",
    "vlle",
    "column design",
    "flowsheet",
    "flowsheet design",
    "process flowsheet design",
    "design a flowsheet",
    "design the flowsheet",
    "流程设计",
    "精馏塔设计",
)


def _reject_unsupported_scope(task_manifest: TaskManifest) -> None:
    searchable = " ".join(
        [
            task_manifest.calculation_type,
            task_manifest.original_question or "",
            *(
                value
                for component in task_manifest.components
                for value in (component.component_id, component.name, *component.aliases)
            ),
        ]
    ).casefold()
    if any(marker in searchable for marker in UNSUPPORTED_SCOPE_MARKERS):
        raise ThermoEquiError(
            FailureType.UNSUPPORTED_MODEL,
            "This task is outside the non-electrolyte molecular VLE/Flash scope of version 0.1.",
            "Use a backend explicitly designed and validated for this phase-equilibrium family.",
        )


def _verify_component_identities(task_manifest: TaskManifest) -> None:
    electrolytes = [component.name for component in task_manifest.components if is_electrolyte_identity(component)]
    if electrolytes:
        raise ThermoEquiError(
            FailureType.UNSUPPORTED_MODEL,
            "Electrolyte components are outside the non-electrolyte molecular scope of version 0.1.",
            "Use a validated electrolyte thermodynamics backend.",
            {"electrolyte_components": electrolytes},
        )
    for component in task_manifest.components:
        if component.cas_number is None:
            continue
        resolved_name = resolve_external_component(component.name)
        if resolved_name is None or resolved_name.cas_number == component.cas_number:
            continue
        raise ThermoEquiError(
            FailureType.SEMANTIC_FAILURE,
            f"Component name {component.name!r} does not match declared CAS {component.cas_number}.",
            "Correct the component identity before calculation.",
            {
                "component_name": component.name,
                "declared_cas": component.cas_number,
                "resolved_cas": resolved_name.cas_number,
            },
        )


def calculate_equilibrium(
    task_manifest: TaskManifest, backend: ThermodynamicBackend | None = None
) -> CalculationResult:
    """Execute one manifest without any LLM or frontend dependency."""
    if task_manifest.original_question is None:
        task_manifest = task_manifest.model_copy(update={"original_question": "Structured Python/CLI submission"})
    _verify_component_identities(task_manifest)
    _reject_unsupported_scope(task_manifest)
    if task_manifest.model_name is None:
        task_manifest = DEFAULT_BACKEND_REGISTRY.route_task(task_manifest)
    selected = backend or DEFAULT_BACKEND_REGISTRY.resolve(task_manifest)
    requested_model = (task_manifest.model_name or "Ideal/Raoult").casefold()
    if task_manifest.composition_basis == "mass_fraction":
        raise ThermoEquiError(
            FailureType.UNSUPPORTED_MODEL,
            "The internal backend does not silently treat mass fractions as mole fractions.",
            "Convert with reviewed molecular weights and submit mole fractions.",
        )
    pressure = task_manifest.conditions.pressure_kPa
    if requested_model in {"ideal", "raoult", "ideal/raoult"} and pressure is not None and pressure > 500:
        raise ThermoEquiError(
            FailureType.PARAMETER_OUT_OF_DOMAIN,
            "Ideal/Raoult is blocked above the configured low-pressure regime (500 kPa).",
            "Select a validated equation-of-state backend for this pressure.",
        )
    operations = {
        "bubble_point": selected.bubble_point,
        "dew_point": selected.dew_point,
        "isobaric_vle": selected.isobaric_vle,
        "isothermal_vle": selected.isothermal_vle,
        "tp_flash": selected.tp_flash,
        "phase_stability": selected.phase_stability,
        "azeotrope": selected.azeotrope,
        "lle": selected.lle,
        "infinite_dilution_activity": selected.infinite_dilution_activity,
    }
    operation = operations.get(task_manifest.calculation_type)
    if operation is None:
        raise ThermoEquiError(
            FailureType.UNSUPPORTED_MODEL,
            f"Calculation type {task_manifest.calculation_type!r} is not implemented.",
            "Choose a supported VLE/Flash calculation type.",
        )
    return operation(task_manifest)


def resolve_backend(task_manifest: TaskManifest) -> ThermodynamicBackend:
    """Resolve the deterministic backend without executing an operation."""
    _verify_component_identities(task_manifest)
    _reject_unsupported_scope(task_manifest)
    return DEFAULT_BACKEND_REGISTRY.resolve(task_manifest)


def route_task_model(task_manifest: TaskManifest) -> TaskManifest:
    """Return a manifest with a conservative deterministic model selection."""
    _verify_component_identities(task_manifest)
    _reject_unsupported_scope(task_manifest)
    return DEFAULT_BACKEND_REGISTRY.route_task(task_manifest)


def validate_equilibrium_result(result: CalculationResult) -> ValidationReport:
    return validate_result(result)

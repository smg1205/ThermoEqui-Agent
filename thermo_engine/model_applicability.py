"""Apply minimal applicability filtering rules to model catalog entries."""

from __future__ import annotations

from schemas.model_applicability import (
    ModelAllowanceRequest,
    ModelAllowanceResult,
    ModelApplicabilityReport,
    ModelApplicabilityRequest,
    ModelApplicabilityResult,
)
from schemas.model_catalog import ModelCatalogEntry
from thermo_engine.model_catalog import load_model_catalog


def is_model_allowed(request: ModelAllowanceRequest) -> ModelAllowanceResult:
    entry = load_model_catalog().get(request.model_name)
    if entry is None:
        return ModelAllowanceResult(
            allowed=False,
            reason=f"Rejected: model {request.model_name!r} is not present in the model catalog.",
        )

    reasons: list[str] = []
    available_parameters = {name.casefold() for name in request.available_parameters}

    if request.calculation_type not in entry.supported_calculation_types:
        reasons.append(f"calculation_type {request.calculation_type!r} is not supported by {entry.name}")
    if request.equilibrium_type not in entry.supported_equilibrium_types:
        reasons.append(f"equilibrium_type {request.equilibrium_type!r} is not supported by {entry.name}")
    if entry.requires_binary_parameters and entry.name.casefold() not in available_parameters:
        reasons.append(f"{entry.name} requires reviewed binary interaction parameters, but none are available")

    if entry.name in {"NRTL", "UNIQUAC"} and request.equilibrium_type == "LLE":
        reasons.append(
            f"{entry.name} is not currently allowed for LLE because the shared activity-coefficient backend "
            "does not provide executable LLE support"
        )
    if entry.name == "Wilson" and request.equilibrium_type == "LLE":
        reasons.append("Wilson explicitly rejects LLE in the current backend")

    if reasons:
        return ModelAllowanceResult(allowed=False, reason="Rejected: " + "; ".join(reasons) + ".")

    return ModelAllowanceResult(
        allowed=True,
        reason=f"Allowed: {entry.name} satisfies the current model applicability rules.",
    )


def evaluate_model_applicability(
    entry: ModelCatalogEntry, request: ModelApplicabilityRequest
) -> ModelApplicabilityResult:
    reasons: list[str] = []
    task = request.task
    available_parameter_models = {name.casefold() for name in request.available_parameter_models}

    if task.calculation_type not in entry.supported_calculation_types:
        reasons.append(f"Excluded: calculation_type {task.calculation_type!r} is not supported by {entry.name}.")
    if task.equilibrium_type not in entry.supported_equilibrium_types:
        reasons.append(f"Excluded: equilibrium_type {task.equilibrium_type!r} is not supported by {entry.name}.")
    if entry.implementation_status == "contract_only":
        reasons.append(f"Excluded: {entry.name} is contract_only and not executable in the current product.")
    if request.production_only and not entry.production_ready:
        reasons.append(f"Excluded: production_only was requested and {entry.name} is not production_ready.")
    if entry.requires_binary_parameters and entry.name.casefold() not in available_parameter_models:
        reasons.append(
            f"Excluded: {entry.name} requires binary parameters, but no reviewed parameter set is available."
        )

    if reasons:
        return ModelApplicabilityResult(model_name=entry.name, decision="exclude", reasons=reasons)

    return ModelApplicabilityResult(
        model_name=entry.name,
        decision="keep",
        reasons=["Kept: the model satisfies the current minimal applicability rules."],
    )


def filter_applicable_models(request: ModelApplicabilityRequest) -> ModelApplicabilityReport:
    catalog = load_model_catalog()
    return ModelApplicabilityReport(
        results=[evaluate_model_applicability(entry, request) for entry in catalog.values()]
    )

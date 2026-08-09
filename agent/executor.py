"""Shared deterministic calculate-validate-evidence workflow."""

from dataclasses import dataclass

from agent.router import (
    available_parameter_models_for_task,
    rank_executable_models,
    recommend_models,
)
from schemas.domain import CalculationEnvelope, CalculationResult, FailureType, TaskManifest
from thermo_engine.backend import ThermodynamicBackend
from thermo_engine.errors import ThermoEquiError
from thermo_engine.service import (
    calculate_equilibrium,
    resolve_backend,
    validate_equilibrium_result,
)


@dataclass(frozen=True)
class TaskExecution:
    """Raw deterministic tool output awaiting the independent validation node."""

    task: TaskManifest
    backend: ThermodynamicBackend
    result: CalculationResult


def calculate_task(
    task: TaskManifest,
    available_parameter_models: set[str] | None = None,
) -> TaskExecution:
    if available_parameter_models is None:
        available_parameter_models = available_parameter_models_for_task(task)
    if task.model_name is None:
        candidates = rank_executable_models(task, available_parameter_models)
        if not candidates:
            raise ThermoEquiError(
                FailureType.PARAMETER_OUT_OF_DOMAIN,
                "No executable model is available for this task.",
                "Import reviewed parameters or select a supported model.",
            )
        selected = candidates[0]
        task = task.model_copy(
            update={
                "model_name": selected.model_name,
                "assumptions": [
                    *task.assumptions,
                    f"Auto-selected {selected.model_name} with score {selected.score:.1f}.",
                ],
            }
        )
    backend = resolve_backend(task)
    result = calculate_equilibrium(task, backend=backend)
    return TaskExecution(task=task, backend=backend, result=result)


def validate_task_execution(
    execution: TaskExecution,
    available_parameter_models: set[str] | None = None,
) -> CalculationEnvelope:
    validation = validate_equilibrium_result(execution.result)
    if available_parameter_models is None:
        available_parameter_models = available_parameter_models_for_task(execution.task)
    return CalculationEnvelope(
        result=execution.result,
        validation=validation,
        parameter_sources=execution.backend.parameter_sources(execution.task),
        model_recommendations=recommend_models(
            execution.task,
            available_parameter_models=available_parameter_models,
        ),
    )


def execute_task(
    task: TaskManifest,
    available_parameter_models: set[str] | None = None,
) -> CalculationEnvelope:
    """Synchronous public seam composing calculation and independent validation."""
    if available_parameter_models is None:
        available_parameter_models = available_parameter_models_for_task(task)
    if task.model_name is not None:
        return validate_task_execution(
            calculate_task(task, available_parameter_models),
            available_parameter_models,
        )

    candidates = rank_executable_models(task, available_parameter_models)[:3]
    if not candidates:
        raise ThermoEquiError(
            FailureType.PARAMETER_OUT_OF_DOMAIN,
            "No executable model is available for this task.",
            "Import reviewed parameters or select a supported model.",
        )

    attempts: list[str] = []
    last_envelope: CalculationEnvelope | None = None
    for candidate in candidates:
        attempt_task = task.model_copy(
            update={
                "model_name": candidate.model_name,
                "assumptions": [
                    *task.assumptions,
                    f"Auto-selected {candidate.model_name} with score {candidate.score:.1f}.",
                ],
            }
        )
        try:
            execution = calculate_task(attempt_task, available_parameter_models)
            envelope = validate_task_execution(execution, available_parameter_models)
        except ThermoEquiError as error:
            if error.detail.failure_type not in {
                FailureType.NUMERICAL_NONCONVERGENCE,
                FailureType.PHYSICAL_VALIDATION_FAILURE,
                FailureType.PHASE_INSTABILITY,
            }:
                raise
            attempts.append(f"{candidate.model_name}: {error.detail.message}")
            continue
        if envelope.validation.overall_status != "failed":
            if attempts:
                envelope = envelope.model_copy(
                    update={
                        "result": envelope.result.model_copy(
                            update={
                                "warnings": [
                                    *envelope.result.warnings,
                                    f"Fell back from: {'; '.join(attempts)}",
                                ]
                            }
                        )
                    }
                )
            return envelope
        attempts.append(f"{candidate.model_name}: validation failed")
        last_envelope = envelope

    if last_envelope is not None:
        return last_envelope
    raise ThermoEquiError(
        FailureType.NUMERICAL_NONCONVERGENCE,
        "All ranked model candidates failed.",
        "Review parameters and model applicability.",
    )

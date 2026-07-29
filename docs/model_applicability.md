# Model Applicability

## Goal

The current model applicability layer provides one place to describe reviewed thermodynamic model
scope and one place to apply minimal candidate filtering before execution logic.

Its current goals are:

- Uniformly manage model applicability metadata.
- Filter candidate models from structured task conditions.
- Return explicit keep or exclude reasons for every catalog entry.

This layer does **not** currently perform automatic routing by itself. Final model routing remains
separate from this metadata and filtering layer.

## Implemented

The following pieces are currently in place:

- `knowledge/model_catalog/*.yaml`
  - Static model catalog entries for the reviewed models currently tracked by the repository.
- `schemas/model_catalog.py`
  - Pydantic schema for catalog entries, including strict `extra="forbid"` validation.
- `thermo_engine/model_catalog.py`
  - YAML loader for the model catalog.
- `schemas/model_applicability.py`
  - Request and result schemas for applicability filtering.
- `thermo_engine/model_applicability.py`
  - Minimal rule-based candidate filtering over the current catalog.
- Tests
  - `tests/test_model_catalog.py`
  - `tests/test_model_applicability.py`

## Current Model Status

| Model | Backend Label | implementation_status | production_ready | Scope Summary |
|---|---|---|---|---|
| `Ideal/Raoult` | `internal` | `available` | `true` | Low-pressure reviewed non-electrolyte VLE and flash baseline within the local pure-property registry |
| `Peng-Robinson` | `thermo` | `available` | `true` | Moderate- to high-pressure VLE and flash for hydrocarbons and reviewed light-gas systems |
| `Phasepy/Peng-Robinson` | `phasepy` | `available` | `false` | Optional external Peng-Robinson backend for VLE and flash |
| `Clapeyron/Peng-Robinson` | `clapeyron` | `available` | `false` | Optional external Peng-Robinson backend for VLE and flash |
| `NRTL` | `internal` | `available` | `false` | Implemented and registered low-/moderate-pressure non-ideal VLE and flash-style backend with limited reviewed binary parameter coverage |
| `UNIQUAC` | `internal` | `available` | `false` | Implemented and registered low-/moderate-pressure non-ideal VLE and flash-style backend with limited reviewed binary parameter coverage |
| `Wilson` | `internal` | `available` | `false` | Implemented and registered low-/moderate-pressure VLE and flash-style backend; LLE is explicitly rejected |

## Current Filtering Rules

The current applicability filter is intentionally minimal. It only applies the following rules:

1. `calculation_type`
   - Exclude a model if the task `calculation_type` is not listed in that catalog entry's
     `supported_calculation_types`.
2. `equilibrium_type`
   - Exclude a model if the task `equilibrium_type` is not listed in that catalog entry's
     `supported_equilibrium_types`.
3. `implementation_status`
   - Exclude a model if its `implementation_status` is `contract_only`.
4. `production_ready`
   - When `production_only=True`, exclude a model if `production_ready=false`.
5. Binary parameter availability
   - Exclude a model if `requires_binary_parameters=true` and its model name is not present in
     `available_parameter_models`.

The filter returns one result for every catalog entry and accumulates all applicable exclusion
reasons. If no exclusion rule is triggered, the result is `keep` with a short positive reason.

## Model Applicability Rules

The repository now also provides a small single-model applicability check for answering:

- whether one named model is allowed for one requested problem shape
- why it is allowed or rejected

This check uses `thermo_engine.model_applicability.is_model_allowed(...)` and currently expects:

- `model_name`
- `calculation_type`
- `equilibrium_type`
- `available_parameters`

It returns:

- `allowed: bool`
- `reason: str`

Current model-specific rules are intentionally conservative and do not change catalog execution
status. This applicability layer only performs candidate-model screening. It does not implement new
thermodynamic backends, does not modify `registry.py`, `service.py`, or `router.py`, and does not
upgrade `production_ready`.

### NRTL

Applicable to:

- Non-ideal liquid-phase VLE
- Flash-style calculations already implemented by the shared activity-coefficient backend
- Cases with valid binary interaction parameters

Rejected when:

- Required parameters are unavailable
- The requested calculation is outside the supported calculation scope
- `LLE` is requested
- The requested equilibrium type is otherwise unsupported

Current status:

- Backend code is implemented and registered in the current code package
- `production_ready` remains `false` because reviewed parameter coverage and execution evidence are still limited

### UNIQUAC

Applicable to:

- Non-ideal liquid-phase VLE
- Flash-style calculations already implemented by the shared activity-coefficient backend
- Cases with valid binary interaction parameters

Rejected when:

- Required parameters are unavailable
- The requested calculation is outside the supported task scope
- `LLE` is requested
- The requested equilibrium type is otherwise unsupported

Current status:

- Backend code is implemented and registered in the current code package
- `production_ready` remains `false` because reviewed parameter coverage and execution evidence are still limited

### Wilson

Applicable to:

- VLE
- Flash-style calculations already implemented by the shared activity-coefficient backend

Rejected when:

- `LLE`
- The requested equilibrium type is otherwise unsupported
- The requested calculation is outside the supported calculation scope
- Required binary parameters are unavailable

Current status:

- Backend code is implemented and registered in the current code package
- `production_ready` remains `false` because reviewed parameter coverage and execution evidence are still limited

## Current Boundary

The current boundary of this feature is intentionally narrow:

- The applicability filter is **not yet connected** to `executor`, `router`, or backend resolution.
- This layer does not change backend execution logic even when a backend already exists in the codebase.
- This document does not evaluate or summarize external traditional-model code quality.
- This document does not describe unconfirmed AI model capabilities.
- This document does not describe unconfirmed `SRK` support in the main project.

Also intentionally out of scope for the current implementation:

- Pressure-threshold rules
- Component-property rules
- Ranking or scoring
- Automatic routing
- Registry synchronization

## Minimal Example

```python
from schemas.domain import ComponentIdentity, TaskManifest, ThermodynamicConditions
from schemas.model_applicability import ModelApplicabilityRequest
from thermo_engine.model_applicability import filter_applicable_models

task = TaskManifest(
    equilibrium_type="VLE",
    calculation_type="isobaric_vle",
    components=[
        ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2"),
        ComponentIdentity(component_id="toluene", name="Toluene", cas_number="108-88-3"),
    ],
    conditions=ThermodynamicConditions(pressure_kPa=101.325),
)

report = filter_applicable_models(
    ModelApplicabilityRequest(
        task=task,
        production_only=True,
        available_parameter_models={"Peng-Robinson"},
    )
)

for item in report.results:
    print(item.model_name, item.decision, item.reasons)
```

This example only performs catalog-based filtering. It does not execute a thermodynamic backend and
does not select a final model automatically.

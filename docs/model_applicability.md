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

## Current Selection Flow

The current project flow is:

- `knowledge/model_cards/*.yaml`
- `agent/router.py`
- `thermo_engine.model_applicability`
- `agent/executor.py`
- `thermo_engine.service`

In practice:

1. `agent/router.py` loads `model_cards` and generates candidate recommendations.
2. During candidate recommendation, the router now calls the model applicability layer for
   per-model hard-constraint checks.
3. Applicability may mark a candidate as non-executable and add explicit exclusion reasons.
4. The router still keeps the recommendation object and original scoring structure.
5. `agent/executor.py` remains responsible for actual execution through `thermo_engine.service`.

This means:

- Applicability is responsible for model-scope checks, execution constraints, and exclusion reasons.
- The router is responsible for candidate generation and final recommendation ordering.
- The executor and `thermo_engine` remain responsible for backend resolution and calculation.

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
| `SRK` | `thermo` | `available` | `false` | Pilot binary SRK VLE/flash requiring an explicit reviewed or user-attested kij ParameterSet; benchmark closure pending |
| `NRTL` | `internal` | `available` | `true` | Low-/moderate-pressure non-ideal VLE and flash for ChemSep-validated ethanol/water and ethanol/benzene binaries; legacy DECHEMA sets remain prototype |
| `UNIQUAC` | `internal` | `available` | `true` | Low-/moderate-pressure non-ideal VLE and flash for ChemSep-validated ethanol/water and ethanol/benzene binaries; UNIQUAC benchmarks exclude pure endpoints |
| `Wilson` | `internal` | `available` | `true` | Low-/moderate-pressure VLE and flash for ChemSep-validated ethanol/water and ethanol/benzene binaries; LLE is explicitly rejected |

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
- Reviewed binary parameters are managed through the production parameter store and seeded with `thermoequi-seed`
- `production_ready` is `true` for the ChemSep-validated ethanol/water and ethanol/benzene binaries benchmarked against experimental isobaric VLE data
- Legacy DECHEMA parameter sets for other binaries remain available but are not yet production-validated

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
- Reviewed binary parameters are managed through the production parameter store and seeded with `thermoequi-seed`
- `production_ready` is `true` for the ChemSep-validated ethanol/water and ethanol/benzene binaries benchmarked against experimental isobaric VLE data
- The UNIQUAC experimental benchmark excludes pure endpoints because combinatorial terms are undefined at x=0/1 in the current backend
- Legacy DECHEMA parameter sets for other binaries remain available but are not yet production-validated

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
- Reviewed binary parameters are managed through the production parameter store and seeded with `thermoequi-seed`
- `production_ready` is `true` for the ChemSep-validated ethanol/water and ethanol/benzene binaries benchmarked against experimental isobaric VLE data
- Legacy DECHEMA parameter sets for other binaries remain available but are not yet production-validated

## Current Boundary

The current boundary of this feature is intentionally narrow:

- The applicability filter is **not yet connected** to `executor`, `router`, or backend resolution.
- This layer does not change backend execution logic even when a backend already exists in the codebase.
- This document does not evaluate or summarize external traditional-model code quality.
- This document does not describe unconfirmed AI model capabilities.
- `SRK` is tracked as a pilot adapter; `production_ready` stays `false` until reviewed kij
  coverage and benchmark closure are complete.

Also intentionally out of scope for the current implementation:

- Pressure-threshold rules
- Component-property rules
- Ranking or scoring
- Automatic routing
- Registry synchronization is instead covered by `tests/test_backend_registry_contract.py`

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

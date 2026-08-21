# Roadmap

## P0 model breadth (2026-08)

Model integration is now prioritized ahead of the pending experimental dataset:

- `UNIFAC` original pilot: predictive group-contribution backend on
  CalebBell/thermo DDBST assignments; no binary `ParameterSet` is required and
  missing group assignments return structured `missing_parameters`.
- `RK` pilot: explicit-kij binary adapter on `thermo.RKMIX`, sharing the new
  `PilotKijCubicEosBackend` base with the existing SRK pilot.
- Multi-model comparison: `POST /api/calculations/compare` runs every executable
  model for one task and returns per-model results, validation reports, parameter
  sources, and structured failures. The endpoint is backend-only; frontend
  adoption is owned by the UI integration task.
- Registry contract tests now cover every registered backend, including the new
  UNIFAC and RK entries, against catalog/card metadata and the public validation
  gate.

Both pilots remain `production_ready=false`. Their graduation checklist is:
parameter provenance review, at least one experimental or software-reference
benchmark, applicability-range checks, and then `production_ready=true`.

## PGSSI first-class pilot (2026-08)

PGSSI (the group's Physics-Guided 3D Solute-Solvent Interaction framework) is now
integrated as a first-class predictive backend, side by side with
NRTL/UNIQUAC/Wilson, not as a dependency of any VLE model:

- `PgssiBackend` (`thermo_engine/pgssi_backend.py`) implements the full
  `ThermodynamicBackend` protocol and is registered in `DEFAULT_BACKEND_REGISTRY`
  under `PGSSI`, supporting the new `infinite_dilution_activity` calculation type.
- It predicts temperature-dependent infinite-dilution activity coefficients
  (`log-gamma_inf = K1 + K2/T`) from solute/solvent SMILES and temperature. It
  requires no binary `ParameterSet`; a missing checkpoint, SMILES, source tree,
  or optional dependency is a structured `missing_parameters` failure.
- VLE/flash operations fail explicitly with `unsupported_model`; PGSSI never
  pretends to be a phase-equilibrium solver, and no VLE backend depends on PGSSI.
- A gamma-infinity-to-NRTL regression bridge (`thermo_engine/pgssi_params.py`)
  converts gamma-infinity data into the production NRTL `a+b/T` parameter form;
  the full chain (gamma-infinity data -> NRTL parameters -> isobaric bubble point
  -> validation gate) has been exercised with real ethanol/water data.
- New endpoint: `POST /api/calculations/infinite-dilution-activity`.

PGSSI remains `production_ready=false`. Graduation requires a trained checkpoint
(reviewed), experimental gamma-infinity or finite-concentration VLE benchmark
closure, and applicability review. The group dataset (`39,840` gamma-infinity
rows) is the candidate benchmark source; any proprietary data must stay out of
the public repository.

## P0 hardening (2026-08)

The parameter pipeline is now closed without waiting for new experimental data:

- Reviewed activity-coefficient parameters moved from hardcoded modules into
  `knowledge/parameters/*.yaml` and seed idempotently with `thermoequi-seed`.
- Backends and routing consume only `ParameterSet` records; missing parameters
  produce structured `missing_parameters` failures.
- Registry contract tests verify capability declarations against the model
  catalog/cards, parameter source reporting, and the public validation gate.
- Peng-Robinson and SRK now share a common `CubicEosBackend` implementation;
  existing SRK/PR behavioral tests plus the new registry contract tests cover
  the refactor.

`production_ready` is now `true` for NRTL, UNIQUAC, and Wilson on the
ChemSep-validated ethanol/water and ethanol/benzene binaries benchmarked against
experimental isobaric VLE data. SRK remains `false` until reviewed kij coverage
and benchmark closure are complete.

Phase 4 will add evidence-backed parameter regression, production NRTL/UNIQUAC binary LLE,
frontend adoption of the multi-model comparison API, sensitivity analysis, PDF reports, and
DWSIM/Aspen configuration support.
SRK now has a pilot `thermo` adapter; enabling it as `production_ready` requires reviewed kij data
and benchmark closure. Later research may evaluate Dortmund-UNIFAC, CPA, PC-SAFT, eNRTL, and Pitzer
adapters. Scope expansion only follows verification coverage; electrolytes, SLE, VLLE, and full
flowsheets remain out of scope for the current release.

Phasepy and Clapeyron.jl now have optional Peng-Robinson adapters for the reviewed non-electrolyte
VLE boundary. The next integration step is evidence-backed Phasepy NRTL/UNIQUAC support and
Clapeyron association/SAFT models; neither may be enabled until parameter provenance and behavioral
validation cases exist. NeqSim remains a potential industrial JVM adapter. Every engine remains
isolated behind `ThermodynamicBackend` and must pass the same evidence and validation gates.

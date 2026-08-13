# Roadmap

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
multi-model comparison, sensitivity analysis, PDF reports, and DWSIM/Aspen configuration support.
SRK now has a pilot `thermo` adapter; enabling it as `production_ready` requires reviewed kij data
and benchmark closure. Later research may evaluate Dortmund-UNIFAC, CPA, PC-SAFT, eNRTL, and Pitzer
adapters. Scope expansion only follows verification coverage; electrolytes, SLE, VLLE, and full
flowsheets remain out of scope for the current release.

Phasepy and Clapeyron.jl now have optional Peng-Robinson adapters for the reviewed non-electrolyte
VLE boundary. The next integration step is evidence-backed Phasepy NRTL/UNIQUAC support and
Clapeyron association/SAFT models; neither may be enabled until parameter provenance and behavioral
validation cases exist. NeqSim remains a potential industrial JVM adapter. Every engine remains
isolated behind `ThermodynamicBackend` and must pass the same evidence and validation gates.

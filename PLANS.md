# ThermoEqui-Agent delivery plan

Last updated: 2026-08-11

## Architecture decisions

- Keep deterministic thermodynamic calculations independent from the LLM, API, and UI.
- Expose calculations through a `ThermodynamicBackend` protocol; ship a tested internal
  Ideal/Raoult adapter first and reject parameterized models when evidence is absent.
- Use Pydantic as the canonical transport contract, SQLAlchemy with SQLite by default, and
  JSON snapshots for reproducibility and a future PostgreSQL migration path.
- Use one conversation orchestrator with deterministic and OpenAI provider implementations.
- Treat validation as a separate mandatory gate. A converged solver can still produce a failed run.
- Build the UI as a Next.js engineering workbench, not as a generic chat page.

## Status

- [x] Phase 0: repository conventions, plans, local skills, dependency manifests, CI skeleton.
- [x] Phase 1: schemas, pure-component data, ideal VLE/Flash adapter, validation, CLI, tests.
- [x] Phase 2: deterministic agent, model router, parameter repository, database, API, evals.
- [x] Phase 3: workbench UI, chart/table, editable conditions, history, JSON/CSV export.
- [~] Phase 4 (model breadth first): UNIFAC/RK pilots and backend multi-model
  comparison are implemented; PGSSI first-class gamma-infinity pilot and the
  gamma-infinity-to-NRTL regression bridge are implemented; frontend adoption,
  parameter regression, production NRTL/UNIQUAC LLE, and simulator integrations
  remain.
- [ ] Phase 4 remainder: parameter regression, production NRTL/UNIQUAC LLE, frontend
  multi-model comparison UI, sensitivity analysis, PDF reports, simulator integrations,
  PGSSI benchmark closure and production_ready graduation.

## Acceptance tracking

The first release is accepted when backend tests, lint/type checks, frontend tests/build, API smoke
tests, and a benzene/toluene isobaric VLE demonstration pass. Docker files are provided; an actual
container smoke test additionally requires Docker on the host.

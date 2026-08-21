# Production parameter repository

This directory is the reviewed source of truth for production `ParameterSet`
records. The repository seed tool (`thermoequi-seed`) loads every `*.yaml` file
here into the application database idempotently.

Rules:

- Entries are YAML mappings compatible with `schemas.domain.ParameterSet`.
- `source_type` must never be `test_fixture`; synthetic and test-only records
  live under `tests/fixtures` and are rejected by the loader.
- Every record carries a stable `parameter_set_id`, component order, units,
  applicable T/P range, equilibrium types, source metadata, and quality level.
- Values must be preserved verbatim from their reviewed source. Do not invent
  DOIs, page numbers, or parameter values. `legacy-registry:` identifiers are
  internal record keys, not external bibliographic identifiers.

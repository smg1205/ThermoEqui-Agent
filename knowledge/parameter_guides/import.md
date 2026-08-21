# Parameter import

Import exact model form, directional component order, units, applicable T/P range, equilibrium type,
and a source identifier. User-supplied and estimated values must remain visibly labeled. Never reuse
parameters across a different equation form without an explicit conversion.

## Production parameter store

Reviewed production records live in `knowledge/parameters/*.yaml` and are seeded
into the application database idempotently:

```bash
thermoequi-seed
thermoequi-seed --check-only
thermoequi-seed --database-url "sqlite:///./thermoequi.db"
```

The loader rejects `test_fixture` records and duplicate `parameter_set_id`
values. Synthetic and test-only parameters must stay under `tests/fixtures` and
never enter the production store.

## Model-specific forms

- NRTL: `tau12_a`, `tau12_b`, `tau21_a`, `tau21_b`, optional `alpha`
  (form `NRTL a+b/T binary`)
- UNIQUAC: `tau12_a`, `tau12_b`, `tau21_a`, `tau21_b`, `r1`, `r2`, `q1`, `q2`
  (form `UNIQUAC exp(a+b/T) binary`)
- Wilson: `lambda12_a`, `lambda12_b`, `lambda21_a`, `lambda21_b`, `v1`, `v2`
  (form `Wilson exp(a+b/T) binary`)
- SRK: `kij` (form `SRK kij`)

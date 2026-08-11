# Benchmarks

This directory holds model-agnostic benchmark cases. Each case defines a
thermodynamic task, reference values, tolerances, and a traceable source.

`source.kind` distinguishes:

- `experimental`: measured VLE/phase-equilibrium data
- `literature`: published table or correlation values
- `simulation`: values produced by another simulator
- `software_reference`: reproducible values from a library reference case

Benchmark cases are loaded by `benchmarks.loader` and executed by
`benchmarks.runner`. The runner uses `thermo_engine.calculate_equilibrium`
and `thermo_engine.validate_equilibrium_result` for every model, so all
benchmark results pass through the same deterministic and validation path.

Experimental data must include a source identifier before it can be used to
promote a model to `production_ready=true`.

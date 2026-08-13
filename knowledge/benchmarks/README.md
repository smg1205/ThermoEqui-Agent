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

Multi-point isobaric VLE datasets use `calculation_type: bubble_point` (or
`dew_point`) with one reference point per experimental row. The runner executes
one point calculation per row, injects the reference liquid (or vapor)
composition into the task conditions, and compares every predicted temperature
and composition against the experimental value. Pure endpoints may be excluded
from a case when a backend is undefined there; the case notes must say so.

Current experimental cases:

- `ideal-benzene-toluene-isobaric-101kpa`: Rollet et al. 1956, 101.325 kPa
- `nrtl-ethanol-benzene-isobaric-53kpa`: Nielsen & Weber 1959, 53.33 kPa
- `uniquac-ethanol-benzene-isobaric-53kpa`: Nielsen & Weber 1959, 53.33 kPa (interior points)
- `wilson-ethanol-benzene-isobaric-53kpa`: Nielsen & Weber 1959, 53.33 kPa

Rows are transcribed from the traceable `DannerGuessIsoB.csv` compilation in
`RafaelSch/vle-ai-benchmarking`; each case's `source.notes` records the original
literature citation.

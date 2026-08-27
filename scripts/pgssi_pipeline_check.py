"""One-shot pipeline validation: gamma-infinity data -> NRTL parameters -> VLE.

Uses real ethanol/water infinite-dilution data from the PGSSI group dataset to
exercise the production chain: regress NRTL (b12, b21) with the pgssi_params
bridge, then run the production NRTL backend on an isobaric bubble point and
cross the independent validation gate.  Not part of the test suite.
"""

from __future__ import annotations

import csv

from schemas.domain import ComponentIdentity, TaskManifest, ThermodynamicConditions
from thermo_engine import calculate_equilibrium, validate_equilibrium_result
from thermo_engine.pgssi_params import regress_nrtl_from_gamma_infinity

DATA = r"D:\CodexDocuments\PGSSI\PGSSI\dataset\NRTL_5fold\IdacRecJaubert+TdeOrginW.csv"

temperatures: list[float] = []
ln_gamma: list[float] = []
with open(DATA, encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        if row["Solute_SMILES"] == "CCO" and row["Solvent_SMILES"] == "O":
            temperatures.append(float(row["T_K"]))
            ln_gamma.append(float(row["ln_gamma_inf"]))

# De-duplicate near-identical temperatures to keep the regression well posed.
pairs = sorted(zip(temperatures, ln_gamma, strict=True))
deduped: list[tuple[float, float]] = []
for t, g in pairs:
    if not deduped or abs(t - deduped[-1][0]) > 1.0:
        deduped.append((t, g))
temperatures = [item[0] for item in deduped]
ln_gamma = [item[1] for item in deduped]

print(f"Ethanol/water gamma-infinity rows: {len(temperatures)}")
parameter_set = regress_nrtl_from_gamma_infinity(
    ["ethanol", "water"],
    temperatures,
    ln_gamma,
    source_type="estimated",
    source_title="PGSSI group dataset NRTL_5fold (IdacRecJaubert+TdeOrginW.csv)",
    source_identifier=DATA,
    quality_level="pgssi-regressed-unreviewed",
)
print(
    f"Regressed NRTL: tau12_b={parameter_set.parameters['tau12_b']:.4f} "
    f"tau21_b={parameter_set.parameters['tau21_b']:.4f} alpha={parameter_set.parameters['alpha']}"
)
print(f"Temperature range: {parameter_set.temperature_range_K}")

task = TaskManifest(
    equilibrium_type="VLE",
    calculation_type="bubble_point",
    components=[
        ComponentIdentity(component_id="ethanol", name="ethanol", cas_number="64-17-5"),
        ComponentIdentity(component_id="water", name="water", cas_number="7732-18-5"),
    ],
    conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
    model_name="NRTL",
    parameters=[parameter_set],
)
result = calculate_equilibrium(task)
report = validate_equilibrium_result(result)
print(f"Bubble T = {result.temperature_K:.3f} K, y = {result.points[0].vapor_composition}")
print(f"Validation: {report.overall_status} | residual {report.maximum_equilibrium_residual:.2e}")
print("PIPELINE-OK" if report.overall_status in {"passed", "warning"} else "PIPELINE-FAILED")

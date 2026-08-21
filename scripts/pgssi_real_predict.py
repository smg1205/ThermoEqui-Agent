"""End-to-end real PGSSI prediction with the actual checkpoint."""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()  # 读 .env 的 PGSSI_* 配置

from schemas.domain import ComponentIdentity, TaskManifest, ThermodynamicConditions
from thermo_engine.pgssi_backend import PgssiBackend
from thermo_engine.service import validate_equilibrium_result

ETHANOL = ComponentIdentity(component_id="ethanol", name="Ethanol", cas_number="64-17-5", smiles="CCO", aliases=[])
WATER = ComponentIdentity(component_id="water", name="Water", cas_number="7732-18-5", smiles="O", aliases=[])


def make_task(conditions: ThermodynamicConditions, points: int = 21) -> TaskManifest:
    return TaskManifest(
        equilibrium_type="VLE",
        calculation_type="infinite_dilution_activity",
        components=[ETHANOL, WATER],
        conditions=conditions,
        points=points,
        model_name="PGSSI",
    )


def main() -> None:
    backend = PgssiBackend()

    print("=== 单点模式: 298.15 K ===")
    result = backend.infinite_dilution_activity(make_task(ThermodynamicConditions(temperature_K=298.15)))
    report = validate_equilibrium_result(result)
    for p in result.gamma_infinity:
        print(f"  γ∞(idx {p.solute_index}→{p.solvent_index}) = {p.gamma_infinity:.4f}  ln={p.ln_gamma_infinity:.4f}")
    print(f"  验证: {report.overall_status} | residual={report.maximum_equilibrium_residual:.2e}")

    print("\n=== 曲线模式: 283–373 K, 11 点 ===")
    curve = backend.infinite_dilution_activity(
        make_task(ThermodynamicConditions(temperature_span_K=(283.15, 373.15)), points=11)
    )
    report2 = validate_equilibrium_result(curve)
    direction_counts: dict[tuple[int, int], int] = {}
    for p in curve.gamma_infinity:
        direction_counts[(p.solute_index, p.solvent_index)] = direction_counts.get((p.solute_index, p.solvent_index), 0) + 1
    print(f"  方向: {dict(direction_counts)}")
    print(f"  总点数: {len(curve.gamma_infinity)}")
    ethanol_to_water = [p for p in curve.gamma_infinity if (p.solute_index, p.solvent_index) == (0, 1)]
    print("  乙醇→水 γ∞(T) 前3点:", [(round(p.temperature_K, 1), round(p.gamma_infinity, 4)) for p in ethanol_to_water[:3]])
    print("  乙醇→水 γ∞(T) 后3点:", [(round(p.temperature_K, 1), round(p.gamma_infinity, 4)) for p in ethanol_to_water[-3:]])
    print(f"  验证: {report2.overall_status}")


if __name__ == "__main__":
    main()

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FlashResultCard } from "./FlashResultCard";

const calculation = {
  result: {
    run_id: "run-flash-1",
    task_id: "task-flash-1",
    calculation_type: "tp_flash",
    input_snapshot: {},
    model_name: "Peng-Robinson",
    points: [],
    phases: [
      { phase: "liquid", fraction: 0.911, composition: [0.9744, 0.0198, 0.0058] },
      { phase: "vapor", fraction: 0.089, composition: [0.8688, 0, 0.1312] },
    ],
    temperature_K: 110,
    pressure_kPa: 100,
    vapor_fraction: 0.089,
    phase_state: "two_phase",
    converged: true,
    residual: 0,
    iterations: 3,
    warnings: [],
    backend_version: "test",
    solver_name: "thermo.FlashVL / PRMIX",
    created_at: "2026-08-04T00:00:00Z",
  },
  validation: {
    overall_status: "warning",
    composition_balance: { passed: true, message: "ok" },
    material_balance: { passed: true, message: "ok" },
    equilibrium_residual: { passed: true, message: "ok" },
    convergence: { passed: true, message: "ok" },
    parameter_applicability: { passed: true, message: "ok" },
    warnings: [],
    maximum_equilibrium_residual: 0,
    mean_equilibrium_residual: 0,
    solver_converged: true,
  },
  parameter_sources: [],
  model_recommendations: [],
} as const;

describe("FlashResultCard", () => {
  it("renders TP Flash phase fractions and compositions without VLE points", () => {
    render(<FlashResultCard calculation={calculation as never} />);
    expect(screen.getAllByText(/Peng-Robinson/).length).toBeGreaterThan(0);
    expect(screen.getByText(/91.1%/)).toBeInTheDocument();
    expect(screen.getByText(/8.9%/)).toBeInTheDocument();
    expect(screen.getByText(/0.9744, 0.0198, 0.0058/)).toBeInTheDocument();
    expect(screen.getByText(/0.8688, 0.0000, 0.1312/)).toBeInTheDocument();
    expect(screen.getByText(/110.000 K/)).toBeInTheDocument();
    expect(screen.getByText(/100.000 kPa/)).toBeInTheDocument();
  });
});

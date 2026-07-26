import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Workbench } from "./Workbench";

vi.mock("./VleChart", () => ({ VleChart: () => <div data-testid="vle-chart">chart</div> }));

const response = {
  conversation_id: "conversation-1",
  intent: "EQUILIBRIUM_CALCULATION",
  execution_steps: [
    { phase: "plan", status: "completed", summary: "Structured task created." },
    {
      phase: "execute",
      status: "completed",
      summary: "Deterministic tool completed.",
      tool_name: "phase_equilibrium",
    },
    {
      phase: "validate",
      status: "completed",
      summary: "Validation passed.",
      tool_name: "phase_equilibrium",
    },
    { phase: "respond", status: "completed", summary: "Grounded response created." },
  ],
  answer: "Calculation complete.",
  statements: [{ category: "Calculation", text: "Deterministic result" }],
  task: {
    task_id: "task-1",
    equilibrium_type: "VLE",
    calculation_type: "isobaric_vle",
    components: [
      { component_id: "benzene", name: "Benzene", aliases: [] },
      { component_id: "toluene", name: "Toluene", aliases: [] },
    ],
    conditions: { pressure_kPa: 101.325 },
    composition_basis: "mole_fraction",
    requested_outputs: [],
    validation_requirements: [],
    assumptions: [],
    model_name: "Ideal/Raoult",
    points: 2,
  },
  calculation: {
    result: {
      run_id: "run-1",
      task_id: "task-1",
      calculation_type: "isobaric_vle",
      model_name: "Ideal/Raoult",
      points: [
        {
          temperature_K: 383.7,
          pressure_kPa: 101.325,
          liquid_composition: [0, 1],
          vapor_composition: [0, 1],
          equilibrium_residual: 0,
        },
      ],
      pressure_kPa: 101.325,
      phase_state: "curve",
      converged: true,
      residual: 0,
      iterations: 2,
      warnings: [],
      backend_version: "test",
      solver_name: "solver",
    },
    validation: {
      overall_status: "passed",
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
  },
};

describe("Workbench", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => response }));
  });

  it("loads the engineering workbench and accepts a conversation task", async () => {
    render(<Workbench />);
    expect(screen.getByText("工程对话")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /运行任务/i }));
    await waitFor(() => expect(screen.getByText("Calculation complete.")).toBeInTheDocument());
    expect(screen.getByTestId("vle-chart")).toBeInTheDocument();
    expect(screen.getByText("Benzene / Toluene")).toBeInTheDocument();
    expect(screen.getByText("1 任务理解")).toBeInTheDocument();
    expect(screen.getByText("4 结果验证")).toBeInTheDocument();
  });

  it("keeps API errors in the diagnostic panel", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, json: async () => ({ error: { message: "Missing parameters" } }) }));
    render(<Workbench />);
    fireEvent.click(screen.getByRole("button", { name: /运行任务/i }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Missing parameters"));
  });

  it("edits pressure, creates a new run, and exposes table and downloads", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: true, json: async () => response })
      .mockResolvedValueOnce({ ok: true, json: async () => response.calculation });
    vi.stubGlobal("fetch", fetchMock);
    render(<Workbench />);
    fireEvent.click(screen.getByRole("button", { name: /运行任务/i }));
    await waitFor(() => expect(screen.getByTestId("vle-chart")).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("压力 kPa"), { target: { value: "80" } });
    fireEvent.click(screen.getByRole("button", { name: "按当前条件重新计算" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const rerunBody = JSON.parse(String(fetchMock.mock.calls[1][1]?.body));
    expect(rerunBody.conditions.pressure_kPa).toBe(80);
    fireEvent.click(screen.getByRole("button", { name: "Table" }));
    expect(screen.getByText("T / K")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "下载 JSON" })).toHaveAttribute("href", expect.stringContaining("format=json"));
    expect(screen.getByRole("link", { name: "下载 CSV" })).toHaveAttribute("href", expect.stringContaining("format=csv"));
  });
});

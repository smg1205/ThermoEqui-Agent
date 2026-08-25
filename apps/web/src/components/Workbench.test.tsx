import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
      input_snapshot: {},
      points: [
        {
          temperature_K: 383.7,
          pressure_kPa: 101.325,
          liquid_composition: [0, 1],
          vapor_composition: [0, 1],
          equilibrium_residual: 0,
        },
      ],
      gamma_infinity: [],
      phases: [],
      pressure_kPa: 101.325,
      phase_state: "curve",
      converged: true,
      residual: 0,
      iterations: 2,
      warnings: [],
      backend_version: "test",
      solver_name: "solver",
      created_at: "2026-08-25T00:00:00Z",
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
    model_recommendations: [
      {
        model_name: "NRTL",
        score: 72,
        executable: false,
        reasons: ["Phase support: not matched."],
        exclusions: ["NRTL does not support LLE"],
        breakdown: {
          phase_support_score: 0,
          system_match_score: 23,
          condition_match_score: 15,
          parameter_availability_score: 0,
          evidence_quality_score: 0,
          extrapolation_penalty: 0,
          numerical_risk_penalty: 12,
        },
      },
      {
        model_name: "Ideal/Raoult",
        score: 81,
        executable: true,
        reasons: ["Phase support: matched."],
        exclusions: [],
        breakdown: {
          phase_support_score: 30,
          system_match_score: 24,
          condition_match_score: 15,
          parameter_availability_score: 15,
          evidence_quality_score: 10,
          extrapolation_penalty: 0,
          numerical_risk_penalty: 0,
        },
      },
    ],
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

  it("clears the textarea after a successful task submission", async () => {
    render(<Workbench />);
    const composer = screen.getByLabelText("任务输入");
    fireEvent.change(composer, { target: { value: "第一行\n第二行" } });
    expect(composer).toHaveValue("第一行\n第二行");
    fireEvent.click(screen.getByRole("button", { name: /运行任务/i }));
    await waitFor(() => expect(screen.getByText("Calculation complete.")).toBeInTheDocument());
    expect(composer).toHaveValue("");
  });

  it("keeps API errors in the diagnostic panel", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, json: async () => ({ error: { message: "Missing parameters" } }) }),
    );
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
    expect(screen.getByRole("link", { name: "下载 JSON" })).toHaveAttribute(
      "href",
      expect.stringContaining("format=json"),
    );
    expect(screen.getByRole("link", { name: "下载 CSV" })).toHaveAttribute(
      "href",
      expect.stringContaining("format=csv"),
    );
  });

  it("shows model applicability feedback in the model tab", async () => {
    render(<Workbench />);
    fireEvent.click(screen.getByRole("button", { name: /运行任务/i }));
    await waitFor(() => expect(screen.getByText("Calculation complete.")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Model" }));
    expect(screen.getAllByText("NRTL").length).toBeGreaterThan(0);
    expect(screen.getByText("NRTL does not support LLE")).toBeInTheDocument();
    expect(screen.getByText("模型不可用")).toBeInTheDocument();
    expect(screen.getByText("建议：选择其他支持模型或补充必要参数。")).toBeInTheDocument();
  });

  it("runs a multi-model comparison and shows per-model status and validation", async () => {
    const compareResponse = {
      task: response.task,
      entries: [
        {
          model_name: "NRTL",
          score: 90,
          executable: true,
          result: { ...response.calculation.result, model_name: "NRTL" },
          validation: response.calculation.validation,
          failure: null,
          parameter_sources: [],
          warnings: [],
        },
        {
          model_name: "UNIQUAC",
          score: 86,
          executable: true,
          result: { ...response.calculation.result, model_name: "UNIQUAC" },
          validation: response.calculation.validation,
          failure: null,
          parameter_sources: [],
          warnings: [],
        },
      ],
      executed_count: 2,
      passed_count: 2,
      warning_count: 0,
      failed_count: 0,
      summary: "对比 2 个可执行模型：2 通过、0 警告、0 失败。",
    };
    const fetchMock = vi.fn().mockImplementation((url: string) =>
      Promise.resolve({
        ok: true,
        json: async () => (url.includes("/api/calculations/compare") ? compareResponse : response),
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<Workbench />);
    fireEvent.click(screen.getByRole("button", { name: /运行任务/i }));
    await waitFor(() => expect(screen.getByText("Calculation complete.")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "模型对比" }));
    await waitFor(() => expect(screen.getByTestId("comparison-status")).toBeInTheDocument());
    expect(screen.getByText("对比 2 个可执行模型：2 通过、0 警告、0 失败。")).toBeInTheDocument();
    expect(screen.getByText("已按当前条件对比 2 个可执行模型。")).toBeInTheDocument();
    const status = screen.getByTestId("comparison-status");
    expect(within(status).getAllByText("passed")).toHaveLength(2);
    expect(within(status).getAllByText("NRTL").length).toBeGreaterThan(0);
    expect(within(status).getAllByText("UNIQUAC").length).toBeGreaterThan(0);
  });
});

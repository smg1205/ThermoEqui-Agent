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

  it("runs a multi-model phase diagram and shows per-model status, failure, and warnings", async () => {
    const phaseDiagramResponse = {
      diagram_type: "TXY",
      entries: [
        {
          model_name: "NRTL",
          status: "passed",
          executable: true,
          result: { ...response.calculation.result, model_name: "NRTL" },
          failure: null,
          warnings: [],
        },
        {
          model_name: "UNIQUAC",
          status: "warning",
          executable: true,
          result: { ...response.calculation.result, model_name: "UNIQUAC" },
          failure: null,
          warnings: ["UNIQUAC 预测性模型，需工程复核。"],
        },
        {
          model_name: "SRK",
          status: "failed",
          executable: false,
          result: null,
          failure: {
            failure_type: "missing_parameters",
            message: "SRK binary parameters are unavailable.",
            recovery_action: "Import reviewed SRK kij parameters.",
            details: {},
          },
          warnings: ["Missing reviewed SRK kij."],
        },
        {
          model_name: "PGSSI",
          status: "unsupported",
          executable: false,
          result: null,
          failure: {
            failure_type: "unsupported_model",
            message: "PGSSI does not support isobaric_vle.",
            recovery_action: "Choose another model.",
            details: {},
          },
          warnings: [],
        },
      ],
      total_models: 4,
      passed_count: 1,
      warning_count: 1,
      failed_count: 1,
      unsupported_count: 1,
      summary: "Compared 4 models: 1 passed, 1 warnings, 1 failed, 1 unsupported.",
    };
    const fetchMock = vi.fn().mockImplementation((url: string) =>
      Promise.resolve({
        ok: true,
        json: async () => (url.includes("/api/calculations/phase-diagram") ? phaseDiagramResponse : response),
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<Workbench />);
    fireEvent.click(screen.getByRole("button", { name: /运行任务/i }));
    await waitFor(() => expect(screen.getByText("Calculation complete.")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "多模型相图" }));
    await waitFor(() => expect(screen.getByTestId("phase-diagram-status")).toBeInTheDocument());
    expect(screen.getByText(/已按当前条件生成多模型相图/)).toBeInTheDocument();
    const status = screen.getByTestId("phase-diagram-status");
    expect(within(status).getAllByText("passed")).toHaveLength(1);
    expect(within(status).getAllByText("warning")).toHaveLength(1);
    expect(within(status).getAllByText("failed")).toHaveLength(1);
    expect(within(status).getAllByText("unsupported")).toHaveLength(1);
    expect(within(status).getByText("SRK binary parameters are unavailable.")).toBeInTheDocument();
    expect(within(status).getByText("Missing reviewed SRK kij.")).toBeInTheDocument();
    expect(within(status).getByText("PGSSI does not support isobaric_vle.")).toBeInTheDocument();

    const select = screen.getByTestId("phase-diagram-model-select");
    expect(within(select).getAllByRole("checkbox")).toHaveLength(4);
    expect(within(select).getByLabelText("选择 NRTL")).toBeChecked();
    expect(within(select).getByLabelText("选择 UNIQUAC")).toBeChecked();
    expect(within(select).getByLabelText("选择 SRK")).toBeDisabled();
    expect(within(select).getByLabelText("选择 PGSSI")).toBeDisabled();
    expect(screen.getByTestId("phase-diagram-selection-count")).toHaveTextContent("已选 2 / 2 个可绘制模型");

    fireEvent.click(within(select).getByLabelText("选择 UNIQUAC"));
    expect(screen.getByTestId("phase-diagram-selection-count")).toHaveTextContent("已选 1 / 2 个可绘制模型");
    expect(screen.queryByText("请至少选择一个模型以显示相图曲线。")).not.toBeInTheDocument();

    fireEvent.click(within(select).getByLabelText("选择 NRTL"));
    expect(screen.getByText("请至少选择一个模型以显示相图曲线。")).toBeInTheDocument();
  });
});

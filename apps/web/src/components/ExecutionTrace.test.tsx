import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ExecutionTrace } from "./ExecutionTrace";

describe("ExecutionTrace", () => {
  it("renders all agent phases and maps waiting and success states", () => {
    render(
      <ExecutionTrace
        loading={false}
        steps={[
          { phase: "plan", status: "completed", summary: "Task structured." },
          { phase: "execute", status: "completed", summary: "Tool executed.", tool_name: "phase_equilibrium" },
        ]}
      />,
    );

    expect(screen.getByText("Plan")).toBeInTheDocument();
    expect(screen.getByText("Execute")).toBeInTheDocument();
    expect(screen.getByText("Validate")).toBeInTheDocument();
    expect(screen.getByText("Respond")).toBeInTheDocument();
    expect(screen.getAllByText("完成")).toHaveLength(2);
    expect(screen.getAllByText("等待")).toHaveLength(2);
  });

  it("shows the next missing phase as running while loading", () => {
    render(
      <ExecutionTrace
        loading={true}
        steps={[{ phase: "plan", status: "completed", summary: "Task structured." }]}
      />,
    );

    expect(screen.getByText("运行中")).toBeInTheDocument();
    expect(screen.getAllByText("等待该阶段开始。").length).toBeGreaterThan(0);
  });
});

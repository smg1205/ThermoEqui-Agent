import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { VleChart } from "./VleChart";

interface PlotProps {
  data: Array<{ y: number[]; name?: string }>;
  layout: { yaxis: { title: { text: string } }; height: number };
}

vi.mock("next/dynamic", () => ({
  default: () => (props: PlotProps) => <pre data-testid="plot-props">{JSON.stringify(props)}</pre>,
}));

const points = [
  { temperature_K: 350, pressure_kPa: 150, liquid_composition: [0.2, 0.8], vapor_composition: [0.4, 0.6], equilibrium_residual: 0 },
  { temperature_K: 350, pressure_kPa: 100, liquid_composition: [0.8, 0.2], vapor_composition: [0.9, 0.1], equilibrium_residual: 0 },
];

describe("VleChart", () => {
  it("uses pressure as the ordinate for isothermal P-x-y results", () => {
    render(<VleChart points={points} model="Ideal/Raoult" temperature={350} calculationType="isothermal_vle" />);
    const props = JSON.parse(screen.getByTestId("plot-props").textContent ?? "{}") as PlotProps;
    expect(props.data[0].y).toEqual([150, 100]);
    expect(props.layout.yaxis.title.text).toBe("压力 / kPa");
    expect(props.layout.height).toBe(448);
  });

  it("renders liquid and vapor traces per model when series is provided", () => {
    render(
      <VleChart
        series={[
          { model_name: "NRTL", points },
          { model_name: "UNIQUAC", points },
        ]}
        temperature={350}
        calculationType="isothermal_vle"
      />,
    );
    const props = JSON.parse(screen.getByTestId("plot-props").textContent ?? "{}") as PlotProps;
    expect(props.data).toHaveLength(4);
    expect(props.data.map((trace) => trace.name)).toEqual([
      "NRTL liquid",
      "NRTL vapor",
      "UNIQUAC liquid",
      "UNIQUAC vapor",
    ]);
    expect(props.data[0].y).toEqual([150, 100]);
  });

  it("draws only passed and warning entries and uses pressure for PXY", () => {
    render(
      <VleChart
        entries={[
          { model_name: "NRTL", status: "passed", executable: true, result: { points }, failure: null, warnings: [] },
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
            model_name: "UNIQUAC",
            status: "warning",
            executable: true,
            result: { points },
            failure: null,
            warnings: [],
          },
        ]}
        diagramType="PXY"
        temperature={350}
        calculationType="isothermal_vle"
      />,
    );
    const props = JSON.parse(screen.getByTestId("plot-props").textContent ?? "{}") as PlotProps;
    expect(props.data).toHaveLength(4);
    expect(props.data.map((trace) => trace.name)).toEqual([
      "NRTL liquid",
      "NRTL vapor",
      "UNIQUAC liquid",
      "UNIQUAC vapor",
    ]);
    expect(props.data[0].y).toEqual([150, 100]);
  });

  it("uses temperature as the ordinate for TXY entries", () => {
    render(
      <VleChart
        entries={[
          { model_name: "NRTL", status: "passed", executable: true, result: { points }, failure: null, warnings: [] },
        ]}
        diagramType="TXY"
        pressure={101.325}
        calculationType="isobaric_vle"
      />,
    );
    const props = JSON.parse(screen.getByTestId("plot-props").textContent ?? "{}") as PlotProps;
    expect(props.data).toHaveLength(2);
    expect(props.data[0].y).toEqual([350, 350]);
    expect(props.layout.yaxis.title.text).toBe("温度 / K");
  });

  it("shows the empty state when every entry is failed or unsupported", () => {
    render(
      <VleChart
        entries={[
          { model_name: "SRK", status: "failed", executable: false, result: null, failure: null, warnings: [] },
          {
            model_name: "PGSSI",
            status: "unsupported",
            executable: false,
            result: null,
            failure: null,
            warnings: [],
          },
        ]}
        diagramType="TXY"
        calculationType="isobaric_vle"
      />,
    );
    expect(screen.getByText("当前结果不包含相图数据。")).toBeInTheDocument();
  });
});

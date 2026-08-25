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
});

"use client";

import dynamic from "next/dynamic";
import type { EquilibriumPoint } from "@/lib/types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export function VleChart({
  points,
  pressure,
  temperature,
  model,
  calculationType,
}: {
  points: EquilibriumPoint[];
  pressure?: number | null;
  temperature?: number | null;
  model: string;
  calculationType: string;
}) {
  if (!points.length) return <div className="empty-chart">当前结果不包含相图数据。</div>;
  const liquid = points.map((point) => point.liquid_composition[0]);
  const vapor = points.map((point) => point.vapor_composition[0]);
  const isIsothermal = calculationType === "isothermal_vle";
  const ordinate = points.map((point) => (isIsothermal ? point.pressure_kPa : point.temperature_K));
  const condition = isIsothermal ? `${temperature?.toFixed(3) ?? "--"} K` : `${pressure?.toFixed(3) ?? "--"} kPa`;

  return (
    <div className="chart-shell" aria-label="VLE 相图" data-testid="vle-chart">
      <Plot
        data={[
          { x: liquid, y: ordinate, type: "scatter", mode: "lines+markers", name: "液相 x1", line: { color: "#2dd4bf", width: 3 } },
          { x: vapor, y: ordinate, type: "scatter", mode: "lines+markers", name: "气相 y1", line: { color: "#fb923c", width: 3 } },
        ]}
        layout={{
          autosize: true,
          height: 448,
          margin: { l: 56, r: 18, t: 36, b: 48 },
          title: { text: `${model} · ${condition}` },
          xaxis: { title: { text: "摩尔分数 x1 / y1" }, range: [0, 1], gridcolor: "#dbe4ec" },
          yaxis: { title: { text: isIsothermal ? "压力 / kPa" : "温度 / K" }, gridcolor: "#dbe4ec" },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "#f8fafc",
          font: { family: "Inter, system-ui, sans-serif", color: "#243244" },
          legend: { orientation: "h", y: 1.08 },
        }}
        config={{ responsive: true, displaylogo: false, toImageButtonOptions: { filename: "thermoequi-vle" } }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

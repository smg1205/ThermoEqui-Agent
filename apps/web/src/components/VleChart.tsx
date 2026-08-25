"use client";

import dynamic from "next/dynamic";
import type { EquilibriumPoint, ModelSeries } from "@/lib/types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const MODEL_COLORS = [
  "#2563eb",
  "#16a34a",
  "#d97706",
  "#dc2626",
  "#7c3aed",
  "#0891b2",
  "#db2777",
  "#65a30d",
];

interface VleTrace {
  x: number[];
  y: number[];
  type: "scatter";
  mode: "lines+markers";
  name: string;
  line: { color: string; width: number; dash?: "dash" };
}

function seriesTraces(series: ModelSeries, index: number, isIsothermal: boolean): VleTrace[] {
  const color = MODEL_COLORS[index % MODEL_COLORS.length];
  const liquid = series.points.map((point) => point.liquid_composition[0]);
  const vapor = series.points.map((point) => point.vapor_composition[0]);
  const ordinate = series.points.map((point) => (isIsothermal ? point.pressure_kPa : point.temperature_K));
  return [
    {
      x: liquid,
      y: ordinate,
      type: "scatter",
      mode: "lines+markers",
      name: `${series.model_name} liquid`,
      line: { color, width: 3 },
    },
    {
      x: vapor,
      y: ordinate,
      type: "scatter",
      mode: "lines+markers",
      name: `${series.model_name} vapor`,
      line: { color, width: 2, dash: "dash" },
    },
  ];
}

function singleTraces(points: EquilibriumPoint[], model: string, isIsothermal: boolean): VleTrace[] {
  const liquid = points.map((point) => point.liquid_composition[0]);
  const vapor = points.map((point) => point.vapor_composition[0]);
  const ordinate = points.map((point) => (isIsothermal ? point.pressure_kPa : point.temperature_K));
  return [
    {
      x: liquid,
      y: ordinate,
      type: "scatter",
      mode: "lines+markers",
      name: "液相 x1",
      line: { color: "#2dd4bf", width: 3 },
    },
    {
      x: vapor,
      y: ordinate,
      type: "scatter",
      mode: "lines+markers",
      name: "气相 y1",
      line: { color: "#fb923c", width: 3 },
    },
  ];
}

export function VleChart({
  points,
  pressure,
  temperature,
  model,
  calculationType,
  series,
}: {
  points?: EquilibriumPoint[];
  pressure?: number | null;
  temperature?: number | null;
  model?: string;
  calculationType: string;
  series?: ModelSeries[];
}) {
  const hasSeries = series !== undefined && series.length > 0;
  if (!hasSeries && !(points && points.length > 0)) {
    return <div className="empty-chart">当前结果不包含相图数据。</div>;
  }
  const isIsothermal = calculationType === "isothermal_vle";
  const condition = isIsothermal ? `${temperature?.toFixed(3) ?? "--"} K` : `${pressure?.toFixed(3) ?? "--"} kPa`;
  const data = hasSeries
    ? series!.flatMap((item, index) => seriesTraces(item, index, isIsothermal))
    : singleTraces(points ?? [], model ?? "Model", isIsothermal);
  const title = hasSeries ? condition : `${model ?? "Model"} · ${condition}`;

  return (
    <div className="chart-shell" aria-label="VLE 相图" data-testid="vle-chart">
      <Plot
        data={data}
        layout={{
          autosize: true,
          height: 448,
          margin: { l: 56, r: 18, t: 36, b: 48 },
          title: { text: title },
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

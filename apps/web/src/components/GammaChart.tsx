"use client";

import dynamic from "next/dynamic";
import type { GammaInfinityPoint } from "@/lib/types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export function GammaChart({
  points,
  model,
  components,
}: {
  points: GammaInfinityPoint[];
  model: string;
  components?: { name?: string }[] | null;
}) {
  if (!points.length) return <div className="empty-chart">当前结果不包含 γ∞ 数据。</div>;

  const byDirection = new Map<string, { x: number[]; y: number[] }>();
  for (const point of points) {
    const solute = components?.[point.solute_index]?.name ?? `组分${point.solute_index}`;
    const solvent = components?.[point.solvent_index]?.name ?? `组分${point.solvent_index}`;
    const key = `${solute}→${solvent}`;
    if (!byDirection.has(key)) byDirection.set(key, { x: [], y: [] });
    byDirection.get(key)!.x.push(point.temperature_K);
    byDirection.get(key)!.y.push(point.gamma_infinity);
  }

  const traces = [...byDirection.entries()].map(([name, data], index) => ({
    x: data.x,
    y: data.y,
    type: "scatter" as const,
    mode: "lines+markers" as const,
    name,
    line: { color: index % 2 === 0 ? "#2dd4bf" : "#fb923c", width: 3 },
  }));

  return (
    <div className="chart-shell" aria-label="γ∞-T 曲线" data-testid="gamma-chart">
      <Plot
        data={traces}
        layout={{
          autosize: true,
          height: 448,
          margin: { l: 56, r: 18, t: 36, b: 48 },
          title: { text: `${model} · 无限稀释活度系数` },
          xaxis: { title: { text: "温度 / K" }, gridcolor: "#dbe4ec" },
          yaxis: { title: { text: "γ∞" }, gridcolor: "#dbe4ec" },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "#f8fafc",
          font: { family: "Inter, system-ui, sans-serif", color: "#243244" },
          legend: { orientation: "h", y: 1.08 },
        }}
        config={{ responsive: true, displaylogo: false, toImageButtonOptions: { filename: "thermoequi-gamma" } }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

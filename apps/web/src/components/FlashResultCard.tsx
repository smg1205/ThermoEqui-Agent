"use client";

import type { CalculationEnvelope, PhaseResult } from "@/lib/types";

function percent(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return "--";
  return `${(value * 100).toFixed(1)}%`;
}

function numberLabel(value?: number | null, digits = 3): string {
  if (value == null || !Number.isFinite(value)) return "--";
  return value.toFixed(digits);
}

function compositionLabel(phase?: PhaseResult): string {
  if (!phase?.composition?.length) return "--";
  return phase.composition.map((item) => item.toFixed(4)).join(", ");
}

export function FlashResultCard({ calculation }: { calculation: CalculationEnvelope }) {
  const liquidPhase = calculation.result.phases.find((item) => item.phase === "liquid");
  const vaporPhase = calculation.result.phases.find((item) => item.phase === "vapor");

  return (
    <section className="result-panel flash-result-card" aria-label="TP Flash result summary">
      <div className="panel-heading">
        <h3>TP Flash 结果</h3>
        <span>{calculation.result.model_name}</span>
      </div>
      <div className="flash-result-grid">
        <article>
          <span>液相比例</span>
          <strong>{percent(liquidPhase?.fraction ?? (calculation.result.vapor_fraction != null ? 1 - calculation.result.vapor_fraction : null))}</strong>
          <p>相态: {liquidPhase ? "液相" : "--"}</p>
        </article>
        <article>
          <span>气相比例</span>
          <strong>{percent(vaporPhase?.fraction ?? calculation.result.vapor_fraction)}</strong>
          <p>汽化分率 β: {numberLabel(calculation.result.vapor_fraction, 4)}</p>
        </article>
        <article>
          <span>液相组成</span>
          <strong>{compositionLabel(liquidPhase)}</strong>
          <p>mol fraction</p>
        </article>
        <article>
          <span>气相组成</span>
          <strong>{compositionLabel(vaporPhase)}</strong>
          <p>mol fraction</p>
        </article>
      </div>
      <div className="flash-result-meta">
        <div>
          <span>温度</span>
          <strong>{numberLabel(calculation.result.temperature_K)} K</strong>
        </div>
        <div>
          <span>压力</span>
          <strong>{numberLabel(calculation.result.pressure_kPa)} kPa</strong>
        </div>
        <div>
          <span>模型</span>
          <strong>{calculation.result.model_name}</strong>
        </div>
        <div>
          <span>求解器</span>
          <strong>{calculation.result.solver_name}</strong>
        </div>
      </div>
    </section>
  );
}

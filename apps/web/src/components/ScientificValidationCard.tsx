"use client";

import type { CalculationEnvelope } from "@/lib/types";

function consistencyLabel(envelope: CalculationEnvelope): string {
  if (envelope.validation.equilibrium_residual.passed && envelope.validation.composition_balance.passed) {
    return "通过";
  }
  if (envelope.validation.overall_status === "warning") {
    return "复核";
  }
  return "检查";
}

export function ScientificValidationCard({ calculation }: { calculation: CalculationEnvelope }) {
  return (
    <section className="scientific-validation-card" aria-label="Scientific validation summary">
      <div className="scientific-validation-header">
        <div>
          <p className="eyebrow">Result Summary</p>
          <h3>科学验证摘要</h3>
        </div>
        <span className={`scientific-validation-badge ${calculation.validation.overall_status}`}>
          {calculation.validation.overall_status}
        </span>
      </div>
      <div className="scientific-validation-grid">
        <article>
          <span>收敛</span>
          <strong>{calculation.result.converged ? "已收敛" : "未收敛"}</strong>
          <p>{calculation.validation.convergence.message}</p>
        </article>
        <article>
          <span>一致性</span>
          <strong>{consistencyLabel(calculation)}</strong>
          <p>{calculation.validation.equilibrium_residual.message}</p>
        </article>
        <article>
          <span>模型</span>
          <strong>{calculation.result.model_name}</strong>
          <p>{calculation.result.solver_name}</p>
        </article>
        <article>
          <span>运行</span>
          <strong>{calculation.result.iterations} iter</strong>
          <p>{calculation.result.run_id.slice(0, 8)}</p>
        </article>
      </div>
    </section>
  );
}

"use client";

import type { AgentStep } from "@/lib/types";

type RuntimeStage = {
  id: "understanding" | "selection" | "engine" | "validation";
  title: string;
  summary: string;
  status: "waiting" | "running" | "success" | "error";
};

function deriveStatus(hasStep: boolean, failed: boolean, running: boolean): RuntimeStage["status"] {
  if (failed) return "error";
  if (running) return "running";
  if (hasStep) return "success";
  return "waiting";
}

function statusLabel(status: RuntimeStage["status"]): string {
  if (status === "running") return "运行中";
  if (status === "success") return "完成";
  if (status === "error") return "异常";
  return "等待";
}

function compactSummary(summary: string): string {
  const cleaned = summary.replace(/\s+/g, " ").trim();
  return cleaned.length > 22 ? `${cleaned.slice(0, 22)}...` : cleaned;
}

export function AgentRuntime({
  steps,
  loading,
  modelName,
}: {
  steps: AgentStep[];
  loading: boolean;
  modelName?: string | null;
}) {
  const plan = steps.find((step) => step.phase === "plan");
  const execute = steps.find((step) => step.phase === "execute");
  const validate = steps.find((step) => step.phase === "validate");
  const respond = steps.find((step) => step.phase === "respond");

  const stages: RuntimeStage[] = [
    {
      id: "understanding",
      title: "1 任务理解",
      summary: compactSummary(plan?.summary ?? "解析工程任务"),
      status: deriveStatus(Boolean(plan), plan?.status === "failed" || plan?.status === "blocked", loading && !plan),
    },
    {
      id: "selection",
      title: "2 模型选择",
      summary: compactSummary(execute?.tool_name ? `${execute.tool_name} / ${modelName ?? "待路由"}` : `模型 ${modelName ?? "待路由"}`),
      status: deriveStatus(
        Boolean(execute),
        execute?.status === "failed" || execute?.status === "blocked",
        loading && Boolean(plan) && !execute,
      ),
    },
    {
      id: "engine",
      title: "3 热力学计算",
      summary: compactSummary(execute?.summary ?? "调用确定性计算内核"),
      status: deriveStatus(
        Boolean(execute),
        execute?.status === "failed" || execute?.status === "blocked",
        loading && Boolean(execute) && !validate,
      ),
    },
    {
      id: "validation",
      title: "4 结果验证",
      summary: compactSummary(validate?.summary ?? respond?.summary ?? "检查收敛与一致性"),
      status: deriveStatus(
        Boolean(validate || respond),
        validate?.status === "failed" ||
          validate?.status === "blocked" ||
          respond?.status === "failed" ||
          respond?.status === "blocked",
        loading && Boolean(validate) && !respond,
      ),
    },
  ];

  return (
    <section className="runtime-panel" aria-label="Agent runtime panel">
      <div className="runtime-panel-header">
        <div>
          <p className="eyebrow">Agent Runtime</p>
          <h3>智能体执行流水线</h3>
        </div>
      </div>
      <div className="runtime-flow">
        {stages.map((stage, index) => (
          <div className="runtime-flow-item" key={stage.id}>
            <article className={`runtime-card ${stage.status}`}>
              <div className="runtime-card-topline">
                <strong>{stage.title}</strong>
                <span className={`runtime-state ${stage.status}`}>{statusLabel(stage.status)}</span>
              </div>
              <p className="runtime-card-summary" title={stage.summary}>
                {stage.summary}
              </p>
            </article>
            {index < stages.length - 1 ? <span className="runtime-arrow">→</span> : null}
          </div>
        ))}
      </div>
    </section>
  );
}

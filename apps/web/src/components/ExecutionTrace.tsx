"use client";

import type { AgentStep } from "@/lib/types";

type Phase = "plan" | "execute" | "validate" | "respond";
type TraceStatus = "waiting" | "running" | "success" | "error";

const phaseOrder: Phase[] = ["plan", "execute", "validate", "respond"];

const phaseLabels: Record<Phase, string> = {
  plan: "Plan",
  execute: "Execute",
  validate: "Validate",
  respond: "Respond",
};

const phaseDescriptions: Record<Phase, string> = {
  plan: "解析工程问题、结构化任务并规划计算路径。",
  execute: "调用受约束工具与确定性求解器执行计算。",
  validate: "检查收敛性、相平衡残差与物理有效性。",
  respond: "汇总结论、证据和后续建议并返回结果。",
};

function getTraceStatus(
  phase: Phase,
  stepsByPhase: Map<Phase, AgentStep>,
  runningPhase: Phase | null,
): TraceStatus {
  const step = stepsByPhase.get(phase);
  if (step) {
    if (step.status === "failed" || step.status === "blocked") return "error";
    return "success";
  }
  if (runningPhase === phase) return "running";
  return "waiting";
}

function getRunningPhase(stepsByPhase: Map<Phase, AgentStep>, loading: boolean): Phase | null {
  if (!loading) return null;
  for (const phase of phaseOrder) {
    if (!stepsByPhase.has(phase)) return phase;
  }
  return phaseOrder[phaseOrder.length - 1];
}

function statusLabel(status: TraceStatus): string {
  if (status === "running") return "运行中";
  if (status === "success") return "完成";
  if (status === "error") return "异常";
  return "等待";
}

export function ExecutionTrace({ steps, loading }: { steps: AgentStep[]; loading: boolean }) {
  const stepsByPhase = new Map<Phase, AgentStep>();
  for (const step of steps) {
    stepsByPhase.set(step.phase, step);
  }
  const runningPhase = getRunningPhase(stepsByPhase, loading);

  return (
    <section className="execution-trace-card" aria-label="Agent execution trace">
      <div className="execution-trace-header">
        <div>
          <p className="eyebrow">Agent Workflow</p>
          <h3>执行轨迹</h3>
        </div>
        <span className="execution-trace-badge">Plan -&gt; Execute -&gt; Validate -&gt; Respond</span>
      </div>
      <div className="execution-trace-summary">
        Agent 正在按固定工程链路完成任务规划、工具调用、确定性计算与结果验证。
      </div>
      <div className="execution-trace-timeline">
        {phaseOrder.map((phase) => {
          const step = stepsByPhase.get(phase);
          const status = getTraceStatus(phase, stepsByPhase, runningPhase);
          return (
            <article className={`trace-step ${status}`} key={phase}>
              <div className="trace-rail" aria-hidden="true">
                <span className={`trace-node ${status}`} />
                {phase !== phaseOrder[phaseOrder.length - 1] && <span className="trace-line" />}
              </div>
              <div className="trace-body">
                <div className="trace-topline">
                  <strong>{phaseLabels[phase]}</strong>
                  <span className={`trace-status ${status}`}>{statusLabel(status)}</span>
                </div>
                <p className="trace-description">{phaseDescriptions[phase]}</p>
                {step?.tool_name ? (
                  <code>{step.tool_name}</code>
                ) : (
                  <span className="trace-tool-placeholder">Agent orchestration</span>
                )}
                <p>{step?.summary ?? "等待该阶段开始。"}</p>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

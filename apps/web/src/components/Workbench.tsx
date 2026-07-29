"use client";

import { FormEvent, useMemo, useState } from "react";
import { exportUrl, rerunTask, sendChat } from "@/lib/api";
import type { AgentStep, CalculationEnvelope, ChatResponse, TaskManifest } from "@/lib/types";
import { AgentRuntime } from "./AgentRuntime";
import { ScientificValidationCard } from "./ScientificValidationCard";
import { VleChart } from "./VleChart";

type DetailTab = "table" | "model" | "parameters" | "validation" | "runs";

const examples = [
  { code: "T01", title: "二元 VLE 曲线", prompt: "计算苯-甲苯在 101.325 kPa 下的 T-x-y 曲线", tag: "VLE" },
  { code: "M02", title: "模型比较", prompt: "NRTL 和 Peng-Robinson 有什么区别？", tag: "Model" },
  { code: "F03", title: "Flash 任务", prompt: "计算给定进料在指定温压下的 Flash 平衡", tag: "Flash" },
];

function riskLabel(calculation?: CalculationEnvelope): string {
  if (!calculation) return "待处理";
  if (calculation.validation.overall_status === "failed") return "高风险";
  if (calculation.validation.overall_status === "warning") return "需复核";
  return "低风险";
}

function agentStatus(steps: AgentStep[], loading: boolean): string {
  if (steps.some((step) => step.status === "failed" || step.status === "blocked")) return "异常";
  if (loading) return "运行中";
  if (steps.length > 0) return "已完成";
  return "待机";
}

export function Workbench() {
  const [messages, setMessages] = useState<Array<{ role: "user" | "agent"; text: string }>>([
    {
      role: "agent",
      text: "请描述体系与工况。我会先结构化任务，再调用确定性内核完成热力学计算与验证。",
    },
  ]);
  const [input, setInput] = useState(examples[0].prompt);
  const [conversationId, setConversationId] = useState<string>();
  const [task, setTask] = useState<TaskManifest>();
  const [calculation, setCalculation] = useState<CalculationEnvelope>();
  const [runs, setRuns] = useState<CalculationEnvelope[]>([]);
  const [executionSteps, setExecutionSteps] = useState<AgentStep[]>([]);
  const [activeTab, setActiveTab] = useState<DetailTab>("table");
  const [loading, setLoading] = useState(false);
  const [diagnostic, setDiagnostic] = useState<string>();

  const risk = useMemo(() => riskLabel(calculation), [calculation]);
  const agentRuntimeStatus = useMemo(() => agentStatus(executionSteps, loading), [executionSteps, loading]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!input.trim() || loading) return;
    const message = input.trim();
    setMessages((current) => [...current, { role: "user", text: message }]);
    setLoading(true);
    setDiagnostic(undefined);
    try {
      const response: ChatResponse = await sendChat(message, conversationId);
      setConversationId(response.conversation_id);
      setExecutionSteps(response.execution_steps);
      setMessages((current) => [...current, { role: "agent", text: response.answer }]);
      if (response.task) setTask(response.task);
      const nextCalculation = response.calculation;
      if (nextCalculation) {
        setCalculation(nextCalculation);
        setRuns((current) => [...current, nextCalculation]);
        setActiveTab("table");
      } else if (response.statements.some((item) => item.category === "Warning")) {
        setDiagnostic(response.statements.map((item) => item.text).join(" "));
      }
    } catch (error) {
      setDiagnostic(error instanceof Error ? error.message : "未知错误");
    } finally {
      setLoading(false);
    }
  }

  async function rerun() {
    if (!task || loading) return;
    setLoading(true);
    setDiagnostic(undefined);
    try {
      const next = await rerunTask(task);
      setCalculation(next);
      setRuns((current) => [...current, next]);
      setMessages((current) => [...current, { role: "agent", text: "已按当前条件重新创建确定性计算运行。" }]);
    } catch (error) {
      setDiagnostic(error instanceof Error ? error.message : "重新计算失败");
    } finally {
      setLoading(false);
    }
  }

  function updatePressure(value: string) {
    if (!task) return;
    const pressure = Number(value);
    setTask({ ...task, conditions: { ...task.conditions, pressure_kPa: Number.isFinite(pressure) ? pressure : null } });
  }

  function updateTemperature(value: string) {
    if (!task) return;
    const temperature = Number(value);
    setTask({ ...task, conditions: { ...task.conditions, temperature_K: Number.isFinite(temperature) ? temperature : null } });
  }

  function updateComposition(value: string) {
    if (!task) return;
    const parsed = value.split(",").map((item) => Number(item.trim()));
    const composition = parsed.length && parsed.every(Number.isFinite) ? parsed : null;
    const field =
      task.calculation_type === "tp_flash"
        ? "feed_composition"
        : task.calculation_type === "dew_point"
          ? "vapor_composition"
          : "liquid_composition";
    setTask({ ...task, conditions: { ...task.conditions, [field]: composition } });
  }

  function updateModel(value: string) {
    if (task) setTask({ ...task, model_name: value });
  }

  const tabs: Array<{ id: DetailTab; label: string }> = [
    { id: "table", label: "Table" },
    { id: "model", label: "Model" },
    { id: "parameters", label: "Parameters" },
    { id: "validation", label: "Validation" },
    { id: "runs", label: "Runs" },
  ];

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <span className="brand-mark">TE</span>
          <div>
            <h1>ThermoEqui-Agent</h1>
            <p>AI4SCIENCE ENGINEERING CONTROL CONSOLE</p>
          </div>
        </div>
        <div className="topbar-status-grid">
          <article className="topbar-status-card">
            <label>Agent 状态</label>
            <strong>{agentRuntimeStatus}</strong>
          </article>
          <article className="topbar-status-card">
            <label>Engine 状态</label>
            <strong>{loading ? "运行中" : "在线"}</strong>
          </article>
          <article className="topbar-status-card">
            <label>Validation 状态</label>
            <strong>{calculation?.validation.overall_status ?? "待处理"}</strong>
          </article>
        </div>
      </header>

      <section className="top-workspace">
        <aside className="sidebar">
          <div className="sidebar-section-title">
            <p className="eyebrow">AI Chemistry Tasks</p>
            <h2>任务模板</h2>
          </div>
          <div className="task-template-grid">
            {examples.map((example) => (
              <button key={example.code} className="task-template-card" onClick={() => setInput(example.prompt)}>
                <div className="task-template-topline">
                  <span>{example.code}</span>
                  <small>{example.tag}</small>
                </div>
                <strong>{example.title}</strong>
                <p>{example.prompt}</p>
              </button>
            ))}
          </div>
          <div className="scope-card">
            <p className="eyebrow">当前边界</p>
            <strong>非电解质分子体系</strong>
            <small>VLE / Flash / 泡露点 / 共沸搜索</small>
          </div>
        </aside>

        <div className="center-column">
          <section className="conversation">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Conversation</p>
                <h2>工程对话</h2>
              </div>
              <span>{conversationId ? "会话进行中" : "新会话"}</span>
            </div>
            <div className="conversation-body">
              <div className="messages" aria-live="polite">
                {messages.map((message, index) => (
                  <article className={message.role} key={`${message.role}-${index}`}>
                    <label>{message.role === "agent" ? "AGENT" : "YOU"}</label>
                    <p>{message.text}</p>
                  </article>
                ))}
                {loading && (
                  <article className="agent">
                    <label>ENGINE</label>
                    <p>正在理解任务、调度模型、调用热力学计算并执行验证...</p>
                  </article>
                )}
              </div>
              <form onSubmit={submit} className="composer composer-sticky">
                <textarea aria-label="任务输入" value={input} onChange={(event) => setInput(event.target.value)} />
                <button disabled={loading}>运行任务</button>
              </form>
            </div>
          </section>

          <AgentRuntime
            steps={executionSteps}
            loading={loading}
            modelName={calculation?.result.model_name ?? task?.model_name}
          />

          <section className="results results-inline">
            <div className="results-header">
              <div>
                <p className="eyebrow">Scientific Output</p>
                <h2>计算结果</h2>
              </div>
              {calculation && (
                <div className="results-actions">
                  <a href={exportUrl(calculation.result.run_id, "json")}>下载 JSON</a>
                  <a href={exportUrl(calculation.result.run_id, "csv")}>下载 CSV</a>
                </div>
              )}
            </div>

            <div className="results-stack">
              {calculation && <ScientificValidationCard calculation={calculation} />}
              {calculation && (
                <section className="result-panel chart-panel">
                  <div className="panel-heading">
                    <h3>相平衡图</h3>
                    <span>{calculation.result.model_name}</span>
                  </div>
                  <VleChart
                    points={calculation.result.points}
                    pressure={calculation.result.pressure_kPa}
                    temperature={calculation.result.temperature_K}
                    model={calculation.result.model_name}
                    calculationType={calculation.result.calculation_type}
                  />
                </section>
              )}
            </div>

            <nav aria-label="结果视图">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  className={activeTab === tab.id ? "active" : ""}
                  onClick={() => setActiveTab(tab.id)}
                >
                  {tab.label}
                </button>
              ))}
            </nav>
            <div className="result-body">
              {!calculation && (
                <div className="empty result-empty">
                  <span>◎</span>
                  <h3>等待确定性计算结果</h3>
                  <p>上半屏聚焦任务输入、Agent 执行和验证摘要，详细图表与数据在其后连续展开。</p>
                </div>
              )}
              {calculation && activeTab === "table" && (
                <section className="result-panel">
                  <div className="panel-heading">
                    <h3>数据表</h3>
                  </div>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>T / K</th>
                          <th>P / kPa</th>
                          <th>x1</th>
                          <th>y1</th>
                          <th>残差</th>
                        </tr>
                      </thead>
                      <tbody>
                        {calculation.result.points.map((point, index) => (
                          <tr key={index}>
                            <td>{point.temperature_K.toFixed(4)}</td>
                            <td>{point.pressure_kPa.toFixed(3)}</td>
                            <td>{point.liquid_composition[0].toFixed(5)}</td>
                            <td>{point.vapor_composition[0].toFixed(5)}</td>
                            <td>{point.equilibrium_residual.toExponential(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}
              {calculation && activeTab === "model" && (
                <section className="result-panel">
                  <div className="panel-heading">
                    <h3>模型推荐</h3>
                  </div>
                  <div className="cards">
                    {calculation.model_recommendations.slice(0, 3).map((item) => (
                      <article key={item.model_name}>
                        <span>{item.executable ? "可执行" : "候选受限"}</span>
                        <h3>{item.model_name}</h3>
                        <strong>{item.score.toFixed(1)} 分</strong>
                        <p>{item.reasons.join(" ")}</p>
                      </article>
                    ))}
                  </div>
                </section>
              )}
              {calculation && activeTab === "parameters" && (
                <section className="result-panel">
                  <div className="panel-heading">
                    <h3>参数来源</h3>
                  </div>
                  <div className="cards">
                    {calculation.parameter_sources.map((source, index) => (
                      <article key={index}>
                        <span>Database</span>
                        <h3>{source.component}</h3>
                        <p>{source.property}</p>
                        <a href={source.source_identifier} target="_blank" rel="noreferrer">
                          {source.source_title}
                        </a>
                        <small>{source.temperature_range_K} K</small>
                      </article>
                    ))}
                  </div>
                </section>
              )}
              {calculation && activeTab === "validation" && (
                <section className="result-panel validation-panel">
                  <div className="panel-heading">
                    <h3>验证报告</h3>
                    <span className={`validation-overview ${calculation.validation.overall_status}`}>
                      {calculation.validation.overall_status}
                    </span>
                  </div>
                  <div className="validation-grid">
                    {Object.entries(calculation.validation)
                      .filter(([, value]) => typeof value === "object" && value && "passed" in value)
                      .map(([key, value]) => {
                        const check = value as { passed: boolean; message: string; metric?: number };
                        return (
                          <article key={key}>
                            <span className={check.passed ? "pass" : "fail"}>
                              {check.passed ? "PASS" : "CHECK"}
                            </span>
                            <h3>{key.replaceAll("_", " ")}</h3>
                            <p>{check.message}</p>
                            <small>metric: {check.metric?.toExponential(3) ?? "--"}</small>
                          </article>
                        );
                      })}
                  </div>
                </section>
              )}
              {calculation && activeTab === "runs" && (
                <section className="result-panel">
                  <div className="panel-heading">
                    <h3>执行记录</h3>
                  </div>
                  <div className="runs">
                    {runs.map((run, index) => (
                      <article key={run.result.run_id}>
                        <span>RUN {String(index + 1).padStart(2, "0")}</span>
                        <div>
                          <strong>{run.result.run_id.slice(0, 8)}</strong>
                          <p>
                            {run.result.model_name} · {run.result.pressure_kPa?.toFixed(3) ?? "--"} kPa
                          </p>
                        </div>
                        <span className={run.validation.overall_status}>{run.validation.overall_status}</span>
                      </article>
                    ))}
                  </div>
                </section>
              )}
            </div>
          </section>
        </div>

        <aside className="task-panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Mission Control</p>
              <h2>实验记录面板</h2>
            </div>
            <span className={`risk-badge ${risk}`}>{risk}</span>
          </div>

          <section className="mission-control-card overview-card">
            <div className="group-heading">
              <h3>Experiment Summary</h3>
            </div>
            <div className="mission-record">
              <div className="mission-record-group">
                <span className="mission-record-label">System</span>
                <strong>{task?.components.map((item) => item.name).join(" / ") ?? "--"}</strong>
              </div>
              <div className="mission-record-group">
                <span className="mission-record-label">Task</span>
                <strong>{task?.calculation_type ?? "--"}</strong>
              </div>
              <div className="mission-record-group">
                <span className="mission-record-label">Method</span>
                <strong>{calculation?.result.model_name ?? task?.model_name ?? "待路由"}</strong>
              </div>
              <div className="mission-record-group">
                <span className="mission-record-label">Conditions</span>
                <strong>
                  {task?.conditions.pressure_kPa ?? "--"} kPa / {task?.conditions.temperature_K ?? "--"} K
                </strong>
              </div>
              <div className="mission-record-group">
                <span className="mission-record-label">Status</span>
                <strong>{calculation?.validation.overall_status ?? "待处理"}</strong>
              </div>
            </div>
          </section>

          <section className="task-group editor-card parameter-panel">
            <div className="group-heading">
              <h3>工况参数</h3>
            </div>
            <div className="parameter-grid">
              <label className="field field-model">
                模型
                <select aria-label="热力学模型" value={task?.model_name ?? "Ideal/Raoult"} onChange={(event) => updateModel(event.target.value)}>
                  <option>Ideal/Raoult</option>
                  <option>Wilson</option>
                  <option>NRTL</option>
                  <option>UNIQUAC</option>
                  <option>Peng-Robinson</option>
                  <option>Phasepy/Peng-Robinson</option>
                  <option>Clapeyron/Peng-Robinson</option>
                </select>
              </label>
              <label className="field field-pressure">
                压力 / kPa
                <input aria-label="压力 kPa" type="number" step="0.001" min="0.001" value={task?.conditions.pressure_kPa ?? ""} onChange={(event) => updatePressure(event.target.value)} />
              </label>
              <label className="field field-temperature">
                温度 / K
                <input aria-label="温度 K" type="number" step="0.01" min="0.01" value={task?.conditions.temperature_K ?? ""} onChange={(event) => updateTemperature(event.target.value)} />
              </label>
              <label className="field field-composition">
                组成
                <input
                  aria-label="摩尔组成"
                  value={(task?.conditions.feed_composition ?? task?.conditions.vapor_composition ?? task?.conditions.liquid_composition ?? []).join(", ")}
                  onChange={(event) => updateComposition(event.target.value)}
                  placeholder="0.5, 0.5"
                />
              </label>
            </div>
            <button className="rerun rerun-compact" onClick={rerun} disabled={!task || loading}>
              按当前条件重新计算
            </button>
          </section>

          {diagnostic && (
            <div className="diagnostic" role="alert">
              <strong>诊断信息</strong>
              <p>{diagnostic}</p>
            </div>
          )}
          {calculation?.result.warnings.length ? (
            <div className="diagnostic warning">
              <strong>适用性警告</strong>
              <p>{calculation.result.warnings[0]}</p>
            </div>
          ) : null}
        </aside>
      </section>
    </main>
  );
}

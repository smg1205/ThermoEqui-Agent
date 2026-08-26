"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { exportUrl, phaseDiagram, rerunTask, sendChat } from "@/lib/api";
import type {
  AgentStep,
  CalculationEnvelope,
  ChatResponse,
  PhaseDiagramEntry,
  PhaseDiagramResponse,
  TaskManifest,
} from "@/lib/types";
import { FlashResultCard } from "./FlashResultCard";
import { GammaChart } from "./GammaChart";
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
  return "待命";
}

function applicabilityAdvice(executable: boolean): string {
  return executable ? "模型可用，可继续参与当前推荐流程。" : "选择其他支持模型或补充必要参数。";
}

function modelStatusLabel(executable: boolean): string {
  return executable ? "模型可用" : "模型不可用";
}

function phaseDiagramEntryStatus(entry: PhaseDiagramEntry): string {
  return entry.status;
}

function isDrawableEntry(entry: PhaseDiagramEntry): boolean {
  return (
    (entry.status === "passed" || entry.status === "warning") &&
    entry.result !== null &&
    entry.result.points.length > 0
  );
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
  const [phaseDiagramResult, setPhaseDiagramResult] = useState<PhaseDiagramResponse>();
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [runs, setRuns] = useState<CalculationEnvelope[]>([]);
  const [executionSteps, setExecutionSteps] = useState<AgentStep[]>([]);
  const [activeTab, setActiveTab] = useState<DetailTab>("table");
  const [loading, setLoading] = useState(false);
  const [diagnostic, setDiagnostic] = useState<string>();
  const composerRef = useRef<HTMLTextAreaElement>(null);

  const risk = useMemo(() => riskLabel(calculation), [calculation]);
  const agentRuntimeStatus = useMemo(() => agentStatus(executionSteps, loading), [executionSteps, loading]);
  const isFlashResult = calculation?.result.calculation_type === "tp_flash";

  useEffect(() => {
    const textarea = composerRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
    textarea.style.overflowY = textarea.scrollHeight > 160 ? "auto" : "hidden";
  }, [input]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!input.trim() || loading) return;
    const message = input.trim();
    setMessages((current) => [...current, { role: "user", text: message }]);
    setLoading(true);
    setDiagnostic(undefined);
    setPhaseDiagramResult(undefined);
    setSelectedModels([]);
    try {
      const response: ChatResponse = await sendChat(message, conversationId);
      setConversationId(response.conversation_id);
      setExecutionSteps(response.execution_steps);
      setMessages((current) => [...current, { role: "agent", text: response.answer }]);
      setInput("");
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
    setPhaseDiagramResult(undefined);
    setSelectedModels([]);
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

  async function phaseDiagramCompare() {
    if (!task || loading) return;
    setLoading(true);
    setDiagnostic(undefined);
    try {
      const next = await phaseDiagram(task);
      setPhaseDiagramResult(next);
      setSelectedModels(next.entries.filter(isDrawableEntry).map((entry) => entry.model_name));
      setMessages((current) => [
        ...current,
        {
          role: "agent",
          text: `已按当前条件生成多模型相图：${next.total_models} 个模型（通过 ${next.passed_count}、警告 ${next.warning_count}、失败 ${next.failed_count}、不支持 ${next.unsupported_count}）。`,
        },
      ]);
    } catch (error) {
      setDiagnostic(error instanceof Error ? error.message : "多模型相图生成失败");
    } finally {
      setLoading(false);
    }
  }

  function toggleSelectedModel(modelName: string) {
    setSelectedModels((current) =>
      current.includes(modelName) ? current.filter((name) => name !== modelName) : [...current, modelName],
    );
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
                  <article className={`message-row ${message.role}`} key={`${message.role}-${index}`}>
                    <div className="message-bubble">
                      <label>{message.role === "agent" ? "AGENT" : "YOU"}</label>
                      <div className="message-markdown">
                        <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                          {message.text}
                        </ReactMarkdown>
                      </div>
                    </div>
                  </article>
                ))}
                {loading && (
                  <article className="message-row agent">
                    <div className="message-bubble">
                      <label>ENGINE</label>
                      <div className="message-markdown">
                        <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                          {"正在理解任务、调度模型、调用热力学计算并执行验证..."}
                        </ReactMarkdown>
                      </div>
                    </div>
                  </article>
                )}
              </div>
              <form onSubmit={submit} className="composer composer-sticky">
                <textarea
                  ref={composerRef}
                  aria-label="任务输入"
                  value={input}
                  rows={1}
                  onChange={(event) => setInput(event.target.value)}
                />
                <button disabled={loading}>运行任务</button>
              </form>
            </div>
          </section>

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
                  <a href={exportUrl(calculation.result.run_id, "dwsim")}>下载 DWSIM</a>
                </div>
              )}
            </div>

            <div className="results-stack">
              {calculation && <ScientificValidationCard calculation={calculation} />}
              {phaseDiagramResult && (
                <section className="result-panel chart-panel" data-testid="phase-diagram-panel">
                  <div className="panel-heading">
                    <h3>多模型相图</h3>
                    <span>
                      {phaseDiagramResult.diagram_type} · {phaseDiagramResult.total_models} 个模型
                    </span>
                  </div>
                  <div className="phase-diagram-controls">
                    <p className="phase-diagram-selection-count" data-testid="phase-diagram-selection-count">
                      已选{" "}
                      {selectedModels.filter((name) =>
                        phaseDiagramResult.entries.some((entry) => entry.model_name === name && isDrawableEntry(entry)),
                      ).length}{" "}
                      / {phaseDiagramResult.entries.filter(isDrawableEntry).length} 个可绘制模型
                    </p>
                    <div className="phase-diagram-model-select" data-testid="phase-diagram-model-select">
                      {phaseDiagramResult.entries.map((entry) => {
                        const drawable = isDrawableEntry(entry);
                        return (
                          <label
                            key={entry.model_name}
                            className={`phase-diagram-model-option ${drawable ? "" : "disabled"}`}
                          >
                            <input
                              type="checkbox"
                              disabled={!drawable}
                              checked={drawable && selectedModels.includes(entry.model_name)}
                              onChange={() => toggleSelectedModel(entry.model_name)}
                              aria-label={`选择 ${entry.model_name}`}
                            />
                            {entry.model_name}
                            <span className={`model-validation-badge ${entry.status}`}>{entry.status}</span>
                          </label>
                        );
                      })}
                    </div>
                  </div>
                  {selectedModels.length === 0 && (
                    <p className="phase-diagram-empty-hint">请至少选择一个模型以显示相图曲线。</p>
                  )}
                  <VleChart
                    calculationType={
                      phaseDiagramResult.task?.calculation_type ?? task?.calculation_type ?? "isobaric_vle"
                    }
                    pressure={
                      phaseDiagramResult.task?.conditions.pressure_kPa ?? task?.conditions.pressure_kPa
                    }
                    temperature={
                      phaseDiagramResult.task?.conditions.temperature_K ?? task?.conditions.temperature_K
                    }
                    entries={phaseDiagramResult.entries.filter((entry) =>
                      selectedModels.includes(entry.model_name),
                    )}
                    diagramType={phaseDiagramResult.diagram_type}
                  />
                  <div className="cards phase-diagram-cards" data-testid="phase-diagram-status">
                    {phaseDiagramResult.entries.map((entry) => (
                      <article key={entry.model_name} className="model-validation-card">
                        <div className="model-validation-topline">
                          <strong>{entry.model_name}</strong>
                          <span className={`model-validation-badge ${entry.status}`}>
                            {phaseDiagramEntryStatus(entry)}
                          </span>
                        </div>
                        {entry.result && (
                          <p>
                            数据点 {entry.result.points.length} · 收敛 {entry.result.converged ? "是" : "否"}
                          </p>
                        )}
                        {entry.failure && <p>{entry.failure.message}</p>}
                        {entry.warnings.length > 0 && <p>{entry.warnings.join(" ")}</p>}
                      </article>
                    ))}
                  </div>
                </section>
              )}
              {calculation && !isFlashResult && calculation.result.gamma_infinity.length > 0 && (
                <section className="result-panel chart-panel">
                  <div className="panel-heading">
                    <h3>γ∞-T 曲线</h3>
                    <span>{calculation.result.model_name}</span>
                  </div>
                  <GammaChart
                    points={calculation.result.gamma_infinity}
                    model={calculation.result.model_name}
                    components={calculation.result.input_snapshot?.components as
                      | { name?: string }[]
                      | undefined}
                  />
                </section>
              )}
              {calculation && !isFlashResult && calculation.result.gamma_infinity.length === 0 && (
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
              {calculation && isFlashResult && <FlashResultCard calculation={calculation} />}
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
                  <span>●</span>
                  <h3>等待确定性计算结果</h3>
                  <p>上方区域聚焦任务输入、Agent 执行和验证摘要，详细图表与数据将在这里展开。</p>
                </div>
              )}
              {calculation && activeTab === "table" && (
                <section className="result-panel">
                  <div className="panel-heading">
                    <h3>数据表</h3>
                  </div>
                  <div className="table-wrap">
                    {calculation.result.gamma_infinity.length > 0 ? (
                      <table>
                        <thead>
                          <tr>
                            <th>T / K</th>
                            <th>溶质 → 溶剂</th>
                            <th>γ∞</th>
                            <th>ln γ∞</th>
                          </tr>
                        </thead>
                        <tbody>
                          {calculation.result.gamma_infinity.map((point, index) => {
                            const components = calculation.result.input_snapshot?.components as
                              | Array<{ name?: string }>
                              | undefined;
                            return (
                              <tr key={index}>
                                <td>{point.temperature_K.toFixed(2)}</td>
                                <td>
                                  {components?.[point.solute_index]?.name ?? point.solute_index} →{" "}
                                  {components?.[point.solvent_index]?.name ?? point.solvent_index}
                                </td>
                                <td>{point.gamma_infinity.toFixed(4)}</td>
                                <td>{point.ln_gamma_infinity.toFixed(4)}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    ) : (
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
                    )}
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
              {calculation && activeTab === "model" && calculation.model_recommendations.length > 0 && (
                <section className="result-panel model-validation-panel">
                  <div className="panel-heading">
                    <h3>模型适用性检查</h3>
                  </div>
                  <div className="model-validation-list">
                    {calculation.model_recommendations.map((item) => (
                      <article key={`validation-${item.model_name}`} className="model-validation-card">
                        <div className="model-validation-topline">
                          <strong>{item.model_name}</strong>
                          <span className={`model-validation-badge ${item.executable ? "available" : "blocked"}`}>
                            {modelStatusLabel(item.executable)}
                          </span>
                        </div>
                        {item.reasons.length > 0 && (
                          <div className="model-validation-reasons">
                            <p>理由</p>
                            <ul>
                              {item.reasons.map((reason, index) => (
                                <li key={`${item.model_name}-reason-${index}`}>{reason}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {item.exclusions.length > 0 && (
                          <div className="model-validation-reasons">
                            <p>排除原因</p>
                            <ul>
                              {item.exclusions.map((reason, index) => (
                                <li key={`${item.model_name}-exclusion-${index}`}>{reason}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                        <p className="model-validation-advice">建议：{applicabilityAdvice(item.executable)}</p>
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
                            {run.result.model_name} 路 {run.result.pressure_kPa?.toFixed(3) ?? "--"} kPa
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
                <select
                  aria-label="热力学模型"
                  value={task?.model_name ?? "Ideal/Raoult"}
                  onChange={(event) => updateModel(event.target.value)}
                >
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
            <div className="parameter-actions">
              <button className="rerun rerun-compact" onClick={rerun} disabled={!task || loading}>
                按当前条件重新计算
              </button>
              <button
                className="rerun rerun-compact"
                onClick={phaseDiagramCompare}
                disabled={
                  !task ||
                  loading ||
                  (task.calculation_type !== "isobaric_vle" && task.calculation_type !== "isothermal_vle")
                }
                title={task ? "仅支持 isobaric_vle（T-x-y）与 isothermal_vle（P-x-y）" : undefined}
              >
                多模型相图
              </button>
            </div>
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

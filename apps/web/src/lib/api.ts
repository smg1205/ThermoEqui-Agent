import type {
  CalculationEnvelope,
  ChatResponse,
  ModelComparisonResponse,
  TaskManifest,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  const payload = await response.json();
  if (!response.ok) {
    const message = payload?.error?.message ?? payload?.detail ?? "请求失败";
    throw new Error(message);
  }
  return payload as T;
}

export function sendChat(message: string, conversationId?: string): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify({ message, conversation_id: conversationId }),
  });
}

export function rerunTask(task: TaskManifest): Promise<CalculationEnvelope> {
  const paths: Record<string, string> = {
    isobaric_vle: "isobaric-vle",
    isothermal_vle: "isothermal-vle",
    bubble_point: "bubble-point",
    dew_point: "dew-point",
    tp_flash: "tp-flash",
    azeotrope: "azeotrope",
  };
  const endpoint = paths[task.calculation_type];
  if (!endpoint) throw new Error(`不支持的计算类型：${task.calculation_type}`);
  return request<CalculationEnvelope>(`/api/calculations/${endpoint}`, {
    method: "POST",
    body: JSON.stringify(task),
  });
}

export function compareModels(task: TaskManifest): Promise<ModelComparisonResponse> {
  return request<ModelComparisonResponse>("/api/calculations/compare", {
    method: "POST",
    body: JSON.stringify(task),
  });
}

export function exportUrl(runId: string, format: "json" | "csv" | "dwsim"): string {
  return `${API_URL}/api/runs/${runId}/export?format=${format}`;
}

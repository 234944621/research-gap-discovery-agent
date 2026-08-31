export const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export interface ResearchRequest {
  topic: string;
  search_api?: string;
  mode?: string;
  task_id?: string;
  resume?: boolean;
}

export interface ResearchStreamEvent {
  type: string;
  [key: string]: unknown;
}

export type TaskSnapshot = {
  task_id: string;
  thread_id: string;
  topic: string;
  status: string;
  current_node?: string;
  termination_reason?: string;
  last_checkpoint_id?: string;
  state?: Record<string, unknown>;
  verification_round?: number;
  max_verification_rounds?: number;
  tool_call_count?: number;
  max_tool_calls?: number;
  warnings?: string[];
  tool_traces?: unknown[];
};

async function readSSE(
  response: Response,
  onEvent: (event: ResearchStreamEvent) => void,
  stopTypes: string[] = ["error", "done", "task_completed", "task_paused", "task_failed"]
): Promise<void> {
  const body = response.body;
  if (!body) {
    throw new Error("浏览器不支持流式响应");
  }

  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);

      if (rawEvent.startsWith("data:")) {
        const dataPayload = rawEvent.slice(5).trim();
        if (dataPayload) {
          try {
            const event = JSON.parse(dataPayload) as ResearchStreamEvent;
            onEvent(event);
            if (stopTypes.includes(String(event.type))) {
              return;
            }
          } catch (error) {
            console.error("SSE parse failed", error);
          }
        }
      }

      boundary = buffer.indexOf("\n\n");
    }

    if (done) {
      break;
    }
  }
}

export async function runResearchStream(
  payload: ResearchRequest,
  onEvent: (event: ResearchStreamEvent) => void,
  options: { signal?: AbortSignal } = {}
): Promise<void> {
  const response = await fetch(`${API_BASE}/research/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      ...payload,
      mode: payload.mode || "gap_discovery",
    }),
    signal: options.signal,
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(errorText || `研究请求失败（${response.status}）`);
  }

  await readSSE(response, onEvent);
}

export async function fetchTaskSnapshot(taskId: string): Promise<TaskSnapshot> {
  const response = await fetch(`${API_BASE}/research/tasks/${taskId}`);
  if (response.status === 404) {
    throw new Error(`task not found: ${taskId}`);
  }
  if (!response.ok) {
    throw new Error(`获取任务失败（${response.status}）`);
  }
  return (await response.json()) as TaskSnapshot;
}

export async function resumeResearchTask(
  taskId: string,
  topic?: string
): Promise<{ task_id: string; resumed?: boolean; reason?: string; status?: string }> {
  const response = await fetch(`${API_BASE}/research/tasks/${taskId}/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic: topic || null }),
  });
  if (response.status === 404) {
    throw new Error(`task not found: ${taskId}`);
  }
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(text || `Resume 失败（${response.status}）`);
  }
  return response.json();
}

export async function streamTaskEvents(
  taskId: string,
  onEvent: (event: ResearchStreamEvent) => void,
  options: { afterSeq?: number; signal?: AbortSignal } = {}
): Promise<void> {
  const after = options.afterSeq || 0;
  const response = await fetch(
    `${API_BASE}/research/tasks/${taskId}/events?after_seq=${after}&stream=true`,
    {
      headers: { Accept: "text/event-stream" },
      signal: options.signal,
    }
  );
  if (response.status === 404) {
    throw new Error(`task not found: ${taskId}`);
  }
  if (!response.ok) {
    throw new Error(`事件流失败（${response.status}）`);
  }
  await readSSE(response, onEvent, ["error", "done", "task_completed", "task_failed"]);
}

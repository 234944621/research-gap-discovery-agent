import type { ToolTrace } from "../types";

type Budget = {
  verificationRound: number;
  maxVerificationRounds: number;
  toolCallCount: number;
  maxToolCalls: number;
};

type Props = {
  taskId: string;
  threadId: string;
  currentNode: string | null;
  budget: Budget;
  warnings: string[];
  toolTraces: ToolTrace[];
  lastCheckpoint: string;
  terminationReason: string;
  paused: boolean;
  onResume: () => void;
};

export function TaskObservability({
  taskId,
  threadId,
  currentNode,
  budget,
  warnings,
  toolTraces,
  lastCheckpoint,
  terminationReason,
  paused,
  onResume,
}: Props) {
  const recent = toolTraces.slice(-12);

  return (
    <section className="task-observability">
      <header className="section-head">
        <div>
          <h2>Task Observability</h2>
          <p>task / thread / 预算 / Tool Trace / Checkpoint</p>
        </div>
        {paused ? (
          <button type="button" className="primary-btn" onClick={onResume}>
            Resume
          </button>
        ) : null}
      </header>

      <dl className="obs-grid">
        <div>
          <dt>task_id</dt>
          <dd className="mono">{taskId || "—"}</dd>
        </div>
        <div>
          <dt>thread_id</dt>
          <dd className="mono">{threadId || "—"}</dd>
        </div>
        <div>
          <dt>current_node</dt>
          <dd>{currentNode || "—"}</dd>
        </div>
        <div>
          <dt>验证轮次</dt>
          <dd>
            {budget.verificationRound} / {budget.maxVerificationRounds || "—"}
          </dd>
        </div>
        <div>
          <dt>工具调用</dt>
          <dd>
            {budget.toolCallCount} / {budget.maxToolCalls || "—"}
          </dd>
        </div>
        <div>
          <dt>termination</dt>
          <dd>{terminationReason || "—"}</dd>
        </div>
        <div>
          <dt>checkpoint</dt>
          <dd className="mono">{lastCheckpoint || "—"}</dd>
        </div>
      </dl>

      {warnings.length ? (
        <div className="obs-warnings">
          <h3>Warnings</h3>
          <ul>
            {warnings.slice(-8).map((w, i) => (
              <li key={`${i}-${w.slice(0, 24)}`}>{w}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="obs-traces">
        <h3>Tool Trace</h3>
        {recent.length === 0 ? (
          <p className="muted">尚无工具调用</p>
        ) : (
          <ol>
            {recent.map((t, idx) => (
              <li key={t.trace_id || `${t.tool_name}-${idx}`}>
                <strong>{t.tool_name || "tool"}</strong>
                <span className={`trace-status ${(t.status || "").toLowerCase()}`}>
                  {t.status || "?"}
                </span>
                {t.error_type ? <em>{t.error_type}</em> : null}
                {t.result_summary ? (
                  <span className="muted"> {String(t.result_summary).slice(0, 80)}</span>
                ) : null}
                {typeof t.retry_index === "number" && t.retry_index > 0 ? (
                  <span> retry={t.retry_index}</span>
                ) : null}
              </li>
            ))}
          </ol>
        )}
      </div>
    </section>
  );
}

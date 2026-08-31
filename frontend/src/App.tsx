import { useCallback, useMemo, useRef, useState } from "react";
import {
  fetchTaskSnapshot,
  resumeResearchTask,
  runResearchStream,
  streamTaskEvents,
} from "./api";
import { HomeCompose } from "./components/HomeCompose";
import { LangGraphCanvas, type GraphEdge } from "./components/LangGraphCanvas";
import { PipelineLog } from "./components/PipelineLog";
import { ReportPanel } from "./components/ReportPanel";
import {
  StateInspector,
  type StatePatch,
} from "./components/StateInspector";
import { TaskObservability } from "./components/TaskObservability";
import type { ArtifactSummary, LogItem, ToolTrace } from "./types";
import { labelNode } from "./types";

export default function App() {
  const [topic, setTopic] = useState("跨链桥智能合约漏洞检测");
  const [started, setStarted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [logs, setLogs] = useState<LogItem[]>([]);
  const [nodesDone, setNodesDone] = useState<string[]>([]);
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [graphNodes, setGraphNodes] = useState<string[]>([]);
  const [graphEdges, setGraphEdges] = useState<GraphEdge[]>([]);
  const [currentState, setCurrentState] = useState<Record<string, unknown> | null>(
    null
  );
  const [patches, setPatches] = useState<StatePatch[]>([]);
  const [artifacts, setArtifacts] = useState<ArtifactSummary[]>([]);
  const [engine, setEngine] = useState<string>("");
  const [report, setReport] = useState("");
  const [reportPulse, setReportPulse] = useState(false);
  const [logsCollapsed, setLogsCollapsed] = useState(false);
  const [receivedDone, setReceivedDone] = useState(false);
  const [taskId, setTaskId] = useState<string>("");
  const [threadId, setThreadId] = useState<string>("");
  const [terminationReason, setTerminationReason] = useState<string>("");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [toolTraces, setToolTraces] = useState<ToolTrace[]>([]);
  const [budget, setBudget] = useState({
    verificationRound: 0,
    maxVerificationRounds: 0,
    toolCallCount: 0,
    maxToolCalls: 0,
  });
  const [lastCheckpoint, setLastCheckpoint] = useState<string>("");
  const [paused, setPaused] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const logId = useRef(0);
  const patchId = useRef(0);
  const lastEventSeq = useRef(0);

  const pushLog = useCallback((text: string, kind: LogItem["kind"] = "status") => {
    logId.current += 1;
    setLogs((prev) => [...prev, { id: `log-${logId.current}`, text, kind }]);
  }, []);

  const resetRun = useCallback(() => {
    setLogs([]);
    setNodesDone([]);
    setActiveNode(null);
    setGraphNodes([]);
    setGraphEdges([]);
    setCurrentState(null);
    setPatches([]);
    setArtifacts([]);
    setEngine("");
    setReport("");
    setError("");
    setReceivedDone(false);
    setReportPulse(false);
    setTaskId("");
    setThreadId("");
    setTerminationReason("");
    setWarnings([]);
    setToolTraces([]);
    setBudget({
      verificationRound: 0,
      maxVerificationRounds: 0,
      toolCallCount: 0,
      maxToolCalls: 0,
    });
    setLastCheckpoint("");
    setPaused(false);
    lastEventSeq.current = 0;
  }, []);

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
  }, []);

  const applyEvent = useCallback(
    (event: Record<string, unknown>) => {
      const type = typeof event.type === "string" ? event.type : "";

      if (typeof event.task_id === "string" && event.task_id) {
        setTaskId(event.task_id);
      }
      if (typeof event.thread_id === "string" && event.thread_id) {
        setThreadId(event.thread_id);
      }
      if (typeof event._seq === "number") {
        lastEventSeq.current = Math.max(lastEventSeq.current, event._seq);
      }

      if (type === "pipeline") {
        if (typeof event.engine === "string") setEngine(event.engine);
        if (Array.isArray(event.nodes)) {
          setGraphNodes(event.nodes.filter((n): n is string => typeof n === "string"));
        }
        if (Array.isArray(event.edges)) {
          setGraphEdges(
            event.edges.filter(
              (e): e is GraphEdge =>
                typeof e === "object" &&
                e !== null &&
                typeof (e as GraphEdge).from === "string" &&
                typeof (e as GraphEdge).to === "string"
            ) as GraphEdge[]
          );
        }
        pushLog(
          `LangGraph 拓扑已就绪${event.engine ? `（engine=${event.engine}）` : ""}`,
          "system"
        );
        return;
      }

      if (type === "state_snapshot" || type === "state_patch") {
        const snap =
          event.state && typeof event.state === "object"
            ? (event.state as Record<string, unknown>)
            : null;
        if (snap) {
          setCurrentState(snap);
          if (typeof snap.task_id === "string") setTaskId(snap.task_id);
          if (typeof snap.thread_id === "string") setThreadId(snap.thread_id);
          if (typeof snap.termination_reason === "string") {
            setTerminationReason(snap.termination_reason);
          }
          if (Array.isArray(snap.warnings)) {
            setWarnings(snap.warnings.map(String));
          }
          if (Array.isArray(snap.tool_traces)) {
            setToolTraces(snap.tool_traces as ToolTrace[]);
          }
          setBudget((prev) => ({
            verificationRound: Number(snap.verification_round ?? prev.verificationRound) || 0,
            maxVerificationRounds:
              Number(snap.max_verification_rounds ?? prev.maxVerificationRounds) || 0,
            toolCallCount: Number(snap.tool_call_count ?? prev.toolCallCount) || 0,
            maxToolCalls: Number(snap.max_tool_calls ?? prev.maxToolCalls) || 0,
          }));
          if (typeof snap.last_checkpoint_id === "string") {
            setLastCheckpoint(snap.last_checkpoint_id);
          }
        }
        if (type === "state_patch") {
          const node = typeof event.node === "string" ? event.node : "node";
          const diff = Array.isArray(event.diff)
            ? (event.diff as StatePatch["diff"])
            : [];
          patchId.current += 1;
          setPatches((prev) => [
            ...prev,
            {
              id: `patch-${patchId.current}`,
              node,
              diff,
              state: snap || {},
            },
          ]);
          if (typeof event.termination_reason === "string") {
            setTerminationReason(event.termination_reason);
          }
          if (Array.isArray(event.warnings)) {
            setWarnings(event.warnings.map(String));
          }
        }
        return;
      }

      if (type === "node_started" || type === "node_start") {
        const node = typeof event.node === "string" ? event.node : "node";
        setActiveNode(node);
        pushLog(`进入节点 · ${labelNode(node)}`, "node");
        return;
      }

      if (type === "tool_started") {
        pushLog(`工具开始 · ${String(event.tool_name || "")}`, "tool");
        return;
      }
      if (type === "tool_completed" || type === "tool_failed") {
        const trace = (event.trace || event) as ToolTrace;
        setToolTraces((prev) => [...prev, trace]);
        pushLog(
          `工具${type === "tool_failed" ? "失败" : "完成"} · ${String(
            trace.tool_name || event.tool_name || ""
          )}`,
          "tool"
        );
        return;
      }

      if (type === "checkpoint_saved") {
        if (typeof event.checkpoint_id === "string") {
          setLastCheckpoint(event.checkpoint_id);
        }
        pushLog(`Checkpoint 已保存 · ${String(event.node || "")}`, "system");
        return;
      }

      if (type === "warning") {
        const msg = String(event.message || "warning");
        setWarnings((prev) => [...prev, msg]);
        pushLog(`警告：${msg}`, "system");
        return;
      }

      if (type === "task_paused") {
        setPaused(true);
        setLoading(false);
        setTerminationReason(String(event.reason || "paused"));
        pushLog(`任务暂停：${String(event.message || event.reason || "")}`, "system");
        return;
      }

      if (type === "task_resumed") {
        setPaused(false);
        pushLog("任务已恢复", "system");
        return;
      }

      if (type === "status") {
        const message =
          typeof event.message === "string" && event.message.trim()
            ? event.message
            : "状态更新";
        const kind = message.startsWith("tool:") ? "tool" : "status";
        pushLog(message, kind);
        return;
      }

      if (type === "node_completed" || type === "node_done") {
        const node = typeof event.node === "string" ? event.node : "node";
        setNodesDone((prev) => (prev.includes(node) ? prev : [...prev, node]));
        setActiveNode(null);
        const fields = Array.isArray(event.updated_fields)
          ? event.updated_fields.filter((f): f is string => typeof f === "string")
          : [];
        pushLog(
          fields.length
            ? `节点完成 · ${labelNode(node)} · 更新 ${fields.join(", ")}`
            : `节点完成 · ${labelNode(node)}`,
          "node"
        );
        return;
      }

      if (type === "artifact") {
        const name =
          typeof event.artifact === "string" ? event.artifact : "artifact";
        setArtifacts((prev) => [...prev, { name }]);
        pushLog(`产出工件 · ${name}`, "artifact");
        return;
      }

      if (type === "report") {
        const md =
          (typeof event.report_markdown === "string" &&
            event.report_markdown.trim()) ||
          (typeof event.report === "string" && event.report.trim()) ||
          "";
        if (md) {
          setReport(md);
          setReportPulse(true);
          window.setTimeout(() => setReportPulse(false), 1200);
          pushLog("证据链报告已生成", "system");
        }
        return;
      }

      if (type === "task_completed" || type === "done") {
        setReceivedDone(true);
        setActiveNode(null);
        setPaused(false);
        if (typeof event.termination_reason === "string") {
          setTerminationReason(event.termination_reason);
        }
        pushLog("研究流程已完整结束", "system");
        return;
      }

      if (type === "task_failed" || type === "error") {
        const detail =
          typeof event.detail === "string"
            ? event.detail
            : typeof event.message === "string"
              ? event.message
              : "研究过程中发生错误";
        setError(detail);
        setActiveNode(null);
        pushLog(`失败：${detail}`, "system");
      }
    },
    [pushLog]
  );

  const reconnectTask = useCallback(
    async (id: string, signal?: AbortSignal) => {
      const snap = await fetchTaskSnapshot(id);
      setTaskId(snap.task_id);
      setThreadId(snap.thread_id);
      setTerminationReason(snap.termination_reason || "");
      setWarnings((snap.warnings || []).map(String));
      setToolTraces((snap.tool_traces || []) as ToolTrace[]);
      setLastCheckpoint(snap.last_checkpoint_id || "");
      setBudget({
        verificationRound: Number(snap.verification_round || 0),
        maxVerificationRounds: Number(snap.max_verification_rounds || 0),
        toolCallCount: Number(snap.tool_call_count || 0),
        maxToolCalls: Number(snap.max_tool_calls || 0),
      });
      if (snap.state) setCurrentState(snap.state);
      if (snap.status === "paused") setPaused(true);
      if (snap.status === "completed") {
        setReceivedDone(true);
        setLoading(false);
      }
      // Continue events after snapshot
      await streamTaskEvents(
        id,
        (event) => applyEvent(event),
        { afterSeq: lastEventSeq.current, signal }
      );
    },
    [applyEvent]
  );

  const start = useCallback(async () => {
    const trimmed = topic.trim();
    if (!trimmed) {
      setError("请输入研究主题");
      return;
    }

    cancel();
    resetRun();
    setStarted(true);
    setLoading(true);
    setError("");

    const controller = new AbortController();
    controllerRef.current = controller;

    try {
      let gotDone = false;
      let gotReport = false;
      let localTaskId = "";
      await runResearchStream(
        { topic: trimmed, search_api: "academic", mode: "gap_discovery" },
        (event) => {
          if (typeof event.task_id === "string") localTaskId = event.task_id;
          applyEvent(event);
          if (event.type === "done" || event.type === "task_completed") gotDone = true;
          if (event.type === "report") gotReport = true;
          if (event.type === "task_paused") {
            gotDone = true;
          }
        },
        { signal: controller.signal }
      );

      setReceivedDone(gotDone);
      if (!gotDone && !gotReport && localTaskId) {
        // SSE dropped — reconnect from snapshot
        pushLog("SSE 断开，正在拉取任务快照并重连事件流…", "system");
        await reconnectTask(localTaskId, controller.signal);
      } else if (!gotDone && !gotReport) {
        setError(
          (prev) =>
            prev ||
            "SSE 提前结束：可能是后端中断。请查看终端日志后重试。"
        );
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        pushLog("已取消当前研究", "system");
      } else {
        setError(err instanceof Error ? err.message : "请求失败");
      }
    } finally {
      setLoading(false);
      setActiveNode(null);
      if (controllerRef.current === controller) {
        controllerRef.current = null;
      }
    }
  }, [applyEvent, cancel, pushLog, reconnectTask, resetRun, topic]);

  const onResume = useCallback(async () => {
    if (!taskId) {
      setError("无 task_id，无法 Resume");
      return;
    }
    setLoading(true);
    setError("");
    setPaused(false);
    try {
      const res = await resumeResearchTask(taskId, topic);
      if (res.resumed === false && res.reason === "already_completed") {
        setReceivedDone(true);
        pushLog("任务已完成，无需 Resume", "system");
        return;
      }
      pushLog("已请求 Resume，拉取快照与后续事件…", "system");
      const controller = new AbortController();
      controllerRef.current = controller;
      await reconnectTask(taskId, controller.signal);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Resume 失败");
    } finally {
      setLoading(false);
    }
  }, [pushLog, reconnectTask, taskId, topic]);

  const goHome = useCallback(() => {
    if (loading) return;
    cancel();
    setStarted(false);
    resetRun();
  }, [cancel, loading, resetRun]);

  const statusLabel = useMemo(() => {
    if (paused) return "已暂停 · 可 Resume";
    if (loading) {
      return activeNode ? `运行中 · ${labelNode(activeNode)}` : "证据链构建中";
    }
    if (receivedDone || report) return "本轮研究已完成";
    if (error) return "流程中断";
    return "等待开始";
  }, [activeNode, error, loading, paused, receivedDone, report]);

  if (!started) {
    return (
      <div className="page home-page">
        <div className="atmosphere" aria-hidden="true" />
        <HomeCompose
          topic={topic}
          onTopicChange={setTopic}
          loading={loading}
          error={error}
          onSubmit={start}
          onCancel={cancel}
        />
      </div>
    );
  }

  return (
    <div className="page workspace-page">
      <div className="atmosphere dim" aria-hidden="true" />

      <aside className="side-rail">
        <button type="button" className="ghost-btn back" onClick={goHome} disabled={loading}>
          ← 返回
        </button>
        <p className="brand-mark compact">科研空白发现智能体</p>
        <h2 className="topic-title">{topic}</h2>
        <p className="side-copy">
          LangGraph 节点图 · Shared State · 逐步 diff
        </p>

        <dl className="meta-list">
          <div>
            <dt>状态</dt>
            <dd>
              <span className={`live-pill ${loading ? "on" : ""}`}>{statusLabel}</span>
            </dd>
          </div>
          <div>
            <dt>引擎</dt>
            <dd>{engine || "langgraph"}</dd>
          </div>
          <div>
            <dt>节点</dt>
            <dd>
              {nodesDone.length} / 11
            </dd>
          </div>
          <div>
            <dt>State patches</dt>
            <dd>{patches.length}</dd>
          </div>
          <div>
            <dt>工件</dt>
            <dd>{artifacts.length}</dd>
          </div>
        </dl>

        <div className="side-actions">
          {loading ? (
            <button type="button" className="ghost-btn" onClick={cancel}>
              取消本轮
            </button>
          ) : paused ? (
            <button type="button" className="primary-btn" onClick={onResume}>
              Resume 继续
            </button>
          ) : (
            <button type="button" className="primary-btn" onClick={start}>
              再跑一轮
            </button>
          )}
        </div>
      </aside>

      <main className="workspace-main">
        {error ? <p className="error-banner">{error}</p> : null}

        <TaskObservability
          taskId={taskId}
          threadId={threadId}
          currentNode={activeNode}
          budget={budget}
          warnings={warnings}
          toolTraces={toolTraces}
          lastCheckpoint={lastCheckpoint}
          terminationReason={terminationReason}
          paused={paused}
          onResume={onResume}
        />

        <LangGraphCanvas
          nodes={graphNodes}
          edges={graphEdges}
          activeNode={activeNode}
          doneNodes={nodesDone}
          engine={engine}
        />

        <StateInspector
          currentState={currentState}
          patches={patches}
          activeNode={activeNode}
        />

        <PipelineLog
          logs={logs}
          nodesDone={nodesDone}
          activeNode={activeNode}
          engine={engine}
          collapsed={logsCollapsed}
          onToggle={() => setLogsCollapsed((v) => !v)}
        />

        {report ? (
          <ReportPanel markdown={report} highlight={reportPulse} />
        ) : (
          <section className="report-placeholder">
            <h2>证据链报告</h2>
            <p>
              {loading
                ? "正在聚合全文局限、后续引用与验证轨迹，报告将在 finalize/report 后出现。"
                : paused
                  ? "任务已暂停。可修改主题后点击 Resume。"
                  : "本轮尚未生成报告。若流程中断，请检查后端终端后重试。"}
            </p>
          </section>
        )}
      </main>
    </div>
  );
}

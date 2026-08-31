import { useEffect, useRef } from "react";
import type { LogItem } from "../types";
import { labelNode } from "../types";

type Props = {
  logs: LogItem[];
  nodesDone: string[];
  activeNode?: string | null;
  engine?: string;
  collapsed: boolean;
  onToggle: () => void;
};

const NODES = [
  "memory_recall",
  "planner",
  "search",
  "paper_reader",
  "analyzer",
  "evidence_chain",
  "gap_discover",
  "gap_verify",
  "cross_domain",
  "finalize",
  "report",
];

export function PipelineLog({
  logs,
  nodesDone,
  activeNode,
  engine,
  collapsed,
  onToggle,
}: Props) {
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs]);

  return (
    <section className="pipeline-panel">
      <header className="section-head">
        <div>
          <h2>Agent 执行轨迹</h2>
          <p>
            {engine ? `编排引擎 ${engine}` : "LangGraph 流水线"}
            {" · "}
            已完成节点 {nodesDone.length}
            {activeNode ? ` · 运行中 ${labelNode(activeNode)}` : ""}
          </p>
        </div>
        <button type="button" className="ghost-btn" onClick={onToggle}>
          {collapsed ? "展开日志" : "收起日志"}
        </button>
      </header>

      {!collapsed && (
        <>
          <ol className="node-rail" aria-label="pipeline nodes">
            {NODES.map((node) => {
              const done = nodesDone.includes(node);
              const active = activeNode === node;
              return (
                <li
                  key={node}
                  className={`${done ? "done" : ""} ${active ? "active" : ""}`}
                >
                  <span className="rail-dot" />
                  <span>{labelNode(node)}</span>
                </li>
              );
            })}
          </ol>

          <div className="log-scroller" ref={scroller}>
            <ul className="log-list">
              {logs.map((log) => (
                <li key={log.id} className={`log-item kind-${log.kind}`}>
                  <span className="log-mark" />
                  <p>{log.text}</p>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </section>
  );
}

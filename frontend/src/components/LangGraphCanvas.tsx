import { PIPELINE_LABELS } from "../types";

export type GraphEdge = {
  from: string;
  to: string;
  kind?: string;
  label?: string;
};

type Props = {
  nodes: string[];
  edges: GraphEdge[];
  activeNode: string | null;
  doneNodes: string[];
  engine?: string;
};

const DEFAULT_NODES = [
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

const LAYOUT: Record<string, { x: number; y: number }> = {
  memory_recall: { x: 40, y: 40 },
  planner: { x: 220, y: 40 },
  search: { x: 400, y: 40 },
  paper_reader: { x: 580, y: 40 },
  analyzer: { x: 40, y: 150 },
  evidence_chain: { x: 220, y: 150 },
  gap_discover: { x: 400, y: 150 },
  gap_verify: { x: 580, y: 150 },
  cross_domain: { x: 400, y: 270 },
  finalize: { x: 580, y: 270 },
  report: { x: 760, y: 270 },
  END: { x: 940, y: 270 },
};

export function LangGraphCanvas({
  nodes,
  edges,
  activeNode,
  doneNodes,
  engine,
}: Props) {
  const nodeList = nodes.length ? nodes : DEFAULT_NODES;
  const edgeList =
    edges.length > 0
      ? edges
      : [
          { from: "memory_recall", to: "planner" },
          { from: "planner", to: "search" },
          { from: "search", to: "paper_reader" },
          { from: "paper_reader", to: "analyzer" },
          { from: "analyzer", to: "evidence_chain" },
          { from: "evidence_chain", to: "gap_discover" },
          { from: "gap_discover", to: "gap_verify" },
          {
            from: "gap_verify",
            to: "cross_domain",
            kind: "conditional",
            label: "needs cross-domain",
          },
          { from: "gap_verify", to: "finalize", kind: "conditional", label: "skip" },
          { from: "cross_domain", to: "finalize" },
          { from: "finalize", to: "report" },
          { from: "report", to: "END" },
        ];

  return (
    <section className="graph-panel">
      <header className="section-head">
        <div>
          <h2>LangGraph StateGraph</h2>
          <p>
            {engine || "langgraph"} · 当前节点高亮 · 条件边在 gap_verify 后分支
          </p>
        </div>
      </header>

      <div className="graph-scroll">
        <svg
          className="graph-svg"
          viewBox="0 0 1080 360"
          role="img"
          aria-label="LangGraph topology"
        >
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#3a4f48" />
            </marker>
          </defs>

          {edgeList.map((edge) => {
            const a = LAYOUT[edge.from];
            const b = LAYOUT[edge.to];
            if (!a || !b) return null;
            const x1 = a.x + 130;
            const y1 = a.y + 28;
            const x2 = b.x;
            const y2 = b.y + 28;
            const midX = (x1 + x2) / 2;
            const midY = (y1 + y2) / 2 - (edge.kind === "conditional" ? 12 : 0);
            const activeEdge =
              activeNode === edge.from ||
              (doneNodes.includes(edge.from) &&
                (doneNodes.includes(edge.to) || activeNode === edge.to));
            return (
              <g key={`${edge.from}-${edge.to}-${edge.label || ""}`}>
                <path
                  d={`M ${x1} ${y1} Q ${midX} ${midY} ${x2} ${y2}`}
                  className={`graph-edge ${edge.kind || "normal"} ${
                    activeEdge ? "lit" : ""
                  }`}
                  markerEnd="url(#arrow)"
                  fill="none"
                />
                {edge.label ? (
                  <text x={midX} y={midY - 6} className="edge-label">
                    {edge.label}
                  </text>
                ) : null}
              </g>
            );
          })}

          {[...nodeList, "END"].map((node) => {
            const pos = LAYOUT[node];
            if (!pos) return null;
            const isActive = activeNode === node;
            const isDone = doneNodes.includes(node) || (node === "END" && doneNodes.includes("report"));
            const label =
              node === "END" ? "END" : PIPELINE_LABELS[node] || node;
            return (
              <g key={node} transform={`translate(${pos.x}, ${pos.y})`}>
                <rect
                  width="130"
                  height="56"
                  rx="4"
                  className={`graph-node ${isActive ? "active" : ""} ${
                    isDone ? "done" : ""
                  }`}
                />
                <text x="65" y="24" textAnchor="middle" className="node-id">
                  {node === "END" ? "END" : node}
                </text>
                <text x="65" y="42" textAnchor="middle" className="node-label">
                  {label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </section>
  );
}

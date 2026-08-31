export type LogItem = {
  id: string;
  text: string;
  kind: "status" | "node" | "artifact" | "tool" | "system";
};

export type ArtifactSummary = {
  name: string;
  detail?: string;
};

export type ToolTrace = {
  trace_id?: string;
  tool_call_id?: string;
  tool_name?: string;
  arguments?: Record<string, unknown>;
  status?: string;
  retry_index?: number;
  result_count?: number;
  result_summary?: string;
  error_type?: string;
  error_message?: string;
  duration_ms?: number;
};

export const PIPELINE_LABELS: Record<string, string> = {
  memory_recall: "Memory 召回",
  planner: "研究规划",
  search: "文献检索",
  paper_reader: "全文 / PaperCard",
  analyzer: "方法分析",
  evidence_chain: "证据链 / 引用批评",
  gap_discover: "Gap 发现",
  gap_verify: "Gap 验证 Agent",
  cross_domain: "跨域启发",
  finalize: "候选方向整理",
  report: "报告生成",
};

export function labelNode(node: string): string {
  return PIPELINE_LABELS[node] || node;
}

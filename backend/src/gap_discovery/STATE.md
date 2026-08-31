# ResearchState 设计

帮助发现与验证**候选** Research Gap；State 只存可序列化数据（无 DB 连接、无模型客户端、无巨型全文对象）。

## 字段总表

| 字段 | 含义 | 主要写入 | 主要读取 | 合并策略 | 持久化 | 前端展示 |
|---|---|---|---|---|---|---|
| `topic` | 研究主题 | init | 全节点 | replace | ✓ | ✓ |
| `task_id` | 业务任务 ID | init / tasks API | runner, tasks | replace | ✓ | ✓ |
| `thread_id` | Checkpoint/会话隔离键 | init / tasks API | graph, tasks | replace | ✓ | ✓ |
| `current_node` | 当前/最近完成节点 | runner / 各节点 touch | SSE, resume | replace | ✓ | ✓ |
| `completed_nodes` | 已完成节点列表（恢复跳过） | runner | iter_pipeline | append unique | ✓ | ✓ |
| `research_questions` | 研究问题 | planner | search | replace | ✓ | ✓ |
| `search_keywords` | 检索词 | planner | search | replace | ✓ | ✓ |
| `plan` | 规划结构 | planner | 后续 | replace | ✓ | 摘要 |
| `papers` | 检索论文列表 | search / verify | reader, analyzer | replace | ✓ | 摘要 |
| `paper_cards` | PaperCard | paper_reader | analyzer…report | replace | ✓ | 摘要 |
| `method_taxonomy` / `analysis` / `limitations` | 分析产物 | analyzer | gap_discover | replace | ✓ | 摘要 |
| `candidate_gaps` | 候选 Gap | gap_discover | gap_verify | replace | ✓ | ✓ |
| `verified_gaps` | 验证后保留的 Gap | gap_verify | finalize, report | replace | ✓ | ✓ |
| `gap_verification_results` | 验证决策记录 | gap_verify | report | replace | ✓ | ✓ |
| `cross_domain_methods` | 跨域启发 | cross_domain | finalize | replace | ✓ | ✓ |
| `final_candidates` | 最终候选方向 | finalize | report | replace | ✓ | ✓ |
| `research_memory` | 召回的记忆条目副本（flat：情节+语义+实体+程序） | memory_recall | planner, verify | replace | ✓ | 摘要 |
| `memory_semantic_lessons` | 语义教训（REJECTED 蒸馏规则） | memory_recall | planner, verify | replace | ✓ | 摘要 |
| `memory_entities` | 实体记忆（stats / rejected_pattern / keep_direction） | memory_recall | planner | replace | ✓ | 摘要 |
| `memory_procedures` | 程序记忆（gap_verify_sop / landscape_search_sop） | memory_recall | planner, verify | replace | ✓ | 摘要 |
| `rag_hits` | 最近 RAG 命中摘要 | gap_verify | report | replace | ✓ | 摘要 |
| `iteration_count` / `max_iterations` | 遗留迭代计数 | gap_verify | route | replace | ✓ | ✓ |
| `verification_round` / `max_verification_rounds` | 验证轮预算 | gap_verify | verify_agent | replace | ✓ | ✓ |
| `tool_call_count` / `max_tool_calls` | 工具调用预算 | verify_agent | verify, SSE | replace | ✓ | ✓ |
| `token_usage` / `max_token_budget` | Token 预算（可选） | verify_agent | verify | replace | ✓ | ✓ |
| `visited_actions` | 已访问工具签名 | tool_runtime | loop guard | append unique | ✓ | ✓ |
| `tool_traces` | 结构化工具 Trace | tool_runtime | SSE, eval | append unique | ✓ | ✓ |
| `retry_counts` | 按工具重试计数 | tool_runtime | SSE | dict merge | ✓ | ✓ |
| `last_error` | 最近结构化错误 | tool_runtime / nodes | report | replace | ✓ | ✓ |
| `warnings` | 警告（含 injection） | safety / runtime | report, UI | append unique | ✓ | ✓ |
| `evidence_status` | 证据充分度 | search / verify | report | replace | ✓ | ✓ |
| `verification_status` | 验证阶段状态 | gap_verify | report, API | replace | ✓ | ✓ |
| `termination_reason` | 终止原因枚举语义 | verify / runner | API, report | replace | ✓ | ✓ |
| `started_at` / `updated_at` | 时间戳 | init / merge | API | replace | ✓ | ✓ |
| `needs_more_evidence` | 是否走 cross_domain | gap_verify | route | replace | ✓ | ✓ |
| `final_report` | 报告 Markdown | report | API | replace | ✓ | ✓ |
| `stage` / `status` | 流水线阶段与任务状态 | 各节点 | SSE | replace | ✓ | ✓ |
| `notices` / `events` | 通知与内部事件 | 各节点 | runner | append unique | ✓ | events→日志 |
| `error` | 失败信息 | 节点 | SSE | replace | ✓ | ✓ |
| `executed_side_effects` | 幂等副作用键 | pipeline | resume | append unique | ✓ | 可选 |
| `last_checkpoint_id` | 最近 checkpoint | runner | UI | replace | ✓ | ✓ |

## 终止 / 验证状态（不重复枚举）

`verification_status` / `termination_reason` 取值语义：

- `COMPLETED` — 正常完成
- `KEEP` / `REFINED` / `REJECTED` — Gap 判决（也可出现在单条 verification 结果中）
- `INSUFFICIENT_EVIDENCE` — 证据不足，禁止确定性 novelty
- `BUDGET_EXCEEDED` — 轮数/工具/超时/Token 超限
- `TOOL_FAILURE` — 工具不可恢复失败
- `NEEDS_USER_INPUT` — 暂停等待用户补充（保留 Checkpoint）

## 合并规则

- `LIST_MERGE_FIELDS`：`events`, `notices`, `warnings`, `visited_actions`, `tool_traces`, `executed_side_effects`, `completed_nodes`
- `retry_counts`：字典浅合并
- 其余字段：节点返回值覆盖（节点应只返回增量字段，由 `merge_state_update` 应用到基线）

## Checkpoint

- 业务表：`workspace/research_tasks.db`（与 `research_memory.db` 表级隔离）
- LangGraph：`SqliteSaver` → `workspace/langgraph_checkpoints.db`
- 恢复：加载最新业务 checkpoint 的 State，按 `completed_nodes` 跳过已完成节点；副作用经 `executed_side_effects` / `side_effect_log` 幂等。

# 科研空白发现智能体 — 架构说明
# Research Gap Discovery Agent — Architecture

## Goal

Help researchers **discover and verify candidate Research Gaps** with an evidence chain.  
Never claim automatic discovery of true novelty.

## Hybrid architecture

```
Deterministic LangGraph Workflow
  memory_recall → planner → search → paper_reader
  → analyzer → evidence_chain → gap_discover
  → gap_verify  ★ (only Tool-Calling / ReAct Agent here)
  → (conditional) cross_domain → finalize → report → END
```

**Why Agent only in `gap_verify`:** earlier stages are data acquisition / structuring; verification needs autonomous tool choice under budgets.  
**Why not multi-agent:** one shared `ResearchState` + one verify loop is enough for interview demo and keeps routing/debugging simple.

## Evidence chain

```
OA Fulltext → PDF sections → PaperCard
  → Forward citations → Citation context (CRITIQUE only)
  → Evolution + Limitation Lifecycle
  → Candidate Gap → Gap Verify Tool Agent → Report
```

## ResearchState

See `backend/src/gap_discovery/STATE.md` for field ownership, reducers, persistence, and frontend exposure.

Key control fields: `task_id`, `thread_id`, `current_node`, `completed_nodes`,
`verification_round`/`max_verification_rounds`, `tool_call_count`/`max_tool_calls`,
`visited_actions`, `tool_traces`, `warnings`, `verification_status`, `termination_reason`.

## Checkpoint / resume / isolation

- Business tasks DB: `workspace/research_tasks.db` (isolated from `research_memory.db`)
- LangGraph SqliteSaver: `workspace/langgraph_checkpoints.db`
- Each task: unique `task_id` and `thread_id`; states never share across threads
- After each node in SSE runner: save checkpoint + structured SSE `checkpoint_saved`
- Resume skips `completed_nodes`; side effects use `side_effect_log` / `executed_side_effects`
- Completed tasks refuse re-execution (idempotent resume)

## Tool-calling loop (`verify_agent.py` + `tool_runtime.py`)

Budgets: max rounds, max tool calls, per-tool timeout, task deadline, empty streak, duplicate action signatures.  
Traces store summaries/refs only (no full papers).  
Failures classified → retry (backoff+jitter) | rewrite query | ask_user | stop.

Termination semantics: `COMPLETED|KEEP|REFINED|REJECTED|INSUFFICIENT_EVIDENCE|BUDGET_EXCEEDED|TOOL_FAILURE|NEEDS_USER_INPUT`.

## Safety

- System / user / untrusted evidence partitions (`safety.py`)
- Tool whitelist only
- Injection phrases flagged into `warnings` (content still usable as academic text, not as control)
- Citation validation before report; strip unsupported novelty wording

## Evals

`evals/` — offline deterministic suite + hard rules (workflow/tool/evidence/report).  
Baseline artifacts under `evals/reports/`.

## Learning path（建议阅读顺序）

### 1. 入口与配置
| 文件 | 作用 |
|---|---|
| `backend/src/main.py` | FastAPI + SSE + task APIs |
| `backend/src/config.py` | 环境变量 / LLM / 搜索配置 |
| `frontend/src/api.ts` → `App.tsx` | 流式事件、Resume、可观测性 |

### 2. 编排骨架
| 文件 | 作用 |
|---|---|
| `gap_discovery/state.py` | 共享 `ResearchState` |
| `gap_discovery/STATE.md` | 字段设计 |
| `gap_discovery/graph.py` | LangGraph 拓扑 + checkpointer |
| `gap_discovery/runner.py` | State → SSE + checkpoint |
| `gap_discovery/tasks.py` | 任务/事件/幂等副作用 |
| `gap_discovery/pipeline.py` | 节点实现 |

### 3. Agent / 工具 / 安全
| 文件 | 作用 |
|---|---|
| `gap_discovery/verify_agent.py` | Tool-calling 验证循环 |
| `gap_discovery/tool_runtime.py` | 重试/超时/Trace |
| `gap_discovery/safety.py` | Injection + 引用安全 |
| `gap_discovery/rag.py` / `memory.py` | RAG / SQLite Memory |

## Fallback

PDF fail → abstract_only · citation fulltext fail → no invent critique  
API fail → seed corpus · embedding fail → lexical RAG  
Budget / tool failure → keep evidence, degrade certainty, no novelty claims

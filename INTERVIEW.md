# Interview Cheat Sheet — Research Gap Discovery Agent

## 30-second pitch

我基于 **LangGraph StateGraph** 构建 Research Gap Discovery Agent：前半段是确定性 Workflow（Planner→Search→PaperReader→Analyzer→GapDiscover）；**Gap Verification 阶段是真正的 Tool-Calling Agent**——模型根据证据缺口自主选择 `search_papers` / `recall_memory` / `retrieve_rag`，观察后再判定 REJECTED/REFINED/KEEP。证据检索使用 **Embedding → Chroma → Top-K → ContextBuilder** 的向量 RAG。系统明确只帮助发现和验证候选 Research Gap，不宣称自动发现真正创新点。

## Q&A

**旧 Deep Research 模式还在吗？**  
已删除。仓库只保留 Gap Discovery；入口一律走 `GapDiscoveryRunner`。

**LangGraph 真的在跑吗？**  
是。`REQUIRE_LANGGRAPH=true` 时必须成功 `graph.compile()`；`run_pipeline()` 走 `compiled.invoke`。SSE 流式路径用同一套节点函数 + `gap_verify` 条件路由，并额外发出 `node_start` / `state_patch` 以便前端高亮与 State 可视化。结构：线性边 + `gap_verify` 的 `add_conditional_edges` → cross_domain|finalize → report → END。

**为什么需要 Agent，而不是普通 Workflow？**  
前半段可固定；Gap 验证需要根据证据不足选择下一步工具，属于带 Observation 的决策循环。

**Tool Calling 具体怎么体现？**  
`verify_agent.py`：`ChatOpenAI.bind_tools([search_papers, recall_memory, retrieve_rag])`，多轮 tool_calls → ToolMessage → 再决策；`tool_trace` 写入结果并可 SSE 展示。

**RAG 在哪里用？**  
不是通用问答。PaperCard chunk 向量化进 Chroma；Gap Verify 的 `retrieve_rag` 工具与 ContextBuilder 使用 Top-K 证据。

**Memory 保存什么？**  
论文卡片、Gap 状态与原因、历史 query。`recall_memory` 工具可主动召回 REJECTED，避免重复。

**系统最大局限？**  
多为 abstract_only；跨域迁移仍需人工；检索覆盖不是全球完备；embedding 依赖 DashScope。

**若做成生产系统下一步？**  
PDF 全文、引用语境、Hybrid/Rerank、人工反馈闭环、Gap 评测集。

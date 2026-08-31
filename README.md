# 科研空白发现智能体
# Research Gap Discovery Agent

帮助研究者在给定主题下**发现并验证「候选」科研空白**（证据链 + 可恢复任务）。  
系统**不能**证明全球首次或绝对 novelty，也不会输出「已确认创新」类确定性结论。

## 它做什么

给定研究主题后自动：

召回记忆 → 规划检索 → 读文献/全文 → 构建局限生命周期与引用批评 → 提出候选空白 → 用工具调用智能体验证 → 生成带证据边界的报告。

## 架构要点

- **外层**：LangGraph 确定性流水线（规划与执行）
- **内层**：仅在「空白验证」环节使用可自主选工具的智能体
- **记忆**：短期任务状态 + 长期情节/语义/程序/实体记忆
- **取证**：论文块向量/词法检索（本轮证据索引）

详见 `ARCHITECTURE.md`、`docs/architecture-onepage.html`。

## 快速启动

```bash
# 后端
cd backend
source .venv/bin/activate   # 或按 pyproject 自行创建环境
cp .env.example .env        # 填入模型与检索相关密钥
python src/main.py          # 默认 :8000

# 前端
cd frontend
npm install
npm run dev                 # 默认 :5174
```

演示可离线种子库：`export FORCE_SEED_CORPUS=1`

## API（节选）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/research/stream` | SSE 流式研究 |
| POST | `/research/tasks` | 创建后台任务 |
| GET | `/research/tasks/{task_id}` | 任务快照 |
| POST | `/research/tasks/{task_id}/resume` | 恢复任务 |
| GET | `/research/tasks/{task_id}/events` | 事件 / 可选流式 |

## 测试

```bash
cd backend && source .venv/bin/activate
PYTHONPATH=src pytest tests/ -q
```

## 许可与声明

本项目用于科研辅助与教学演示；输出均为**候选方向**，需人工复核文献与新颖性。

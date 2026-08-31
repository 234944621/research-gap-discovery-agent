"""FastAPI entrypoint for Research Gap Discovery Agent."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

_BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BACKEND_DIR / ".env", override=True)
load_dotenv(override=False)

from config import Configuration, SearchAPI
from gap_discovery.graph import describe_pipeline
from gap_discovery.runner import GapDiscoveryRunner
from gap_discovery.tasks import get_task_store

logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <4}</level> | "
    "<cyan>{function}</cyan> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
    colorize=True,
)


class ResearchRequest(BaseModel):
    topic: str = Field(..., description="Research topic")
    search_api: SearchAPI | None = Field(
        default=None,
        description="Optional search backend override",
    )
    mode: str | None = Field(
        default="gap_discovery",
        description="Only gap_discovery is supported",
    )
    task_id: str | None = Field(default=None, description="Optional existing task id")
    resume: bool = Field(default=False, description="Resume from checkpoint if task_id set")


class ResearchResponse(BaseModel):
    report_markdown: str = Field(default="", description="Final markdown report")
    todo_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Unused legacy field kept for frontend compatibility",
    )
    task_id: str | None = None
    thread_id: str | None = None


class CreateTaskRequest(BaseModel):
    topic: str
    search_api: SearchAPI | None = None


class ResumeTaskRequest(BaseModel):
    topic: str | None = Field(
        default=None,
        description="Optional narrowed topic when resuming NEEDS_USER_INPUT",
    )


def _mask_secret(value: Optional[str], visible: int = 4) -> str:
    if not value:
        return "unset"
    if len(value) <= visible * 2:
        return "*" * len(value)
    return f"{value[:visible]}...{value[-visible:]}"


def _build_config(payload: ResearchRequest | CreateTaskRequest) -> Configuration:
    overrides: Dict[str, Any] = {}
    if payload.search_api is not None:
        overrides["search_api"] = payload.search_api
    return Configuration.from_env(overrides=overrides)


def create_app() -> FastAPI:
    app = FastAPI(title="Research Gap Discovery Agent")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def log_startup_configuration() -> None:
        config = Configuration.from_env()
        if config.llm_provider == "ollama":
            base_url = config.sanitized_ollama_url()
        elif config.llm_provider == "lmstudio":
            base_url = config.lmstudio_base_url
        else:
            base_url = config.llm_base_url or "unset"
        logger.info(
            "Gap Discovery config: provider=%s model=%s base_url=%s search_api=%s api_key=%s",
            config.llm_provider,
            config.resolved_model() or "unset",
            base_url,
            (
                config.search_api.value
                if isinstance(config.search_api, SearchAPI)
                else config.search_api
            ),
            _mask_secret(config.llm_api_key),
        )
        # Ensure task DB is ready
        get_task_store()

    @app.get("/healthz")
    def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/gap-discovery/pipeline")
    def gap_pipeline() -> Dict[str, Any]:
        return {
            "name": "Research Gap Discovery",
            "disclaimer": "帮助发现和验证候选 Research Gap，不宣称自动发现真正创新点",
            "nodes": describe_pipeline(),
        }

    @app.post("/research", response_model=ResearchResponse)
    def run_research(payload: ResearchRequest) -> ResearchResponse:
        try:
            config = _build_config(payload)
            runner = GapDiscoveryRunner(config=config)
            state = runner.run(payload.topic)
            return ResearchResponse(
                report_markdown=state.get("final_report") or "",
                task_id=state.get("task_id"),
                thread_id=state.get("thread_id"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            logger.exception("Research failed")
            raise HTTPException(status_code=500, detail="Research failed") from exc

    def _sse_from_iter(events: Iterator[dict[str, Any]]) -> Iterator[str]:
        try:
            for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                yield ": keepalive\n\n"
        except Exception as exc:  # pragma: no cover
            logger.exception("Streaming research failed")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"

    @app.post("/research/stream")
    def stream_research(payload: ResearchRequest) -> StreamingResponse:
        try:
            config = _build_config(payload)
            runner = GapDiscoveryRunner(config=config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            events = runner.run_stream(
                payload.topic,
                task_id=payload.task_id,
                resume=payload.resume,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"task not found: {exc}") from exc
        except RuntimeError as exc:
            if str(exc) == "TASK_ALREADY_COMPLETED":
                raise HTTPException(status_code=409, detail="task already completed") from exc
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return StreamingResponse(
            _sse_from_iter(events),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/gap-research/stream")
    def stream_gap_research(payload: ResearchRequest) -> StreamingResponse:
        """Alias of /research/stream for explicit Gap Discovery clients."""

        return stream_research(payload)

    # ---- Task APIs (checkpoint / resume / isolation) ----

    @app.post("/research/tasks")
    def create_task(payload: CreateTaskRequest) -> Dict[str, Any]:
        try:
            config = _build_config(payload)
            runner = GapDiscoveryRunner(config=config)
            created = runner.start_background_task(payload.topic)
            return {**created, "message": "task started"}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/research/tasks/{task_id}")
    def get_task(task_id: str) -> Dict[str, Any]:
        runner = GapDiscoveryRunner()
        try:
            return runner.get_task_snapshot(task_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"task not found: {task_id}") from None

    @app.post("/research/tasks/{task_id}/resume")
    def resume_task(task_id: str, payload: ResumeTaskRequest | None = None) -> Dict[str, Any]:
        runner = GapDiscoveryRunner()
        try:
            return runner.resume_background_task(
                task_id, topic=(payload.topic if payload else None)
            )
        except KeyError:
            raise HTTPException(status_code=404, detail=f"task not found: {task_id}") from None
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/research/tasks/{task_id}/events")
    def task_events(
        task_id: str,
        after_seq: int = Query(0, ge=0),
        stream: bool = Query(False),
    ):
        store = get_task_store()
        if not store.get_task(task_id):
            raise HTTPException(status_code=404, detail=f"task not found: {task_id}")

        if not stream:
            return {"task_id": task_id, "events": store.list_events(task_id, after_seq=after_seq)}

        def _event_sse() -> Iterator[str]:
            seq = after_seq
            idle = 0
            while idle < 600:  # ~10 min with 1s sleep
                batch = store.list_events(task_id, after_seq=seq, limit=100)
                if not batch:
                    task = store.get_task(task_id)
                    if task and task["status"] in {"completed", "failed"}:
                        yield f"data: {json.dumps({'type': 'done', 'task_id': task_id, 'status': task['status']}, ensure_ascii=False)}\n\n"
                        return
                    idle += 1
                    time.sleep(1)
                    yield ": keepalive\n\n"
                    continue
                idle = 0
                for ev in batch:
                    seq = int(ev.get("_seq") or seq)
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'error', 'detail': 'event stream idle timeout'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            _event_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    # Demo: no reload so in-flight SSE is not killed by file watchers.
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )

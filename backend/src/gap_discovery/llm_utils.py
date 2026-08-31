"""Shared LLM helpers for Research Gap Discovery."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from config import Configuration

logger = logging.getLogger(__name__)


def build_llm(config: Optional[Configuration] = None):
    """Create HelloAgentsLLM from Configuration / env."""

    from hello_agents import HelloAgentsLLM

    cfg = config or Configuration.from_env()
    kwargs: dict[str, Any] = {"temperature": 0.2}
    model_id = cfg.llm_model_id or cfg.local_llm
    if model_id:
        kwargs["model"] = model_id
    provider = (cfg.llm_provider or "").strip()
    if provider:
        kwargs["provider"] = provider
    if provider == "ollama":
        kwargs["base_url"] = cfg.sanitized_ollama_url()
        kwargs["api_key"] = cfg.llm_api_key or "ollama"
    elif provider == "lmstudio":
        kwargs["base_url"] = cfg.lmstudio_base_url
        if cfg.llm_api_key:
            kwargs["api_key"] = cfg.llm_api_key
    else:
        if cfg.llm_base_url:
            kwargs["base_url"] = cfg.llm_base_url
        if cfg.llm_api_key:
            kwargs["api_key"] = cfg.llm_api_key
    return HelloAgentsLLM(**kwargs)


def llm_chat(llm, system: str, user: str) -> str:
    """Single-turn chat returning plain text."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    # HelloAgentsLLM.invoke returns str in recent versions
    try:
        result = llm.invoke(messages)
    except TypeError:
        result = llm.invoke(messages, stream=False)
    if isinstance(result, str):
        return result.strip()
    # fallback stream collector
    if hasattr(result, "__iter__") and not isinstance(result, (dict, str)):
        chunks = []
        for item in result:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and "content" in item:
                chunks.append(str(item["content"]))
        return "".join(chunks).strip()
    return str(result).strip()


def extract_json(text: str) -> Any:
    """Extract JSON object/array from LLM output."""

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if not match:
        raise ValueError(f"No JSON found in LLM output: {text[:200]}")
    return json.loads(match.group(1))


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

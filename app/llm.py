"""LLM client abstraction + robust JSON handling.

Uses the OpenAI-compatible chat API, which Ollama exposes natively at
http://localhost:11434/v1 — so the same client works against Ollama,
LM Studio, Groq, OpenAI, etc. by changing LLM_BASE_URL / LLM_MODEL.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from openai import OpenAI

from .config import Config, get_config


class ChatLLM(Protocol):
    """Interface the graph nodes depend on (tests inject a fake)."""

    def chat(self, system: str, messages: list[dict[str, str]], json_mode: bool = False) -> str: ...


class OpenAICompatLLM:
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._client = OpenAI(
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
        )

    def chat(self, system: str, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        kwargs: dict[str, Any] = {}
        if json_mode:
            # Supported by Ollama's OpenAI-compatible endpoint and by OpenAI.
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature if not json_mode else 0.0,
            messages=[{"role": "system", "content": system}, *messages],
            **kwargs,
        )
        return response.choices[0].message.content or ""


class EmbeddingClient:
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._client = OpenAI(
            base_url=self.config.embed_base_url,
            api_key=self.config.embed_api_key,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.config.embed_model, input=texts)
        return [item.embedding for item in response.data]


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of an LLM reply, tolerating code fences
    and surrounding prose. Small local models don't always honor json_mode."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find the first balanced {...} block.
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None

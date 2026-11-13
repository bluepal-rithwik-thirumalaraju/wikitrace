"""Direct OpenAI transport for a small, separately configured WikiTrace smoke test."""

from __future__ import annotations

from typing import Any

import httpx

from wikitrace.harness.llm.credentials import env_value
from wikitrace.harness.llm.openrouter import ChatResponse, EmbedResponse, ModelNotInCatalogError


def _estimate_cost(model: str, data: dict[str, Any]) -> None:
    """Attach an estimated USD cost; OpenAI's responses report tokens but not charges."""
    usage = data.get("usage") or {}
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    if model == "gpt-6-luna":
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens", 0) or 0)
        written = int(details.get("cache_write_tokens", 0) or 0)
        uncached = max(0, prompt - cached - written)
        long_context = prompt > 272000
        input_scale = 2 if long_context else 1
        output_scale = 1.5 if long_context else 1
        usage["cost"] = (
            uncached * 0.10 * input_scale + cached * 0.01 * input_scale
            + written * 0.125 * input_scale + completion * 0.50 * output_scale
        ) / 1_000_000
    elif model == "text-embedding-3-small":
        usage["cost"] = prompt * 0.02 / 1_000_000
    else:
        return
    usage["cost_estimated"] = True
    data["usage"] = usage


class OpenAIClient:
    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 120,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._api_key = api_key

    def _headers(self) -> dict[str, str]:
        key = self._api_key or env_value("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment or .env")
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def validate_models(self, model_ids: list[str]) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/models", headers=self._headers())
            response.raise_for_status()
            catalog = {row["id"] for row in response.json().get("data", []) if "id" in row}
        requested = list(dict.fromkeys(model_ids))
        missing = [model for model in requested if model not in catalog]
        if missing:
            raise ModelNotInCatalogError(f"Configured model id(s) not in OpenAI catalog: {missing}")
        return {"validated": True, "requested_models": requested, "catalog_size": len(catalog)}

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 8192,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        effort = (reasoning_effort or "").lower()
        if effort:
            payload["reasoning_effort"] = "none" if effort == "off" else effort
        if not effort or effort in ("none", "off"):
            payload["temperature"] = temperature
        if response_format is not None:
            payload["response_format"] = response_format
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions", headers=self._headers(), json=payload,
            )
            response.raise_for_status()
            data = response.json()
        _estimate_cost(model, data)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices in OpenAI response: {data}")
        choice = choices[0]
        message = choice.get("message") or {}
        return ChatResponse(
            content=message.get("content") or "",
            raw=data,
            generation_id=data.get("id", "") or "",
            model_returned=data.get("model", model) or model,
            finish_reason=choice.get("finish_reason", "") or "",
            provider="OpenAI",
        )

    @staticmethod
    def generation_provider(generation_id: str) -> str:
        return "OpenAI" if generation_id else ""

    def embed(self, model: str, texts: list[str]) -> EmbedResponse:
        direct_model = model.removeprefix("openai/")
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/embeddings", headers=self._headers(),
                json={"model": direct_model, "input": list(texts)},
            )
            response.raise_for_status()
            data = response.json()
        _estimate_cost(direct_model, data)
        items = data.get("data") or []
        if len(items) != len(texts):
            raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(items)}")
        vectors = [list(item["embedding"]) for item in sorted(items, key=lambda i: i.get("index", 0))]
        return EmbedResponse(vectors=vectors, raw=data, model_returned=data.get("model", direct_model))

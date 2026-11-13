"""Direct OpenAI smoke configuration and transport, without billable calls."""

from __future__ import annotations

import json

import httpx
import pytest

from wikitrace.config import load_config
from wikitrace.harness.llm.credentials import env_value
from wikitrace.harness.llm.openai import OpenAIClient
from wikitrace.harness.terminal.runner import PiTraverseRunner
from wikitrace.harness.terminal.usage import _write_pi_agent_dir
from wikitrace.run.evaluate import _default_traverse_runner


def test_smoke_config_uses_direct_openai_and_tiny_split():
    cfg = load_config("configs/plancraft-openai-smoke.yaml")
    assert (cfg.models.default.provider, cfg.models.default.model) == ("openai", "gpt-6-luna")
    assert cfg.models.default.reasoning_effort == "none"
    assert cfg.dataset.limit_explicit_split is True
    assert (cfg.dataset.train_size, cfg.dataset.test_size) == (1, 1)
    assert cfg.llm.base_url == cfg.traverse.base_url == "https://api.openai.com/v1"
    assert cfg.traverse.provider == "openai"
    assert cfg.llm.openrouter_providers == []


def test_openai_key_is_read_from_dotenv_without_overriding_environment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text('OPENAI_API_KEY="sk-from-file"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert env_value("OPENAI_API_KEY") == "sk-from-file"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-process")
    assert env_value("OPENAI_API_KEY") == "sk-from-process"


def test_openai_chat_uses_chat_completions_schema_and_estimates_cost(monkeypatch):
    requests = []

    def send(self, request, **kwargs):
        requests.append(request)
        return httpx.Response(200, request=request, json={
            "id": "chatcmpl-test", "model": "gpt-6-luna",
            "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                      "prompt_tokens_details": {"cached_tokens": 40}},
        })

    monkeypatch.setattr(httpx.Client, "send", send)
    response = OpenAIClient(api_key="sk-test").chat(
        model="gpt-6-luna", messages=[{"role": "user", "content": "JSON, please"}],
        max_tokens=2048, reasoning_effort="none",
        response_format={"type": "json_schema", "json_schema": {"name": "Smoke", "schema": {}}},
    )
    payload = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/chat/completions"
    assert payload["model"] == "gpt-6-luna"
    assert payload["reasoning_effort"] == "none"
    assert payload["max_completion_tokens"] == 2048
    assert payload["response_format"]["type"] == "json_schema"
    assert response.content == '{"ok":true}'
    assert response.provider == "OpenAI"
    assert response.raw["usage"]["cost_estimated"] is True
    assert response.raw["usage"]["cost"] == pytest.approx((60 * 0.1 + 40 * 0.01 + 20 * 0.5) / 1e6)


def test_openai_catalog_and_embedding_model_name(monkeypatch):
    requests = []

    def send(self, request, **kwargs):
        requests.append(request)
        if request.url.path.endswith("/models"):
            data = {"data": [{"id": "gpt-6-luna"}]}
        else:
            data = {"model": "text-embedding-3-small", "data": [{"index": 0, "embedding": [0.5]}],
                    "usage": {"prompt_tokens": 10}}
        return httpx.Response(200, request=request, json=data)

    monkeypatch.setattr(httpx.Client, "send", send)
    client = OpenAIClient(api_key="sk-test")
    assert client.validate_models(["gpt-6-luna"])["validated"] is True
    response = client.embed("openai/text-embedding-3-small", ["task"])
    assert json.loads(requests[1].content)["model"] == "text-embedding-3-small"
    assert response.vectors == [[0.5]]
    assert response.raw["usage"]["cost"] == pytest.approx(10 * 0.02 / 1e6)


def test_pi_registration_and_dotenv_key_forwarding(tmp_path, monkeypatch):
    settings = _write_pi_agent_dir(
        tmp_path, "gpt-6-luna", 2048, provider_name="openai",
        base_url="https://api.openai.com/v1",
    ) / "models.json"
    provider = json.loads(settings.read_text(encoding="utf-8"))["providers"]["openai"]
    assert provider["api"] == "openai-responses"
    assert provider["apiKey"] == "OPENAI_API_KEY"
    assert provider["models"][0]["id"] == "gpt-6-luna"
    assert provider["models"][0]["maxTokens"] == 2048
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-from-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.warns(RuntimeWarning, match="full network"):
        runner = PiTraverseRunner(
            pi_cli="unused", model="gpt-6-luna", provider="openai",
            base_url="https://api.openai.com/v1", sandbox="none",
        )
    assert runner._env()["OPENAI_API_KEY"] == "sk-from-file"
    assert runner._env()["OPENAI_BASE_URL"] == "https://api.openai.com/v1"


def test_openai_pi_output_budget_uses_smoke_model_limit():
    cfg = load_config("configs/plancraft-openai-smoke.yaml")
    cfg = cfg.model_copy(update={"traverse": cfg.traverse.model_copy(update={"sandbox": "none"})})
    with pytest.warns(RuntimeWarning, match="full network"):
        runner = _default_traverse_runner(cfg, None)
    assert runner.max_output_tokens == cfg.models.default.max_tokens == 2048
    assert runner.extra_args[-2:] == ["--thinking", "off"]

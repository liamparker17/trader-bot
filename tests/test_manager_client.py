"""
Task 13 — src/manager/client.py: ManagerClient.

Covers:
- Champion prompt loading (champion.txt -> named file) and fallback to
  the highest vNNN.md when champion.txt is missing/invalid.
- call() happy path: forced tool_choice, parses tool_use block directly
  (no json.loads), returns (proposals, rationale, usage).
- Cost math: exact ZAR value for known token counts.
- Typed exception handling (RateLimitError / APIConnectionError /
  APIStatusError) all wrap into ManagerAPIUnavailable.
- No network: the anthropic client is always a fake/mock injected via
  constructor.
"""
import anthropic
import pytest

from src.config import Config
from src.manager.client import (
    ManagerAPIUnavailable,
    ManagerClient,
    cost_zar,
    load_champion_prompt,
    PRICE_INPUT_USD_PER_TOKEN,
    PRICE_OUTPUT_USD_PER_TOKEN,
)


def _config(**overrides):
    settings = {
        "manager": {
            "model": "claude-opus-4-8",
            "usd_zar_rate": 18.0,
        }
    }
    settings["manager"].update(overrides)
    return Config(settings=settings)


class FakeToolBlock:
    def __init__(self, input_dict, type_="tool_use"):
        self.type = type_
        self.input = input_dict


class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeResponse:
    def __init__(self, content, input_tokens, output_tokens):
        self.content = content
        self.usage = FakeUsage(input_tokens, output_tokens)


class FakeMessages:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return self._response


class FakeAnthropicClient:
    def __init__(self, response=None, exc=None):
        self.messages = FakeMessages(response=response, exc=exc)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def test_load_champion_prompt_reads_named_file(tmp_path):
    (tmp_path / "v001.md").write_text("PROMPT ONE", encoding="utf-8")
    (tmp_path / "champion.txt").write_text("v001.md", encoding="utf-8")
    assert load_champion_prompt(tmp_path) == "PROMPT ONE"


def test_load_champion_prompt_falls_back_when_champion_missing(tmp_path):
    (tmp_path / "v001.md").write_text("ONE", encoding="utf-8")
    (tmp_path / "v002.md").write_text("TWO", encoding="utf-8")
    (tmp_path / "v010.md").write_text("TEN", encoding="utf-8")
    # No champion.txt at all -> numeric (not lexical) highest wins.
    assert load_champion_prompt(tmp_path) == "TEN"


def test_load_champion_prompt_falls_back_when_champion_invalid(tmp_path):
    (tmp_path / "v001.md").write_text("ONE", encoding="utf-8")
    (tmp_path / "v003.md").write_text("THREE", encoding="utf-8")
    (tmp_path / "champion.txt").write_text("does_not_exist.md", encoding="utf-8")
    assert load_champion_prompt(tmp_path) == "THREE"


def test_load_champion_prompt_raises_when_no_prompts(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_champion_prompt(tmp_path)


# ---------------------------------------------------------------------------
# Cost math
# ---------------------------------------------------------------------------

def test_cost_zar_exact_value():
    # 1000 input tokens, 500 output tokens, rate 18.0
    usd = 1000 * PRICE_INPUT_USD_PER_TOKEN + 500 * PRICE_OUTPUT_USD_PER_TOKEN
    expected = usd * 18.0
    assert cost_zar(1000, 500, 18.0) == pytest.approx(expected)
    # Sanity: known numbers -> concrete float
    assert cost_zar(1000, 500, 18.0) == pytest.approx((1000 * 5e-6 + 500 * 25e-6) * 18.0)


# ---------------------------------------------------------------------------
# call() happy path
# ---------------------------------------------------------------------------

def test_call_happy_path_parses_tool_use_block(tmp_path):
    (tmp_path / "v001.md").write_text("system prompt text", encoding="utf-8")
    (tmp_path / "champion.txt").write_text("v001.md", encoding="utf-8")

    tool_input = {
        "adjustments": [
            {"key": "risk.risk_per_trade_pct", "value": 1.2, "reason": "drawdown rising"},
        ],
        "rationale": "conservative cycle",
    }
    response = FakeResponse(
        content=[FakeTextBlock("thinking out loud"), FakeToolBlock(tool_input)],
        input_tokens=1200,
        output_tokens=300,
    )
    fake_client = FakeAnthropicClient(response=response)

    manager_client = ManagerClient(_config(), client=fake_client, prompts_dir=tmp_path)
    proposals, rationale, usage = manager_client.call({"balance": 1000})

    assert proposals == tool_input["adjustments"]
    assert rationale == "conservative cycle"
    assert usage["input_tokens"] == 1200
    assert usage["output_tokens"] == 300
    expected_cost = (1200 * PRICE_INPUT_USD_PER_TOKEN + 300 * PRICE_OUTPUT_USD_PER_TOKEN) * 18.0
    assert usage["cost_zar"] == pytest.approx(expected_cost)

    # Verify the API call shape: forced tool_choice, no thinking/temperature/etc.
    call_kwargs = fake_client.messages.calls[0]
    assert call_kwargs["model"] == "claude-opus-4-8"
    assert call_kwargs["max_tokens"] == 2048
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "propose_adjustments"}
    for forbidden in ("thinking", "temperature", "top_p", "top_k"):
        assert forbidden not in call_kwargs
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call_kwargs["system"][0]["text"] == "system prompt text"
    tool_def = call_kwargs["tools"][0]
    assert tool_def["name"] == "propose_adjustments"
    assert tool_def["strict"] is True
    assert tool_def["input_schema"]["additionalProperties"] is False
    assert "adjustments" in tool_def["input_schema"]["required"]


def test_call_raises_manager_api_unavailable_when_no_tool_use_block(tmp_path):
    (tmp_path / "v001.md").write_text("sys", encoding="utf-8")
    response = FakeResponse(content=[FakeTextBlock("no tool call")], input_tokens=10, output_tokens=5)
    fake_client = FakeAnthropicClient(response=response)
    manager_client = ManagerClient(_config(), client=fake_client, prompts_dir=tmp_path)
    with pytest.raises(ManagerAPIUnavailable):
        manager_client.call({"balance": 1000})


# ---------------------------------------------------------------------------
# Typed exception handling
# ---------------------------------------------------------------------------

def _fake_httpx_request():
    import httpx
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _fake_httpx_response(status_code):
    import httpx
    request = _fake_httpx_request()
    return httpx.Response(status_code, request=request, json={"error": {"message": "boom"}})


@pytest.mark.parametrize("exc_factory", [
    lambda: anthropic.RateLimitError(
        message="rate limited", response=_fake_httpx_response(429), body=None
    ),
    lambda: anthropic.APIConnectionError(request=_fake_httpx_request()),
    lambda: anthropic.APIStatusError(
        message="server error", response=_fake_httpx_response(500), body=None
    ),
])
def test_call_wraps_typed_exceptions_in_manager_api_unavailable(tmp_path, exc_factory):
    (tmp_path / "v001.md").write_text("sys", encoding="utf-8")
    fake_client = FakeAnthropicClient(exc=exc_factory())
    manager_client = ManagerClient(_config(), client=fake_client, prompts_dir=tmp_path)
    with pytest.raises(ManagerAPIUnavailable):
        manager_client.call({"balance": 1000})

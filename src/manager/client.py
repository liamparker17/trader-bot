"""
Manager client — talks to the Anthropic API (Claude Opus) to get proposed
config adjustments for one manager cycle.

No trading logic lives here: this module only (1) loads the champion
system prompt, (2) makes one forced tool-call to the model with the
briefing JSON, and (3) parses the result into (proposals, rationale,
usage). Validation/clamping is `src.manager.policy`; scheduling is
`src.manager.scheduler`; wiring is `src.manager.runner`.

The Anthropic client is always injectable (constructor `client=` param)
so tests never touch the network — they pass a fake object exposing
`.messages.create(**kwargs)`.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import anthropic

from src.config import PROJECT_ROOT

logger = logging.getLogger("traderbot.manager.client")

PROMPTS_DIR = PROJECT_ROOT / "src" / "manager" / "prompts"

# Opus 4.8 per-token USD pricing — verify against current pricing.
PRICE_INPUT_USD_PER_TOKEN = 5e-6    # $5.00 / 1M input tokens
PRICE_OUTPUT_USD_PER_TOKEN = 25e-6  # $25.00 / 1M output tokens

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_USD_ZAR_RATE = 18.0
MAX_TOKENS = 2048
SDK_MAX_RETRIES = 3

TOOL_NAME = "propose_adjustments"

PROPOSE_ADJUSTMENTS_TOOL = {
    "name": TOOL_NAME,
    "description": (
        "Propose zero or more bounded config adjustments for this cycle, "
        "plus an overall rationale. An empty adjustments array is valid "
        "and often the correct choice."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "adjustments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Dotted lever key, e.g. 'risk.risk_per_trade_pct' or 'weight.XAU_USD'.",
                        },
                        "value": {
                            "type": "number",
                            "description": "Proposed new value for this lever.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Concrete, briefing-grounded reason for this specific adjustment.",
                        },
                    },
                    "required": ["key", "value", "reason"],
                    "additionalProperties": False,
                },
            },
            "rationale": {
                "type": "string",
                "description": "Overall rationale for this cycle's decision, even if adjustments is empty.",
            },
        },
        "required": ["adjustments", "rationale"],
        "additionalProperties": False,
    },
    "strict": True,
}

_VERSION_RE = re.compile(r"^v(\d+)\.md$")


class ManagerAPIUnavailable(Exception):
    """Raised when the Anthropic API is unavailable after SDK retries, or
    returns a response the client cannot use (e.g. no tool_use block)."""


def cost_zar(input_tokens: int, output_tokens: int, usd_zar_rate: float) -> float:
    """
    cost_zar = (input_tokens * price_in + output_tokens * price_out) * usd_zar_rate
    """
    usd = input_tokens * PRICE_INPUT_USD_PER_TOKEN + output_tokens * PRICE_OUTPUT_USD_PER_TOKEN
    return usd * usd_zar_rate


def _highest_version_file(prompts_dir: Path) -> Optional[Path]:
    candidates = []
    for path in prompts_dir.glob("v*.md"):
        match = _VERSION_RE.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[-1][1]


def load_champion_prompt(prompts_dir: Path = PROMPTS_DIR) -> str:
    """
    Load the system prompt named in `champion.txt`. If champion.txt is
    missing, empty, or names a file that doesn't exist, fall back to the
    highest-numbered `vNNN.md` file in `prompts_dir` (numeric, not
    lexical, sort). Raises FileNotFoundError if no prompt can be found.
    """
    champion_path = prompts_dir / "champion.txt"
    if champion_path.exists():
        name = champion_path.read_text(encoding="utf-8").strip()
        candidate = prompts_dir / name if name else None
        if candidate is not None and candidate.exists():
            return candidate.read_text(encoding="utf-8")
        logger.warning(
            "champion.txt names %r which doesn't exist under %s; falling back "
            "to highest vNNN.md", name, prompts_dir,
        )

    fallback = _highest_version_file(prompts_dir)
    if fallback is None:
        raise FileNotFoundError(f"No manager system prompt found under {prompts_dir}")
    return fallback.read_text(encoding="utf-8")


class ManagerClient:
    """
    Wraps one Anthropic Messages API call per manager cycle: forced
    `propose_adjustments` tool call, prompt-cached system prompt, briefing
    JSON as the (uncached) user message.
    """

    def __init__(
        self,
        config,
        client=None,
        prompts_dir: Path = PROMPTS_DIR,
    ):
        self.config = config
        self.model = config.get("manager.model", DEFAULT_MODEL)
        self.usd_zar_rate = config.get("manager.usd_zar_rate", DEFAULT_USD_ZAR_RATE)
        self.system_prompt = load_champion_prompt(prompts_dir)

        # Injectable so tests never hit the network. In production, the
        # Anthropic SDK reads ANTHROPIC_API_KEY from the environment
        # (loaded via python-dotenv in src.config.load_config()).
        self.client = client if client is not None else anthropic.Anthropic(max_retries=SDK_MAX_RETRIES)

    def call(self, briefing: dict) -> tuple[list, str, dict]:
        """
        One manager cycle's model call. Returns (proposals, rationale, usage)
        where `proposals` is the raw `adjustments` list (not yet validated
        or clamped — that's `src.manager.policy.validate_and_clamp`), and
        `usage` is `{input_tokens, output_tokens, cost_zar}`.

        Raises `ManagerAPIUnavailable` if the API call fails (after SDK
        retries) or the response has no usable tool_use block.
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {"role": "user", "content": json.dumps(briefing, default=str)},
                ],
                tools=[PROPOSE_ADJUSTMENTS_TOOL],
                tool_choice={"type": "tool", "name": TOOL_NAME},
            )
        except anthropic.RateLimitError as e:
            raise ManagerAPIUnavailable(f"rate limited: {e}") from e
        except anthropic.APIConnectionError as e:
            raise ManagerAPIUnavailable(f"connection error: {e}") from e
        except anthropic.APIStatusError as e:
            raise ManagerAPIUnavailable(f"API error: {e}") from e

        tool_block = next(
            (block for block in response.content if getattr(block, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            raise ManagerAPIUnavailable("Anthropic response contained no tool_use block")

        data = tool_block.input  # already a parsed dict — never json.loads a string
        proposals = data.get("adjustments", [])
        rationale = data.get("rationale", "")

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_zar": cost_zar(input_tokens, output_tokens, self.usd_zar_rate),
        }
        return proposals, rationale, usage

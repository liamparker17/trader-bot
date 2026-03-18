"""
AI Analyst — Claude-powered strategic trading layer.

This module provides an optional Claude API integration that acts as a
"senior trader" reviewing XGBoost signals before execution. It degrades
gracefully: if no API key is set, credits run out, or the API is down,
the bot continues trading on XGBoost + rules alone.

Usage in the trading pipeline:
    XGBoost Signal -> analyst.review_trade() -> Approve/Reject -> Executor
"""

import json
import logging
import os
import time
import threading
from typing import Optional

from src.config import Config

logger = logging.getLogger("traderbot.ai.analyst")


class AIAnalyst:
    """
    Claude-powered trade analyst.

    Gracefully degrades to pass-through mode when:
    - No API key configured
    - API credits exhausted
    - Network/API errors
    - Explicitly disabled in config
    """

    def __init__(self, config: Config):
        self.config = config
        self._client = None
        self._enabled = False
        self._model = "claude-sonnet-4-6"
        self._system_prompt: str | None = None
        self._consecutive_errors = 0
        self._max_errors_before_disable = 5
        self._error_cooldown_until = 0.0
        self._lock = threading.Lock()

        # Stats
        self.total_calls = 0
        self.total_approvals = 0
        self.total_rejections = 0
        self.total_errors = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        self._initialize()

    def _initialize(self):
        """Set up the Claude client if API key is available."""
        ai_config = self.config.get("ai_analyst", {})

        if not ai_config.get("enabled", True):
            logger.info("AI Analyst disabled in config.")
            return

        api_key = os.getenv(
            ai_config.get("api_key_env", "ANTHROPIC_API_KEY"), ""
        )
        if not api_key:
            logger.info(
                "AI Analyst: No ANTHROPIC_API_KEY found. "
                "Running in standalone ML-only mode."
            )
            return

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            self._model = ai_config.get("model", "claude-sonnet-4-6")
            self._enabled = True
            logger.info(
                f"AI Analyst initialized (model={self._model}). "
                "Claude will review trade signals."
            )
        except ImportError:
            logger.warning(
                "anthropic package not installed. "
                "Run: pip install anthropic"
            )
        except Exception as e:
            logger.error(f"AI Analyst init failed: {e}")

    @property
    def is_active(self) -> bool:
        """Whether the analyst is active and able to make API calls."""
        if not self._enabled or self._client is None:
            return False
        if time.time() < self._error_cooldown_until:
            return False
        return True

    def review_trade(
        self,
        instrument: str,
        direction: str,
        ml_confidence: float,
        features: dict,
        open_positions: list[dict] | None = None,
        daily_pnl: float = 0.0,
        balance: float = 0.0,
        atr_value: float = 0.0,
        spread: float = 0.0,
        recent_trades: list[dict] | None = None,
    ) -> dict:
        """
        Ask Claude to review a proposed trade.

        Returns:
            dict with keys:
                - approved (bool): Whether to take the trade
                - confidence (float): Claude's confidence (0-1)
                - reasoning (str): Why approved/rejected
                - modifications (dict): Any SL/TP/size adjustments
                - source (str): "ai_analyst" or "passthrough"
        """
        if not self.is_active:
            return self._passthrough("AI Analyst inactive")

        # Don't review low-confidence trades (save tokens)
        require_review = self.config.get("ai_analyst.require_approval", False)
        min_confidence = self.config.get(
            "ai_analyst.min_ml_confidence_for_review", 0.45
        )
        if not require_review and ml_confidence < min_confidence:
            return self._passthrough(
                f"ML confidence {ml_confidence:.1%} below review threshold"
            )

        from src.ai.prompts import build_trade_review_prompt, get_system_prompt

        prompt = build_trade_review_prompt(
            instrument=instrument,
            direction=direction,
            ml_confidence=ml_confidence,
            features=features,
            open_positions=open_positions or [],
            daily_pnl=daily_pnl,
            balance=balance,
            atr_value=atr_value,
            spread=spread,
            recent_trades=recent_trades,
        )

        response = self._call_claude(prompt, max_tokens=300)
        if response is None:
            return self._passthrough("API call failed")

        return self._parse_trade_review(response)

    def session_briefing(
        self,
        session: str,
        instruments: list[str],
        market_data: dict[str, dict],
        balance: float,
        daily_pnl: float = 0.0,
    ) -> dict | None:
        """
        Get a pre-session market briefing from Claude.

        Returns parsed briefing dict or None if unavailable.
        """
        if not self.is_active:
            return None

        from src.ai.prompts import build_session_briefing_prompt, get_system_prompt

        prompt = build_session_briefing_prompt(
            session=session,
            instruments=instruments,
            market_data=market_data,
            balance=balance,
            daily_pnl=daily_pnl,
        )

        response = self._call_claude(prompt, max_tokens=600)
        if response is None:
            return None

        return self._parse_json_response(response)

    def check_regime(
        self,
        instrument: str,
        features: dict,
        recent_candles_summary: str,
    ) -> dict | None:
        """
        Quick regime classification for an instrument.

        Returns parsed regime dict or None if unavailable.
        """
        if not self.is_active:
            return None

        from src.ai.prompts import build_regime_check_prompt

        prompt = build_regime_check_prompt(
            instrument=instrument,
            features=features,
            recent_candles_summary=recent_candles_summary,
        )

        # Use Haiku for regime checks (cheaper, fast enough)
        haiku_model = self.config.get("ai_analyst.regime_model", "claude-haiku-4-5-20251001")
        response = self._call_claude(prompt, max_tokens=200, model_override=haiku_model)
        if response is None:
            return None

        return self._parse_json_response(response)

    def session_review(
        self,
        session: str,
        trades: list[dict],
        daily_pnl: float,
        balance: float,
        win_rate: float,
    ) -> dict | None:
        """
        Post-session review and analysis.

        Returns parsed review dict or None if unavailable.
        """
        if not self.is_active:
            return None

        from src.ai.prompts import build_session_review_prompt

        prompt = build_session_review_prompt(
            session=session,
            trades=trades,
            daily_pnl=daily_pnl,
            balance=balance,
            win_rate=win_rate,
        )

        response = self._call_claude(prompt, max_tokens=500)
        if response is None:
            return None

        return self._parse_json_response(response)

    def get_stats(self) -> dict:
        """Return usage statistics."""
        return {
            "enabled": self._enabled,
            "active": self.is_active,
            "model": self._model,
            "total_calls": self.total_calls,
            "total_approvals": self.total_approvals,
            "total_rejections": self.total_rejections,
            "total_errors": self.total_errors,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": self._estimate_cost(),
        }

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _call_claude(
        self,
        user_prompt: str,
        max_tokens: int = 300,
        model_override: str | None = None,
    ) -> str | None:
        """
        Make an API call to Claude. Returns response text or None on failure.

        Thread-safe. Tracks token usage. Auto-disables after repeated errors.
        """
        from src.ai.prompts import get_system_prompt

        model = model_override or self._model

        with self._lock:
            try:
                start = time.time()
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=get_system_prompt(),
                    messages=[{"role": "user", "content": user_prompt}],
                )
                elapsed = time.time() - start

                # Track usage
                self.total_calls += 1
                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens
                self._consecutive_errors = 0

                text = response.content[0].text
                logger.debug(
                    f"Claude response ({model}, {elapsed:.1f}s, "
                    f"{response.usage.input_tokens}+{response.usage.output_tokens} tokens): "
                    f"{text[:100]}..."
                )
                return text

            except Exception as e:
                self.total_errors += 1
                self._consecutive_errors += 1
                logger.warning(f"Claude API error ({self._consecutive_errors}): {e}")

                if self._consecutive_errors >= self._max_errors_before_disable:
                    cooldown = 300  # 5 minutes
                    self._error_cooldown_until = time.time() + cooldown
                    logger.warning(
                        f"AI Analyst entering cooldown for {cooldown}s "
                        f"after {self._consecutive_errors} consecutive errors."
                    )

                return None

    def _parse_trade_review(self, response_text: str) -> dict:
        """Parse Claude's trade review response into a structured result."""
        parsed = self._parse_json_response(response_text)

        if parsed is None:
            logger.warning(f"Failed to parse trade review: {response_text[:200]}")
            return self._passthrough("Failed to parse AI response")

        decision = parsed.get("decision", "reject").lower()
        approved = decision == "approve"
        confidence = float(parsed.get("confidence", 0.5))
        reasoning = parsed.get("reasoning", "No reasoning provided")
        modifications = parsed.get("modifications", {})
        warnings = parsed.get("warnings", [])

        if approved:
            self.total_approvals += 1
        else:
            self.total_rejections += 1

        # Handle "modify" as approved with adjustments
        if decision == "modify":
            approved = True

        return {
            "approved": approved,
            "confidence": confidence,
            "reasoning": reasoning,
            "modifications": {
                "adjust_sl": modifications.get("adjust_sl"),
                "adjust_tp": modifications.get("adjust_tp"),
                "reduce_size": modifications.get("reduce_size"),
            },
            "warnings": warnings,
            "source": "ai_analyst",
        }

    def _parse_json_response(self, text: str) -> dict | None:
        """Extract JSON from Claude's response (handles markdown code blocks)."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from ```json ... ``` block
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

        # Try extracting from ``` ... ``` block
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

        # Try finding { ... } in the text
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        return None

    def _passthrough(self, reason: str) -> dict:
        """Return a passthrough result (trade proceeds without AI review)."""
        return {
            "approved": True,
            "confidence": 0.5,
            "reasoning": reason,
            "modifications": {
                "adjust_sl": None,
                "adjust_tp": None,
                "reduce_size": None,
            },
            "warnings": [],
            "source": "passthrough",
        }

    def _estimate_cost(self) -> float:
        """Estimate USD cost based on token usage (Sonnet 4.6 pricing)."""
        input_cost = self.total_input_tokens / 1_000_000 * 3.0
        output_cost = self.total_output_tokens / 1_000_000 * 15.0
        return round(input_cost + output_cost, 4)

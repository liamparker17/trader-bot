"""
Configuration loader for TraderBot.
Loads settings.yaml, instruments.yaml, and .env into a single Config object.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import yaml
from dotenv import load_dotenv


# Project root is one level up from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class Config:
    """Central configuration object for TraderBot."""

    settings: dict = field(default_factory=dict)
    instruments: dict = field(default_factory=dict)

    # MT5 / Exness credentials
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = ""
    mt5_terminal_path: str = ""

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @property
    def broker_environment(self) -> str:
        return self.get("broker.environment", "demo")

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """
        Access nested settings with dot notation.
        Example: config.get("risk.risk_per_trade_pct") -> 1.5
        """
        keys = dotted_key.split(".")
        value = self.settings
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_instrument(self, instrument: str) -> dict:
        """Get config for a specific instrument (e.g., 'EUR_USD')."""
        return self.instruments.get("instruments", {}).get(instrument, {})

    def get_enabled_instruments(self) -> list[str]:
        """Return list of enabled instrument names."""
        return [
            name
            for name, cfg in self.instruments.get("instruments", {}).items()
            if cfg.get("enabled", False)
        ]

    # Backward compat aliases for code that still references oanda_environment
    @property
    def oanda_environment(self) -> str:
        return self.broker_environment


def load_config() -> Config:
    """Load all configuration from files and environment variables."""
    # Load .env file
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Load YAML configs
    settings = _load_yaml(CONFIG_DIR / "settings.yaml")
    instruments = _load_yaml(CONFIG_DIR / "instruments.yaml")

    # Build config object
    # Parse MT5 login as int (0 if not set)
    mt5_login_str = os.getenv("MT5_LOGIN", "0")
    try:
        mt5_login = int(mt5_login_str)
    except ValueError:
        mt5_login = 0

    config = Config(
        settings=settings,
        instruments=instruments,
        mt5_login=mt5_login,
        mt5_password=os.getenv("MT5_PASSWORD", ""),
        mt5_server=os.getenv("MT5_SERVER", ""),
        mt5_terminal_path=os.getenv("MT5_TERMINAL_PATH", ""),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
    )

    # Ensure data directories exist
    (DATA_DIR / "historical").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "trade_logs").mkdir(parents=True, exist_ok=True)

    return config

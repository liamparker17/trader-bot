"""
Indicator Registry — Plugin system for adding/removing indicators dynamically.

Each indicator implements the BaseIndicator interface and registers itself.
The engine uses the registry to discover and run all active indicators.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

logger = logging.getLogger("traderbot.indicators")


class BaseIndicator(ABC):
    """
    Base class for all technical indicators.

    Every indicator must:
    1. Accept config params in __init__
    2. Implement calculate() which takes a candle DataFrame and returns
       a dict of named Series (one per output feature)
    3. Provide a name and list of output feature names
    """

    def __init__(self, **params):
        self.params = params

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique indicator name (e.g., 'ema', 'rsi')."""
        ...

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """List of output feature column names this indicator produces."""
        ...

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """
        Calculate indicator values from candle data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
                and a DatetimeIndex.

        Returns:
            Dict mapping feature name → Series of values (same index as df).
        """
        ...

    @property
    def min_periods(self) -> int:
        """Minimum number of candles needed before output is valid."""
        return max(self.params.values(), default=14)


class IndicatorRegistry:
    """
    Registry of available indicators.

    Supports:
    - Registering indicator classes
    - Instantiating indicators with config params
    - Enabling/disabling indicators at runtime
    """

    def __init__(self):
        self._classes: dict[str, type[BaseIndicator]] = {}
        self._instances: dict[str, BaseIndicator] = {}

    def register_class(self, indicator_cls: type[BaseIndicator]):
        """Register an indicator class (does not instantiate yet)."""
        # Create a temporary instance to get the name
        temp = indicator_cls.__new__(indicator_cls)
        temp.params = {}
        name = indicator_cls.__name__.lower()
        self._classes[name] = indicator_cls
        logger.debug(f"Registered indicator class: {name}")

    def instantiate(self, name: str, **params) -> BaseIndicator:
        """Create and activate an indicator instance with given params."""
        name = name.lower()
        if name not in self._classes:
            raise KeyError(f"Unknown indicator: {name}. Available: {list(self._classes.keys())}")

        instance = self._classes[name](**params)
        self._instances[instance.name] = instance
        logger.info(f"Activated indicator: {instance.name} with params {params}")
        return instance

    def remove(self, name: str):
        """Deactivate an indicator."""
        name = name.lower()
        if name in self._instances:
            del self._instances[name]
            logger.info(f"Deactivated indicator: {name}")

    def get_active(self) -> dict[str, BaseIndicator]:
        """Get all active indicator instances."""
        return dict(self._instances)

    def get_all_feature_names(self) -> list[str]:
        """Get combined list of all feature names from active indicators."""
        features = []
        for indicator in self._instances.values():
            features.extend(indicator.feature_names)
        return features

    def list_available(self) -> list[str]:
        """List all registered indicator class names."""
        return list(self._classes.keys())

    def list_active(self) -> list[str]:
        """List all active (instantiated) indicator names."""
        return list(self._instances.keys())


# Global registry instance
registry = IndicatorRegistry()

"""
Position Sizer — Calculates trade size based on account balance, risk rules,
and current market conditions.

Core formula:
    position_size = risk_amount / (sl_distance_pips * pip_value)

Where:
    risk_amount = balance * risk_per_trade_pct
    sl_distance_pips = ATR * sl_atr_multiplier (clamped to min/max)
    pip_value = value of 1 pip per 1 unit in account currency (ZAR)

Adjustments:
    - Halve size after N consecutive losses
    - Reduce by 25% in high volatility
    - Always round DOWN (never risk more than intended)
"""

import logging
import math

from src.config import Config

logger = logging.getLogger("traderbot.risk.sizer")


class PositionSizer:
    """
    Calculates position sizes and SL/TP prices for trades.

    All calculations prioritize safety:
    - Always rounds position size DOWN
    - Enforces minimum and maximum stop-loss distances
    - Caps effective leverage
    - Adjusts for volatility and losing streaks
    """

    def __init__(self, config: Config):
        self.config = config
        self.risk_pct = config.get("risk.risk_per_trade_pct", 1.5) / 100
        self.sl_atr_mult = config.get("risk.sl_atr_multiplier", 1.5)
        self.tp_atr_mult = config.get("risk.tp_atr_multiplier", 2.25)
        self.min_sl_pips = config.get("risk.min_sl_pips", 5)
        self.max_sl_pips = config.get("risk.max_sl_pips", 20)
        self.max_leverage = config.get("risk.max_effective_leverage", 5.0)
        self.consec_loss_reduce = config.get("risk.consecutive_loss_reduce_at", 3)
        self.high_vol_ratio = config.get("risk.high_volatility_atr_ratio", 2.0)

    def calculate(
        self,
        balance: float,
        instrument: str,
        direction: str,
        entry_price: float,
        atr_value: float,
        atr_ratio: float = 1.0,
        consecutive_losses: int = 0,
        current_spread: float = 0.0,
    ) -> dict | None:
        """
        Calculate position size, stop-loss, and take-profit.

        Args:
            balance: Current account balance in ZAR
            instrument: e.g. "EUR_USD"
            direction: "buy" or "sell"
            entry_price: Expected entry price
            atr_value: Current ATR value (in price units, not pips)
            atr_ratio: ATR / baseline ATR (volatility multiplier)
            consecutive_losses: Current consecutive loss streak
            current_spread: Current bid-ask spread (price units)

        Returns:
            Dict with position parameters, or None if trade should be skipped.
            Keys: units, stop_loss, take_profit, risk_amount, sl_pips,
                  tp_pips, effective_leverage, adjustments
        """
        inst_config = self.config.get_instrument(instrument)
        if not inst_config:
            logger.error(f"No config for instrument: {instrument}")
            return None

        pip_location = inst_config.get("pip_location", -4)
        pip_size = 10 ** pip_location  # e.g. 0.0001 for EUR/USD

        # Per-instrument SL/TP/risk overrides (fall back to global config)
        inst_sl_atr_mult = inst_config.get("sl_atr_multiplier", self.sl_atr_mult)
        inst_tp_atr_mult = inst_config.get("tp_atr_multiplier", self.tp_atr_mult)
        inst_risk_pct = inst_config.get("risk_per_trade_pct", self.risk_pct * 100) / 100

        # Step 1: Calculate SL distance in pips
        sl_pips = atr_value * inst_sl_atr_mult / pip_size
        inst_min_sl = inst_config.get("atr_sl_min_pips", self.min_sl_pips)
        inst_max_sl = inst_config.get("atr_sl_max_pips", self.max_sl_pips)
        sl_pips = max(inst_min_sl, min(sl_pips, inst_max_sl))

        # Step 2: Calculate TP distance
        tp_pips = sl_pips * (inst_tp_atr_mult / inst_sl_atr_mult)

        # Step 3: Base risk amount (per-instrument risk %)
        risk_amount = balance * inst_risk_pct

        adjustments = []

        # Step 4: Consecutive loss adjustment
        if consecutive_losses >= self.consec_loss_reduce:
            risk_amount *= 0.5
            adjustments.append(f"halved_risk_consec_losses_{consecutive_losses}")

        # Step 5: Volatility adjustment
        if atr_ratio > self.high_vol_ratio:
            risk_amount *= 0.5
            adjustments.append(f"halved_risk_high_vol_{atr_ratio:.2f}")
        elif atr_ratio > 1.5:
            risk_amount *= 0.75
            adjustments.append(f"reduced_risk_elevated_vol_{atr_ratio:.2f}")

        # Step 6: Calculate pip value in ZAR
        pip_value_per_unit = self._pip_value_zar(instrument, entry_price, pip_size)

        if pip_value_per_unit <= 0:
            logger.error(f"Invalid pip value for {instrument}")
            return None

        # Step 7: Position size (in units)
        sl_distance_price = sl_pips * pip_size
        units = risk_amount / (sl_pips * pip_value_per_unit)
        units = max(1, math.floor(units))  # Always round DOWN, minimum 1 unit

        # Step 8: Leverage check
        # For USD-base pairs (USD/JPY), 1 unit = 1 USD, notional = units * USD_ZAR
        # For XXX/USD pairs (EUR/USD), notional = units * entry_price * USD_ZAR
        if instrument.startswith("USD_") or instrument.startswith("USD"):
            position_value_zar = units * self._usd_to_zar()
        else:
            position_value_zar = units * entry_price * self._usd_to_zar()
        effective_leverage = position_value_zar / balance if balance > 0 else 999

        if effective_leverage > self.max_leverage:
            if instrument.startswith("USD_") or instrument.startswith("USD"):
                max_units = (balance * self.max_leverage) / self._usd_to_zar()
            else:
                max_units = (balance * self.max_leverage) / (entry_price * self._usd_to_zar())
            units = max(1, math.floor(max_units))
            adjustments.append(f"leverage_capped_{effective_leverage:.1f}x_to_{self.max_leverage}x")
            effective_leverage = self.max_leverage

        # Step 9: Calculate SL/TP prices
        if direction == "buy":
            stop_loss = entry_price - sl_distance_price
            take_profit = entry_price + (tp_pips * pip_size)
        else:
            stop_loss = entry_price + sl_distance_price
            take_profit = entry_price - (tp_pips * pip_size)

        # Step 10: Spread sanity check
        max_spread = inst_config.get("max_spread_pips", 3.0) * pip_size
        if current_spread > max_spread:
            logger.warning(
                f"Spread too wide for {instrument}: "
                f"{current_spread / pip_size:.1f} pips > max {max_spread / pip_size:.1f} pips"
            )
            return None

        result = {
            "units": units if direction == "buy" else -units,
            "abs_units": units,
            "stop_loss": round(stop_loss, abs(pip_location) + 1),
            "take_profit": round(take_profit, abs(pip_location) + 1),
            "risk_amount": risk_amount,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "reward_risk_ratio": tp_pips / sl_pips if sl_pips > 0 else 0,
            "effective_leverage": effective_leverage,
            "pip_value_per_unit": pip_value_per_unit,
            "adjustments": adjustments,
        }

        logger.info(
            f"Position sized: {instrument} {direction} | "
            f"{units} units | SL: {sl_pips:.1f} pips | TP: {tp_pips:.1f} pips | "
            f"Risk: R{risk_amount:.2f} | Leverage: {effective_leverage:.1f}x"
            + (f" | Adjustments: {adjustments}" if adjustments else "")
        )

        return result

    def _pip_value_zar(self, instrument: str, price: float, pip_size: float) -> float:
        """
        Calculate the value of 1 pip for 1 unit in ZAR.

        For USD-denominated accounts, this would be simpler. Since we're in ZAR,
        we need the USD/ZAR exchange rate to convert.

        Simplified approach:
        - For XXX/USD pairs (e.g. EUR/USD): pip_value = pip_size * USD/ZAR rate
        - For USD/XXX pairs (e.g. USD/JPY): pip_value = (pip_size / price) * USD/ZAR rate
        - For XAU/USD: pip_value = pip_size * USD/ZAR rate
        """
        usd_zar = self._usd_to_zar()

        inst = instrument.replace("_", "")
        if inst.endswith("USD"):
            # Quote currency is USD (EUR/USD, GBP/USD, XAU/USD)
            return pip_size * usd_zar
        elif inst.startswith("USD"):
            # Base currency is USD (USD/JPY)
            return (pip_size / price) * usd_zar
        elif inst.endswith("JPY"):
            # Cross pair with JPY quote (GBP/JPY): convert via USD/JPY
            usdjpy_approx = 150.0
            return (pip_size / usdjpy_approx) * usd_zar
        elif inst.endswith("GBP"):
            # Cross pair with GBP quote (EUR/GBP): convert via GBP/USD
            gbpusd_approx = 1.26
            return (pip_size * gbpusd_approx) * usd_zar
        else:
            # Other cross pairs — approximate
            return pip_size * usd_zar

    def _usd_to_zar(self) -> float:
        """
        Get approximate USD/ZAR rate.

        In production, this should fetch the live rate.
        For now, using a reasonable approximation.
        """
        # TODO: Fetch live USD/ZAR rate from OANDA
        # For now, approximate. At ~R18.5/USD as of 2026.
        return 18.5

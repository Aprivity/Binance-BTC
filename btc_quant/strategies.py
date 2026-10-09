"""Research-only entry/exit generators; existing paper trading stays EMA9/21."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import signals as original_ema_signals

STRATEGIES = ("ema", "triple_ema", "donchian", "supertrend_cci")
STRATEGY_DESCRIPTIONS = {
    "ema": "EMA9/21 crossover: original v0.2.0 strategy",
    "triple_ema": "EMA9/21 crossover confirmed by close > EMA200",
    "donchian": "20-bar prior-high breakout; 10-bar prior-low breakdown exit",
    "supertrend_cci": "ATR(14)x3 SuperTrend bullish + CCI(20)>0; bearish or CCI<-100 exit",
}


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               atr: pd.Series, multiplier: float = 3.0) -> pd.Series:
    """Completed-bar SuperTrend direction; never reads a future bar."""
    upper = ((high + low) / 2 + multiplier * atr).to_numpy(dtype=float)
    lower = ((high + low) / 2 - multiplier * atr).to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    trend = np.zeros(len(close), dtype=int)
    final_upper = np.full(len(close), np.nan)
    final_lower = np.full(len(close), np.nan)
    for i in range(len(close)):
        if not np.isfinite(upper[i]) or not np.isfinite(lower[i]):
            continue
        if i == 0 or trend[i - 1] == 0:
            final_upper[i] = upper[i]
            final_lower[i] = lower[i]
            trend[i] = 1 if c[i] >= (high.iloc[i] + low.iloc[i]) / 2 else -1
            continue
        final_upper[i] = (upper[i] if upper[i] < final_upper[i - 1]
                          or c[i - 1] > final_upper[i - 1] else final_upper[i - 1])
        final_lower[i] = (lower[i] if lower[i] > final_lower[i - 1]
                          or c[i - 1] < final_lower[i - 1] else final_lower[i - 1])
        if trend[i - 1] == 1:
            trend[i] = -1 if c[i] < final_lower[i] else 1
        else:
            trend[i] = 1 if c[i] > final_upper[i] else -1
    return pd.Series(trend, index=close.index, name="supertrend_direction")


def strategy_signals(df: pd.DataFrame, cfg, strategy: str = "ema") -> pd.DataFrame:
    """Emit buy_signal/sell_signal/atr using information known by each candle's close."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unsupported research strategy: {strategy}")
    x = original_ema_signals(df, cfg)
    if strategy == "ema":
        return x
    close = x["close"]
    if strategy == "triple_ema":
        ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
        x["ema200"] = ema200
        x["buy_signal"] = x.buy_signal & (close > ema200)
        # Exit unchanged: a fast/slow cross-down or the original protective bracket.
    elif strategy == "donchian":
        # Crucial shift: the signal bar MUST NOT be included in its own channel.
        upper = x.high.rolling(20, min_periods=20).max().shift(1)
        lower = x.low.rolling(10, min_periods=10).min().shift(1)
        breakout = close > upper
        breakdown = close < lower
        x["channel_high"] = upper
        x["channel_low"] = lower
        x["buy_signal"] = breakout & ~breakout.shift(1, fill_value=False) & x.atr.notna()
        x["sell_signal"] = breakdown & ~breakdown.shift(1, fill_value=False) & x.atr.notna()
    else:
        trend = supertrend(x.high, x.low, close, x.atr)
        typical = (x.high + x.low + close) / 3
        avg = typical.rolling(20, min_periods=20).mean()
        mad = typical.rolling(20, min_periods=20).apply(
            lambda z: np.mean(np.abs(z - np.mean(z))), raw=True)
        cci = (typical - avg) / (0.015 * mad.replace(0, np.nan))
        eligible = (trend == 1) & (cci > 0)
        bearish = (trend == -1) | (cci < -100)
        x["supertrend_direction"] = trend
        x["cci"] = cci
        x["buy_signal"] = eligible & ~eligible.shift(1, fill_value=False) & x.atr.notna()
        x["sell_signal"] = bearish & ~bearish.shift(1, fill_value=False) & x.atr.notna()
    x["buy_signal"] = x.buy_signal.fillna(False).astype(bool)
    x["sell_signal"] = x.sell_signal.fillna(False).astype(bool)
    return x

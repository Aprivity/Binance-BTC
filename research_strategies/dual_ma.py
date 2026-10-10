"""Research-only, researcher-defined dual moving average BTC spot baselines.

Not a verified transcription of https://www.youtube.com/watch?v=F2FJeRsUzNE.
No futures, shorts, leverage, API orders or automatic activation.
Signals are based only on each completed candle.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Parameters:
    fast: int
    slow: int
    kind: str = "ema"
    confirm_bars: int = 1

    def __post_init__(self):
        if not (2 <= self.fast < self.slow <= 200):
            raise ValueError("2 <= fast < slow <= 200 required")
        if self.kind not in ("ema", "sma") or not (1 <= self.confirm_bars <= 5):
            raise ValueError("Unknown moving average or confirmation bars")


def calculate(frame: pd.DataFrame, params: Parameters) -> pd.DataFrame:
    close = pd.to_numeric(frame["close"], errors="raise").astype(float)
    if params.kind == "ema":
        fast = close.ewm(span=params.fast, min_periods=params.fast, adjust=False).mean()
        slow = close.ewm(span=params.slow, min_periods=params.slow, adjust=False).mean()
    else:
        fast = close.rolling(params.fast, min_periods=params.fast).mean()
        slow = close.rolling(params.slow, min_periods=params.slow).mean()
    above = fast.gt(slow) & fast.notna() & slow.notna()
    confirmed = above.rolling(params.confirm_bars, min_periods=params.confirm_bars).sum().eq(params.confirm_bars)
    buy = confirmed & ~confirmed.shift(1, fill_value=False)
    sell = fast.lt(slow) & fast.notna() & slow.notna()
    return pd.DataFrame({"buy_signal": buy.astype(bool), "sell_signal": sell.astype(bool)}, index=frame.index)


def ema_9_21(frame, cfg):
    return calculate(frame, Parameters(9, 21, "ema"))


def ema_9_21_confirm2(frame, cfg):
    return calculate(frame, Parameters(9, 21, "ema", 2))


def ema_12_26(frame, cfg):
    return calculate(frame, Parameters(12, 26, "ema"))


def ema_20_60(frame, cfg):
    return calculate(frame, Parameters(20, 60, "ema"))


def sma_20_60(frame, cfg):
    return calculate(frame, Parameters(20, 60, "sma"))


CANDIDATES = {
    "ema_9_21": ema_9_21,
    "ema_9_21_confirm2": ema_9_21_confirm2,
    "ema_12_26": ema_12_26,
    "ema_20_60": ema_20_60,
    "sma_20_60": sma_20_60,
}

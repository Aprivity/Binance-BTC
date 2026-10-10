"""Research approximation of the video's six-MA system, BTC SPOT LONG ONLY.

Source: https://www.youtube.com/watch?v=vdCdg4MdwBs
Video explicitly uses MA/EMA(20,60,120), (1) MA compression entry at zone
retest and (2) FIRST MA20 pullback after dispersion, zone/MA20 structural
stops and fixed 3R as one of three exit methods. It does NOT specify exact
numerical thresholds, confirmation bars, validity periods or price buffers.
ALL threshold values below are researcher choices, NOT the creator's rules.
No leverage/shorts, no hindsight, no auto-parameter search and no exchange IO.

Entry at next opening bar via generic BTC-only backtesting framework.
Profit target is computed at *actual simulated fill* as fill+3*(fill-stop).
Stops are executable simulated price barriers, not the video's subjective
'有效跌破' discretionary judgement. Caveat: OHLC stop-first resolution.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResearchSettings:
    # Below are experimental quantification choices absent from the video.
    compression_spread_pct: float = 1.2   # max(six MAs)-min(six MAs) / close * 100
    min_compression_bars: int = 3
    zone_lookback_bars: int = 12           # only last N compressed bars to freeze zone
    zone_lifetime_bars: int = 24           # abandon old setups
    breakout_margin_pct: float = 0.5      # close above frozen zone top to confirm breakout
    retest_tolerance_pct: float = 0.2     # low may stay this much above zone top
    stop_buffer_pct: float = 0.1          # below support
    reward_risk: float = 3.0

    def __post_init__(self):
        if (not (0 < self.compression_spread_pct < 25)
                or not (2 <= self.min_compression_bars <= self.zone_lookback_bars)
                or not (1 <= self.zone_lifetime_bars <= 500)
                or not (0 <= self.breakout_margin_pct <= 5)
                or not (0 <= self.retest_tolerance_pct <= 5)
                or not (0 <= self.stop_buffer_pct <= 5)
                or not (1 <= self.reward_risk <= 100)):
            raise ValueError("Invalid experimental quantification settings")


FIXED_RESEARCH = ResearchSettings()


def six_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Six completed-candle SMA/EMA prices; no centered/future window."""
    result = pd.DataFrame(index=df.index)
    for period in (20,60,120):
        result[f"ma{period}"] = df.close.rolling(period, min_periods=period).mean()
        result[f"ema{period}"] = df.close.ewm(span=period, min_periods=period, adjust=False).mean()
    return result


def _signals(df: pd.DataFrame, cfg, mode: str, opt: ResearchSettings) -> pd.DataFrame:
    if cfg.symbols != ("BTC/USDT",):
        raise ValueError("Video adaptation is BTC spot only")
    if mode not in ("compression_retest", "first_ma20_retest"):
        raise ValueError("Unknown research signal")
    ma = six_lines(df)
    n = len(df)
    buy = np.zeros(n, dtype=bool)
    sell = np.zeros(n, dtype=bool)  # exit only at preplanned price brackets
    stop = np.full(n, np.nan)
    rr = np.full(n, np.nan)
    dense = ((ma.max(axis=1)-ma.min(axis=1)) / df.close * 100 <= opt.compression_spread_pct)
    ready = ma.notna().all(axis=1).to_numpy(dtype=bool)
    dense = (dense & ready).to_numpy(dtype=bool)
    high_ma = ma.max(axis=1).to_numpy(dtype=float)
    low_ma = ma.min(axis=1).to_numpy(dtype=float)
    ma20 = ma.ma20.to_numpy(dtype=float)
    valid_uptrend = ((ma.ma20 > ma.ma60) & (ma.ma60 > ma.ma120) &
                     (ma.ema20 > ma.ema60) & (ma.ema60 > ma.ema120)).to_numpy(dtype=bool)
    opens = df.open.to_numpy(dtype=float)
    lows = df.low.to_numpy(dtype=float)
    closes = df.close.to_numpy(dtype=float)
    span, block, setup = 0, [], None
    # setup = {stage, top, bottom, breakout_index, start_index, trend_armed}
    for i in range(n):
        if not ready[i]:
            continue
        # Check breakout against the prior *frozen* zone BEFORE checking whether
        # the MAs are still compressed on today's candle. The two can coexist.
        if setup is not None and setup["stage"] == "zone":
            if closes[i] < setup["bottom"]:
                setup = None
            elif (i > setup["start_index"] and
                  closes[i] > setup["top"]*(1+opt.breakout_margin_pct/100)
                  and closes[i] > opens[i]):
                setup["stage"] = "broken"
                setup["breakout_index"] = i
                continue
        if setup is not None and setup["stage"] == "broken":
            if i - setup["breakout_index"] > opt.zone_lifetime_bars:
                setup = None
                continue
            if i <= setup["breakout_index"]:
                continue
            top,bottom = setup["top"],setup["bottom"]
            if closes[i] < bottom:
                setup = None
                continue
            if mode == "compression_retest":
                touched = lows[i] <= top*(1+opt.retest_tolerance_pct/100)
                held = closes[i] > top and closes[i] > opens[i]
                if touched:
                    if held:
                        candidate = bottom*(1-opt.stop_buffer_pct/100)
                        if math.isfinite(candidate) and 0 < candidate < closes[i]:
                            buy[i] = True
                            stop[i] = candidate
                            rr[i] = opt.reward_risk
                    setup = None  # only first attempt; avoid arbitrary reentries
            else:
                if not setup["trend_armed"]:
                    if valid_uptrend[i] and closes[i] > ma20[i]:
                        setup["trend_armed"] = True
                    continue
                touched = lows[i] <= ma20[i]*(1+opt.retest_tolerance_pct/100)
                if touched:
                    held = (valid_uptrend[i] and closes[i] > ma20[i]
                            and closes[i] > opens[i])
                    if held:
                        candidate = ma20[i]*(1-opt.stop_buffer_pct/100)
                        if math.isfinite(candidate) and 0 < candidate < closes[i]:
                            buy[i] = True
                            stop[i] = candidate
                            rr[i] = opt.reward_risk
                    setup = None  # first touch only, successful or not
            continue
        if dense[i]:
            span += 1
            block.append((low_ma[i], high_ma[i]))
            block = block[-opt.zone_lookback_bars:]
            if span >= opt.min_compression_bars:
                setup = {"stage":"zone","top":max(item[1] for item in block),
                         "bottom":min(item[0] for item in block),
                         "start_index":i,"breakout_index":None,"trend_armed":False}
        else:
            span = 0
            block = []
            if setup is not None and i-setup["start_index"] > opt.zone_lifetime_bars:
                setup = None
    return pd.DataFrame({"buy_signal":buy,"sell_signal":sell,
                         "initial_stop_price":stop,"reward_risk":rr},index=df.index)


def compression_retest(frame: pd.DataFrame, cfg) -> pd.DataFrame:
    """First bullish retest of a six-MA compression zone after breakout."""
    return _signals(frame,cfg,"compression_retest",FIXED_RESEARCH)


def first_ma20_retest(frame: pd.DataFrame, cfg) -> pd.DataFrame:
    """First confirmed MA20 touch after compression breakout and bull alignment."""
    return _signals(frame,cfg,"first_ma20_retest",FIXED_RESEARCH)
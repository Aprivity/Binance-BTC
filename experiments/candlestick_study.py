"""BTC candlestick event study. Research only; no live trading or default strategy.

Pre-registered rules (do not optimize on evaluation period):
- BTC/USDT Binance spot 1h/4h/1d completed OHLCV, 2020-01-01 .. 2026-10-10 exclusive.
- 2020-2023 descriptive development; 2024-2026 evaluation.
- Four fixed entry patterns; event at closed candle t, enter open[t+1].
- Primary horizon six bars, diagnostics three/twelve bars; exit open[t+1+h].
- 10 bps fee + 5 bps slippage each side.
- Disjoint events per strategy/horizon; calendar-month-matched random-entry placebo.
- This is an event study, not a compounded account backtest; does not estimate realized portfolio CAGR.
"""
from __future__ import annotations

import json
import math
import os
import time
import requests
from pathlib import Path

import numpy as np
import pandas as pd

from btc_quant.config import STEPS

def fetch_gap_aware(timeframe: str):
    """Fetch authentic public Binance spot bars; detect gaps and never impute them."""
    step = STEPS[timeframe]
    cursor = int(START.timestamp()*1000)
    end_ms = int(END.timestamp()*1000)
    url = "https://data-api.binance.vision/api/v3/klines"
    session = requests.Session()
    raw = []
    while cursor < end_ms:
        params = {"symbol":"BTCUSDT", "interval":timeframe,
                  "startTime":cursor, "endTime":end_ms-1, "limit":1000}
        for retry in range(4):
            try:
                r = session.get(url, params=params, timeout=25)
                r.raise_for_status()
                batch = r.json()
                if not isinstance(batch, list):
                    raise ValueError("Binance response must be a list of candles")
                break
            except (requests.RequestException, ValueError):
                if retry == 3:
                    raise
                time.sleep(2**retry)
        if not batch:
            break
        raw.extend(batch)
        new_cursor = int(batch[-1][0]) + step
        if new_cursor <= cursor:
            raise RuntimeError("Stalled pagination")
        cursor = new_cursor
        time.sleep(0.04)
    rows = []
    for k in raw:
        t = int(k[0])
        if t < int(START.timestamp()*1000) or t + step > end_ms:
            continue
        rows.append((t,*[float(x) for x in k[1:6]],t+step-1))
    cols = ["open_time_ms","open","high","low","close","base_volume","close_time_ms"]
    df = pd.DataFrame(rows,columns=cols).sort_values("open_time_ms").reset_index(drop=True)
    if df.empty or df.open_time_ms.duplicated().any():
        raise RuntimeError("Empty or duplicate raw Binance candles")
    values = df[["open","high","low","close","base_volume"]].to_numpy()
    if (not np.isfinite(values).all() or (values[:,:4] <= 0).any()
        or (values[:,4] < 0).any() or (df.high < df[["open","low","close"]].max(axis=1)).any()
        or (df.low > df[["open","high","close"]].min(axis=1)).any()
        or (df.open_time_ms % step != 0).any()):
        raise RuntimeError("Invalid raw Binance OHLCV")
    delta = df.open_time_ms.diff().fillna(step)
    if (delta <= 0).any() or (delta % step != 0).any():
        raise RuntimeError("Irregular Binance timestamp intervals")
    gaps = (delta != step).to_numpy()
    if np.sum(gaps) > max(100, 0.002*len(df)):
        raise RuntimeError("Too many Binance gaps to study safely")
    segments = np.cumsum(gaps)
    return df, int(np.sum(gaps)), segments

START = pd.Timestamp("2020-01-01", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
END = pd.Timestamp("2026-10-10", tz="UTC")
TIMEFRAMES = ("1h", "4h", "1d")
PATTERNS = ("morning_star", "bullish_engulfing", "bullish_harami", "breakout_20")
HORIZONS = (3, 6, 12)
COST_BPS_SIDE = 15  # 10 fee + 5 slippage
SEED = 260101
NULL_DRAWS = 400


def signals(df: pd.DataFrame) -> dict[str, np.ndarray]:
    op = df.open.to_numpy(float)
    cl = df.close.to_numpy(float)
    hi = df.high.to_numpy(float)
    lo = df.low.to_numpy(float)
    body = np.abs(cl - op)
    rng = np.maximum(hi - lo, 1e-12)
    out = {k: np.zeros(len(df), dtype=bool) for k in PATTERNS}
    for t in range(20, len(df)):
        # Traditional gap requirements intentionally omitted for 24/7 BTC.
        large_first = op[t-2] > cl[t-2] and body[t-2] >= 0.50*rng[t-2]
        small_middle = body[t-1] <= 0.35*body[t-2] if body[t-2] else False
        good_third = cl[t] > op[t] and body[t] >= 0.50*rng[t]
        falling = cl[t-2] < cl[t-6]
        if (large_first and small_middle and good_third and falling
                and lo[t-1] <= cl[t-2] and cl[t] >= (op[t-2]+cl[t-2])/2):
            out["morning_star"][t] = True
        if (op[t-1] > cl[t-1] and cl[t] > op[t]
                and op[t] <= cl[t-1] and cl[t] >= op[t-1]
                and body[t-1] >= 0.35*rng[t-1] and body[t] >= 0.35*rng[t]):
            out["bullish_engulfing"][t] = True
        if (op[t-1] > cl[t-1] and body[t-1] >= 0.50*rng[t-1]
                and cl[t] > op[t] and op[t] >= cl[t-1]
                and cl[t] <= op[t-1] and body[t] <= 0.60*body[t-1]):
            out["bullish_harami"][t] = True
        if cl[t] > np.max(hi[t-20:t]):
            out["breakout_20"][t] = True
    return out


def thin(indices: np.ndarray, horizon: int) -> np.ndarray:
    """Accept next setup only after previous holding window has finished."""
    kept = []
    last_exit = -1
    for i in indices:
        if i > last_exit:
            kept.append(int(i))
            last_exit = int(i) + 1 + horizon
    return np.asarray(kept, dtype=int)


def sample_stats(df: pd.DataFrame, mask: np.ndarray, timeframe: str,
                 pattern: str, horizon: int, phase: str, rng: np.random.Generator, segments: np.ndarray) -> dict:
    # A signal at index i fills at open[i+1] and exits at open[i+1+h].
    n = len(df)
    opens = df.open.to_numpy(float)
    timestamps = pd.to_datetime(df.open_time_ms, unit="ms", utc=True)
    valid_idx = np.flatnonzero(mask & (np.arange(n) + 1 + horizon < n)
                               & (np.arange(n) >= 30))
    # Exclude every trade whose 20-bar pattern history or full holding window crosses a feed gap.
    valid_idx = valid_idx[segments[valid_idx-20] == segments[valid_idx+1+horizon]]
    events = thin(valid_idx, horizon)
    if not len(events):
        return dict(timeframe=timeframe, pattern=pattern, horizon=horizon, phase=phase,
                    n=0, mean_gross_pct=None, mean_net_pct=None, median_net_pct=None,
                    win_rate_pct=None, random_mean_net_pct=None, lift_vs_random_pp=None,
                    permutation_p=None, month_bootstrap_ci95_pct=[None, None])
    buy = opens[events+1]
    sell = opens[events+1+horizon]
    fee = COST_BPS_SIDE / 1e4
    net = (sell*(1-fee))/(buy*(1+fee))-1
    gross = sell/buy-1
    months = timestamps.dt.to_period("M").to_numpy()
    # Random placebo includes all valid entry points within each event's calendar month.
    # This controls broad month-level market regimes, not momentum or volatility.
    all_candidates = np.arange(n, dtype=int)
    all_candidates = all_candidates[(all_candidates >= 30) & (all_candidates + 1 + horizon < n)]
    all_candidates = all_candidates[segments[all_candidates-20] == segments[all_candidates+1+horizon]]
    by_month = {}
    for month in sorted(set(months[events])):
        eligible = all_candidates[months[all_candidates] == month]
        by_month[month] = eligible
    month_counts = {month: int(np.sum(months[events] == month)) for month in by_month}
    null = np.empty(NULL_DRAWS, dtype=float)
    for b in range(NULL_DRAWS):
        picks = np.concatenate([
            rng.choice(by_month[m], size=count, replace=(count > len(by_month[m])))
            for m,count in month_counts.items()
        ])
        null[b] = np.mean((opens[picks+1+horizon]*(1-fee))
                          /(opens[picks+1]*(1+fee))-1)
    # Resample months rather than individual overlapping events.
    month_groups = [net[months[events] == m] for m in month_counts]
    bootstrap = np.empty(400, dtype=float)
    for b in range(400):
        draws = rng.integers(0, len(month_groups), size=len(month_groups))
        bootstrap[b] = np.concatenate([month_groups[j] for j in draws]).mean()
    mean_net = float(np.mean(net))
    return {
        "timeframe": timeframe, "pattern": pattern, "horizon": horizon, "phase": phase,
        "n": int(len(events)), "mean_gross_pct": round(float(np.mean(gross)*100), 4),
        "mean_net_pct": round(mean_net*100, 4),
        "median_net_pct": round(float(np.median(net)*100), 4),
        "win_rate_pct": round(float(np.mean(net>0)*100), 2),
        "random_mean_net_pct": round(float(np.mean(null)*100), 4),
        "lift_vs_random_pp": round(float((mean_net - np.mean(null))*100), 4),
        "permutation_p": round(float((1 + np.sum(null >= mean_net))/(NULL_DRAWS+1)), 4),
        "month_bootstrap_ci95_pct": [round(float(v*100), 4) for v in np.quantile(bootstrap,[.025,.975])],
    }


def main() -> None:
    out = Path("outputs/candlestick-research")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    records = []
    source = []
    for timeframe in TIMEFRAMES:
        print(f"FETCH_START timeframe={timeframe}", flush=True)
        # 2550d covers Jan 2020 for run date Oct 2026; cache only in current runner.
        df, gap_count, segments = fetch_gap_aware(timeframe)
        stamp = pd.to_datetime(df.open_time_ms, unit="ms", utc=True)
        df = df.loc[(stamp >= START) & (stamp < END)].reset_index(drop=True)
        stamp = pd.to_datetime(df.open_time_ms, unit="ms", utc=True)
        if df.empty or stamp.iloc[0] > START or stamp.iloc[-1] < END - pd.Timedelta({"1h": "2h", "4h": "8h", "1d": "2d"}[timeframe]):
            raise RuntimeError(f"Incomplete history for {timeframe}: {stamp.min()} .. {stamp.max()}")
        if len(df) < 1000:
            raise RuntimeError(f"Suspiciously short history for {timeframe}: {len(df)}")
        source.append({"timeframe":timeframe,"rows":len(df),"first":stamp.iloc[0].isoformat(),
                       "last":stamp.iloc[-1].isoformat(),"source":"Binance public spot klines",
                       "missing_intervals":int((df.open_time_ms.diff().fillna(STEPS[timeframe]) / STEPS[timeframe] - 1).sum()),
                       "gap_spans":gap_count})
        print(f"DATA_OK {source[-1]}", flush=True)
        pattern_map = signals(df)
        for phase, phase_start, phase_end in (
            ("development_2020_2023", START, TEST_START),
            ("evaluation_2024_2026", TEST_START, END),
        ):
            phase_mask = ((stamp >= phase_start) & (stamp < phase_end)).to_numpy()
            for pattern in PATTERNS:
                for h in HORIZONS:
                    r = sample_stats(df, pattern_map[pattern] & phase_mask,
                                     timeframe, pattern, h, phase, rng, segments)
                    records.append(r)
                    if h == 6:
                        print("RESULT " + json.dumps(r, ensure_ascii=False), flush=True)

    tab = pd.DataFrame(records)
    tab.to_csv(out/"summary.csv", index=False)
    primary = tab[(tab.phase == "evaluation_2024_2026") & (tab.horizon == 6)]
    # Bonferroni over the 12 pre-specified primary comparisons.
    primary = primary.copy()
    primary["bonferroni_p_12"] = primary.permutation_p.map(lambda x: min(1.0,round(float(x)*12,4)) if pd.notna(x) else None)
    primary.to_csv(out/"primary_evaluation.csv",index=False)
    manifest = {
        "source": source,
        "frozen_range_utc": [START.isoformat(),END.isoformat()],
        "holdout_start_utc":TEST_START.isoformat(),
        "unmodified_strategy_framework":True,
        "design":"event study, no actual portfolio equity, no live orders",
        "entry":"next candle open after pattern close",
        "exit":"next open 3/6/12 candles after entry; primary=6",
        "total_round_trip_cost_bps_approx":30,
        "primary_comparisons":12,
        "permutation_draws":NULL_DRAWS,
        "null":"month-matched random-entry sample; one-sided actual > placebo",
        "ci":"calendar-month cluster bootstrap (descriptive; not a trading guarantee)",
        "limitations":[
          "Pattern identification and event selection use this fixed code; no parameter optimization.",
          "Historical range is inspected in October 2026; 2024-2026 is chronological holdout only, not genuinely untouched prospectively.",
          "Monthly random placebo does not control momentum/volatility conditional on the pattern.",
          "Trading opportunities may overlap across different strategies and horizons.",
          "Execution assumes open prices can be obtained at modeled fees and slippage.",
          "No exchange trading, compounding, protective ATR stops, or tax are included.",
        ],
    }
    (out/"methodology.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf8")
    print("PRIMARY_EVALUATION_TABLE",flush=True)
    print(primary.to_string(index=False),flush=True)
    print("RESEARCH_COMPLETE",flush=True)


if __name__ == "__main__":
    main()

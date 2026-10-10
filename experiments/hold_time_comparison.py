"""Paired exit-horizon comparison for BTC 4h morning-star; research only.

Exactly the same entry dates and OHLCV prices are compared at 12h / 24h / 48h.
Long-only, no live orders, no stop-loss/take-profit.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.candlestick_study import (
    START, TEST_START, END, COST_BPS_SIDE, fetch_gap_aware, signals,
)

TIMEFRAME = "4h"
HORIZON_BARS = (3, 6, 12)  # 12h, 24h, 48h
MAX_HORIZON = max(HORIZON_BARS)
FEE_SLIP = COST_BPS_SIDE / 10_000
BOOTSTRAP_DRAWS = 4000
SEED = 20261010


def pick_common_entries(mask: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Disjoint signals, all usable under longest exit, with intact 20-bar history."""
    n = len(mask)
    idx = np.flatnonzero(mask)
    good = []
    last_max_exit = -1
    for i in idx:
        entry = int(i) + 1
        max_exit = entry + MAX_HORIZON
        if i < 20 or max_exit >= n:
            continue
        # Use no candle data from across gaps anywhere in the 20-bar lookback or hold.
        if segments[i - 20] != segments[max_exit]:
            continue
        # Next position may begin only after previous hypothetical 48h exit.
        if entry <= last_max_exit:
            continue
        good.append(int(i))
        last_max_exit = max_exit
    return np.asarray(good, dtype=int)


def bootstrap_means(arr: np.ndarray, month_keys: np.ndarray, rng: np.random.Generator):
    unique_months = sorted(set(month_keys))
    groups = [arr[month_keys == m] for m in unique_months]
    draws = np.empty(BOOTSTRAP_DRAWS)
    for b in range(BOOTSTRAP_DRAWS):
        chosen = rng.integers(0, len(groups), size=len(groups))
        draws[b] = np.concatenate([groups[j] for j in chosen]).mean()
    return np.percentile(draws, [2.5, 97.5])


def sign_flip_p(diff: np.ndarray, month_keys: np.ndarray, rng: np.random.Generator):
    """Approximate clustered two-sided randomization; assumes symmetric month-level null."""
    unique_months = sorted(set(month_keys))
    group_sums = np.asarray([diff[month_keys == m].sum() for m in unique_months])
    actual = abs(float(diff.mean()))
    hits = 0
    for _ in range(BOOTSTRAP_DRAWS):
        signs = rng.choice([-1, 1], size=len(group_sums))
        value = abs(float(np.sum(group_sums * signs) / len(diff)))
        hits += (value >= actual)
    return float((hits + 1) / (BOOTSTRAP_DRAWS + 1))


def phase_results(df: pd.DataFrame, segments: np.ndarray, detected: np.ndarray,
                  phase: str, start: pd.Timestamp, end: pd.Timestamp,
                  rng: np.random.Generator):
    ts = pd.to_datetime(df.open_time_ms, unit="ms", utc=True)
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    valid = detected.copy()
    opentimes = df.open_time_ms.to_numpy(dtype=np.int64)
    # Exit at open of the later bar must remain within the same phase.
    valid &= (opentimes >= start_ms) & (opentimes < end_ms)
    for h in HORIZON_BARS:
        last_idx = np.minimum(np.arange(len(df)) + 1 + h, len(df) - 1)
        valid &= opentimes[last_idx] < end_ms
    indices = pick_common_entries(valid, segments)
    if len(indices) < 4:
        raise RuntimeError(f"Too few comparable entry signals in {phase}: {len(indices)}")
    open_px = df.open.to_numpy(float)
    entry_price = open_px[indices + 1]
    time_signal = ts.iloc[indices].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()
    time_entry = ts.iloc[indices+1].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()
    month_keys = ts.iloc[indices+1].dt.strftime("%Y-%m").to_numpy()

    trade_frame = pd.DataFrame({
        "phase":phase, "signal_utc":time_signal, "entry_utc":time_entry,
        "entry_open":entry_price, "entry_month_utc":month_keys,
    })
    returns = {}
    summary = []
    for h in HORIZON_BARS:
        hours = 4*h
        exit_idx = indices + 1 + h
        exit_price = open_px[exit_idx]
        gross = exit_price / entry_price - 1.0
        net = exit_price * (1-FEE_SLIP) / (entry_price * (1+FEE_SLIP)) - 1.0
        returns[hours] = net
        trade_frame[f"exit_{hours}h_utc"] = ts.iloc[exit_idx].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()
        trade_frame[f"exit_{hours}h_open"] = exit_price
        trade_frame[f"net_{hours}h_pct"] = net*100
        ci = bootstrap_means(net,month_keys,rng)
        s = {
            "phase":phase, "hold_hours":hours, "trades":int(len(net)),
            "mean_gross_pct":round(float(np.mean(gross)*100),4),
            "mean_net_pct":round(float(np.mean(net)*100),4),
            "median_net_pct":round(float(np.median(net)*100),4),
            "win_rate_pct":round(float(np.mean(net>0)*100),2),
            "worst_net_pct":round(float(np.min(net)*100),3),
            "best_net_pct":round(float(np.max(net)*100),3),
            "bootstrap_ci95_mean_net_pct": [round(float(ci[0]*100),4),round(float(ci[1]*100),4)],
            "gross_exposure_hours": int(hours*len(net)),
            "hypothetical_all_in_compound_pct":round(float((np.prod(1+net)-1)*100),3),
        }
        summary.append(s)
        print("PAIRED_RESULT "+json.dumps(s),flush=True)

    contrasts = []
    for a,b in ((12,24),(24,48),(12,48)):
        difference = returns[b]-returns[a]
        ci = bootstrap_means(difference,month_keys,rng)
        record = {
            "phase":phase, "compare":f"{b}h minus {a}h", "trades":len(indices),
            "mean_difference_percentage_points":round(float(difference.mean()*100),4),
            "bootstrap_ci95_pp":[round(float(ci[0]*100),4),round(float(ci[1]*100),4)],
            "cluster_sign_flip_p_two_sided":round(sign_flip_p(difference,month_keys,rng),4),
            "pct_events_later_better":round(float(np.mean(difference>0)*100),2),
        }
        contrasts.append(record)
        print("PAIRED_CONTRAST "+json.dumps(record),flush=True)

    yearly=[]
    years=ts.iloc[indices+1].dt.year.to_numpy()
    for year in sorted(set(years)):
        sel=(years==year)
        for h in HORIZON_BARS:
            r=returns[4*h][sel]
            yearly.append({"phase":phase,"year":int(year),"hold_hours":int(4*h),
                           "trades":int(len(r)),"mean_net_pct":round(float(r.mean()*100),4),
                           "win_rate_pct":round(float(np.mean(r>0)*100),2)})
    return trade_frame,summary,contrasts,yearly


def main():
    out=Path("outputs/hold-time-comparison")
    out.mkdir(parents=True,exist_ok=True)
    frame, gap_spans, segments=fetch_gap_aware(TIMEFRAME)
    stamps=pd.to_datetime(frame.open_time_ms,unit="ms",utc=True)
    if frame.empty or stamps.iloc[0] > START or stamps.iloc[-1] < END - pd.Timedelta("8h"):
        raise RuntimeError("4h historical data incomplete")
    events=signals(frame)["morning_star"]
    rng=np.random.default_rng(SEED)
    all_trades, all_stats, all_contrasts, all_years=[],[],[],[]
    for args in [
        ("development_2020_2023", START, TEST_START),
        ("evaluation_2024_2026", TEST_START, END),
    ]:
        trades,stats,contrasts,years=phase_results(frame,segments,events,*args,rng)
        all_trades.append(trades)
        all_stats.extend(stats)
        all_contrasts.extend(contrasts)
        all_years.extend(years)
    pd.concat(all_trades,ignore_index=True).to_csv(out/"same_entries_trades.csv",index=False)
    pd.DataFrame(all_stats).to_csv(out/"hold_times_summary.csv",index=False)
    pd.DataFrame(all_contrasts).to_csv(out/"paired_differences.csv",index=False)
    pd.DataFrame(all_years).to_csv(out/"yearly_results.csv",index=False)
    missing=int((frame.open_time_ms.diff().fillna(14_400_000)/14_400_000-1).sum())
    meta={
        "symbol":"BTC/USDT spot", "timeframe":"4h","start":START.isoformat(),
        "evaluation_starts":TEST_START.isoformat(),"end_exclusive":END.isoformat(),
        "source":"Binance public historical klines", "rows":len(frame),
        "missing_4h_bars":missing,"gap_spans":gap_spans,
        "buy":"4h morning star confirmed at bar close; next bar open",
        "sell":"exit at open, exactly 3, 6 or 12 complete 4h bars after entry",
        "same_entry_cohort":True,
        "cohort_rule":"only signals with a 20-bar intact history and 48h uninterrupted future; earliest signal is retained; skip all signals while any candidate 48h trade is open",
        "fee_bps_each_side":10,"slippage_bps_each_side":5,
        "primary_horizons_hours":[12,24,48],
        "bootstrap":"resample calendar months containing entry signals, 4000 draws",
        "difference_test":"clustered calendar-month sign flips, two sided, 4000 draws; exploratory",
        "limitations":[
          "Study is historical and all horizons were chosen after the original 24h result was inspected.",
          "The true forward-looking untouched test does not exist; holdout is temporal only.",
          "Bootstrap confidence intervals and p-values are descriptive under modeling assumptions.",
          "Entries do not overlap up to 48h, so results differ from earlier horizon-specific event sets.",
          "Fixed all-in illustrative compounding lacks intratrade drawdown, taxes, market impact and portfolio risk model.",
          "No ATR stop, no profit target, no shorting, no paper or real orders.",
          "No guarantee price or slippage can actually be executed at modeled next-bar opens.",
        ],
    }
    (out/"methodology.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf8")
    print("COMPARISON_COMPLETE",flush=True)


if __name__=="__main__":
    main()

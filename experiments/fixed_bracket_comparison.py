"""Fixed take-profit / stop-loss comparisons for 4h BTC morning star.

Research only. All combinations use identical nonoverlapping entries, fees,
slippage, and a maximum 48-hour holding period. No live trading.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.candlestick_study import START, TEST_START, END, fetch_gap_aware, signals
from experiments.hold_time_comparison import pick_common_entries

TIMEFRAME = "4h"
TIME_STOP_BARS = 12  # 48 hours
TAKES_PCT = (1.0, 2.0, 3.0, 5.0)
STOPS_PCT = (1.0, 2.0, 3.0)
FEE = 0.001
SLIPPAGE = 0.0005
SEED = 20261010
BOOTSTRAP_SAMPLES = 3000


def simulate_bracket(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                     entry_idx: int, take_pct: float, stop_pct: float,
                     horizon: int = TIME_STOP_BARS):
    """Return net and gross return, exit reason, exit bar, ambiguous candle flag.

    Entry is next-open after confirmed signal, market buy with positive slippage.
    Stop/take thresholds are percentages from the simulated BUY fill price.
    Inspect entry candle and next horizon-1 candles, then time-exit at open
    entry_idx+horizon. Gaps through stop fill at worse open; favorable gaps
    through take fill at limit target. OHLC stop/take collision: STOP FIRST.
    No attempt is made to infer an unknown intra-bar event timestamp.
    """
    buy_fill = opens[entry_idx] * (1+SLIPPAGE)
    buy_cost = buy_fill * (1+FEE)
    stop = buy_fill * (1-stop_pct/100)
    take = buy_fill * (1+take_pct/100)
    raw_exit = float(opens[entry_idx+horizon])
    reason = "time_48h"
    exit_idx = entry_idx + horizon
    ambiguous = False

    for k in range(entry_idx,entry_idx+horizon):
        o,h,l = float(opens[k]),float(highs[k]),float(lows[k])
        if o <= stop:
            raw_exit,reason,exit_idx = o,"gap_stop",k
            break
        if o >= take:
            raw_exit,reason,exit_idx = take,"take_profit",k
            break
        if l <= stop:
            ambiguous = h >= take
            raw_exit,reason,exit_idx = stop,"stop_loss",k
            break
        if h >= take:
            raw_exit,reason,exit_idx = take,"take_profit",k
            break

    sell_fill = raw_exit*(1-SLIPPAGE)
    sell_after_fee = sell_fill*(1-FEE)
    return (sell_after_fee/buy_cost - 1,
            raw_exit/opens[entry_idx]-1, reason, exit_idx, ambiguous)


def baseline_net(opens: np.ndarray, entry_idx: np.ndarray, bars: int):
    entry = opens[entry_idx]
    exits = opens[entry_idx+bars]
    return exits*(1-SLIPPAGE)*(1-FEE)/(entry*(1+SLIPPAGE)*(1+FEE))-1


def block_bootstrap_ci(v: np.ndarray, months: np.ndarray, rng, reps=BOOTSTRAP_SAMPLES):
    month_ids=np.unique(months)
    blocks=[v[months==m] for m in month_ids]
    sampling=rng.integers(0,len(blocks),size=(reps,len(blocks)))
    avg=np.array([np.concatenate([blocks[j] for j in draw]).mean() for draw in sampling])
    return (float(np.quantile(avg,.025)),float(np.quantile(avg,.975)))


def paired_signflip_p(diffs: np.ndarray, months: np.ndarray, rng,
                      reps=BOOTSTRAP_SAMPLES):
    unique_months=np.unique(months)
    totals=np.asarray([diffs[months==m].sum() for m in unique_months])
    observed=np.abs(diffs.mean())
    signs=rng.choice((-1,1),size=(reps,len(totals)))
    null=np.abs((signs@totals)/len(diffs))
    return float((1+np.count_nonzero(null>=observed))/(reps+1))


def cohort(df,segments,detected,phase_start,phase_end):
    opens_ms=df.open_time_ms.to_numpy(dtype=np.int64)
    phase_start_ms=int(phase_start.timestamp()*1000)
    phase_end_ms=int(phase_end.timestamp()*1000)
    eligible=detected & (opens_ms>=phase_start_ms) & (opens_ms<phase_end_ms)
    last_idx=np.minimum(np.arange(len(df))+1+TIME_STOP_BARS,len(df)-1)
    eligible &= opens_ms[last_idx]<phase_end_ms
    indices=pick_common_entries(eligible,segments)
    if len(indices)<4:
        raise ValueError("Cohort too small to compare exits")
    return indices


def report_metrics(v,months,phase,label,reason_counts,rng,reference=None):
    ci=block_bootstrap_ci(v,months,rng)
    row={
        "phase":phase,"rule":label,"trades":len(v),
        "mean_net_pct":round(float(v.mean()*100),4),
        "median_net_pct":round(float(np.median(v)*100),4),
        "win_rate_pct":round(float(np.mean(v>0)*100),2),
        "worst_net_pct":round(float(v.min()*100),3),
        "best_net_pct":round(float(v.max()*100),3),
        "ci95_mean_net_pct":[round(ci[0]*100,4),round(ci[1]*100,4)],
        "take_profit_trades":int(reason_counts.get("take_profit",0)),
        "stop_loss_trades":int(reason_counts.get("stop_loss",0)),
        "gap_stop_trades":int(reason_counts.get("gap_stop",0)),
        "time_exit_trades":int(reason_counts.get("time_48h",0)),
        "same_bar_ambiguous_trades":int(reason_counts.get("ambiguous",0)),
    }
    if reference is not None:
        diff=v-reference
        lo,hi=block_bootstrap_ci(diff,months,rng)
        row.update({
            "vs_fixed_48h_mean_diff_pp":round(float(diff.mean()*100),4),
            "vs_fixed_48h_ci95_pp":[round(lo*100,4),round(hi*100,4)],
            "vs_fixed_48h_p_two_sided":round(paired_signflip_p(diff,months,rng),4),
        })
    print("BRACKET_RESULT "+json.dumps(row,ensure_ascii=False),flush=True)
    return row


def sanity_checks():
    # Breach of both thresholds within one candle exits pessimistically at stop.
    op=np.array([100.,100.,100.,100.])
    hi=np.array([103.,100.,100.,100.])
    lo=np.array([97.,100.,100.,100.])
    n,_,why,idx,amb=simulate_bracket(op,hi,lo,0,2,1,3)
    assert why=="stop_loss" and idx==0 and amb and n < 0
    # Stop gaps must fill worse than the stop threshold (not idealized stop).
    op=np.array([100.,95.,96.,96.])
    hi=np.array([100.,96.,97.,97.])
    lo=np.array([100.,94.,95.,95.])
    n,g,why,idx,amb=simulate_bracket(op,hi,lo,0,2,1,3)
    assert why=="gap_stop" and idx==1 and not amb and g < -0.04
    # Favorable gap cannot claim above take-profit limit order.
    op=np.array([100.,105.,105.,105.])
    hi=np.array([100.5,105.,105.,105.])
    lo=np.array([99.5,104.,104.,104.])
    _,_,why,idx,amb=simulate_bracket(op,hi,lo,0,2,1,3)
    assert why=="take_profit" and idx==1 and not amb
    # Neither barrier reached -> stop at 48h opening price.
    op=np.array([100.,100.,100.,100.,100.])
    hi=np.array([100.4]*5)
    lo=np.array([99.5]*5)
    _,_,why,idx,amb=simulate_bracket(op,hi,lo,0,3,2,3)
    assert why=="time_48h" and idx==3 and not amb


def main():
    sanity_checks()
    out=Path("outputs/fixed-bracket-comparison")
    out.mkdir(parents=True,exist_ok=True)
    df,gaps,segments=fetch_gap_aware(TIMEFRAME)
    ts=pd.to_datetime(df.open_time_ms,unit="ms",utc=True)
    if ts.iloc[0]>START or ts.iloc[-1]<END-pd.Timedelta("8h"):
        raise RuntimeError("Incomplete BTC/USDT 4h candle series")
    trigger=signals(df)["morning_star"]
    opens=df.open.to_numpy(float)
    highs=df.high.to_numpy(float)
    lows=df.low.to_numpy(float)
    rng=np.random.default_rng(SEED)
    stats=[]
    trades=[]
    yearly=[]
    for phase,start,end in (
        ("development_2020_2023",START,TEST_START),
        ("evaluation_2024_2026",TEST_START,END)
    ):
        entry_signals=cohort(df,segments,trigger,start,end)
        # Identical cohort to the prior 12/24/48h research. Fail closed if changed.
        expected=53 if phase.startswith("development") else 44
        if len(entry_signals)!=expected:
            raise RuntimeError(f"Entry cohort drift: {phase} {len(entry_signals)} vs {expected}")
        entry_idx=entry_signals+1
        months=ts.iloc[entry_idx].dt.strftime("%Y-%m").to_numpy()
        years=ts.iloc[entry_idx].dt.year.to_numpy()
        baseline48=baseline_net(opens,entry_idx,12)
        for baseline_bars in (3,6,12):
            net=baseline_net(opens,entry_idx,baseline_bars)
            stats.append(report_metrics(net,months,phase,f"fixed_{4*baseline_bars}h",
                                       {},rng,baseline48 if baseline_bars<12 else None))
            for y in sorted(set(years)):
                v=net[years==y]
                yearly.append({"phase":phase,"year":int(y),"rule":f"fixed_{4*baseline_bars}h",
                               "trades":len(v),"mean_net_pct":round(float(v.mean()*100),4)})
        base_rows=[]
        for j,i in enumerate(entry_signals):
            base_rows.append({
                "phase":phase,"signal_utc":ts.iloc[i].isoformat(),
                "entry_utc":ts.iloc[i+1].isoformat(),
                "entry_market_open":round(float(opens[i+1]),4),
                "fixed_12h_net_pct":round(float(baseline_net(opens,np.array([i+1]),3)[0]*100),4),
                "fixed_24h_net_pct":round(float(baseline_net(opens,np.array([i+1]),6)[0]*100),4),
                "fixed_48h_net_pct":round(float(baseline48[j]*100),4),
            })
        for tp in TAKES_PCT:
            for sl in STOPS_PCT:
                label=f"TP{tp:g}_SL{sl:g}_48hcap"
                net=[]
                gross=[]
                reason_count={}
                for j,signal_idx in enumerate(entry_signals):
                    r,g,reason,exit_idx,amb=simulate_bracket(opens,highs,lows,int(signal_idx)+1,tp,sl)
                    net.append(r)
                    gross.append(g)
                    reason_count[reason]=reason_count.get(reason,0)+1
                    if amb:
                        reason_count["ambiguous"]=reason_count.get("ambiguous",0)+1
                    base_rows[j][f"{label}_net_pct"]=round(r*100,4)
                    base_rows[j][f"{label}_reason"]=reason
                net=np.array(net)
                row=report_metrics(net,months,phase,label,reason_count,rng,baseline48)
                row.update({"take_pct":tp,"stop_pct":sl,"max_hold_hours":48})
                stats.append(row)
                for y in sorted(set(years)):
                    v=net[years==y]
                    yearly.append({"phase":phase,"year":int(y),"rule":label,
                                   "trades":len(v),"mean_net_pct":round(float(v.mean()*100),4),
                                   "win_rate_pct":round(float(np.mean(v>0)*100),2)})
        trades.extend(base_rows)
    pd.DataFrame(stats).to_csv(out/"bracket_summary.csv",index=False)
    pd.DataFrame(trades).to_csv(out/"same_entry_bracket_trades.csv",index=False)
    pd.DataFrame(yearly).to_csv(out/"yearly_bracket_results.csv",index=False)
    manifest={
        "dataset":"Binance public BTCUSDT spot 4h OHLCV",
        "data_time_utc":[START.isoformat(),END.isoformat()],
        "holdout_split_utc":TEST_START.isoformat(),
        "gap_spans":gaps,
        "strategy":"Original 4h morning star from experiments/candlestick_study.py",
        "same_entry_cohorts":{"development":53,"evaluation":44},
        "entry":"next 4h candle open after signal, with positive 5bps buy slippage",
        "fixed_take_profit_pct":list(TAKES_PCT),
        "fixed_stop_loss_pct":list(STOPS_PCT),
        "max_holding_hours":48,
        "barrier_threshold_reference":"buy execution price after slippage",
        "within_bar_rule":"inspect entry candle plus subsequent 11 candles; gap stop fills at open, favorable take gap fills at limit; if TP/SL both touched, SL first; 48h time exit at next opening",
        "cost":"10bps fee and 5bps slippage each side; both charged per trade",
        "baseline":"12h/24h/48h fixed time exits use identical entry signals; 48h is primary comparator",
        "analysis":"calendar month block bootstrap 3000 resamples; exploratory month-level two-sided sign flip",
        "testing":"sanity_checks() for collisions, gap stop, limit take, time expiration",
        "warnings":[
            "2024+ interval was previously inspected, hence not truly untouched evaluation for new TP/SL grid.",
            "12 combinations tested; selecting the best retrospectively incurs serious data snooping risk.",
            "Risk and return are per matched event, not full portfolio equity or annualized performance.",
            "Different TP/SL settings use the same fixed signal cohort even when some positions exit early.",
            "Bar OHLC cannot establish order of high and low; assumed pessimistic stop-first within same candle.",
            "Next-opening execution prices and intrabar stop orders are modeling assumptions; extreme gaps can fill worse.",
            "Historical execution does not model order book depth, funding, taxes, 24/7 operation of automated orders, or exchange failures.",
            "No live orders, no default strategy, no leverage, no change to master/main."
        ],
    }
    (out/"methodology.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf8")
    print("BRACKET_COMPARISON_COMPLETE",flush=True)


if __name__=="__main__":
    main()

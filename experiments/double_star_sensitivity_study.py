"""Study an extra two-star (4-candle) Morning-Star variant with unchanged ATR exit.

Research only. It does NOT alter production signals, order placement, configuration
defaults, main branch or live/paper trading behavior.

Control: original 3-candle BTC Morning Star, unchanged.
Experimental: large bearish candle + TWO consecutive small star bodies + large
bullish confirmation candle (4 candles). Reuse the original relative thresholds,
previous-downtrend check and midpoint recovery with the first/last candles.

Compare chronological union, standalone two-star signal and a baseline-preserving
add-only union that cannot displace any original signal. All candidates use
ATR(14) x1.5 initial stop, TP5%, 48h max, constant 30%-of-equity spot allocation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.candlestick_study import START, TEST_START, END, fetch_gap_aware, signals
from experiments.fixed_bracket_comparison import cohort, simulate_bracket
from experiments.atr_risk_sized_comparison import wilder_atr

OUT=Path("outputs/double-star-morningstar")
ATR_MULTIPLIER=1.5
CAPITAL_FRACTION=0.30
TP_PERCENT=5.0
EXIT_BARS=12
SEED=20261010
BOOTSTRAP=4000
PATTERN_TYPES=("original_three_star","double_star_only","either_star_chronological","original_plus_nonconflicting_double")


def double_star_signals(frame: pd.DataFrame) -> np.ndarray:
    """4-candle Morning Star, two SMALL consecutive consolidation bodies.

    At completed candle t, first bearish=t-3, two star candles=t-2,t-1,
    final bullish=t. All terms use [t-7,t] only; no future information.
    Consciously no gap requirement (BTC trades 24/7).
    """
    op=frame.open.to_numpy(float)
    cl=frame.close.to_numpy(float)
    hi=frame.high.to_numpy(float)
    lo=frame.low.to_numpy(float)
    bodies=np.abs(cl-op)
    ranges=np.maximum(hi-lo,1e-12)
    ans=np.zeros(len(frame),dtype=bool)
    for t in range(20,len(frame)):
        first=t-3
        if not (op[first]>cl[first] and bodies[first]>=.50*ranges[first]):
            continue
        if not (bodies[t-2]<=.35*bodies[first] and bodies[t-1]<=.35*bodies[first]):
            continue
        if not (cl[t]>op[t] and bodies[t]>=.50*ranges[t]):
            continue
        # Same lookback as prior 3-bar signal: the first bearish close
        # must be lower than four bars before that bearish candle.
        if not cl[first]<cl[first-4]:
            continue
        if not min(lo[t-2],lo[t-1])<=cl[first]:
            continue
        if not cl[t]>=(op[first]+cl[first])/2:
            continue
        ans[t]=True
    return ans


def eligible_signals(frame,segments,mask,phase_start,phase_end):
    """All usable signal indices BEFORE any overlap thinning."""
    ids=np.flatnonzero(mask)
    n=len(frame)
    stamps=frame.open_time_ms.to_numpy(np.int64)
    start=int(phase_start.timestamp()*1000)
    end=int(phase_end.timestamp()*1000)
    chosen=[]
    for k in ids:
        e=int(k)+1
        last=e+EXIT_BARS
        if k<20 or last>=n or stamps[k]<start or stamps[k]>=end or stamps[last]>=end:
            continue
        if segments[k-20]!=segments[last]:
            continue
        chosen.append(int(k))
    return np.asarray(chosen,dtype=int)


def nonconflicting_baseline_plus(original: np.ndarray, new_candidates: np.ndarray) -> np.ndarray:
    """Freeze original 48h-reserved entries, admit only additions in free slots.

    Sorted chronological additions NEVER displace a selected original trade.
    """
    old=sorted(int(i) for i in original)
    occupied=[(i+1,i+1+EXIT_BARS) for i in old]
    accepted=[]
    for sig in sorted(set(int(i) for i in new_candidates)):
        e=sig+1
        finish=e+EXIT_BARS
        if sig in old:
            continue
        if all(finish < a or e > b for a,b in occupied):
            occupied.append((e,finish))
            accepted.append(sig)
    final=np.array(sorted(old+accepted),dtype=int)
    assert set(old).issubset(set(final))
    for first,second in zip(final[:-1],final[1:]):
        assert second+1 > first+1+EXIT_BARS
    return final


def run_entries(frame,atr,idx,ts,labels):
    op=frame.open.to_numpy(float)
    hi=frame.high.to_numpy(float)
    lo=frame.low.to_numpy(float)
    rows=[]
    for signal_idx in idx:
        j=int(signal_idx)
        entry=j+1
        buy_fill=op[entry]*(1+.0005)
        stop_pct=float(100*ATR_MULTIPLIER*atr[j]/buy_fill)
        if not np.isfinite(stop_pct) or stop_pct<=0:
            raise ValueError(f"No non-lookahead ATR for signal {j}")
        net,gross,reason,exit_idx,amb=simulate_bracket(
            op,hi,lo,entry,TP_PERCENT,stop_pct,EXIT_BARS
        )
        rows.append({
            "signal_index":j,
            "signal_utc":ts.iloc[j].isoformat(),
            "entry_utc":ts.iloc[entry].isoformat(),
            "entry_month_utc":ts.iloc[entry].strftime("%Y-%m"),
            "entry_year_utc":int(ts.iloc[entry].year),
            "pattern_source":("both" if labels["original"][j] and labels["double"][j]
                              else "original_3" if labels["original"][j] else "double_4"),
            "entry_market_open":round(float(op[entry]),4),
            "signal_atr14":round(float(atr[j]),5),
            "stop_distance_pct":round(stop_pct,4),
            "position_net_pct":round(float(net*100),4),
            "account_net_pct":round(float(CAPITAL_FRACTION*net*100),4),
            "account_return":float(CAPITAL_FRACTION*net),
            "exit_reason":str(reason),
            "exit_utc":ts.iloc[exit_idx].isoformat(),
            "same_bar_ambiguous":bool(amb),
        })
    return rows


def outcome(phase,name,entries):
    r=np.asarray([x["account_return"] for x in entries],dtype=float)
    position=r/CAPITAL_FRACTION
    n=len(r)
    if n==0:
        res={"phase":phase,"variant":name,"trades":0}
        print("DOUBLE_STAR_RESULT "+json.dumps(res),flush=True)
        return res
    equity=np.r_[1.,np.cumprod(1+r)]
    mdd=float(np.min(equity/np.maximum.accumulate(equity)-1))
    tails=max(1,int(np.ceil(.05*n)))
    wins=position>0
    streak=0;longest=0
    for x in r:
        if x<0:
            streak+=1;longest=max(streak,longest)
        else:
            streak=0
    counts=pd.Series([e["pattern_source"] for e in entries]).value_counts().to_dict()
    reasons=pd.Series([e["exit_reason"] for e in entries]).value_counts().to_dict()
    res={
        "phase":phase,"variant":name,
        "trades":n,
        "source_counts":{str(k):int(v) for k,v in counts.items()},
        "position_mean_net_pct":round(float(position.mean()*100),4),
        "position_median_net_pct":round(float(np.median(position)*100),4),
        "win_rate_pct":round(float(wins.mean()*100),2),
        "account_mean_signal_net_pct":round(float(r.mean()*100),4),
        "account_compounded_return_pct":round(float((equity[-1]-1)*100),4),
        "account_trade_close_mdd_pct":round(mdd*100,4),
        "account_worst_event_pct":round(float(r.min()*100),4),
        "account_bottom5pct_es_pct":round(float(np.sort(r)[:tails].mean()*100),4),
        "account_top3_event_sum_pp":round(float(np.sum(np.sort(r)[-3:])*100),4),
        "max_consecutive_losses":longest,
        "tp_exits":int(reasons.get("take_profit",0)),
        "stop_exits":int(reasons.get("stop_loss",0))+int(reasons.get("gap_stop",0)),
        "time_exits":int(reasons.get("time_48h",0)),
    }
    print("DOUBLE_STAR_RESULT "+json.dumps(res),flush=True)
    return res


def month_log_growth(trades,months):
    logmonthly={m:0. for m in months}
    for t in trades:
        logmonthly[t["entry_month_utc"]]+=float(np.log1p(t["account_return"]))
    return np.array([logmonthly[m] for m in months],dtype=float)


def month_paired_comparison(phase,name,baseline,test,phase_start,phase_end,rng):
    # Includes non-trading months in both arms; compare growth across identical
    # calendar months. No resampling of individual correlated trade outcomes.
    months=pd.period_range(start=phase_start.strftime("%Y-%m"),end=(phase_end-pd.Timedelta(days=1)).strftime("%Y-%m"),freq="M").astype(str).tolist()
    ba=month_log_growth(baseline,months)
    ne=month_log_growth(test,months)
    d=ne-ba
    sampling=rng.integers(0,len(d),size=(BOOTSTRAP,len(d)))
    samples=d[sampling].mean(axis=1)
    # Mean *monthly* difference in log return, percentage points; not CAGR.
    ci=np.quantile(samples,[.025,.975])
    observed=abs(float(d.mean()))
    flips=rng.choice([-1,1],size=(BOOTSTRAP,len(d)))
    null=np.abs((flips*d).mean(axis=1))
    p=(1+int(np.sum(null>=observed)))/(1+BOOTSTR)
    result={
        "phase":phase,
        "comparison":name+" minus original_three_star",
        "calendar_months_including_no_trades":len(months),
        "cumulative_account_return_delta_pp":round(
            100*((np.exp(ne.sum())-1)-(np.exp(ba.sum())-1)),4),
        "mean_monthly_log_growth_diff_pp":round(float(d.mean()*100),5),
        "bootstrap95_month_log_diff_pp":[round(float(x*100),5) for x in ci],
        "exploratory_month_signflip_p":round(float(p),4),
        "warning":"Many months without trades; prospective verification required, time-varying returns.",
    }
    print("DOUBLE_STAR_PAIRED "+json.dumps(result),flush=True)
    return result


def sanity_tests():
    # Four-candle pattern: bearish, small, small, confirm bullish.
    n=40
    op=np.full(n,100.)
    cl=np.full(n,100.)
    hi=np.full(n,101.)
    lo=np.full(n,99.)
    t=30
    op[t-7]=110.;cl[t-7]=110.;hi[t-7]=111.;lo[t-7]=109.
    op[t-3]=110.;cl[t-3]=100.;hi[t-3]=112.;lo[t-3]=98.
    op[t-2]=100.;cl[t-2]=99.5;hi[t-2]=100.5;lo[t-2]=98.
    op[t-1]=99.5;cl[t-1]=99.8;hi[t-1]=101.;lo[t-1]=98.
    op[t]=100.;cl[t]=107.;hi[t]=108.;lo[t]=99.
    frame=pd.DataFrame({"open":op,"close":cl,"high":hi,"low":lo})
    double=double_star_signals(frame)
    assert double[t] and not signals(frame)["morning_star"][t], "Double pattern should be distinct"
    # More recent candles may NEVER change prior signal state.
    alt=frame.copy()
    alt.loc[t+1:,"open"]=500.
    alt.loc[t+1:,"high"]=501.
    assert double_star_signals(alt)[t]
    invalid=frame.copy()
    invalid.loc[t-1,"close"]=105.
    assert not double_star_signals(invalid)[t]
    # Baseline-preserving add-only cannot evict an original signal.
    orig=np.array([40,90])
    additions=np.array([39,52,70,100,120])
    combined=nonconflicting_baseline_plus(orig,additions)
    assert set(orig).issubset(set(combined))
    assert 70 in combined and 39 not in combined and 52 not in combined
    # Always use original unmodified signal code for the control.


def main():
    sanity_tests()
    OUT.mkdir(parents=True,exist_ok=True)
    df,gaps,segments=fetch_gap_aware("4h")
    ts=pd.to_datetime(df.open_time_ms,unit="ms",utc=True)
    if ts.iloc[0]>START or ts.iloc[-1]<END-pd.Timedelta("8h"):
        raise RuntimeError("BTCUSDT 4h dataset incomplete")
    atr=wilder_atr(df.high.to_numpy(float),df.low.to_numpy(float),
                   df.close.to_numpy(float),segments)
    old=signals(df)["morning_star"]
    double=double_star_signals(df)
    labels={"original":old,"double":double}
    rng=np.random.default_rng(SEED)
    summary=[];individual=[];calendar=[];paired=[];attribution=[]
    for phase,start,end in (
        ("development_2020_2023",START,TEST_START),
        ("evaluation_2024_2026",TEST_START,END),
    ):
        original=cohort(df,segments,old,start,end)
        expected=53 if phase.startswith("development") else 44
        if len(original)!=expected:
            raise RuntimeError(f"Original baseline drift {len(original)} != {expected}")
        ds_eligible=eligible_signals(df,segments,double,start,end)
        valid_old=eligible_signals(df,segments,old,start,end)
        if not np.isfinite(atr[np.r_[original,ds_eligible]]).all():
            raise ValueError("Missing preentry ATR in any signal")
        double_only=cohort(df,segments,double,start,end) if len(ds_eligible)>0 else np.array([],dtype=int)
        union=cohort(df,segments,old|double,start,end)
        preserved=nonconflicting_baseline_plus(original,ds_eligible)
        cases={
            "original_three_star":original,
            "double_star_only":double_only,
            "either_star_chronological":union,
            "original_plus_nonconflicting_double":preserved,
        }
        if not np.array_equal(cases["original_three_star"],original):
            raise AssertionError("Original cohort mutated")
        # Difference between signal inventory and executed trades is not zero:
        # overlapping signals are explicitly skipped, never double-counted.
        inventory={
            "phase":phase,
            "raw_eligible_old_signal_count":len(valid_old),
            "raw_eligible_double_signal_count":len(ds_eligible),
            "raw_signal_intersection":len(set(valid_old).intersection(ds_eligible)),
            "original_selected":len(original),
            "union_selected":len(union),
            "baseline_preserving_added":len(preserved)-len(original),
            "baseline_signals_displaced_in_chronological_union":len(set(original)-set(union)),
        }
        print("DOUBLE_STAR_INVENTORY "+json.dumps(inventory),flush=True)
        attribution.append(inventory)
        events={}
        for name,ids in cases.items():
            rows=run_entries(df,atr,ids,ts,labels)
            events[name]=rows
            for e in rows:
                individual.append({"phase":phase,"variant":name,**e})
            summary.append(outcome(phase,name,rows))
            for y in sorted(set(x["entry_year_utc"] for x in rows)):
                selected=[x for x in rows if x["entry_year_utc"]==y]
                ar=np.asarray([x["account_return"] for x in selected],dtype=float)
                res={
                    "phase":phase,"variant":name,"year":y,"trades":len(selected),
                    "account_compounded_year_pct":round(float((np.prod(1+ar)-1)*100),4),
                    "account_mean_per_signal_pct":round(float(ar.mean()*100),4),
                    "double_only_trade_count":sum(x["pattern_source"]=="double_4" for x in selected),
                }
                calendar.append(res)
                print("DOUBLE_STAR_YEAR "+json.dumps(res),flush=True)
        for other in ("double_star_only","either_star_chronological",
                      "original_plus_nonconflicting_double"):
            paired.append(month_paired_comparison(
                phase,other,events["original_three_star"],events[other],
                start,end,rng))
        # Incremental contributions from eligible standalone 4-bar events not
        # already selected by the original baseline; no postselection by PnL.
        added_ids=set(preserved)-set(original)
        additions=[r for r in events["original_plus_nonconflicting_double"]
                   if r["signal_index"] in added_ids]
        add_pnl=np.asarray([r["account_return"] for r in additions],float)
        ad={
            "phase":phase,"additional_nonoverlapping_trades":len(additions),
            "incremental_mean_net_trade_pct":round(
                float(add_pnl.mean()/CAPITAL_FRACTION*100),4) if len(add_pnl) else None,
            "incremental_win_pct":round(float((add_pnl>0).mean()*100),2) if len(add_pnl) else None,
            "incremental_compounded_account_factor_pct":round(
                float((np.prod(1+add_pnl)-1)*100),4),
            "worst_added_account_trade_pct":round(
                float(add_pnl.min()*100),4) if len(add_pnl) else None,
        }
        attribution.append(ad)
        print("DOUBLE_STAR_ADDED "+json.dumps(ad),flush=True)
    pd.DataFrame(summary).to_csv(OUT/"comparative_summary.csv",index=False)
    pd.DataFrame(individual).to_csv(OUT/"all_trade_events.csv",index=False)
    pd.DataFrame(calendar).to_csv(OUT/"yearly_strategies.csv",index=False)
    pd.DataFrame(paired).to_csv(OUT/"calendar_matched_differences.csv",index=False)
    pd.DataFrame(attribution).to_csv(OUT/"signal_inventory_and_incremental_value.csv",index=False)
    manifest={
        "research_goal":"Compare unchanged 3-candle Morning Star versus added strict four-candle two-star setup.",
        "data":"Binance public BTCUSDT spot 4h OHLCV; no synthetic market data",
        "history_utc":[START.isoformat(),END.isoformat()],
        "evaluation_start_utc":TEST_START.isoformat(),
        "three_candle_signal":"Unmodified experiments.candlestick_study.signals()['morning_star']",
        "four_candle_rule":{
            "first":"At t-3 bearish candle with body >=50% high-low and close below close[t-7]",
            "both_middle":"t-2 and t-1 candle bodies each <=35% of FIRST bearish body; minimum low of two <= first close",
            "confirmation":"t bullish body >=50% candle high-low and close >= mid of first bearish real body",
            "no_gap_requirement":"24/7 spot market",
            "known_at":"End of completed candle t, enter next candle t+1 open",
        },
        "no_trend_volume_or_news_filter":True,
        "atr":"Wilder ATR14 from completed signal candle t, never from the new entry candle",
        "atr_multiple":ATR_MULTIPLIER,
        "stop":"Initial volatility-scaled stop remains FIXED after entry, not a trailing stop",
        "take_profit_percent":TP_PERCENT,
        "maximum_hold_4h_candles":EXIT_BARS,
        "every_trade_account_fraction":CAPITAL_FRACTION,
        "fees_bps_each_side":10,"slippage_bps_each_side":5,
        "stop_model":"Stop on intrabar low, worse open if gap, TP/SL collision assumed stop first; impossible to verify execution path using OHLC alone",
        "cohorts":{
            "original":"Exactly same original matched 53/44 48h-cohort",
            "double_star_only":"Two-star signals alone with independent nonoverlap thinning",
            "either_star_chronological":"OR of both signals, chronological 48h-cap independent thinning; can displace original signals",
            "original_plus_nonconflicting_double":"Freeze all original selected trades, admit additional 2-star trades only when their full 48h reservation does not overlap ANY original or already accepted additional trade",
        },
        "paired_inference":"All inclusive calendar months, compare monthly summed log compounded equity increments, calendar-month bootstrap 4000 draws, exploratory sign-flip; NOT independent validation",
        "caveats":[
            "All earlier 2020-2026 data, notably 2024-2026, already inspected repeatedly to select the base and ATR exit. Adding the 4-bar setup is further multiple testing and data snooping; it has NO untouched out-of-sample evidence.",
            "A four-candle pattern is not a canonical standard Morning Star; this research-only definition retains the old first-star and confirmation thresholds.",
            "Four-candle confirmation is one 4h bar later in pattern formation than a three-candle setup. Execution always on next bar; no hindsight.",
            "Account sizing always 30%, including periods of wide ATR. No 0.9% stop-risk constraint.",
            "Historical PnL uses 4h OHLC pessimistic bar-collision model, not level2 liquidity, 1m events, outage, mark-to-market drawdown or guarantee of stop fills.",
            "Research strategies not activated in default framework, and absolutely no real or paper orders.",
            "No merge to main branch.",
        ],
        "feed_gaps":gaps,
    }
    (OUT/"methodology.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf8")
    print("DOUBLE_STAR_RESEARCH_COMPLETE",flush=True)


if __name__=="__main__":
    main()

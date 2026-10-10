"""BTCUSDT SPOT 4h bullish candle ensembles, no orders, no default strategy.

Frozen forward entry/exit: authentic Binance spot 4h, strict completed bars;
next-bar open, ATR14 Wilder x1.5 static initial stop, take +5%, 48h max hold,
30% of current equity per trade, 10bps fee + 5bps slippage per SIDE.
No relative volume, heatmap, lookahead, leverage, short or order submission.

Candidate additions: original engulfing, original harami, confirmed bullish
hammer, piercing line, three white soldiers, original 20bar upside breakout.
Ensembles are CHRONOLOGICAL OR unions with 48h reservation after each entry.
A new early pattern may displace a later original Morning Star: this is
measured, not swept away by impossible "protect future baseline" hindsight.
Same fixed params/cohorts for all variants. Development and inspected 2024+
historical phases reported; NONE are a prospective untouched evaluation.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.candlestick_study import START, TEST_START, END, fetch_gap_aware, signals
from experiments.fixed_bracket_comparison import simulate_bracket
from experiments.atr_risk_sized_comparison import wilder_atr

OUT=Path("outputs/bullish-pattern-ensembles")
WINDOW_BARS=12
STARTUP_HISTORY=30
ATR_MULT=1.5
TP_PCT=5.
ALLOCATION=.30
FEE_BPS=10
SLIP_BPS=5
SEED=20261010
BOOT=2000
BUCKETS=(
    "morning_star","bullish_engulfing","bullish_harami",
    "hammer_bullish","piercing_line","three_white_soldiers","breakout_20")
# Explicit fixed research list. No automatic parameter tuning.
ARMS=[
  ("baseline_morning_star",("morning_star",)),
  ("solo_bullish_engulfing",("bullish_engulfing",)),
  ("solo_bullish_harami",("bullish_harami",)),
  ("solo_hammer_bullish",("hammer_bullish",)),
  ("solo_piercing_line",("piercing_line",)),
  ("solo_three_white_soldiers",("three_white_soldiers",)),
  ("solo_breakout_20",("breakout_20",)),
  ("combo_star_engulfing",("morning_star","bullish_engulfing")),
  ("combo_star_harami",("morning_star","bullish_harami")),
  ("combo_star_hammer",("morning_star","hammer_bullish")),
  ("combo_star_piercing",("morning_star","piercing_line")),
  ("combo_star_soldiers",("morning_star","three_white_soldiers")),
  ("combo_star_breakout",("morning_star","breakout_20")),
  ("combo_star_engulfing_hammer",("morning_star","bullish_engulfing","hammer_bullish")),
  ("combo_star_harami_piercing",("morning_star","bullish_harami","piercing_line")),
  ("combo_star_soldiers_breakout",("morning_star","three_white_soldiers","breakout_20")),
  ("combo_all_bullish",BUCKETS)
]


def additional_patterns(frame):
    """Explicit deterministic definitions, decisions use <= signal index t."""
    op=frame.open.to_numpy(float);cl=frame.close.to_numpy(float)
    hi=frame.high.to_numpy(float);lo=frame.low.to_numpy(float)
    n=len(frame)
    span=np.maximum(hi-lo,1.e-10)
    body=np.abs(cl-op)
    upper=hi-np.maximum(op,cl)
    lower=np.minimum(op,cl)-lo
    hammer=np.zeros(n,bool)
    piercing=np.zeros(n,bool)
    soldiers=np.zeros(n,bool)
    for t in range(20,n):
        # Bullish hammer/pinbar: upper-third close, long lower rejection
        # wick, mild downtrend in the already completed four previous bars.
        prior_down=cl[t-1] < cl[t-5]
        if (prior_down and cl[t]>op[t] and body[t]<=.35*span[t] and
            lower[t]>=2*max(body[t],1.e-12) and
            lower[t]>=.50*span[t] and upper[t]<=.25*span[t] and
            cl[t]>=lo[t]+.65*span[t]):
            hammer[t]=True

        # Crypto-adapted piercing line: large bearish previous body,
        # strong current bullish body opening at/below preceding close,
        # current close ABOVE bearish first midpoint, but BELOW its open.
        if (cl[t-1]<op[t-1] and body[t-1]>=.50*span[t-1]
            and cl[t-1]<cl[t-5] and cl[t]>op[t]
            and body[t]>=.40*span[t] and op[t]<=cl[t-1]
            and cl[t]>.5*(op[t-1]+cl[t-1]) and cl[t]<op[t-1]):
            piercing[t]=True

        # Three White Soldiers: three consecutive strong rising greens,
        # bodies >=40% of own ranges; last two open within previous
        # candle real bodies, each closes in top 25% of own range.
        last3=range(t-2,t+1)
        if (cl[t-3]<cl[t-8] and
            all(cl[k]>op[k] and body[k]>=.40*span[k]
                and upper[k]<=.25*span[k] for k in last3)
            and cl[t-2]<cl[t-1]<cl[t]
            and op[t-2]<=op[t-1]<=cl[t-2]
            and op[t-1]<=op[t]<=cl[t-1]):
            soldiers[t]=True
    return {
        "hammer_bullish":hammer,
        "piercing_line":piercing,
        "three_white_soldiers":soldiers
    }


def detect(frame):
    old=signals(frame)
    return {**{k:old[k] for k in
        ("morning_star","bullish_engulfing","bullish_harami","breakout_20")},
            **additional_patterns(frame)}


def pick_events(mask,segment,opens_ms,start,end):
    """Chronological 48h reservation after each next-open fill.

    Only include signal if [lookback 20 bars, full 48h following entry]
    remains in contiguous real bars and full 48h exit inside phase.
    """
    lower=int(start.timestamp()*1000)
    upper=int(end.timestamp()*1000)
    reserved_through=-1
    picked=[]
    for ix in np.flatnonzero(mask):
        i=int(ix);entry=i+1;exit_idx=entry+WINDOW_BARS
        if i<STARTUP_HISTORY or exit_idx>=len(mask):continue
        if not(lower<=int(opens_ms[i])<upper and int(opens_ms[exit_idx])<upper):continue
        if segment[i-20]!=segment[exit_idx]:continue
        if entry<=reserved_through:continue
        picked.append(i)
        reserved_through=exit_idx
    return np.asarray(picked,dtype=int)


def classify_trigger(index,pattern_masks,chosen):
    names=[name for name in chosen if pattern_masks[name][index]]
    if not names:raise AssertionError("Unlabeled signal")
    # Name Morning Star first for the same candle, otherwise chosen order.
    if "morning_star" in names:return "morning_star"
    return names[0]


def trades_for(frame,segments,atr,masks,arm,phase,start,end):
    name,pattern_names=arm
    mask=np.logical_or.reduce([masks[p] for p in pattern_names])
    opens_ms=frame.open_time_ms.to_numpy(np.int64)
    original=pick_events(masks["morning_star"],segments,opens_ms,start,end)
    idx=pick_events(mask,segments,opens_ms,start,end)
    op=frame.open.to_numpy(float)
    h=frame.high.to_numpy(float)
    l=frame.low.to_numpy(float)
    rows=[]
    for i in idx:
        if not np.isfinite(atr[i]) or atr[i]<=0:raise RuntimeError("Signal ATR missing")
        entry=i+1
        filled_open=op[entry]*(1+SLIP_BPS/10000)
        stop_pct=float(100*ATR_MULT*atr[i]/filled_open)
        pnl,_,reason,exit_bar,amb=simulate_bracket(op,h,l,entry,TP_PCT,
                                   stop_pct,WINDOW_BARS)
        timestamp=pd.Timestamp(int(opens_ms[entry]),unit="ms",tz="UTC")
        rows.append({
            "phase":phase,"method":name,
            "trigger":classify_trigger(int(i),masks,pattern_names),
            "signal_open_ms":int(opens_ms[i]),
            "entry_open_ms":int(opens_ms[entry]),
            "exit_bar_open_ms":int(opens_ms[exit_bar]),
            "utc_entry":timestamp.isoformat(),
            "month":timestamp.strftime("%Y-%m"),
            "year":timestamp.year,
            "net_position_return":float(pnl),
            "net_position_pct":round(float(pnl*100),5),
            "account_contribution_pct":round(float(ALLOCATION*pnl*100),5),
            "exit_reason":reason,
            "atr_signal":float(atr[i]),
            "initial_stop_pct":round(stop_pct,5),
            "ambiguous_stop_first":bool(amb)
        })
    original_set=set(int(x) for x in original)
    proposed_set=set(int(x) for x in idx)
    return rows,{
        "baseline_trades_in_phase":len(original),
        "baseline_signals_kept":len(proposed_set&original_set),
        "baseline_signals_displaced":len(original_set-proposed_set),
        "nonbaseline_signals_added":len(proposed_set-original_set),
        "raw_pattern_fires":{
            p:int(np.sum(masks[p] & (opens_ms>=int(start.timestamp()*1000))
                                 & (opens_ms<int(end.timestamp()*1000))))
            for p in pattern_names
        },
    }


def phase_months(start,end):
    return pd.period_range(start.strftime("%Y-%m"),
                           (end-pd.Timedelta(days=1)).strftime("%Y-%m"),freq="M").astype(str).to_list()


def log_growth_by_month(rows,months):
    """Month-match log-account growth; no-trade months are exactly zero."""
    table={m:0. for m in months}
    for t in rows:
        net=ALLOCATION*t["net_position_return"]
        if net<=-1:raise RuntimeError("Nonpositive account event growth")
        table[t["month"]]+=np.log1p(net)
    return np.array([table[m] for m in months],float)


def performance(rows,months,original_rows,rng,info):
    nets=np.array([t["net_position_return"] for t in rows],float)
    rewards=ALLOCATION*nets
    eq=np.r_[1.,np.cumprod(1.+rewards)]
    draw=eq/np.maximum.accumulate(eq)-1
    orig_log=log_growth_by_month(original_rows,months)
    alt_log=log_growth_by_month(rows,months)
    difference=alt_log-orig_log
    samples=rng.integers(0,len(months),size=(BOOT,len(months)))
    means=np.mean(difference[samples],axis=1)
    boot_lo,boot_hi=np.quantile(means,[.025,.975])
    null=np.mean(difference*rng.choice((-1,1),size=(BOOT,len(months))),axis=1)
    obs=abs(float(np.mean(difference)))
    p=(1+int(np.sum(np.abs(null)>=obs)))/(BOOT+1)
    cnt_years=len(months)/12
    profit_per_trade=float(np.mean(nets)*100) if len(nets) else None
    diff_counts=info
    return {
        "trades":len(nets),
        "monthly_trades":round(len(nets)/len(months),3),
        "account_compounded_pct":round(float((eq[-1]-1)*100),4),
        "trade_win_rate_pct":round(float(np.mean(nets>0)*100),2) if len(nets) else None,
        "position_avg_net_pct":round(profit_per_trade,4) if len(nets) else None,
        "trade_close_account_mdd_pct":round(float(np.min(draw)*100),4),
        "worst_account_event_pct":round(float(np.min(rewards)*100),4) if len(nets) else None,
        "take_count":sum(t["exit_reason"]=="take_profit" for t in rows),
        "stop_count":sum("stop" in t["exit_reason"] for t in rows),
        "48h_time_count":sum(t["exit_reason"]=="time_48h" for t in rows),
        "same_bar_ambiguous_count":sum(t["ambiguous_stop_first"] for t in rows),
        "nonbaseline_signals_added":info["nonbaseline_signals_added"],
        "baseline_signals_displaced":info["baseline_signals_displaced"],
        "baseline_signals_kept":info["baseline_signals_kept"],
        "paired_mean_monthly_log_growth_diff_pp":round(float(np.mean(difference)*100),5),
        "paired_bootstrap_month95_diff_pp":[round(float(boot_lo*100),5),round(float(boot_hi*100),5)],
        "paired_exploratory_signflip_p":round(p,4),
    }


def synthetic_tests():
    from experiments.candlestick_study import signals as original_signals
    # Old strict Morning Star must be preserved, not loosened.
    f=pd.DataFrame({"open":np.full(42,100.),
        "close":np.full(42,100.),"high":np.full(42,101.),
        "low":np.full(42,99.)})
    t=32
    f.loc[t-5,"close"]=104.
    f.loc[t-2,["open","close","high","low"]]=[102.,98.,103.,97.]
    f.loc[t-1,["open","close","high","low"]]=[97.9,98.1,98.7,97.7]
    f.loc[t,["open","close","high","low"]]=[98.,101.3,102.3,97.8]
    assert np.array_equal(detect(f)["morning_star"],original_signals(f)["morning_star"])
    assert detect(f)["morning_star"][t]
    # Hammer requires previously declining market and a long lower wick.
    f.loc[t,["open","close","high","low"]]=[100.,100.3,100.35,98.0]
    f.loc[t-1,"close"]=95.
    assert detect(f)["hammer_bullish"][t]
    # 48h reserve must not backfill signals skipped during an earlier trade.
    mask=np.zeros(90,bool);mask[[31,32,44,45,60]]=True
    series=(np.arange(90)*4*3600*1000)+int(START.timestamp()*1000)
    seg=np.zeros(90,int)
    res=pick_events(mask,seg,series,START,TEST_START)
    assert res.tolist()==[31,44,60],res  # entry(44)=45 is AFTER reserved exit bar index44
    # Reject broken contiguous price history:
    seg[42:]=1
    out=pick_events(mask,seg,series,START,TEST_START)
    assert 31 not in out


def main():
    synthetic_tests()
    OUT.mkdir(parents=True,exist_ok=True)
    f,gaps,seg=fetch_gap_aware("4h")
    if f.empty:raise RuntimeError("No authentic Binance 4h spot history")
    times=pd.to_datetime(f.open_time_ms,unit="ms",utc=True)
    if times.iloc[0]>START or times.iloc[-1]<(END-pd.Timedelta(hours=8)):
        raise RuntimeError("Short spot history for study")
    masks=detect(f)
    atr=wilder_atr(f.high.to_numpy(float),f.low.to_numpy(float),
                   f.close.to_numpy(float),seg)
    # ATR intentionally stays undefined at the beginning of each real-data
    # contiguous segment after an OHLCV gap; per-signal strict validity below.
    outputs=[];trades=[];years=[];raw_count=[];phase_bases={}
    for phase,start,end in (("development_2020_2023",START,TEST_START),
                            ("inspected_2024_2026",TEST_START,END)):
        months=phase_months(start,end)
        base,bi=trades_for(f,seg,atr,masks,ARMS[0],phase,start,end)
        expect=53 if phase.startswith("development") else 44
        assert len(base)==expect, f"Changed ORIGINAL 4h frozen signal cohort: {len(base)} vs {expect}"
        phase_bases[phase]=base
        rng=np.random.default_rng(SEED+(0 if phase.startswith("development") else 1))
        for arm in ARMS:
            rows,info=trades_for(f,seg,atr,masks,arm,phase,start,end)
            if arm[0]==ARMS[0][0] and len(rows)!=len(base):
                raise AssertionError("Baseline mismatch")
            perf=performance(rows,months,base,rng,info)
            res={"phase":phase,"method":arm[0],"patterns_or":";".join(arm[1]),**perf}
            outputs.append(res)
            trades.extend(rows)
            raw_count.append({"phase":phase,"method":arm[0],**info})
            print("BULLISH_METHOD "+json.dumps({k:v for k,v in res.items() if
                k in ("phase","method","trades","monthly_trades",
                    "account_compounded_pct","trade_win_rate_pct",
                    "position_avg_net_pct","trade_close_account_mdd_pct",
                    "baseline_signals_displaced","nonbaseline_signals_added")},
                ensure_ascii=False),flush=True)
            for yr in sorted({t["year"] for t in rows}):
                sub=[t for t in rows if t["year"]==yr]
                a=np.array([t["net_position_return"] for t in sub])
                eq=np.r_[1.,np.cumprod(1+ALLOCATION*a)]
                ymonths=[s for s in months if s.startswith(str(yr))]
                years.append({"phase":phase,"method":arm[0],"year":yr,
                  "trades":len(sub),"account_compounded_pct":
                     round(float((eq[-1]-1)*100),4),
                  "win_rate_pct":round(float(np.mean(a>0)*100),2),
                  "avg_position_net_pct":round(float(np.mean(a)*100),4),
                  "months_in_study_year":len(ymonths)})
    frame=pd.DataFrame(outputs)
    dev=frame[frame.phase=="development_2020_2023"]
    later=frame[frame.phase=="inspected_2024_2026"]
    best_dev=dev.sort_values(["account_compounded_pct","trades"],
        ascending=[False,False]).iloc[0].to_dict()
    best_later=later.sort_values(["account_compounded_pct","trades"],
        ascending=[False,False]).iloc[0].to_dict()
    selected_later=later[later.method==best_dev["method"]].iloc[0].to_dict()
    zero_original_2024=later[later.method==ARMS[0][0]].iloc[0].to_dict()
    print("BULLISH_WINNERS "+json.dumps({
        "highest_development":best_dev,
        "chosen_development_method_in_later_history":selected_later,
        "highest_INSPECTED_later_hindsight":best_later,
        "unchanged_original_later":zero_original_2024,
        "warning":"Retrospective search on inspected 2024+ not a forward/OOS guarantee"
    },ensure_ascii=False),flush=True)
    frame.to_csv(OUT/"all_bullish_method_comparison.csv",index=False)
    pd.DataFrame(trades).to_csv(OUT/"per_trade_audit.csv",index=False)
    pd.DataFrame(years).to_csv(OUT/"yearly_performance.csv",index=False)
    pd.DataFrame(raw_count).to_csv(OUT/"signal_replacement_diagnostics.csv",index=False)
    (OUT/"winners.json").write_text(json.dumps({
        "development_selection":best_dev,
        "later_performance_of_dev_selected":selected_later,
        "hindsight_best_later":best_later,"unchanged_baseline_later":zero_original_2024
    },ensure_ascii=False,indent=2),encoding="utf8")
    notes={
       "research_only":True,"any_orders":False,
       "spot_data":"Authentic public BTCUSDT Binance spot 4h OHLCV",
       "period_UTC":"2020-01-01 to 2026-10-10 exclusive",
       "phases":"2020-23 development vs 2024-26 repeatedly inspected (not untouched)",
       "original_morningstar_cohorts":{"2020_23":53,"2024_26":44},
       "frozen_entry":"Completed 3rd signal candle, enter following 4h open",
       "frozen_exit":"Initial Wilder ATR14×1.5 stop, fixed +5% take, max48h",
       "frozen_position":"30% equity fixed virtual spot exposure per transaction, no leverage",
       "fees_slippage":"10 basis points fees + 5 basis points adverse price slippage at entry and exit",
       "OHLC_ambiguity":"If both stop and TP in same 4h bar: stop first. Worse opening fill on stop gaps.",
       "cohort":"All arms use same 48h full hold window reservation after EACH accepted signal, even if early stop or take. Signal skipped during reserved windows is NOT backfilled later. Must have 20 completed prior bars and full future 48h OHLC bars in same continuous segment and phase.",
       "patterns":{
         "morning_star":"exact frozen 2020-26 original 3bar",
         "bullish_engulfing":"exact existing candle-study 2bar engulfing",
         "bullish_harami":"exact existing candle-study 2bar harami",
         "hammer_bullish":"prior 4bar downtrend; current bullish body<=35% range; lower tail >=2body and >=50% range; upper<=25% range; close upper>=65% of range",
         "piercing_line":"prior large bearish body>=50% candle range, prior downturn 4bars; current green body>=40% range; open<=prior close; close>prior bearish midpoint but <prior open; gap optional for 24x7",
         "three_white_soldiers":"prior decline, three consecutively rising strong bullish candles bodies>=40% range upper wick<=25%, opens in prior real body",
         "breakout_20":"exact existing candle-study 20-bar high breakout"
       },
       "arms":[{"name":name,"patterns_or":list(groups)} for name,groups in ARMS],
       "volume_heatmap_filter":False,
       "paired_month_comparison":"For each phase and arm, compare monthly sum log account growth vs original; no-trade months included. Exploratory iid month bootstrap 2000 reps and signflip 2000 reps, uncorrected for 17 model comparisons or serial dependence.",
       "interpretation":["The OR signal order is chronological and can displace original Morning Star events; nonbaseline additions and displaced baseline are separately counted.",
       "Do not pick positive results from inspected 2024-26 as if independent out-of-sample. The original strategy has also already been selected via repeated backtesting.",
       "Only closed-trade-account drawdown computed, excludes unrealized intratrade peak-to-trough; nonindependent correlated patterns.",
       "No trading orders, do not auto-enable any new signal in the user's currently running paper simulator.",
       "Some seemingly traditional candle patterns are defined explicitly for 24x7 BTC with no session gaps.",
       "A winner with more trades may still be unprofitable after execution deviations; true new-data forward validation needed."],
       "gap_count":gaps,
       "raw_pattern_count_by_phase":raw_count
    }
    (OUT/"methodology.json").write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding="utf8")
    print("BULLISH_RESEARCH_COMPLETE "+json.dumps({
      "methods":len(ARMS),"study_4h_bars":len(f),
      "original":{phase:len(rows) for phase,rows in phase_bases.items()},
      "artifacts":6}),flush=True)


if __name__=="__main__":
    main()

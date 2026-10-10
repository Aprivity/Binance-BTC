"""ATR(14) risk-sized BTC/USDT 4h morning-star exit research. NO ORDERS.

Identical entry signals to prior 48-hour studies. Uses solely completed signal
candles when computing ATR, same conservative 4h OHLC bracket fills, and a
30% cap on equity deployed for each position (the rest remains idle cash).
This is retrospective exploratory research, NOT out-of-sample evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.candlestick_study import START, TEST_START, END, fetch_gap_aware, signals
from experiments.fixed_bracket_comparison import cohort, simulate_bracket
from experiments.hold_time_comparison import bootstrap_means, sign_flip_p

OUTPUT = Path("outputs/atr-risk-sized-comparison")
HOLD_BARS = 12  # 48h, same as previous cohort
ATR_PERIOD = 14
TAKE_PROFIT_PCT = 5.
CAPITAL_CAP = .30
STOP_RISK_BUDGET = .30*.03  # 0.90% of current equity at nominal stop distance
MULTIPLIERS = (1.5, 2.0, 2.5)  # 2.0x = primary; others sensitivity only
SEED = 20261010


def wilder_atr(high, low, close, segments, n=ATR_PERIOD):
    """True-range Wilder ATR using complete candles; restart after feed gaps.

    First n values of each contiguous segment seed ATR with mean of n TRs.
    ATR at signal bar i contains OHLC up to i, never from the entry bar i+1.
    """
    high, low, close, segments = (np.asarray(x) for x in (high,low,close,segments))
    atr=np.full(len(close),np.nan,dtype=float)
    if not (len(high)==len(low)==len(close)==len(segments)):
        raise ValueError("ATR arrays length mismatch")
    st=0
    while st<len(close):
        en=st+1
        while en<len(close) and segments[en]==segments[st]:
            en+=1
        tr=np.empty(en-st,dtype=float)
        tr[0]=high[st]-low[st]  # no previous close across gap
        if en-st>1:
            prev=close[st:en-1]
            tr[1:]=np.maximum.reduce((
                high[st+1:en]-low[st+1:en],
                np.abs(high[st+1:en]-prev),
                np.abs(low[st+1:en]-prev),
            ))
        if len(tr)>=n:
            current=float(np.mean(tr[:n]))
            atr[st+n-1]=current
            for j in range(n,len(tr)):
                current=(current*(n-1)+tr[j])/n
                atr[st+j]=current
        st=en
    return atr


def allocation(stop_distance_fraction):
    """Never deploy more than 30%; nominal stop-price loss budget <=0.9%.

    The budget does not cover fees, stop slippage, gaps or intrabar failures.
    """
    if not np.isfinite(stop_distance_fraction) or stop_distance_fraction<=0:
        raise ValueError("Non-positive stop distance")
    return float(min(CAPITAL_CAP,STOP_RISK_BUDGET/stop_distance_fraction))


def equity_path(account_returns):
    equity=np.concatenate(([1.],np.cumprod(1+np.asarray(account_returns,float))))
    peaks=np.maximum.accumulate(equity)
    return equity, float(np.min(equity/peaks-1.0))


def summarize(phase,name,unit,acc,alloc,stop_dists,reason,months,rng):
    _,mdd=equity_path(acc)
    idx=sorted(range(len(unit)),key=lambda j:unit[j])
    worst_n=max(1,int(np.ceil(.05*len(unit))))
    es=float(np.mean(unit[np.array(idx[:worst_n])]))
    es_account=float(np.mean(acc[np.argsort(acc)[:worst_n]]))
    profits=float(np.sum(acc[acc>0]))
    losses=float(-np.sum(acc[acc<0]))
    ci=bootstrap_means(acc,months,rng)*100
    sorted_acc=np.sort(acc)
    nwin=np.sum(acc>0)
    streak=0;max_streak=0
    for v in acc:
        if v<0:
            streak+=1;max_streak=max(max_streak,streak)
        else:
            streak=0
    row={
        "phase":phase,"strategy":name,"trades":int(len(unit)),
        "mean_trade_net_pct":round(float(unit.mean()*100),4),
        "median_trade_net_pct":round(float(np.median(unit)*100),4),
        "win_pct":round(float(nwin/len(unit)*100),2),
        "worst_trade_net_pct":round(float(np.min(unit)*100),4),
        "historical_worst_5pct_trade_avg_pct":round(es*100,4),
        "account_mean_per_signal_pct":round(float(acc.mean()*100),4),
        "account_worst_signal_pct":round(float(np.min(acc)*100),4),
        "account_historical_es05_pct":round(es_account*100,4),
        "account_compounded_return_pct":round(float((np.prod(1+acc)-1)*100),4),
        "account_trade_close_mdd_pct":round(mdd*100,4),
        "account_month_block_ci95_mean_pct":[round(float(x),4) for x in ci],
        "allocation_mean_pct":round(float(np.mean(alloc)*100),4),
        "allocation_min_pct":round(float(np.min(alloc)*100),4),
        "allocation_max_pct":round(float(np.max(alloc)*100),4),
        "stop_distance_mean_pct":round(float(np.nanmean(stop_dists)*100),4) if np.isfinite(stop_dists).any() else None,
        "profit_factor_account":round(profits/losses,4) if losses>0 else None,
        "longest_loss_streak":int(max_streak),
        "stop_exits":int(np.isin(reason,["stop_loss","gap_stop"]).sum()),
        "take_exits":int(np.sum(reason=="take_profit")),
        "time_exits":int(np.sum(reason=="time_48h")),
    }
    print("ATR_RESULT "+json.dumps(row,ensure_ascii=False),flush=True)
    return row


def deterministic_tests():
    # All history after signal candle should not alter its ATR or allocation.
    h=np.array([101.+i for i in range(26)])
    l=h-3.;c=h-1.
    seg=np.zeros(26,dtype=int)
    reference=wilder_atr(h,l,c,seg)
    changed_h=h.copy();changed_h[21:]+=1000
    changed_l=l.copy();changed_l[21:]-=500
    changed_c=c.copy();changed_c[21:]+=500
    replay=wilder_atr(changed_h,changed_l,changed_c,seg)
    assert np.isclose(reference[20],replay[20])
    assert np.isnan(reference[12]) and np.isfinite(reference[13])
    assert np.isclose(allocation(.02),.30)
    assert np.isclose(allocation(.06),.15)
    assert np.isclose(allocation(.03),.30)
    assert allocation(.2)<=.0450000001
    # A discontinuous segment should not use earlier candles' ATR.
    seg2=seg.copy();seg2[18:]=1
    assert np.isnan(wilder_atr(h,l,c,seg2)[20])
    # Simulated order price model is identical to the fixed-bracket baseline.
    op=np.array([100.,101.,100.,101.])
    hi=np.array([100.5,101.5,102.,102.])
    lo=np.array([99.,99.,99.,99.])
    a=simulate_bracket(op,hi,lo,0,5,3,3)
    assert a[2]=="time_48h"
    # Account only loses its allocated fraction if a spot trade loses.
    assert np.isclose(.3*(-.03),-.009)


def run():
    deterministic_tests()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    df,gaps,segments=fetch_gap_aware("4h")
    ts=pd.to_datetime(df.open_time_ms,unit="ms",utc=True)
    if ts.iloc[0]>START or ts.iloc[-1]<END-pd.Timedelta("8h"):
        raise RuntimeError("BTCUSDT real 4h history incomplete")
    op=df.open.to_numpy(float)
    hi=df.high.to_numpy(float)
    lo=df.low.to_numpy(float)
    cl=df.close.to_numpy(float)
    atr=wilder_atr(hi,lo,cl,segments)
    triggers=signals(df)["morning_star"]
    rng=np.random.default_rng(SEED)
    summaries=[];trades=[];years=[];paired=[];stress=[]
    config=[
        ("fixed_tp5_sl3",None,3.0),
        ("fixed_tp5_sl2",None,2.0),
        ("atr14_x1_5",1.5,None),
        ("atr14_x2_0_primary",2.0,None),
        ("atr14_x2_5",2.5,None),
    ]
    for phase,start,end in (
        ("development_2020_2023",START,TEST_START),
        ("evaluation_2024_2026",TEST_START,END),
    ):
        signal_idx=cohort(df,segments,triggers,start,end)
        expected=53 if phase.startswith("development") else 44
        if len(signal_idx)!=expected:
            raise RuntimeError(f"Changed entry cohort: {phase}, {len(signal_idx)} != {expected}")
        if not np.isfinite(atr[signal_idx]).all():
            raise RuntimeError("Missing signal-candle ATR, cannot compare same entries")
        entry_idx=signal_idx+1
        months=ts.iloc[entry_idx].dt.strftime("%Y-%m").to_numpy()
        entry_years=ts.iloc[entry_idx].dt.year.to_numpy()
        variants={}
        for name,multiplier,fixed_sl in config:
            stop_distance=np.where(multiplier is None, fixed_sl/100 if fixed_sl else np.nan,
                                   multiplier*atr[signal_idx]/(op[entry_idx]*(1+.0005)) if multiplier else np.nan)
            stop_distance=np.full(len(signal_idx),fixed_sl/100,dtype=float) if multiplier is None else np.asarray(stop_distance,dtype=float)
            if np.any(~np.isfinite(stop_distance)) or np.any(stop_distance<=0):
                raise ValueError("Invalid ATR-derived stop level")
            frac=np.asarray([allocation(d) if multiplier is not None else CAPITAL_CAP for d in stop_distance])
            if np.any(frac>CAPITAL_CAP+1e-10) or np.any(frac<=0) or np.any(frac*stop_distance>STOP_RISK_BUDGET+1e-10):
                raise AssertionError("Capital cap or nominal stop risk budget violated")
            results=[simulate_bracket(op,hi,lo,int(e),TAKE_PROFIT_PCT,float(d*100),HOLD_BARS) for e,d in zip(entry_idx,stop_distance)]
            unit=np.asarray([r[0] for r in results])
            acct=frac*unit
            reasons=np.asarray([r[2] for r in results],dtype=object)
            exit_i=np.asarray([r[3] for r in results],int)
            amb=np.asarray([r[4] for r in results],bool)
            variants[name]=(unit,acct,frac,reasons,stop_distance,exit_i,amb)
            row=summarize(phase,name,unit,acct,frac,stop_distance,reasons,months,rng)
            row["ambiguous_same_bar"]=int(amb.sum())
            summaries.append(row)
            for y in sorted(set(entry_years)):
                sel=entry_years==y
                rv=acct[sel]
                _,mdd=equity_path(rv)
                yr={"phase":phase,"year":int(y),"strategy":name,
                    "trades":int(np.sum(sel)),
                    "account_mean_per_signal_pct":round(float(np.mean(rv)*100),4),
                    "trade_mean_pct":round(float(np.mean(unit[sel])*100),4),
                    "compounded_account_return_pct":round(float((np.prod(1+rv)-1)*100),4),
                    "trade_close_mdd_pct":round(float(mdd*100),4)}
                years.append(yr)
                print("ATR_YEAR "+json.dumps(yr),flush=True)
            for shock in (.005,.01,.03):
                # Shock only stop fills: adverse execution price is (1-shock)
                # times the already-modeled sale fill. No "fixed loss cap".
                stressed=unit.copy()
                hit=np.isin(reasons,["stop_loss","gap_stop"])
                stressed[hit]=(1+stressed[hit])*(1-shock)-1
                acc_stressed=frac*stressed
                sr={"phase":phase,"strategy":name,
                    "additional_stop_execution_impact_pct":shock*100,
                    "stops_affected":int(hit.sum()),
                    "mean_account_signal_pct":round(float(acc_stressed.mean()*100),4),
                    "compounded_account_pct":round(float((np.prod(1+acc_stressed)-1)*100),4),
                    "worst_account_trade_pct":round(float(acc_stressed.min()*100),4)}
                stress.append(sr)
                print("ATR_STRESS "+json.dumps(sr),flush=True)
        # Same-event account return comparisons; exploratory (dates already used for selection).
        for other in ("atr14_x1_5","atr14_x2_0_primary","atr14_x2_5","fixed_tp5_sl2"):
            a=variants[other][1]
            b=variants["fixed_tp5_sl3"][1]
            dif=a-b
            ci=bootstrap_means(dif,months,rng)
            pr={"phase":phase,"comparison":f"{other} - fixed_tp5_sl3",
                "matched_events":len(dif),"account_mean_difference_pp":round(float(dif.mean()*100),4),
                "ci95_month_block_pp":[round(float(v*100),4) for v in ci],
                "exploratory_sign_flip_p_two_sided":round(float(sign_flip_p(dif,months,rng)),4)}
            paired.append(pr)
            print("ATR_PAIRED "+json.dumps(pr),flush=True)
        for j,i in enumerate(signal_idx):
            rec={"phase":phase,"signal_utc":ts.iloc[i].isoformat(),
                 "entry_utc":ts.iloc[entry_idx[j]].isoformat(),
                 "entry_open":round(float(op[entry_idx[j]]),4),
                 "atr14_signal_close":round(float(atr[i]),5),
                 "atr14_signal_pct_of_next_open":round(float(atr[i]/op[entry_idx[j]]*100),4)}
            for name in variants:
                unit,acct,frac,reason,d,exit_i,amb=variants[name]
                rec.update({
                    name+"_stop_distance_pct":round(float(d[j]*100),4),
                    name+"_allocation_pct":round(float(frac[j]*100),4),
                    name+"_trade_net_pct":round(float(unit[j]*100),4),
                    name+"_account_net_pct":round(float(acct[j]*100),4),
                    name+"_exit_reason":str(reason[j]),
                    name+"_exit_utc":ts.iloc[exit_i[j]].isoformat(),
                    name+"_ambiguous_bar":bool(amb[j]),
                })
            trades.append(rec)
    pd.DataFrame(summaries).to_csv(OUTPUT/"atr_vs_fixed_summary.csv",index=False)
    pd.DataFrame(years).to_csv(OUTPUT/"yearly_account_comparison.csv",index=False)
    pd.DataFrame(stress).to_csv(OUTPUT/"stop_execution_stress.csv",index=False)
    pd.DataFrame(paired).to_csv(OUTPUT/"paired_account_differences.csv",index=False)
    pd.DataFrame(trades).to_csv(OUTPUT/"same_entry_trade_audit.csv",index=False)
    meta={
        "source":"Real Binance public BTCUSDT spot 4h OHLCV, not generated data",
        "start_utc":START.isoformat(),"split_utc":TEST_START.isoformat(),
        "end_exclusive_utc":END.isoformat(),
        "market_gap_segments":gaps,
        "same_entry_cohorts":{"development":53,"evaluation":44},
        "entry":"Original 4h morning-star detected at complete signal candle i; buy next open i+1.",
        "no_trend_filter":True,
        "ATR":"True range with previous close, Wilder RMA period 14. ATR at signal completed candle i, never from entry candle i+1. Reset after data gaps.",
        "ATR_multipliers":{"primary":2.0,"sensitivity_only":[1.5,2.5]},
        "fixed_comparators":["TP5 SL3","TP5 SL2"],
        "stop":"STATIC per-trade volatility-scaled initial stop, not a trailing stop; stop distance = multiplier * ATR(signal i). Does not ratchet during trade.",
        "take_profit_pct":5.0,"max_bars":12,
        "allocation":"risk_budget = 0.9% of current account equity = 30% * 3% reference stop; min(30%, 0.9% / ATR_stop_fraction), never leveraged.",
        "fees":{"entry":10,"exit":10,"unit":"bps"},
        "slippage":{"entry":5,"exit":5,"unit":"bps"},
        "conservative_intrabar_fills":"Sell at stop when low touches, including on entry candle; if TP and SL in same 4h candle choose stop first. Gap below stop fills worse at candle opening. Stop order not a guaranteed loss ceiling.",
        "account_math":"sequential spot trades, cash not allocated remains constant; account factor each trade = 1 + position_fraction * unit_trade_net. Realized trade-close drawdown only; no taxes, yield on idle cash, missed fills, or intratrade mark-to-market.",
        "tests":"synthetic anti-lookahead, gap reset, cap and risk budget, identical event counts, OHLC assumptions",
        "confidence_and_comparison":"month-clustered bootstrap and sign-flip pairwise exploratory; repeated historical analysis invalidates untouched out-of-sample claim.",
        "limits":[
            "ATR is only based on completed candle at entry, not later volatility and not a trailing stop.",
            "Risk budget excludes fees and gap/slippage losses; true realized account loss can exceed 0.9%.",
            "A single 4h OHLC cannot identify order of TP/SL hits; pessimistically stop-first.",
            "The 2024-2026 period and original Morning Star signal were already inspected repeatedly. Parameter comparisons are data-snooped; no fresh independent evaluation.",
            "Bootstrap CI and historical Expected Shortfall are not future guarantees with only 44 evaluation events.",
            "Cohort is determined under a hypothetical 48h overlap, even if some orders exit early; gives apples-to-apples fixed entries but not maximum trade frequency.",
            "Volatility-adaptive sizing mixes exit and sizing effects; fixed-30%-allocation ATR ablation should be considered separately if attributing causal improvement.",
            "No live, paper, or real orders; no leverage, no main branch merge.",
        ]
    }
    (OUTPUT/"methodology.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf8")
    print("ATR_RISK_RESEARCH_COMPLETE",flush=True)


if __name__=="__main__":
    run()

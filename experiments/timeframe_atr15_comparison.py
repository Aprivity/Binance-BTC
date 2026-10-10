"""Research ONLY: same Morning Star / ATR1.5 / TP5 / 48h / 30% exposure,
different BTCUSDT spot OHLCV candle interval: 1h, 4h, 8h, 1d.

No default strategy changes, API orders, simulation order writes or merges.
Historical comparisons are exploratory; 2024+ has been inspected repeatedly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.candlestick_study import START, TEST_START, END, fetch_gap_aware, signals
from experiments.atr_risk_sized_comparison import wilder_atr
from experiments.fixed_bracket_comparison import simulate_bracket, cohort as legacy_4h_cohort

OUT=Path("outputs/timeframe-atr15-comparison")
CANDLE_HOURS={"1h":1,"4h":4,"8h":8,"1d":24}
PCT_TAKE=5.0
ATR_MULTIPLE=1.5
ALLOCATION=.30
HORIZON_HOURS=48
SEED=20261010
BOOTSTRAP=4000


def resample_8h_from_4h(src):
    """Build complete 8h UTC OHLCV from exactly two contiguous authentic 4h bars.

    Unix-epoch aligned start 00:00/08:00/16:00 UTC. Drop incomplete groups,
    do NOT bridge missing data with synthetic or forward filled bars.
    """
    step4=4*3_600_000
    step8=8*3_600_000
    c=src.copy()
    c["bucket"]=(c.open_time_ms//step8).astype(np.int64)
    agg=c.groupby("bucket",sort=True).agg(
        start=("open_time_ms","first"),end=("open_time_ms","last"),
        count=("open_time_ms","size"),
        open=("open","first"),high=("high","max"),low=("low","min"),
        close=("close","last"),base_volume=("base_volume","sum")
    )
    good=(agg["count"]==2)&(agg["start"]==agg.index.to_numpy(dtype=np.int64)*step8)&(
        agg["end"]==agg["start"]+step4)
    valid=agg.loc[good].copy()
    if valid.empty:
        raise RuntimeError("No complete 8h groups")
    out=valid[["start","open","high","low","close","base_volume"]].rename(
        columns={"start":"open_time_ms"}).reset_index(drop=True)
    out["open_time_ms"]=out["open_time_ms"].astype(np.int64)
    out["close_time_ms"]=out["open_time_ms"]+step8-1
    segments=np.cumsum(out.open_time_ms.diff().fillna(step8).to_numpy()!=step8)
    return out,int(np.sum(np.diff(out.open_time_ms.to_numpy())!=step8)),segments


def select_signals(frame,segments,mask,hold_bars,start,end):
    """Select nonoverlapping max-48h windows with gap-free history and exits."""
    n=len(frame)
    stamps=frame.open_time_ms.to_numpy(np.int64)
    start_ms=int(start.timestamp()*1000);end_ms=int(end.timestamp()*1000)
    kept=[]
    last_reserved_exit=-1
    for k in np.flatnonzero(mask):
        s=int(k);entry=s+1;ending=entry+hold_bars
        if s<20 or ending>=n or stamps[s]<start_ms or stamps[s]>=end_ms:
            continue
        if stamps[ending]>=end_ms or segments[s-20]!=segments[ending]:
            continue
        if entry<=last_reserved_exit:
            continue
        kept.append(s)
        last_reserved_exit=ending
    return np.asarray(kept,dtype=int)


def stats(trades):
    if not trades:
        return {
            "trades":0,"win_rate_pct":None,"position_mean_net_pct":None,
            "account_compounded_pct":None,"account_close_mdd_pct":None
        }
    p=np.asarray([t["position_net_fraction"] for t in trades],dtype=float)
    a=p*ALLOCATION
    eq=np.r_[1.,np.cumprod(1+a)]
    mdd=eq/np.maximum.accumulate(eq)-1
    tail=max(1,int(np.ceil(.05*len(a))))
    current=longest=0
    for r in a:
        if r<0:
            current+=1
            longest=max(current,longest)
        else:
            current=0
    n_tp=sum(t["reason"]=="take_profit" for t in trades)
    n_sl=sum(t["reason"] in ("stop_loss","gap_stop") for t in trades)
    n_t=sum(t["reason"]=="time_48h" for t in trades)
    return {
        "trades":len(a),"win_rate_pct":round(float(np.mean(p>0)*100),2),
        "position_mean_net_pct":round(float(np.mean(p)*100),4),
        "position_median_net_pct":round(float(np.median(p)*100),4),
        "position_worst_net_pct":round(float(np.min(p)*100),4),
        "position_historical_es05_pct":round(float(np.sort(p)[:tail].mean()*100),4),
        "account_mean_signal_pct":round(float(np.mean(a)*100),4),
        "account_worst_event_pct":round(float(np.min(a)*100),4),
        "account_historical_es05_pct":round(float(np.sort(a)[:tail].mean()*100),4),
        "account_compounded_pct":round(float((eq[-1]-1)*100),4),
        "account_close_mdd_pct":round(float(np.min(mdd)*100),4),
        "account_longest_loss_streak":int(longest),
        "average_stop_distance_pct":round(float(np.mean([t["stop_pct"] for t in trades])),4),
        "max_stop_distance_pct":round(float(np.max([t["stop_pct"] for t in trades])),4),
        "tp_count":n_tp,"stop_count":n_sl,"time_count":n_t,
        "ambiguous_count":sum(t["same_bar_ambiguous"] for t in trades),
    }


def monthly_logreturns(trades,month_keys):
    m={str(k):0. for k in month_keys}
    for t in trades:
        m[t["entry_month"]]+=float(np.log1p(ALLOCATION*t["position_net_fraction"]))
    return np.array([m[str(k)] for k in month_keys])


def compare_monthly(phase,baseline,other,base_trades,other_trades,period_start,period_end,rng):
    months=pd.period_range(start=period_start.strftime("%Y-%m"),
        end=(period_end-pd.Timedelta(days=1)).strftime("%Y-%m"),freq="M").astype(str).tolist()
    base=monthly_logreturns(base_trades,months)
    var=monthly_logreturns(other_trades,months)
    dif=var-base
    sample_idx=rng.integers(0,len(dif),size=(BOOTSTRAP,len(dif)))
    means=dif[sample_idx].mean(axis=1)
    signs=rng.choice((-1.,1.),size=(BOOTSTRAP,len(dif)))
    perm=np.abs((signs*dif).mean(axis=1))
    obs=np.abs(np.mean(dif))
    return {
        "phase":phase,"candidate":other,"reference":baseline,
        "month_count_including_no_signals":len(dif),
        "difference_avg_month_log_pp":round(float(dif.mean()*100),5),
        "bootstrap_ci95_month_log_diff_pp":[round(float(v*100),5) for v in np.quantile(means,[.025,.975])],
        "exploratory_month_signflip_p":round(float((1+np.sum(perm>=obs))/(1+BOOTSTRAP)),4),
        "warning":"No matched entry cohort across K-line timeframes; calendar-month aligned returns. Historical selection invalidates OOS interpretation."
    }


def trade_events(frame,segments,timeframe,phase,begin,end):
    hours=CANDLE_HOURS[timeframe]
    horizon=HORIZON_HOURS//hours
    op=frame.open.to_numpy(float)
    h=frame.high.to_numpy(float)
    l=frame.low.to_numpy(float)
    c=frame.close.to_numpy(float)
    ts=pd.to_datetime(frame.open_time_ms,unit="ms",utc=True)
    atr=wilder_atr(h,l,c,segments)
    sig=signals(frame)["morning_star"]
    indices=select_signals(frame,segments,sig,horizon,begin,end)
    if timeframe=="4h":
        # Fail closed on 4h reference drift against previous frozen cohort.
        reference=legacy_4h_cohort(frame,segments,sig,begin,end)
        if not np.array_equal(indices,reference):
            raise RuntimeError("4h cohort selection drift from original 53/44 reference")
        expected=53 if phase=="development_2020_2023" else 44
        if len(indices)!=expected:
            raise RuntimeError(f"4h entry count drift {len(indices)} expected {expected}")
    if np.any(~np.isfinite(atr[indices])):
        raise RuntimeError(f"{timeframe} undefined ATR on confirmed signal")
    result=[]
    for i in indices:
        entry=int(i)+1
        buy_fill=op[entry]*1.0005
        stop_distance_pct=float(100*ATR_MULTIPLE*atr[i]/buy_fill)
        net,gross,reason,exit_idx,amb=simulate_bracket(
            op,h,l,entry,PCT_TAKE,stop_distance_pct,horizon)
        r={
            "phase":phase,"timeframe":timeframe,
            "signal_utc":ts.iloc[i].isoformat(),
            "entry_utc":ts.iloc[entry].isoformat(),
            "exit_reference_utc":ts.iloc[exit_idx].isoformat(),
            "entry_month":ts.iloc[entry].strftime("%Y-%m"),
            "entry_year":int(ts.iloc[entry].year),
            "signal_atr14":round(float(atr[i]),5),
            "entry_market_open":round(float(op[entry]),4),
            "stop_pct":round(stop_distance_pct,5),
            "position_net_fraction":float(net),
            "position_net_pct":round(float(net*100),4),
            "account_net_pct":round(float(ALLOCATION*net*100),4),
            "reason":reason,
            "same_bar_ambiguous":bool(amb)
        }
        result.append(r)
    return result


def tests():
    # 8h direct candle examples: exact opening and OHLC aggregation.
    frame=pd.DataFrame({
        "open_time_ms":np.array([0,14_400_000,28_800_000,43_200_000],dtype=np.int64),
        "open":[100.,101.,103.,102.],"high":[105.,106.,106.,108.],
        "low":[99.,100.,100.,98.],"close":[101.,103.,102.,107.],
        "base_volume":[2.,3.,1.,4.]
    })
    result,gaps,segments=resample_8h_from_4h(frame)
    assert len(result)==2 and gaps==0 and result.iloc[0]["open"]==100.
    assert result.iloc[0]["close"]==103. and result.iloc[0]["base_volume"]==5.
    assert result.iloc[1]["high"]==108. and result.iloc[1]["low"]==98.
    deleted=frame.drop(index=1).reset_index(drop=True)
    result2,gaps2,segments2=resample_8h_from_4h(deleted)
    assert len(result2)==1 and result2.iloc[0]["open_time_ms"]==28_800_000
    assert HORIZON_HOURS//CANDLE_HOURS["1d"]==2
    assert HORIZON_HOURS//CANDLE_HOURS["4h"]==12
    assert HORIZON_HOURS//CANDLE_HOURS["1h"]==48
    assert HORIZON_HOURS//CANDLE_HOURS["8h"]==6


def main():
    tests()
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED)
    market={}
    frames={}
    for tf in ("1h","4h","1d"):
        print(f"TIMEFRAME_FETCH {tf}",flush=True)
        df,gaps,seg=fetch_gap_aware(tf)
        timestamp=pd.to_datetime(df.open_time_ms,unit="ms",utc=True)
        if timestamp.iloc[0]>START or timestamp.iloc[-1]<END-pd.Timedelta(hours=2*CANDLE_HOURS[tf]):
            raise RuntimeError(f"Incomplete {tf} history")
        frames[tf]=(df,int(gaps),seg)
        market[tf]={"source":"Binance genuine spot kline","candles":len(df),
                    "gaps":int(gaps),"first_utc":timestamp.iloc[0].isoformat(),
                    "last_utc":timestamp.iloc[-1].isoformat()}
    df8,gaps8,seg8=resample_8h_from_4h(frames["4h"][0])
    frames["8h"]=(df8,gaps8,seg8)
    market["8h"]={"source":"UTC-aligned exact 2x contiguous genuine Binance 4h candles",
        "candles":len(df8),"gaps":int(gaps8)}
    all_trades=[];summaries=[];yearly=[];phase_trades={};comparisons=[]
    for tf in CANDLE_HOURS:
        frame,gaps,segments=frames[tf]
        for phase,start,end in (
            ("development_2020_2023",START,TEST_START),
            ("evaluation_2024_2026",TEST_START,END)
        ):
            events=trade_events(frame,segments,tf,phase,start,end)
            phase_trades[(phase,tf)]=events
            for event in events:
                all_trades.append(event)
            row={"phase":phase,"timeframe":tf,
                "candle_hours":CANDLE_HOURS[tf],
                "atr_lookback_nominal_hours":14*CANDLE_HOURS[tf],
                "holding_horizon_hours":HORIZON_HOURS,
                "signal_first_candle_lookback_hours":7*CANDLE_HOURS[tf],
                **stats(events)}
            summaries.append(row)
            print("TIMEFRAME_RESULT "+json.dumps(row,ensure_ascii=False),flush=True)
            for y in sorted(set(e["entry_year"] for e in events)):
                subset=[e for e in events if e["entry_year"]==y]
                r=np.array([e["position_net_fraction"] for e in subset],dtype=float)
                result={"phase":phase,"timeframe":tf,"year":y,
                        "trade_count":len(r),
                        "account_year_compound_pct":round(float((np.prod(1+ALLOCATION*r)-1)*100),4),
                        "position_mean_net_pct":round(float(r.mean()*100),4),
                        "win_rate_pct":round(float((r>0).mean()*100),2)}
                yearly.append(result)
                print("TIMEFRAME_YEAR "+json.dumps(result),flush=True)
    for phase,start,end in (
        ("development_2020_2023",START,TEST_START),
        ("evaluation_2024_2026",TEST_START,END),
    ):
        for tf in CANDLE_HOURS:
            if tf=="4h":
                continue
            pair=compare_monthly(phase,"4h",tf,phase_trades[(phase,"4h")],
                                 phase_trades[(phase,tf)],start,end,rng)
            comparisons.append(pair)
            print("TIMEFRAME_PAIRED "+json.dumps(pair),flush=True)
    pd.DataFrame(summaries).to_csv(OUT/"period_summaries.csv",index=False)
    pd.DataFrame(all_trades).to_csv(OUT/"all_trade_events.csv",index=False)
    pd.DataFrame(yearly).to_csv(OUT/"yearly_comparison.csv",index=False)
    pd.DataFrame(comparisons).to_csv(OUT/"month_matched_comparisons.csv",index=False)
    notes={
        "research_only":True,"main_unchanged":True,"real_paper_or_live_orders":False,
        "market":"BTCUSDT Binance public spot candles",
        "history_utc":[START.isoformat(),END.isoformat()],"split_utc":TEST_START.isoformat(),
        "source_coverage":market,
        "timeframes_hours":CANDLE_HOURS,
        "entry":"Original unmodified 3-candle morning_star from candlestick_study.py; closed signal candle, next candle open entry",
        "filter":"No trend/news/volume filters",
        "atr":"Wilder ATR14 computed with completed signal candle at same candle interval (14 bars, not matched physical hours)",
        "atr_multiplier":ATR_MULTIPLE,
        "stop":"Static initial price = simulated buy execution fill - 1.5x signal ATR, NOT trailing",
        "take_profit_percent":PCT_TAKE,
        "hold_max_hours":HORIZON_HOURS,
        "time_exit":"Next opening after hold 48h, every timeframe. Daily bars -> 2 days, 8h -> 6, 4h -> 12, 1h -> 48.",
        "position_allocation_pct":30,
        "portfolio_model":"Only one spot position per pre-reserved 48h interval for each timeframe; remaining 70% cash. Compound each result at exit; close-to-close MDD only.",
        "transaction_cost":{"per_side_fee_bps":10,"per_side_slippage_bps":5},
        "execution":"Intrabar stop/TP evaluated against OHLC. If both stop and take hit in one bar, stop first. Gap at worse opening. Real fill, 1m order path, depth not verified.",
        "timeframe_comparability_warnings":[
            "Only discrete K-line interval setting remains variable, but ATR14 in real-world lookback hours changes: 1h=14h,4h=56h,8h=112h,1d=336h. This is a bundled timeframe+volatility-estimator-window comparison, NOT isolated changes to chart display.",
            "Morning Star historical pattern covers different physical hours in each timeframe; 1d can be a multi-day chart pattern whereas 1h is very short.",
            "Signal counts differ between intervals. Cumulative returns reward opportunity frequency and are not fair per-trade comparisons; report per-event expectancy AND risk/cumulative account outcomes.",
            "All intervals were looked at in older candlestick studies and 2024+ observations have been repeatedly inspected, so not a true untouched independent out-of-sample confirmation.",
            "4h entries have fail-closed assertion that original cohorts remain exactly 53/44.",
            "Use 8h resampling only from complete UTC-aligned consecutive pairs of genuine Binance 4h bars; no made-up prices, no forward fill.",
            "Historical OHLC does not resolve actual intrabar order, 1d bar stop-first especially conservative and path-ambiguous; live execution may be worse.",
            "Price-unit ATR and static TP5 can lead to very uneven stop-to-reward ratio by interval. No stop-risk cap; 30% fixed exposure may incur significant loss on wide daily bars.",
            "Month-matched comparisons use each strategy's own calendar-month aggregate log account growth; signal entries are NOT matched across timeframes. Confidence intervals and sign-flip are exploratory, many zero-trade months.",
            "This research is neither a live trading recommendation nor realized annual returns; no passive buy-and-hold comparison yet."
        ]
    }
    (OUT/"methodology.json").write_text(json.dumps(notes,indent=2,ensure_ascii=False),encoding="utf8")
    print("TIMEFRAME_RESEARCH_COMPLETE",flush=True)


if __name__=="__main__":
    main()

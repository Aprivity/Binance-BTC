"""Point-in-time 4h BTC Morning Star confirmation candle volume ablation.

Research only: same frozen 3-candle pattern, ATR(14)*1.5 initial stop,
TP5%, maximum 48h, fixed 30% of account per signal, no default strategy.

Compare RVOL on the *completed bullish confirmation candle* at signal i:
  RVOL = volume[i] / mean(volume[i-20:i]).
Two predetermined volume gates >=1.0x and >=1.5x, and an original no-gate
baseline. NO order-book/liquidation heatmap is claimed or reconstructed:
OHLCV does not contain historical resting liquidity or liquidation estimates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.candlestick_study import START,TEST_START,END,fetch_gap_aware,signals
from experiments.fixed_bracket_comparison import cohort,simulate_bracket
from experiments.atr_risk_sized_comparison import wilder_atr

OUT=Path("outputs/morningstar-volume-ablation")
LOOKBACK=20
FILTERS=(("original_no_volume_gate",None),("confirmation_RVOL_ge_1_0",1.0),
         ("confirmation_RVOL_ge_1_5",1.5))
BINS=(("rvol_below_0_75",0.,.75),("rvol_0_75_to_1_0",.75,1.),
      ("rvol_1_0_to_1_5",1.,1.5),("rvol_above_1_5",1.5,np.inf))
ATR_MULT=1.5
ALLOCATION=.30
TP_PCT=5.
HOLD_BARS=12
SEED=20261010
BOOTSTRAP=4000


def rolling_rvol(vol,seg,lookback=LOOKBACK):
    """Compare completed signal volume only with PRECEDING lookback candles.

    Resets window on gaps; no incomplete/current entry bar reads.
    """
    volume=np.asarray(vol,dtype=float)
    segments=np.asarray(seg)
    ratio=np.full(len(volume),np.nan)
    for i in range(lookback,len(volume)):
        if segments[i-lookback]!=segments[i]:
            continue
        baseline=float(volume[i-lookback:i].mean())
        if baseline>0 and np.isfinite(baseline) and np.isfinite(volume[i]):
            ratio[i]=float(volume[i]/baseline)
    return ratio


def data(frame,seg):
    atr=wilder_atr(frame.high.to_numpy(float),frame.low.to_numpy(float),
                   frame.close.to_numpy(float),seg)
    rel=rolling_rvol(frame.base_volume.to_numpy(float),seg)
    return atr,rel


def summarise(trades,phase,arm):
    if not trades:
        return {"phase":phase,"arm":arm,"trades":0}
    net=np.asarray([r["trade_net_fraction"] for r in trades],dtype=float)
    account=ALLOCATION*net
    eq=np.r_[1.,np.cumprod(1+account)]
    drawdown=eq/np.maximum.accumulate(eq)-1
    tail=max(1,int(np.ceil(len(net)*.05)))
    losses=0;max_losses=0
    for x in net:
        if x<0:
            losses+=1
            max_losses=max(max_losses,losses)
        else:
            losses=0
    reasons=pd.Series([r["exit_reason"] for r in trades]).value_counts().to_dict()
    return {
        "phase":phase,"arm":arm,"trades":int(len(net)),
        "trade_mean_net_pct":round(float(net.mean()*100),4),
        "trade_median_net_pct":round(float(np.median(net)*100),4),
        "win_rate_pct":round(float(np.mean(net>0)*100),2),
        "account_mean_signal_pct":round(float(account.mean()*100),4),
        "account_compounded_pct":round(float((eq[-1]-1)*100),4),
        "account_trade_close_mdd_pct":round(float(drawdown.min()*100),4),
        "account_worst_single_pct":round(float(account.min()*100),4),
        "account_es05_sample_pct":round(float(np.sort(account)[:tail].mean()*100),4),
        "longest_losses":int(max_losses),
        "tp_count":int(reasons.get("take_profit",0)),
        "stop_count":int(reasons.get("stop_loss",0)+reasons.get("gap_stop",0)),
        "time_count":int(reasons.get("time_48h",0)),
        "rvol_mean":round(float(np.mean([t["confirmation_RVOL"] for t in trades])),4),
    }


def monthly_account_logs(trades,all_months):
    groups={str(k):0.0 for k in all_months}
    for t in trades:
        groups[t["month"]]+=float(np.log1p(ALLOCATION*t["trade_net_fraction"]))
    return np.asarray([groups[str(m)] for m in all_months],dtype=float)


def month_paired(baseline,filtered,start,end,name,phase,rng):
    # Include zero-signal months, so filtering away a losing trade is still
    # properly compared with cash holdings throughout the same calendar time.
    months=pd.period_range(start=start.strftime("%Y-%m"),
             end=(end-pd.Timedelta(days=1)).strftime("%Y-%m"),freq="M").astype(str).tolist()
    b=monthly_account_logs(baseline,months)
    f=monthly_account_logs(filtered,months)
    d=f-b
    smpl=rng.integers(0,len(d),size=(BOOTSTRAP,len(d)))
    ci=np.quantile(d[smpl].mean(axis=1),[.025,.975])
    sign=rng.choice((-1.,1.),size=(BOOTSTRAP,len(d)))
    p=(1+np.sum(np.abs((sign*d).mean(axis=1))>=abs(float(d.mean()))))/(BOOTSTRAP+1)
    return {
        "phase":phase,"comparison":name+" minus original_no_volume_gate",
        "calendar_months":len(months),
        "mean_monthly_log_growth_difference_pp":round(float(d.mean()*100),5),
        "month_bootstrap95_diff_pp":[round(float(x*100),5) for x in ci],
        "exploratory_sign_flip_two_sided_p":round(float(p),4),
        "account_total_return_difference_pp":round(
            float(((np.expm1(f.sum()))-(np.expm1(b.sum())))*100),4),
    }


def checks():
    vol=np.full(30,100.,dtype=float)
    seg=np.zeros(30,dtype=int)
    vol[20]=150.
    r=rolling_rvol(vol,seg)
    assert np.isnan(r[19]) and np.isclose(r[20],1.5)
    # Changing the future entry volume cannot modify the confirmed signal.
    mutated=vol.copy()
    mutated[21:]=1e9
    assert np.isclose(rolling_rvol(mutated,seg)[20],r[20])
    seg[20:]=1
    assert np.isnan(rolling_rvol(vol,seg)[20])
    assert len(FILTERS)==3


def main():
    checks()
    OUT.mkdir(parents=True,exist_ok=True)
    df,gaps,segments=fetch_gap_aware("4h")
    ts=pd.to_datetime(df.open_time_ms,unit="ms",utc=True)
    if ts.iloc[0]>START or ts.iloc[-1]<END-pd.Timedelta("8h"):
        raise RuntimeError("Insufficient Binance BTCUSDT 4h coverage")
    signal=signals(df)["morning_star"]
    atr,rvol=data(df,segments)
    op=df.open.to_numpy(float)
    hi=df.high.to_numpy(float)
    lo=df.low.to_numpy(float)
    rng=np.random.default_rng(SEED)
    rows=[];trades=[];yearly=[];comparisons=[];volume_buckets=[]
    for phase,start,end in (
        ("development_2020_2023",START,TEST_START),
        ("evaluation_2024_2026",TEST_START,END),
    ):
        ids=cohort(df,segments,signal,start,end)
        expected=53 if phase.startswith("development") else 44
        if len(ids)!=expected:
            raise AssertionError(f"Frozen original signal cohort changed {len(ids)} vs {expected}")
        if not np.isfinite(rvol[ids]).all() or not np.isfinite(atr[ids]).all():
            raise AssertionError("Undefined finished-candle rvol or ATR for eligible signals")
        base=[]
        for signal_idx in ids:
            i=int(signal_idx)
            entry=i+1
            buyfill=op[entry]*(1+.0005)
            stop_pct=float(100*ATR_MULT*atr[i]/buyfill)
            net,gross,reason,exit_index,ambiguous=simulate_bracket(
                op,hi,lo,entry,TP_PCT,stop_pct,HOLD_BARS)
            rec={
                "phase":phase,"signal_utc":ts.iloc[i].isoformat(),
                "entry_utc":ts.iloc[entry].isoformat(),
                "exit_utc":ts.iloc[exit_index].isoformat(),
                "year":int(ts.iloc[entry].year),
                "month":ts.iloc[entry].strftime("%Y-%m"),
                "signal_candle_btc_volume":round(float(df.base_volume.iloc[i]),6),
                "previous_20_candle_mean_btc_volume":round(float(df.base_volume.iloc[i-20:i].mean()),6),
                "confirmation_RVOL":round(float(rvol[i]),6),
                "atr14_signal":round(float(atr[i]),6),
                "stop_distance_pct":round(stop_pct,4),
                "trade_net_fraction":float(net),
                "position_net_pct":round(float(net*100),4),
                "account_net_pct":round(float(ALLOCATION*net*100),4),
                "exit_reason":reason,"same_bar_ambiguous":bool(ambiguous),
            }
            for name,thr in FILTERS:
                rec["keep_"+name]=(thr is None or float(rvol[i])>=thr)
            trades.append(rec)
            base.append(rec)
        by_arm={}
        for arm,thr in FILTERS:
            selected=[t for t in base if thr is None or t["confirmation_RVOL"]>=thr]
            by_arm[arm]=selected
            summ=summarise(selected,phase,arm)
            summ.update({"rvol_threshold":thr,"discarded":len(base)-len(selected),
                         "base_signals":len(base)})
            rows.append(summ)
            print("VOLUME_RESULT "+json.dumps(summ,ensure_ascii=False),flush=True)
            for year in sorted(set(t["year"] for t in base)):
                subset=[t for t in selected if t["year"]==year]
                arr=np.asarray([t["trade_net_fraction"] for t in subset],float)
                rec={
                    "phase":phase,"year":year,"arm":arm,
                    "signals":len(arr),
                    "mean_net_pct":round(float(arr.mean()*100),4) if len(arr) else None,
                    "compounded_account_pct":round(float((np.prod(1+ALLOCATION*arr)-1)*100),4)
                }
                yearly.append(rec)
                print("VOLUME_YEAR "+json.dumps(rec),flush=True)
        for arm,_ in FILTERS[1:]:
            cmp=month_paired(by_arm[FILTERS[0][0]],by_arm[arm],start,end,arm,phase,rng)
            comparisons.append(cmp)
            print("VOLUME_PAIRED "+json.dumps(cmp),flush=True)
        for label,lower,upper in BINS:
            subset=[t for t in base if lower<=t["confirmation_RVOL"]<upper]
            diag=summarise(subset,phase,label)
            diag["volume_bucket"]=[lower,upper if np.isfinite(upper) else None]
            volume_buckets.append(diag)
            print("VOLUME_BUCKET "+json.dumps(diag),flush=True)
    pd.DataFrame(rows).to_csv(OUT/"volume_filter_comparison.csv",index=False)
    pd.DataFrame(trades).to_csv(OUT/"frozen_signal_trade_audit.csv",index=False)
    pd.DataFrame(yearly).to_csv(OUT/"yearly_comparisons.csv",index=False)
    pd.DataFrame(comparisons).to_csv(OUT/"calendar_matched_differences.csv",index=False)
    pd.DataFrame(volume_buckets).to_csv(OUT/"volume_bucket_diagnostics.csv",index=False)
    manifest={
        "date_utc":[START.isoformat(),END.isoformat()],"split_utc":TEST_START.isoformat(),
        "source":"Genuine Binance BTCUSDT spot 4h OHLCV including base traded BTC volume, fetched via existing gap-aware loader",
        "original_unmodified_pattern":"signals(frame)['morning_star']",
        "identical_frozen_48h_reserved_entry_cohorts":{"development":53,"evaluation":44},
        "volume_signal":"Completed bullish third-candle BTC base volume / mean of exactly 20 PREVIOUS completed 4h candles (excluding third candle itself), reject gaps. No future candle volume.",
        "filters":{"base":"no volume gate","candidates":["RVOL>=1.0","RVOL>=1.5"]},
        "diagnostics":"RVOL [0,0.75), [0.75,1), [1,1.5), >=1.5, descriptive only; no post-hoc rule selection.",
        "entry":"Next 4h open AFTER full confirmation candle closes",
        "stop":"Wilder 4h ATR14 *1.5 at signal close, initial STATIC exit price fixed until exit (not trailing)",
        "take_profit_pct":5,"max_holding_hours":48,
        "position_fraction_of_equity":.30,
        "fees_bps_side":10,"modeled_slippage_bps_side":5,
        "same_bar_conflict":"STOP first if TP and SL both touched within candle; worse open if stop gap",
        "cohort_comparison":"Select original nonoverlapping cohort FIRST, then filter its 53/44 entries; filtered arms do NOT backfill skipped slots with newly eligible later setups, isolating effect of volume only.",
        "inference":"All calendar months including inactive months; paired log account growth months with 4000 bootstrap/sign-flip draws. Exploratory, not multiple-test-adjusted.",
        "limitations":[
            "2024-2026 has been repeatedly inspected and strategy parameters were chosen using these same years; results NOT untouched out-of-sample.",
            "Adding volume thresholds is additional multiple testing. Do not select an ex-post successful threshold as prospective rule without fresh independent data.",
            "A single bar volume contains no order-book resting liquidity, order cancellations, depth heatmap snapshots, true aggressive trade imbalance, futures open interest or liquidation heatmap levels.",
            "Spot BTC volume can vary by venue and across periods; relative volume is only a local proxy for participation, not directly bullish pressure.",
            "Thirty percent position allocation means risk can exceed .9% of equity for wide ATR stops, even before execution slippage.",
            "OHLC can conceal exact stop/take order; 4h stop-first assumed pessimistically. No order book or genuine stop-market fill simulation.",
            "Few candidate filtered signals; historical ES calculated from tiny bottom sample not a tail-probability estimator.",
            "Trade-close drawdown neglects intratrade marked-to-market variation; no borrowing/yield/taxes or exchange failures.",
            "No default strategy, no paper/live orders, no main-branch merge."
        ]
    }
    (OUT/"methodology.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf8")
    print("VOLUME_ABLATION_COMPLETE",flush=True)


if __name__=="__main__":
    main()

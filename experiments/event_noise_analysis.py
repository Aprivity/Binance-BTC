"""Ex-post noise/outlier attribution on matched BTC 4h morning-star events.

SAME entry selection and trading models as previous studies, for comparability.
No live orders. Pure diagnostics; not a predictive news-based trading system.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.candlestick_study import START, TEST_START, END, fetch_gap_aware, signals
from experiments.fixed_bracket_comparison import simulate_bracket, baseline_net, cohort

MAX_BARS=12
VARIANTS=("fixed_48h","tp5_sl2","tp5_sl3")
OUTPUT=Path("outputs/event-noise-analysis")


def variant_result(df, entry_idx):
    op=df.open.to_numpy(float)
    high=df.high.to_numpy(float)
    low=df.low.to_numpy(float)
    net48=baseline_net(op,entry_idx,12)
    res2=[]
    res3=[]
    flags2=[]
    flags3=[]
    for e in entry_idx:
        a=simulate_bracket(op,high,low,int(e),5,2)
        b=simulate_bracket(op,high,low,int(e),5,3)
        res2.append(a[0])
        res3.append(b[0])
        flags2.append((a[2], a[3], a[4]))
        flags3.append((b[2], b[3], b[4]))
    return (np.asarray(net48), np.asarray(res2), np.asarray(res3),flags2,flags3)


def metrics(v):
    return {"n":len(v),"mean_pct":round(float(np.mean(v)*100),4),
            "median_pct":round(float(np.median(v)*100),4),
            "win_rate_pct":round(float(np.mean(v>0)*100),2),
            "sum_pct":round(float(np.sum(v)*100),3)}


def trimmed_metric(v,p):
    k=int(np.floor(len(v)*p))
    q=np.sort(v)
    return metrics(q[k:len(q)-k]) if len(q)>2*k else None


def summary_sensitivity(phase, variant, values):
    v=np.asarray(values,float)
    sorted_desc=np.argsort(-v)
    sorted_asc=np.argsort(v)
    best_one=float(v[sorted_desc[0]]*100)
    best_three=float(np.sum(v[sorted_desc[:3]])*100)
    positive=float(v[v>0].sum()*100)
    original=metrics(v)
    out={
      "phase":phase,"rule":variant,
      **original,
      "top_1_trade_return_pct":round(best_one,4),
      "top_3_sum_pct":round(best_three,4),
      "sum_positive_trades_pct":round(positive,4),
      "top_3_share_positive_gains_pct":round(100*best_three/positive,2) if positive>0 else None,
      "remove_best_1":metrics(v[sorted_desc[1:]]),
      "remove_best_3":metrics(v[sorted_desc[3:]]),
      "remove_best_5":metrics(v[sorted_desc[5:]]),
      "remove_worst_1":metrics(v[sorted_asc[1:]]),
      "trim_symmetric_10pct":trimmed_metric(v,0.10),
      "trim_symmetric_20pct":trimmed_metric(v,0.20),
      "winsorized_10pct_mean_pct":round(float(np.mean(np.clip(v,np.quantile(v,.1),np.quantile(v,.9)))*100),4),
    }
    print("OUTLIER_SUMMARY "+json.dumps(out,ensure_ascii=False),flush=True)
    return out


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    df,gap_spans,segments=fetch_gap_aware("4h")
    timestamps=pd.to_datetime(df.open_time_ms,unit="ms",utc=True)
    if timestamps.iloc[0] > START or timestamps.iloc[-1] < END - pd.Timedelta("8h"):
        raise ValueError("Incomplete real BTCUSDT 4h data")
    events=signals(df)["morning_star"]
    op=df.open.to_numpy(float)
    hi=df.high.to_numpy(float)
    lo=df.low.to_numpy(float)
    vol=df.base_volume.to_numpy(float)
    rows=[]
    yearly=[]
    sens=[]
    for phase,start,end in [
        ("development_2020_2023",START,TEST_START),
        ("evaluation_2024_2026",TEST_START,END),
    ]:
        signal_indices=cohort(df,segments,events,start,end)
        if len(signal_indices)!=(53 if phase.startswith("development") else 44):
            raise RuntimeError("Entry signal cohort changed; abort")
        entry_idx=signal_indices+1
        a,b,c,flags2,flags3=variant_result(df,entry_idx)
        nets={"fixed_48h":a,"tp5_sl2":b,"tp5_sl3":c}
        for name in VARIANTS:
            sens.append(summary_sensitivity(phase,name,nets[name]))
        by_year=timestamps.iloc[entry_idx].dt.year.to_numpy()
        for yr in sorted(set(by_year)):
            for name in VARIANTS:
                x=nets[name][by_year==yr]
                yearly.append({"phase":phase,"year":int(yr),"rule":name,**metrics(x)})
        for j,i in enumerate(signal_indices):
            e=int(i)+1
            end_idx=e+12
            path_open=op[e:end_idx+1]
            path_high=hi[e:end_idx]
            path_low=lo[e:end_idx]
            # Last 12 full candles starting at entry, 48 hours total.
            peak_offset=int(np.argmax(path_high))
            trough_offset=int(np.argmin(path_low))
            max_high=float(np.max(path_high))
            min_low=float(np.min(path_low))
            previous=op[e-12:e]
            prev_net48=op[e]/op[e-12]-1 if e>=12 else None
            prior_v=vol[max(0,e-30):e]
            median_v=np.median(prior_v) if len(prior_v) else np.nan
            future_range=(max_high-min_low)/op[e]
            event_rows={
                "phase":phase,
                "signal_utc":timestamps.iloc[int(i)].isoformat(),
                "entry_utc":timestamps.iloc[e].isoformat(),
                "exit_48h_utc":timestamps.iloc[end_idx].isoformat(),
                "entry_open_usdt":round(float(op[e]),2),
                "fixed_48h_exit_open_usdt":round(float(op[end_idx]),2),
                "fixed_48h_net_pct":round(float(a[j]*100),4),
                "tp5_sl2_net_pct":round(float(b[j]*100),4),
                "tp5_sl3_net_pct":round(float(c[j]*100),4),
                "tp5_sl2_exit_reason":flags2[j][0],
                "tp5_sl2_exit_utc":timestamps.iloc[flags2[j][1]].isoformat(),
                "tp5_sl3_exit_reason":flags3[j][0],
                "tp5_sl3_exit_utc":timestamps.iloc[flags3[j][1]].isoformat(),
                "peak_gain_from_open_pct":round(float((max_high/op[e]-1)*100),3),
                "max_adverse_from_open_pct":round(float((min_low/op[e]-1)*100),3),
                "max_48h_high_utc":timestamps.iloc[e+peak_offset].isoformat(),
                "min_48h_low_utc":timestamps.iloc[e+trough_offset].isoformat(),
                "intraperiod_price_range_pct":round(float(future_range*100),3),
                "preceding_48h_open_return_pct":round(float(prev_net48*100),3),
                "signal_base_volume_vs_prior_30bar_median":round(float(vol[int(i)]/median_v),3) if np.isfinite(median_v) and median_v>0 else None,
            }
            event_rows["swing_vs_close_gain_pp"]=round(float((max_high/op[e]-op[end_idx]/op[e])*100),3)
            rows.append(event_rows)
        frame_for_phase=pd.DataFrame([r for r in rows if r["phase"]==phase])
        for name in ("fixed_48h","tp5_sl2","tp5_sl3"):
            col=name+"_net_pct"
            leaders=frame_for_phase.sort_values(col,ascending=False).head(8)
            losers=frame_for_phase.sort_values(col).head(6)
            for tag,sub in (("TOP_WIN",leaders),("TOP_LOSS",losers)):
                for _,r in sub.iterrows():
                    print("EVENT "+json.dumps({
                        "tag":tag,"phase":phase,"rule":name,
                        "entry":r["entry_utc"],"exit":r["exit_48h_utc"],
                        "net_pct":r[col],
                        "other_fixed_48h":r["fixed_48h_net_pct"],
                        "other_tp5_sl2":r["tp5_sl2_net_pct"],
                        "other_tp5_sl3":r["tp5_sl3_net_pct"],
                        "peak_gain":r["peak_gain_from_open_pct"],
                        "max_adverse":r["max_adverse_from_open_pct"],
                        "max_high_utc":r["max_48h_high_utc"],
                        "preceding_48h":r["preceding_48h_open_return_pct"],
                        "volume_factor":r["signal_base_volume_vs_prior_30bar_median"],
                    },ensure_ascii=False),flush=True)

    dataframe=pd.DataFrame(rows)
    dataframe.to_csv(OUTPUT/"all_matched_events_with_spikes.csv",index=False)
    pd.DataFrame(sens).to_csv(OUTPUT/"outlier_impact.csv",index=False)
    pd.DataFrame(yearly).to_csv(OUTPUT/"by_calendar_year.csv",index=False)
    # Additional robustness: retain only no-exception candles, remove a priori extreme +/-10% 48h range
    # and run conditional regime analysis. These are sensitivity checks, not a strategy.
    regimes=[]
    for phase in dataframe.phase.unique():
        dfp=dataframe[dataframe.phase==phase].copy()
        for condition,mask in [
            ("all",np.ones(len(dfp),dtype=bool)),
            ("exclude_48h_high_low_range_gt_8pct",(dfp.intraperiod_price_range_pct<=8).to_numpy()),
            ("exclude_48h_high_low_range_gt_10pct",(dfp.intraperiod_price_range_pct<=10).to_numpy()),
            ("exclude_48h_high_low_range_gt_15pct",(dfp.intraperiod_price_range_pct<=15).to_numpy()),
            ("signal_volume_no_spike_below_2x",(dfp.signal_base_volume_vs_prior_30bar_median<2).fillna(False).to_numpy()),
            ("prior_48h_down_only",(dfp.preceding_48h_open_return_pct<0).to_numpy()),
            ("prior_48h_up_or_flat_only",(dfp.preceding_48h_open_return_pct>=0).to_numpy()),
        ]:
            for v in VARIANTS:
                vals=dfp.loc[mask,v+"_net_pct"].to_numpy(dtype=float)/100
                if len(vals):
                    record={"phase":phase,"filter":condition,"rule":v,**metrics(vals)}
                    regimes.append(record)
                    print("SENSITIVITY "+json.dumps(record),flush=True)
    pd.DataFrame(regimes).to_csv(OUTPUT/"volatility_and_regime_filters.csv",index=False)

    top=dataframe.loc[dataframe.phase=="evaluation_2024_2026"].sort_values("fixed_48h_net_pct",ascending=False).head(10)
    report={
        "market":"Binance BTCUSDT spot, 4h candle",
        "period_utc":[START.isoformat(),END.isoformat()],
        "same_entry_cohorts":{"2020-2023":53,"2024-2026":44},
        "source":"Binance public 4h OHLCV, one historic bar missing, gap-straddling signals rejected",
        "exit_variants":list(VARIANTS),
        "transaction_costs":"10bps fee and 5bps slippage each side",
        "top_10_evaluation_fixed_48h_entry_dates":top.entry_utc.tolist(),
        "news_matching":"No news assumed; dates only for independent public source verification",
        "methodology":"Compare net-per-trade returns under identical entries; top-k winner deletions, two-sided trimming, winsorization, per-year returns, ex-post OHLC volatility filters, signal volume flag.",
        "important_limits":[
            "Removing top gainers is an ex-post robustness analysis and is not a tradable exit rule.",
            "Filtering on future 48h realized range is expressly noncausal/look-ahead and diagnostic only.",
            "Signal volume ratio uses current already closed signal candle, is observable at entry.",
            "News association cannot prove a market move was caused by that event; contemporaneous macro, ETF and market factors may overlap.",
            "Some intrabar spikes are not executable as market exits; bracket simulations use conservative stop-first treatment.",
            "The 2024-2026 period has been repeatedly examined and cannot be called an untouched out-of-sample test.",
            "No real or paper orders."
        ],
    }
    (OUTPUT/"methodology.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print("NOISE_ANALYSIS_COMPLETE",flush=True)


if __name__=="__main__":
    main()

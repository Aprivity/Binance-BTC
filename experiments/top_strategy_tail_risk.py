"""Tail risk and robustness on preselected top BTC morning-star exit candidates.

Research-only: no strategy enabled, no paper/live orders, no use of hindsight
filters for constructing trading signals. All exits share the same BTC 4h entry
cohorts. Historical risk quantiles are DESCRIPTIVE, not probability guarantees.
"""
from __future__ import annotations

import json
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.candlestick_study import START, TEST_START, END, fetch_gap_aware, signals
from experiments.fixed_bracket_comparison import baseline_net, simulate_bracket, cohort

OUT=Path("outputs/top-strategy-risk")
CANDIDATES={
    "TP5_SL3_48h":"tp5_sl3",
    "TP5_SL2_48h":"tp5_sl2",
    "FIXED_48h":"fixed_48h",
    "TP5_SL1_48h":"tp5_sl1",
    "FIXED_24h":"fixed_24h",
}
PHASES=(("development_2020_2023",START,TEST_START),
        ("evaluation_2024_2026",TEST_START,END))
EXTRA_EXIT_IMPACT=(0.005,0.01,0.03)


def metrics(r):
    r=np.asarray(r,dtype=float)
    n=len(r)
    ascending=np.sort(r)
    q05=np.quantile(r,0.05)
    q10=np.quantile(r,0.10)
    es05=np.mean(ascending[:max(1,int(np.ceil(n*0.05)))])
    es10=np.mean(ascending[:max(1,int(np.ceil(n*0.10)))])
    winners=r[r>0]
    losers=r[r<0]
    pf=(np.sum(winners)/abs(np.sum(losers))) if len(losers) else None
    longest=0
    now=0
    for x in r:
        if x<0:
            now+=1
            longest=max(longest,now)
        else:
            now=0
    m={
        "n":n,
        "mean_net_pct":round(float(np.mean(r)*100),4),
        "median_net_pct":round(float(np.median(r)*100),4),
        "win_pct":round(float(np.mean(r>0)*100),2),
        "sd_pct":round(float(np.std(r,ddof=1)*100),4),
        "worst_pct":round(float(np.min(r)*100),4),
        "best_pct":round(float(np.max(r)*100),4),
        "p05_net_pct":round(float(q05*100),4),
        "p10_net_pct":round(float(q10*100),4),
        "historical_es05_pct":round(float(es05*100),4),
        "historical_es10_pct":round(float(es10*100),4),
        "es05_tail_n":max(1,int(np.ceil(n*0.05))),
        "es10_tail_n":max(1,int(np.ceil(n*0.10))),
        "profit_factor":round(float(pf),3) if pf is not None else None,
        "loss_streak_max":longest,
        "top3_sum_pct":round(float(np.sum(np.sort(r)[-3:])*100),3),
        "all_sum_pct":round(float(np.sum(r)*100),3),
    }
    for f in (1.0,0.2):
        equity=np.concatenate([[1.0],np.cumprod(1+f*r)])
        running_peak=np.maximum.accumulate(equity)
        drawdown=equity/running_peak-1.0
        key="all_in" if f==1 else "20pct_position"
        m[key+"_model_end_total_return_pct"]=round(float((equity[-1]-1)*100),3)
        m[key+"_trade_close_max_drawdown_pct"]=round(float(np.min(drawdown)*100),3)
    return m


def market_tail(candles,segments):
    ts=pd.to_datetime(candles.open_time_ms,unit="ms",utc=True)
    c=candles.close.to_numpy(dtype=float)
    r=c[1:]/c[:-1]-1
    intact=segments[1:]==segments[:-1]
    valid=(ts.iloc[1:]>=TEST_START).to_numpy()
    for phase,mask in (("full_2020_2026",intact),
                       ("2024_2026",intact & valid),
                       ("2020_2023",intact & ~valid)):
        x=r[mask]
        mu=float(np.mean(x));sigma=float(np.std(x,ddof=1))
        z=(x-mu)/sigma
        upper3=np.mean(z>=3)
        lower3=np.mean(z<=-3)
        above5=np.mean(np.abs(z)>=5)
        normal_two_tail_3=erfc(3/sqrt(2))
        normal_two_tail_5=erfc(5/sqrt(2))
        res={
            "phase":phase,"n_contiguous_4h_returns":len(x),
            "mean_4h_pct":round(mu*100,5),
            "std_4h_pct":round(sigma*100,4),
            "historical_1pct_return":round(float(np.quantile(x,.01)*100),4),
            "historical_99pct_return":round(float(np.quantile(x,.99)*100),4),
            "fraction_abs_z_ge3":round(float(np.mean(np.abs(z)>=3)),6),
            "normal_expected_fraction_abs_z_ge3":round(normal_two_tail_3,6),
            "fraction_abs_z_ge5":round(float(above5),6),
            "normal_expected_fraction_abs_z_ge5":round(normal_two_tail_5,9),
            "negative_3sigma_count":int(np.sum(z<=-3)),
            "positive_3sigma_count":int(np.sum(z>=3)),
            "warning":"A global sigma across years ignores time-varying volatility; normal comparison is descriptive.",
        }
        print("MARKET_TAIL "+json.dumps(res),flush=True)
        yield res


def stress_trade_returns(original,stops):
    # Additional negative execution price impact beyond original 5bps sell slip.
    # All-exit stress differs from stop-only shock model. No invented shock probabilities.
    results=[]
    for extra in EXTRA_EXIT_IMPACT:
        all_bad=(1+original)*(1-extra)-1
        stop_only=np.where(stops,(1+original)*(1-extra)-1,original)
        results.append({
            "extra_execution_impact_pct":extra*100,
            "all_exits_mean_net_pct":round(float(np.mean(all_bad)*100),4),
            "all_exits_worst_net_pct":round(float(np.min(all_bad)*100),4),
            "stop_exits_only_mean_net_pct":round(float(np.mean(stop_only)*100),4),
            "stop_exits_only_worst_net_pct":round(float(np.min(stop_only)*100),4),
        })
    return results


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    df,gaps,segments=fetch_gap_aware("4h")
    ts=pd.to_datetime(df.open_time_ms,unit="ms",utc=True)
    if ts.iloc[0]>START or ts.iloc[-1]<END-pd.Timedelta("8h"):
        raise RuntimeError("Missing entire 4h history")
    market=list(market_tail(df,segments))
    o=df.open.to_numpy(dtype=float)
    high=df.high.to_numpy(dtype=float)
    low=df.low.to_numpy(dtype=float)
    event_mask=signals(df)["morning_star"]
    records=[]
    trade_rows=[]
    stress_rows=[]
    yearly=[]
    for phase,lower,upper in PHASES:
        event_indices=cohort(df,segments,event_mask,lower,upper)
        n=len(event_indices)
        if n != (53 if phase.startswith("development") else 44):
            raise RuntimeError(f"Matched cohort changed: {phase} {n}")
        entry_idx=event_indices+1
        ret={}
        reasons={}
        for label,cfg in CANDIDATES.items():
            if cfg=="fixed_48h":
                net=baseline_net(o,entry_idx,12)
                rs=np.asarray(["time_48h"]*n)
            elif cfg=="fixed_24h":
                net=baseline_net(o,entry_idx,6)
                rs=np.asarray(["time_24h"]*n)
            else:
                sl=float(cfg.split("sl")[1])
                tuples=[simulate_bracket(o,high,low,int(i),5.0,sl) for i in entry_idx]
                net=np.asarray([t[0] for t in tuples])
                rs=np.asarray([t[2] for t in tuples])
            ret[label]=net
            reasons[label]=rs
            m=metrics(net)
            m.update({"phase":phase,"candidate":label})
            records.append(m)
            print("RISK_RESULT "+json.dumps(m),flush=True)
            for year in sorted(set(ts.iloc[entry_idx].dt.year.to_numpy())):
                selected=(ts.iloc[entry_idx].dt.year.to_numpy()==year)
                yrm=metrics(net[selected])
                row={"phase":phase,"year":int(year),"candidate":label,
                     "n":yrm["n"],"mean_net_pct":yrm["mean_net_pct"],
                     "win_pct":yrm["win_pct"],"worst_pct":yrm["worst_pct"]}
                yearly.append(row)
                print("YEAR_RESULT "+json.dumps(row),flush=True)
            sl_mask=np.isin(rs,["stop_loss","gap_stop"])
            for row in stress_trade_returns(net,sl_mask):
                row.update({"phase":phase,"candidate":label,"stop_events":int(sl_mask.sum())})
                stress_rows.append(row)
                print("EXECUTION_STRESS "+json.dumps(row),flush=True)
        for j,i in enumerate(entry_idx):
            trade_rows.append({
                "phase":phase,"entry_utc":ts.iloc[i].isoformat(),
                **{name+"_return_pct":round(float(ret[name][j])*100,4) for name in CANDIDATES},
                **{name+"_exit_reason":reasons[name][j] for name in CANDIDATES},
            })
    pd.DataFrame(records).to_csv(OUT/"risk_metrics_by_phase.csv",index=False)
    pd.DataFrame(stress_rows).to_csv(OUT/"execution_slippage_stress.csv",index=False)
    pd.DataFrame(yearly).to_csv(OUT/"calendar_year_stability.csv",index=False)
    pd.DataFrame(trade_rows).to_csv(OUT/"candidate_trades.csv",index=False)
    pd.DataFrame(market).to_csv(OUT/"btc_return_tail_vs_normal.csv",index=False)
    manifest={
        "source":"Real public Binance BTCUSDT spot candles, 4h",
        "date_start_utc":str(START),"date_end_exclusive_utc":str(END),
        "method":"Pre-existing identical 53/44 4h morning-star entry signals. Each candidate exactly matches prior exit simulator.",
        "candidates":list(CANDIDATES),
        "returns":"Simulated net fractional per trade, 10bps commission + 5bps slippage each side",
        "historical_es05":"Average of bottom ceil(5% * n) trades, not fitted 99% tail prediction",
        "mdd":"Per-position 100% or 20% account capital reallocated at every entry, realized trade-close equity only. No intratrade peak-to-trough or mark-to-market.",
        "impact_stress":"Extra loss 0.5, 1, 3% in exit price, in two scenarios: all exits, and stop exits only. No probabilities assigned. Applied over already modeled standard execution costs.",
        "gap_handling":gaps,
        "known_constraints":[
            "Small 44-event evaluation cohort: historical CVaR/ES is highly uncertain.",
            "Market return normal tails compare unconditional global historical sigma; BTC volatility nonstationary.",
            "All candidates were chosen after viewing the evaluation years, so this is NOT fresh out-of-sample validation.",
            "Full-allocation compounded result is hypothetical, assumes all funds deployable and no additional portfolio frictions.",
            "Trade-close drawdown excludes intra-position losses; stops are NOT guaranteed fills.",
            "No news causality inference or future-leaking filtering here.",
            "No live orders, no strategy as default, no main branch merge."
        ]
    }
    (OUT/"methodology.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf8")
    print("TOP_RISK_STUDY_COMPLETE",flush=True)


if __name__=="__main__":
    main()

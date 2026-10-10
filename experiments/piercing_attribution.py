"""Point-in-time Piercing Line attribution, frozen 4h Morning Star baseline.

Descriptive diagnostics on known-inspected 2020-2026 spot BTC historical data.
No search/tuning of exit rules or entry filter, no real/paper orders.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.candlestick_study import START, TEST_START, END, fetch_gap_aware
from experiments.atr_risk_sized_comparison import wilder_atr
from experiments.bullish_pattern_ensembles import (
    detect, pick_events, trades_for, ARMS, ALLOCATION, TP_PCT, ATR_MULT,
    WINDOW_BARS, FEE_BPS, SLIP_BPS,
)

OUT=Path("outputs/piercing-attribution")
BPS=10000.
PHASES=(("development_2020_2023",START,TEST_START),
        ("inspected_2024_2026",TEST_START,END))
SPOT=4*3600*1000


def account_growth(trades):
    rets=np.asarray([t["net_position_return"] for t in trades],float)
    if len(rets)==0:return 0.
    return 100*float(np.prod(1.+ALLOCATION*rets)-1.)


def log_growth(trades):
    return float(sum(np.log1p(ALLOCATION*t["net_position_return"]) for t in trades))


def trade_summary(trades):
    r=np.asarray([t["net_position_return"] for t in trades])
    if len(r)==0:
        return dict(trades=0,win_rate_pct=None,mean_position_net_pct=None,
                    account_compounded_pct=0,account_log_growth_pct=0)
    eq=np.r_[1.,np.cumprod(1+ALLOCATION*r)]
    worst=eq/np.maximum.accumulate(eq)-1
    return {
        "trades":len(r),
        "win_rate_pct":round(100*float(np.mean(r>0)),2),
        "mean_position_net_pct":round(100*float(np.mean(r)),4),
        "median_position_net_pct":round(100*float(np.median(r)),4),
        "account_compounded_pct":round(float((eq[-1]-1)*100),4),
        "account_log_growth_pct":round(100*log_growth(trades),4),
        "account_trade_close_drawdown_pct":round(100*float(worst.min()),4),
        "take_profit":sum(t["exit_reason"]=="take_profit" for t in trades),
        "stop_loss":sum("stop" in t["exit_reason"] for t in trades),
        "time_exit":sum(t["exit_reason"]=="time_48h" for t in trades),
        "worst_position_net_pct":round(float(100*r.min()),4),
        "best_position_net_pct":round(float(100*r.max()),4)
    }


def split_by_source(baseline,combined):
    base={t["signal_open_ms"]:t for t in baseline}
    union={t["signal_open_ms"]:t for t in combined}
    retained=[t for t in combined if t["signal_open_ms"] in base]
    added=[t for t in combined if t["signal_open_ms"] not in base]
    displaced=[t for t in baseline if t["signal_open_ms"] not in union]
    for t in retained:
        old=base[t["signal_open_ms"]]["net_position_return"]
        if not np.isclose(t["net_position_return"],old,rtol=0,atol=1.e-12):
            raise RuntimeError("Retained trade outcome differs despite identical settings")
    difference=log_growth(combined)-log_growth(baseline)
    parts={"new_piercing_log_growth_pct":100*log_growth(added),
        "minus_displaced_baseline_log_growth_pct":-100*log_growth(displaced),
        "combined_minus_baseline_log_growth_pct":100*difference,
        "addition_count":len(added),
        "retained_stars":len(retained),
        "displaced_stars":len(displaced)}
    if not np.isclose(parts["new_piercing_log_growth_pct"]+
            parts["minus_displaced_baseline_log_growth_pct"],
            parts["combined_minus_baseline_log_growth_pct"],atol=1e-9):
        raise RuntimeError("Accounting decomposition does not balance")
    return retained,added,displaced,parts


def cost_stress_return(frame,row,atr,s_fee,s_slip):
    """Recompute full path-based TP/stop with user-selected fee/slippage.

    Stop is ATR relative to NEW simulated buy fill, TP+5% from NEW buy fill.
    All else including 4h stop-first, 48h and gap convention identical.
    """
    stamps=frame.open_time_ms.to_numpy(np.int64)
    i=int(np.searchsorted(stamps,row["signal_open_ms"]))
    if i>=len(stamps) or stamps[i]!=row["signal_open_ms"]:
        raise AssertionError("Unknown signal timestamp")
    op=frame.open.to_numpy(float)
    hi=frame.high.to_numpy(float)
    lo=frame.low.to_numpy(float)
    entry=i+1
    sf=s_slip/BPS;fee=s_fee/BPS
    fill=op[entry]*(1+sf)
    stop=fill-ATR_MULT*atr[i]
    take=fill*(1+TP_PCT/100.)
    reason="time_48h";exit_raw=float(op[entry+WINDOW_BARS])
    for j in range(entry,entry+WINDOW_BARS):
        opening=float(op[j]);low=float(lo[j]);high=float(hi[j])
        if opening<=stop:
            exit_raw,reason=opening,"gap_stop";break
        if opening>=take:
            exit_raw,reason=take,"take_profit";break
        if low<=stop:
            exit_raw,reason=stop,"stop_loss";break
        if high>=take:
            exit_raw,reason=take,"take_profit";break
    return exit_raw*(1-sf)*(1-fee)/(fill*(1+fee))-1


def regime_info(frame,segments,t):
    """Context BEFORE the two-candle pattern starts; nothing forward."""
    stamps=frame.open_time_ms.to_numpy(np.int64)
    idx=int(np.searchsorted(stamps,t["signal_open_ms"]))
    close=frame.close.to_numpy(float)
    ret={}
    for days in (7,30):
        backward=days*6
        a=idx-2-backward
        b=idx-2
        if a<0 or segments[a]!=segments[b]:
            ret[f"prepattern_{days}d_return_pct"]=None
            ret[f"prepattern_{days}d_direction"]="history_gap_or_start"
        else:
            val=100*(close[b]/close[a]-1)
            ret[f"prepattern_{days}d_return_pct"]=round(float(val),4)
            ret[f"prepattern_{days}d_direction"]="positive" if val>0 else "nonpositive"
    return ret


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frame,gap_count,seg=fetch_gap_aware("4h")
    masks=detect(frame)
    atr=wilder_atr(frame.high.to_numpy(float),
                   frame.low.to_numpy(float),
                   frame.close.to_numpy(float),seg)
    candles=frame.open_time_ms.to_numpy(np.int64)
    orig=("baseline_morning_star",("morning_star",))
    solo=("solo_piercing_line",("piercing_line",))
    both=("combo_star_piercing",("morning_star","piercing_line"))
    yearly=[];decomps=[];by_sources=[];regime=[];stress=[];audit=[]
    for phase,start,end in PHASES:
        base,_=trades_for(frame,seg,atr,masks,orig,phase,start,end)
        solo_trades,_=trades_for(frame,seg,atr,masks,solo,phase,start,end)
        combined,_=trades_for(frame,seg,atr,masks,both,phase,start,end)
        retained,added,displaced,parts=split_by_source(base,combined)
        print("PIERCING_DECOMP "+json.dumps({"phase":phase,**parts},ensure_ascii=False),flush=True)
        decomps.append({"phase":phase,**parts})
        groups={
            "baseline_original":base,
            "piercing_standalone":solo_trades,
            "combined":combined,
            "combined_retained_original":retained,
            "combined_added_piercing":added,
            "baseline_displaced":displaced,
        }
        for name,rows in groups.items():
            result={"phase":phase,"source":name,**trade_summary(rows)}
            by_sources.append(result)
            print("PIERCING_SOURCE "+json.dumps(result),flush=True)
            for yr in sorted(set([t["year"] for t in rows])):
                sub=[t for t in rows if t["year"]==yr]
                yearly.append({"phase":phase,"source":name,"year":yr,
                               **trade_summary(sub)})
        # Realized trade event contribution; not personalized trading advice.
        for name,rows in groups.items():
            for t in rows:
                audit.append({"phase":phase,"source":name,**t})
        # Point-in-time 7/30d baseline market regime, not a filter.
        for src in ("combined_added_piercing","combined_retained_original"):
            rows=groups[src]
            for t in rows:
                context=regime_info(frame,seg,t)
                regime.append({"phase":phase,"source":src,"signal_open_ms":t["signal_open_ms"],
                               "net_position_return":t["net_position_return"],
                               **context})
        for fee,slip in ((10,5),(10,10),(20,10)):
            for src in ("baseline_original","piercing_standalone","combined"):
                rows=groups[src]
                stress_net=np.asarray([
                    cost_stress_return(frame,t,atr,fee,slip) for t in rows])
                if fee==10 and slip==5 and len(rows):
                    if not np.allclose(stress_net,
                        [t["net_position_return"] for t in rows],atol=1e-10,rtol=0):
                        raise RuntimeError("Stress replay fails frozen original model parity")
                growth=100*(np.prod(1+ALLOCATION*stress_net)-1)
                r={"phase":phase,"source":src,
                   "fee_bps_per_side":fee,"slip_bps_per_side":slip,
                   "account_compounded_pct":round(float(growth),4),
                   "trades":len(rows)}
                stress.append(r)
                print("PIERCING_COST_STRESS "+json.dumps(r),flush=True)
    reg=pd.DataFrame(regime)
    regimes=[]
    for (phase,src),g in reg.groupby(["phase","source"]):
        for days in (7,30):
            key=f"prepattern_{days}d_direction"
            for direction,gg in g.groupby(key):
                a=gg.net_position_return.to_numpy(float)
                rec={"phase":phase,"source":src,
                     "lookback_days_before_pattern":days,"direction":direction,
                     "n":len(a),
                     "win_rate_pct":round(float(np.mean(a>0)*100),2),
                     "avg_position_net_pct":round(float(np.mean(a)*100),4),
                     "account_log_growth_pct":round(100*float(np.log1p(ALLOCATION*a).sum()),4)}
                regimes.append(rec)
                print("PIERCING_REGIME "+json.dumps(rec),flush=True)
    year_df=pd.DataFrame(yearly)
    for y in sorted(year_df.year.unique()):
        v=year_df[(year_df.year==y)&(year_df.source.isin(["baseline_original","combined"]))]
        print("PIERCING_YEAR "+json.dumps(v.to_dict("records"),flush=True))
    pd.DataFrame(decomps).to_csv(OUT/"account_log_return_decomposition.csv",index=False)
    pd.DataFrame(by_sources).to_csv(OUT/"pattern_source_performance.csv",index=False)
    pd.DataFrame(yearly).to_csv(OUT/"yearly_source_performance.csv",index=False)
    pd.DataFrame(regime).to_csv(OUT/"prepattern_regime_trade_details.csv",index=False)
    pd.DataFrame(regimes).to_csv(OUT/"prepattern_regime_summary.csv",index=False)
    pd.DataFrame(stress).to_csv(OUT/"fee_slippage_stress.csv",index=False)
    pd.DataFrame(audit).to_csv(OUT/"signal_source_trade_audit.csv",index=False)
    methodology={
        "research_only":True,"orders_sent":False,
        "source":"Binance public BTCUSDT spot 4h OHLCV, 2020-01-01 to 2026-10-10 exclusive",
        "original_strict_morning_star_unchanged":True,
        "piercing_candle_rule":"EXACT experiments/bullish_pattern_ensembles.additional_patterns definition; downtrend, large bearish first, bullish close above midpoint but below first open; no session gap",
        "frozen_strategy":{"ATR_period":14,"stop_multiple":1.5,"TP_pct":5,"48h_full_signal_reservation":True,"account_fraction":.30,"fee_bps_per_side":10,"slippage_bps_per_side":5},
        "historical_phases":["2020-2023","inspected 2024-2026; NOT untouched evaluation"],
        "decomposition":"For each exact signal timestamp: original retained unchanged, new piercing trades added, original displaced by earlier piercing under chronological 48h nonoverlap. Difference in log(1+equity_return) sums exactly equals added logs minus displaced logs; not same as arithmetic portfolio percent diff.",
        "regime":"Prior 7/30-day close-to-close prepattern return ends two full 4h bars before signal closes; phase and contiguous lookback checks. Purely exploratory with no fit thresholds. No regime-conditioned deployment.",
        "stress":"Replay ALL original 4h bar high/low stop-first exits with model fee/slippage 10/5 vs 10/10 vs 20/10 bps per side, ATR stop and 5% TP recalculated relative to simulated buy fill; not futures funding or liquidity."
    }
    (OUT/"methodology.json").write_text(json.dumps(methodology,ensure_ascii=False,indent=2),encoding="utf8")
    print("PIERCING_ATTRIBUTION_COMPLETE "+json.dumps({"historical_bars":len(frame),
        "feed_gaps":gap_count,"source_rows":len(by_sources),"year_rows":len(yearly)},flush=True))


if __name__=="__main__":
    main()

"""Isolate ATR initial-stop placement from position sizing, BTCUSDT 4h.

NO LIVE OR PAPER ORDERS; no strategy activation. Every historical candidate
enters at the EXACT SAME original 4h Morning Star signals and uses a FIXED
30% allocation per trade. Only stop placement changes. 5% TP / 48h cap.
ATR uses fully completed signal candle, so no future data enters the stop.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.candlestick_study import START, TEST_START, END, fetch_gap_aware, signals
from experiments.fixed_bracket_comparison import cohort, simulate_bracket
from experiments.atr_risk_sized_comparison import wilder_atr
from experiments.hold_time_comparison import bootstrap_means, sign_flip_p

OUT = Path("outputs/atr-stop-fixed-30pct")
ALLOCATION = 0.30
TAKE_PROFIT_PCT = 5.0
MAX_BARS = 12
SEED = 20261010
VARIANTS = {
    "fixed_SL2": ("fixed", 2.0),
    "fixed_SL3_reference": ("fixed", 3.0),
    "ATR14_x1_5": ("atr", 1.5),
    "ATR14_x2_primary": ("atr", 2.0),
    "ATR14_x2_5": ("atr", 2.5),
}


def metrics(net: np.ndarray, stop_dist: np.ndarray, exit_reason: np.ndarray,
            months: np.ndarray, rng: np.random.Generator):
    """Report both per-position and fixed 30%-deployed account units."""
    account = ALLOCATION*net
    equity = np.r_[1.,np.cumprod(1.+account)]
    drawdown = equity/np.maximum.accumulate(equity)-1
    loss_seq = longest = 0
    for r in net:
        if r < 0:
            loss_seq += 1
            longest = max(longest,loss_seq)
        else:
            loss_seq = 0
    tail_n=max(1,int(np.ceil(0.05*len(net))))
    cilo,cihi = bootstrap_means(account,months,rng)
    return {
        "trades":int(len(net)),
        "position_mean_net_pct":round(float(np.mean(net)*100),4),
        "position_median_net_pct":round(float(np.median(net)*100),4),
        "position_win_rate_pct":round(float(np.mean(net>0)*100),2),
        "position_worst_net_pct":round(float(np.min(net)*100),4),
        "position_historical_ES05_pct":round(float(np.sort(net)[:tail_n].mean()*100),4),
        "fixed_allocation_pct":30.,
        "account_mean_per_signal_pct":round(float(np.mean(account)*100),4),
        "account_worst_signal_pct":round(float(np.min(account)*100),4),
        "account_historical_ES05_pct":round(float(np.sort(account)[:tail_n].mean()*100),4),
        "account_model_compounded_total_pct":round(float((equity[-1]-1)*100),4),
        "account_trade_close_MDD_pct":round(float(np.min(drawdown)*100),4),
        "account_mean_ci95_month_block_pct":[round(float(cilo*100),4),round(float(cihi*100),4)],
        "mean_initial_stop_distance_pct":round(float(np.mean(stop_dist)*100),4),
        "min_initial_stop_distance_pct":round(float(np.min(stop_dist)*100),4),
        "max_initial_stop_distance_pct":round(float(np.max(stop_dist)*100),4),
        "take_count":int(np.sum(exit_reason=="take_profit")),
        "stop_count":int(np.isin(exit_reason,["stop_loss","gap_stop"]).sum()),
        "gap_stop_count":int(np.sum(exit_reason=="gap_stop")),
        "time_count":int(np.sum(exit_reason=="time_48h")),
        "max_loss_streak":int(longest),
    }


def deterministic_tests():
    op=np.array([100.,100.,100.,100.,100.])
    high=np.array([100.2,102.,103.,100.3,100.2])
    low=np.array([98.,99.8,99.7,99.4,99.5])
    stop_a = simulate_bracket(op,high,low,0,TAKE_PROFIT_PCT,2.0,3)
    stop_b = simulate_bracket(op,high,low,0,TAKE_PROFIT_PCT,3.0,3)
    assert stop_a[2]=="stop_loss" and stop_b[2]=="time_48h"
    assert np.isclose(ALLOCATION,0.3)
    # ATR at candle 20 cannot know any candle with index >20.
    h=np.full(35,101.); l=np.full(35,99.); c=np.full(35,100.)
    segments=np.zeros(35,dtype=int)
    base=wilder_atr(h,l,c,segments)
    hi2=h.copy(); hi2[21:]=500.
    modified=wilder_atr(hi2,l,c,segments)
    assert np.isclose(base[20],modified[20])


def main():
    deterministic_tests()
    OUT.mkdir(parents=True,exist_ok=True)
    frame,gaps,segments=fetch_gap_aware("4h")
    times=pd.to_datetime(frame.open_time_ms,unit="ms",utc=True)
    if times.iloc[0]>START or times.iloc[-1]<END-pd.Timedelta("8h"):
        raise RuntimeError("Incomplete Binance BTCUSDT spot candle history")
    o=frame.open.to_numpy(float)
    h=frame.high.to_numpy(float)
    l=frame.low.to_numpy(float)
    c=frame.close.to_numpy(float)
    atr=wilder_atr(h,l,c,segments)
    signal=signals(frame)["morning_star"]
    all_rows=[];all_trades=[];yearly=[];paired=[];stress=[]
    rng=np.random.default_rng(SEED)
    for phase,start,end in (
        ("development_2020_2023",START,TEST_START),
        ("evaluation_2024_2026",TEST_START,END)
    ):
        signal_indices=cohort(frame,segments,signal,start,end)
        expected=53 if start==START else 44
        assert len(signal_indices)==expected, f"Changed signal cohort: {len(signal_indices)} != {expected}"
        if not np.isfinite(atr[signal_indices]).all():
            raise ValueError("Signal ATR undefined")
        entry_indices=signal_indices+1
        months=times.iloc[entry_indices].dt.strftime("%Y-%m").to_numpy()
        years=times.iloc[entry_indices].dt.year.to_numpy()
        stored={}
        for name,(kind,value) in VARIANTS.items():
            stop_dist=(np.full(len(entry_indices),value/100,dtype=float)
                       if kind=="fixed"
                       else value*atr[signal_indices]/(o[entry_indices]*(1.+.0005)))
            if not (np.isfinite(stop_dist).all() and (stop_dist>0).all()):
                raise ValueError("Invalid stop distance")
            events=[simulate_bracket(o,h,l,int(e),TAKE_PROFIT_PCT,float(s*100),MAX_BARS)
                    for e,s in zip(entry_indices,stop_dist)]
            net=np.asarray([t[0] for t in events],dtype=float)
            exits=np.asarray([t[3] for t in events],dtype=int)
            reasons=np.asarray([t[2] for t in events],dtype=object)
            amb=np.asarray([t[4] for t in events],dtype=bool)
            row={"phase":phase,"strategy":name,**metrics(net,stop_dist,reasons,months,rng),
                 "same_bar_stop_first_ambiguous":int(amb.sum())}
            all_rows.append(row)
            print("FIXED_ALLOCATION_RESULT "+json.dumps(row),flush=True)
            stored[name]=(net,stop_dist,reasons,exits,amb)
            for year in sorted(set(years)):
                chosen=years==year
                eq=np.r_[1.,np.cumprod(1.+ALLOCATION*net[chosen])]
                mr=eq/np.maximum.accumulate(eq)-1
                yr={
                    "phase":phase,"year":int(year),"strategy":name,
                    "n_trades":int(np.sum(chosen)),
                    "position_mean_net_pct":round(float(np.mean(net[chosen])*100),4),
                    "account_year_compounded_pct":round(float((eq[-1]-1)*100),4),
                    "account_year_trade_close_MDD_pct":round(float(np.min(mr)*100),4),
                }
                yearly.append(yr)
                print("FIXED_ALLOCATION_YEAR "+json.dumps(yr),flush=True)
            for shock in (.005,.01,.03):
                hit=np.isin(reasons,["stop_loss","gap_stop"])
                shocked=net.copy()
                shocked[hit]=(1+shocked[hit])*(1-shock)-1
                account_shock=ALLOCATION*shocked
                sr={
                    "phase":phase,"strategy":name,
                    "additional_stop_exit_slippage_pct":shock*100,
                    "affected_stops":int(np.sum(hit)),
                    "mean_account_per_signal_pct":round(float(np.mean(account_shock)*100),4),
                    "account_compounded_total_pct":round(float((np.prod(1.+account_shock)-1)*100),4),
                    "worst_account_signal_pct":round(float(np.min(account_shock)*100),4)
                }
                stress.append(sr)
                print("FIXED_ALLOCATION_STRESS "+json.dumps(sr),flush=True)
        baseline=stored["fixed_SL3_reference"][0]
        for name in VARIANTS:
            if name=="fixed_SL3_reference":
                continue
            dif=stored[name][0]-baseline
            ci=bootstrap_means(dif,months,rng)
            pair={
                "phase":phase,"comparison":name+" - fixed_SL3_reference",
                "same_entries":int(len(dif)),
                "mean_difference_position_pp":round(float(np.mean(dif)*100),4),
                "mean_difference_account_pp":round(float(ALLOCATION*np.mean(dif)*100),4),
                "month_block_ci95_account_diff_pp":[round(float(x*100*ALLOCATION),4) for x in ci],
                "exploratory_month_sign_flip_p":round(float(sign_flip_p(dif,months,rng)),4),
            }
            paired.append(pair)
            print("FIXED_ALLOCATION_PAIRED "+json.dumps(pair),flush=True)
        for j,s in enumerate(signal_indices):
            rec={
                "phase":phase,
                "signal_utc":times.iloc[s].isoformat(),
                "entry_utc":times.iloc[entry_indices[j]].isoformat(),
                "entry_open_usdt":round(float(o[entry_indices[j]]),4),
                "signal_atr14_usdt":round(float(atr[s]),4),
                "account_deployed_pct":30.
            }
            for name in VARIANTS:
                net,dist,reasons,exits,amb=stored[name]
                rec.update({
                    name+"_stop_pct":round(float(dist[j]*100),4),
                    name+"_position_net_pct":round(float(net[j]*100),4),
                    name+"_account_net_pct":round(float(ALLOCATION*net[j]*100),4),
                    name+"_exit_reason":str(reasons[j]),
                    name+"_exit_utc":times.iloc[exits[j]].isoformat(),
                    name+"_same_bar_ambiguous":bool(amb[j]),
                })
            all_trades.append(rec)
    pd.DataFrame(all_rows).to_csv(OUT/"fixed30_stop_summary.csv",index=False)
    pd.DataFrame(all_trades).to_csv(OUT/"matched_signal_trades.csv",index=False)
    pd.DataFrame(yearly).to_csv(OUT/"annual_comparison.csv",index=False)
    pd.DataFrame(paired).to_csv(OUT/"paired_differences.csv",index=False)
    pd.DataFrame(stress).to_csv(OUT/"stop_slippage_stress.csv",index=False)
    meta={
        "purpose":"One variable change: stop placement ONLY; NO volatility-dependent position sizing.",
        "market":"Binance BTCUSDT SPOT 4h, public REST OHLCV",
        "range_utc":[START.isoformat(),END.isoformat()],
        "development_cohort":53,"exploratory_evaluation_cohort":44,
        "entry_rule":"Original 4h morning-star signal at candle close; enter next 4h candle open",
        "position_fraction_every_trade":0.30,
        "no_leverage":True,
        "exit_rules":{
            "take_profit":"5% from simulated buy execution fill",
            "static_stop_pct":[2.0,3.0],
            "atr_stop":"ATR14 (Wilder) at completed signal candle, not at future entry candle; entry fill minus multiplier * pre-entry ATR14",
            "atr_multipliers":[1.5,2.0,2.5],
            "horizon":"48h at open after 12 full 4h candles",
            "same_bar":"If TP and SL both reached within OHLC, STOP FIRST, regardless of true unknown order"
        },
        "broker_model":{
            "entry_and_exit_fee_bps":10,"entry_and_exit_slippage_bps":5,
            "gap_stop":"If open breaches SL, sell at unfavorable next-bar opening price",
            "no_guaranteed_stop_execution":True
        },
        "caveats":[
            "ATR stop is an entry-time dynamic volatility-based STOP LEVEL, NOT a trailing stop; remains fixed while open.",
            "All candidates use identical preselected 53/44 entry dates and constant 30% account fraction, remainder idle cash.",
            "Fixed 30% allocation means nominal portfolio loss at stop is NOT capped to 0.9% anymore, and can be much larger for wide ATR stops.",
            "Reinvest next signal after preceding 48h reserved cohort interval; account drawdown is trade-close only, excludes intratrade price excursion.",
            "Historical 2024+ data repeatedly used for parameter selection and is NOT untouched OOS.",
            "Small sample; 5% ES means worst 3 trades among 44 in evaluation, not a precise true tail probability.",
            "Fees and slips are modeled; no real orderbook, outage, liquidation, tax, idle cash interest or forward fill guarantees.",
            "No live/paper orders, no main merge, no default strategy activation."
        ],
        "data_feed_gap_count":gaps,
    }
    (OUT/"methodology.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf8")
    print("FIXED_30PCT_STOP_ONLY_COMPARISON_COMPLETE",flush=True)


if __name__=="__main__":
    main()

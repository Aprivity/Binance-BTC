"""Explicit opt-in BTC 4h three-candle Morning Star forward PAPER simulator.

Research-only: no authenticated Binance APIs, no order submission or live
positions. Unlike generic 'btc_quant paper', this module freezes the studied
ATR(14) x1.5 INITIAL static stop, fixed +5% take, max48h time exit and
30% capital allocation. Full reservation of a 48h window, even if early exit.

IMPORTANT: This forward simulator polls public ticker prices, which may miss
intrabar stop/take breaches while offline; its fills are NOT guaranteed and
must not be treated as a real protective order or exact historic backtest.
A late signal is skipped rather than pretended to fill at next 4h open.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from btc_quant.data import fetch, validate
from btc_quant.paper import exclusive, atomic_save
from btc_quant.config import STEPS
from experiments.candlestick_study import signals
from experiments.atr_risk_sized_comparison import wilder_atr

LOG=logging.getLogger(__name__)
STEP=STEPS["4h"]
HORIZON_MS=48*3600*1000
ENTRY_WINDOW_MS=120_000
GAP_WARNING_MS=10*60_000
PRICE_POLL_SECONDS=60
STRATEGY_ID="research:btc-4h-original-morningstar-atr14x1.5-tp5-48h-fixed30-v1"
FEE=.001
SLIP=.0005
START_CASH=1000.0
ACCOUNT_FRACTION=.30
MIN_STOP=1e-6
HARD_EQUITY_DRAWDOWN=.12
SCHEMA=1


def fresh_state(initial_usdt=START_CASH):
    if not math.isfinite(initial_usdt) or initial_usdt<=0:
        raise ValueError("Positive initial virtual USDT required")
    return {
        "schema_version":SCHEMA,"strategy_id":STRATEGY_ID,
        "starting_usdt":float(initial_usdt),
        "cash_usdt":float(initial_usdt),"position":None,
        "last_closed_open_ms":None,"last_tick_ms":None,
        "reservation_ends_ms":None,"peak_equity_usdt":float(initial_usdt),
        "halted":False,"unobserved_price_gap":False,
        "events":[],"equity_curve":[],
    }


def read_state(path,initial_usdt):
    p=Path(path)
    if not p.exists():return fresh_state(initial_usdt)
    d=json.loads(p.read_text(encoding="utf8"))
    if (d.get("schema_version")!=SCHEMA or d.get("strategy_id")!=STRATEGY_ID
            or float(d.get("starting_usdt",-1))!=initial_usdt):
        raise ValueError("Invalid or mismatched forward paper state; choose a fresh path")
    return d


def utc(ms):
    return datetime.fromtimestamp(ms/1000,timezone.utc).isoformat()


def equity(state,observed_price):
    pos=state["position"]
    return state["cash_usdt"]+(pos["quantity_btc"]*observed_price if pos else 0.)


def closed_signal_at_last_bar(frame):
    """Re-use EXACT original 4h Morning Star morphology and true Wilder ATR.

    200+ closed bars needed for warmup. ATR's seed on local 240/500 bar
    window may differ minutely from 2020 full-history Wilder seed.
    """
    if len(frame)<220:raise ValueError("Need at least 220 closed 4h candles")
    detected=signals(frame)["morning_star"]
    n=len(frame)
    if n<15 or not detected[-1]:return False,None
    seg=np.zeros(n,dtype=np.int64)
    atr=wilder_atr(frame.high.to_numpy(float),frame.low.to_numpy(float),
                   frame.close.to_numpy(float),seg)
    val=float(atr[-1])
    if not np.isfinite(val) or val<=0:
        raise ValueError("Undefined preentry ATR14")
    return True,val


def simulate_tick(state,frame,observed_price,now_ms):
    """Pure deterministic update, with no network, file, or exchange effects.

    Frame contains only closed, contiguous 4h bars. Signal at last closed
    candle may enter ONLY immediately after the NEXT candle begins.
    """
    if state.get("schema_version")!=SCHEMA or state.get("strategy_id")!=STRATEGY_ID:
        raise ValueError("Wrong strategy schema/version")
    if not math.isfinite(float(observed_price)) or observed_price<=0:
        raise ValueError("Invalid observed BTC price")
    if not isinstance(now_ms,int):raise ValueError("now_ms integer required")
    if state["last_tick_ms"] is not None and now_ms<=state["last_tick_ms"]:
        raise ValueError("Non-increasing snapshot timestamp")
    frame=validate(frame,"4h",require_closed=True,now_ms=now_ms)
    if len(frame)<220:raise ValueError("Less than 220 fully closed 4h candles")
    candle_open_ms=int(frame.open_time_ms.iloc[-1])
    close_boundary=candle_open_ms+STEP
    if not 0<=now_ms-close_boundary<=STEP+60_000:
        raise ValueError("Public 4h feed stale; no account state committed")
    old=state["last_closed_open_ms"]
    if old is not None and candle_open_ms<old:
        raise ValueError("Public candle time regressed")
    s=copy.deepcopy(state)
    s["last_tick_ms"]=int(now_ms)

    last_tick=state["last_tick_ms"]
    if (last_tick is not None and now_ms-last_tick>GAP_WARNING_MS
            and state["position"] is not None):
        s["unobserved_price_gap"]=True
        s["halted"]=True
        s["events"].append({"utc":utc(now_ms),"kind":"OBSERVATION_GAP_HALT",
                "gap_seconds":(now_ms-last_tick)/1000,
                "warning":"Unobserved price path; actual stop/take ordering unknowable."})

    pos=s["position"]
    exited=False
    if pos is not None:
        outcome=None
        if observed_price<=pos["stop_usdt"]:
            outcome="observed_stop"
        elif observed_price>=pos["take_usdt"]:
            outcome="observed_take"
        elif now_ms>=pos["deadline_ms"]:
            outcome="time_48h"
        if outcome:
            raw_fill=(min(observed_price,pos["stop_usdt"]) if outcome=="observed_stop"
                      else pos["take_usdt"] if outcome=="observed_take"
                      else observed_price)
            sell_fill=raw_fill*(1-SLIP)
            proceeds=pos["quantity_btc"]*sell_fill*(1-FEE)
            pnl=proceeds-pos["purchase_debit_usdt"]
            s["cash_usdt"]+=proceeds
            s["position"]=None
            s["events"].append({
                "utc":utc(now_ms),"kind":"PAPER_SELL","reason":outcome,
                "price_fill_usdt":sell_fill,"quantity_btc":pos["quantity_btc"],
                "pnl_usdt":pnl,"signal_open_ms":pos["signal_open_ms"],
                "warning":"Paper sampled-price execution; no real trade or stop guarantee.",
            })
            exited=True

    current=equity(s,observed_price)
    s["peak_equity_usdt"]=max(s["peak_equity_usdt"],current)
    if current<=s["peak_equity_usdt"]*(1-HARD_EQUITY_DRAWDOWN):
        s["halted"]=True

    # Bootstrap and missed candle(s) do NOT backfill earlier signals.
    if old is None:
        s["last_closed_open_ms"]=candle_open_ms
    elif candle_open_ms>old:
        s["last_closed_open_ms"]=candle_open_ms
        one_new_bar=(candle_open_ms==old+STEP)
        fresh_boundary=(now_ms-close_boundary<=ENTRY_WINDOW_MS)
        after_reservation=(s["reservation_ends_ms"] is None
                           or close_boundary>s["reservation_ends_ms"])
        if (one_new_bar and fresh_boundary and not s["halted"] and
                not exited and s["position"] is None and after_reservation):
            detected,atr=closed_signal_at_last_bar(frame)
            if detected:
                raw_price=float(observed_price)
                buy_fill=raw_price*(1+SLIP)
                stop=buy_fill-ATR_MULTIPLIER*atr
                take=buy_fill*(1+TP_PCT/100)
                if stop<=MIN_STOP:
                    s["events"].append({"utc":utc(now_ms),
                        "kind":"SKIPPED_INVALID_STOP"})
                else:
                    budget=current*ACCOUNT_FRACTION
                    budget=min(budget,s["cash_usdt"])
                    qty=budget/(buy_fill*(1+FEE))
                    if not (np.isfinite(qty) and qty>0):
                        raise ValueError("Invalid paper quantity")
                    s["cash_usdt"]-=budget
                    deadline=close_boundary+HORIZON_MS
                    s["position"]={
                        "signal_open_ms":candle_open_ms,
                        "entry_utc":utc(now_ms),"entry_boundary_utc":utc(close_boundary),
                        "entry_fill_usdt":buy_fill,"purchase_debit_usdt":budget,
                        "quantity_btc":qty,"atr_signal_usdt":atr,
                        "stop_usdt":stop,"take_usdt":take,
                        "deadline_ms":deadline
                    }
                    s["reservation_ends_ms"]=deadline
                    s["events"].append({
                        "utc":utc(now_ms),"kind":"PAPER_BUY",
                        "signal_open_ms":candle_open_ms,
                        "quantity_btc":qty,"price_fill_usdt":buy_fill,
                        "stop_usdt":stop,"take_usdt":take,
                        "atr14_usdt":atr,"debit_usdt":budget,
                        "delay_from_next_open_seconds":(now_ms-close_boundary)/1000,
                        "warning":"Paper entry at sampled price, not guaranteed historical next open.",
                    })
        elif one_new_bar and not fresh_boundary:
            s["events"].append({"utc":utc(now_ms),"kind":"LATE_BAR_OBSERVED_NO_ENTRY",
                "signal_open_ms":candle_open_ms})
        elif not one_new_bar:
            s["events"].append({"utc":utc(now_ms),"kind":"CANDLE_GAP_NO_ENTRY",
                "skipped_4h_bars":max(0,(candle_open_ms-old)//STEP-1)})
    eq=equity(s,observed_price)
    s["peak_equity_usdt"]=max(s["peak_equity_usdt"],eq)
    if eq<=s["peak_equity_usdt"]*(1-HARD_EQUITY_DRAWDOWN):
        s["halted"]=True
    s["equity_curve"].append({"utc":utc(now_ms),"equity_usdt":eq,
        "cash_usdt":s["cash_usdt"],"BTC_price_usdt":float(observed_price),
        "has_position":s["position"] is not None})
    return s


def tick_public_market():
    """Public read-only market snapshot; this is NOT a stop-market service."""
    now_ms=int(time.time()*1000)
    frame=fetch("BTC/USDT","4h",days=30,now_ms=now_ms)
    import requests
    r=requests.get("https://data-api.binance.vision/api/v3/ticker/price",
        params={"symbol":"BTCUSDT"},timeout=20)
    r.raise_for_status()
    price=float(r.json()["price"])
    return frame,price,int(time.time()*1000)


def summary(state):
    last=state["equity_curve"][-1] if state["equity_curve"] else None
    return {
        "strategy_id":state["strategy_id"],"mode":"LOCAL PAPER ONLY - NO ORDERS",
        "simulated_cash_usdt":round(state["cash_usdt"],4),
        "last_observed_equity_usdt":round(last["equity_usdt"],4) if last else None,
        "active_paper_position":state["position"],
        "paper_buy_events":sum(e["kind"]=="PAPER_BUY" for e in state["events"]),
        "paper_sell_events":sum(e["kind"]=="PAPER_SELL" for e in state["events"]),
        "halted":state["halted"],
        "unobserved_price_gap":state["unobserved_price_gap"],
        "last_observation_utc":utc(state["last_tick_ms"]) if state["last_tick_ms"] else None,
        "warning":"Sampled ticker prices can miss stop/take between observations; no real orders or guaranteed fills."
    }


def run(initial_usdt,path,once=False,poll_seconds=PRICE_POLL_SECONDS):
    if poll_seconds<30:raise ValueError("Poll frequency cannot be under 30 seconds")
    with exclusive(path):
        state=read_state(path,initial_usdt)
        while True:
            try:
                frame,price,ts=tick_public_market()
                new=simulate_tick(state,frame,price,ts)
                atomic_save(path,new)
                state=new
                LOG.info("PAPER ONLY %s",json.dumps(summary(state),ensure_ascii=False))
                if once:
                    print(json.dumps(summary(state),ensure_ascii=False,indent=2))
                    return
            except (KeyboardInterrupt,SystemExit):
                raise
            except Exception:
                LOG.exception("Market observation failed; no state commit or orders")
                if once:raise
            time.sleep(poll_seconds)


def main(argv=None):
    p=argparse.ArgumentParser(description="BTC Morning Star frozen forward PAPER; no orders")
    p.add_argument("--paper",action="store_true",
        help="Explicitly opt in to local VIRTUAL money simulation")
    p.add_argument("--once",action="store_true")
    p.add_argument("--initial-usdt",type=float,default=START_CASH)
    p.add_argument("--poll-seconds",type=int,default=PRICE_POLL_SECONDS)
    p.add_argument("--state",default="outputs/morningstar-forward-v1/state.json")
    p.add_argument("--status",action="store_true")
    args=p.parse_args(argv)
    if args.status:
        s=read_state(args.state,args.initial_usdt)
        print(json.dumps(summary(s),indent=2,ensure_ascii=False))
        return
    if not args.paper:
        p.error("No simulation will start without --paper; this is not a live-order tool")
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    run(args.initial_usdt,args.state,args.once,args.poll_seconds)


ATR_MULTIPLIER=1.5
TP_PCT=5.0

if __name__=="__main__":
    main()

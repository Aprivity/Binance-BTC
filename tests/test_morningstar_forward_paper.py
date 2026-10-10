"""Offline tests for the opt-in, read-only Morning Star paper runner."""
import math
import pytest
import pandas as pd

from btc_quant.__main__ import synthetic
from experiments.candlestick_study import signals
from experiments.morningstar_forward_paper import (
    STEP,HORIZON_MS,STRATEGY_ID,ATR_MULTIPLIER,TP_PCT,
    fresh_state,simulate_tick,summary,closed_signal_at_last_bar
)


def fixtures(n=330,signal_index=301):
    x=synthetic("4h",n)
    # Make the fixture globally flat with realistic and internally consistent
    # candle high/lows; then inject a confirmed original three-candle pattern.
    x["open"]=100.
    x["close"]=100.1
    x["high"]=101.
    x["low"]=99.
    x["base_volume"]=10.
    t=signal_index
    x.loc[t-6,["open","close","high","low"]]=[104.,104.,105.,103.]
    x.loc[t-2,["open","close","high","low"]]=[103.,99.,104.,98.]
    x.loc[t-1,["open","close","high","low"]]=[98.8,99.,99.6,98.7]
    x.loc[t,["open","close","high","low"]]=[99.,102.,103.,98.5]
    assert signals(x)["morning_star"][t]
    return x


def at_after_bar(frame,index,delay_ms=30_000):
    return int(frame.open_time_ms.iloc[index])+STEP+delay_ms


def test_first_start_is_observer_without_retroactive_buy():
    x=fixtures()
    original=fresh_state()
    t=301
    now=at_after_bar(x,t)
    st=simulate_tick(original,x.iloc[:t+1],101.,now)
    assert st["position"] is None
    assert st["cash_usdt"]==1000.
    assert all(e["kind"]!="PAPER_BUY" for e in st["events"])


def test_exact_morning_star_only_and_30pct_atr15_fixed_tp5():
    x=fixtures()
    t=301
    initial=simulate_tick(fresh_state(),x.iloc[:t],
        100.,at_after_bar(x,t-1))
    now=at_after_bar(x,t)
    st=simulate_tick(initial,x.iloc[:t+1],101.,now)
    p=st["position"]
    assert p and p["atr_signal_usdt"]>0
    assert len([e for e in st["events"] if e["kind"]=="PAPER_BUY"])==1
    assert p["purchase_debit_usdt"]==pytest.approx(300.)
    assert st["cash_usdt"]==pytest.approx(700.)
    assert p["stop_usdt"]==pytest.approx(p["entry_fill_usdt"]-1.5*p["atr_signal_usdt"])
    assert p["take_usdt"]==pytest.approx(p["entry_fill_usdt"]*1.05)
    assert p["deadline_ms"]==int(x.open_time_ms.iloc[t])+STEP+HORIZON_MS
    assert st["reservation_ends_ms"]==p["deadline_ms"]
    assert summary(st)["mode"]=="LOCAL PAPER ONLY - NO ORDERS"
    assert STRATEGY_ID in summary(st)["strategy_id"]


def test_repeat_same_candle_does_not_double_buy_and_stop_sells_once():
    x=fixtures()
    t=301
    one=simulate_tick(fresh_state(),x.iloc[:t],100.,at_after_bar(x,t-1))
    bought=simulate_tick(one,x.iloc[:t+1],101.,at_after_bar(x,t))
    same=simulate_tick(bought,x.iloc[:t+1],101.,at_after_bar(x,t)+60_000)
    assert len([e for e in same["events"] if e["kind"]=="PAPER_BUY"])==1
    stop=same["position"]["stop_usdt"]
    closed=simulate_tick(same,x.iloc[:t+1],stop-1.,at_after_bar(x,t)+120_000)
    assert closed["position"] is None
    assert closed["cash_usdt"]<1000.
    again=simulate_tick(closed,x.iloc[:t+1],stop-1.,at_after_bar(x,t)+180_000)
    assert len([e for e in again["events"] if e["kind"]=="PAPER_SELL"])==1
    assert again["reservation_ends_ms"]==bought["reservation_ends_ms"]


def test_48_hour_time_exit():
    x=fixtures()
    t=301
    one=simulate_tick(fresh_state(),x.iloc[:t],100.,at_after_bar(x,t-1))
    bought=simulate_tick(one,x.iloc[:t+1],101.,at_after_bar(x,t))
    # At the exact 48h deadline, there are now 12 more completed 4h bars.
    # The actual deadline is 48h after the entry BAR OPENS; the last closed
    # candle is the bar that opened at the deadline-4h.
    when=bought["position"]["deadline_ms"]+30_000
    bars=x[x.open_time_ms+STEP<=when]
    closed=simulate_tick(bought,bars,101.,when)
    assert closed["position"] is None
    assert sum(e["kind"]=="PAPER_SELL" for e in closed["events"])==1
    assert any(e.get("reason")=="time_48h" for e in closed["events"])


def test_offline_gap_halts_new_entries_and_preserves_alert():
    x=fixtures()
    t=301
    init=simulate_tick(fresh_state(),x.iloc[:t],100.,at_after_bar(x,t-1))
    bought=simulate_tick(init,x.iloc[:t+1],101.,at_after_bar(x,t))
    gap=simulate_tick(bought,x.iloc[:t+1],101.,
                      at_after_bar(x,t)+11*60_000)
    assert gap["halted"] and gap["unobserved_price_gap"]
    assert any(e["kind"]=="OBSERVATION_GAP_HALT" for e in gap["events"])


def test_late_new_candle_never_enters():
    x=fixtures()
    t=301
    before=simulate_tick(fresh_state(),x.iloc[:t],100.,at_after_bar(x,t-1))
    new=simulate_tick(before,x.iloc[:t+1],101.,at_after_bar(x,t,180_000))
    assert new["position"] is None
    assert any(e["kind"]=="LATE_BAR_OBSERVED_NO_ENTRY" for e in new["events"])


def test_lookahead_and_repeated_timestamp_rejected():
    x=fixtures()
    t=301
    before=simulate_tick(fresh_state(),x.iloc[:t],100.,at_after_bar(x,t-1))
    with pytest.raises(ValueError,match="Non-increasing"):
        simulate_tick(before,x.iloc[:t],100.,at_after_bar(x,t-1))
    with pytest.raises(ValueError,match="Unclosed"):
        simulate_tick(before,x.iloc[:t+2],100.,at_after_bar(x,t))

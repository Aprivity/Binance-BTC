"""Synthetic correctness tests only; they are not evidence of trading profitability."""
import math

import numpy as np
import pandas as pd
import pytest

from btc_quant.config import Config, STEPS
from btc_quant.core import Account, backtest
from btc_quant.paper import paper_tick, state_for
from btc_quant.strategy import prepare_signals
from research_strategies.video_six_ma import (
    ResearchSettings, compression_retest, first_ma20_retest, six_lines,
)


def scenario(n=300):
    step = STEPS["4h"]
    first = 1_700_000_000_000//step*step
    close = np.full(n, 100.0)
    opens = close.copy()
    highs = close + .2
    lows = close - .2
    # Flat 260 closed bars establish six-line density. A large green breakout
    # at 260 is followed by the FIRST successful retest at 262.
    if n > 260:
        opens[260], close[260], highs[260], lows[260] = 100, 105, 105.2, 99.9
    if n > 261:
        opens[261], close[261], highs[261], lows[261] = 105, 106, 106.3, 104.6
    if n > 262:
        opens[262], close[262], highs[262], lows[262] = 103, 105, 105.4, 100.0
    if n > 263:
        opens[263], close[263], highs[263], lows[263] = 104, 105, 105.5, 103.8
    for j in range(264,n):
        opens[j], close[j] = 105, 105
        highs[j], lows[j] = 105.3, 104.7
    t = first+np.arange(n)*step
    return pd.DataFrame({"open_time_ms":t,"open":opens,"high":highs,
                         "low":lows,"close":close,"base_volume":100.,
                         "close_time_ms":t+step-1})


def test_exact_video_indicator_lengths_and_no_lookahead():
    df = scenario()
    lines = six_lines(df)
    assert list(lines) == ["ma20","ema20","ma60","ema60","ma120","ema120"]
    assert pd.isna(lines.ma120.iloc[118])
    assert lines.ma120.iloc[119] == 100
    changed = df.copy()
    changed.loc[265:,"close"] = 200
    pd.testing.assert_frame_equal(lines.iloc[:265],six_lines(changed).iloc[:265])


@pytest.mark.parametrize("fn",[compression_retest,first_ma20_retest])
def test_entry_requires_prior_compression_and_completed_retest(fn):
    df = scenario()
    sig = fn(df,Config())
    hits = list(np.flatnonzero(sig.buy_signal.to_numpy()))
    assert hits == [262], (fn.__name__,hits)
    assert not sig.sell_signal.any()
    assert math.isfinite(sig.initial_stop_price.iloc[262])
    assert sig.initial_stop_price.iloc[262]<df.close.iloc[262]
    assert sig.reward_risk.iloc[262]==3
    # The signal at 262 is impossible before the bar has closed.
    earlier=fn(df.iloc[:262].copy(),Config())
    assert not earlier.buy_signal.any()
    later=df.copy()
    later.loc[263:,"close"] = 130
    pd.testing.assert_frame_equal(sig.iloc[:263],fn(later,Config()).iloc[:263])


@pytest.mark.parametrize("fn",[compression_retest,first_ma20_retest])
def test_next_open_3r_and_position_risk_uses_structure_stop(fn):
    cfg=Config()
    df=scenario()
    signals=prepare_signals(df,cfg,fn)
    assert signals.buy_signal.iloc[262]
    book=backtest({"BTC/USDT":df},cfg,signal_fn=fn)
    buys=[e for e in book.events if e["side"]=="BUY"]
    assert len(buys)==1
    assert buys[0]["time"]==pd.to_datetime(int(df.iloc[263].open_time_ms),unit="ms",utc=True).isoformat()
    pos=book.positions["BTC/USDT"]
    assert pos.entry_price>pos.stop
    assert (pos.take-pos.entry_price)/(pos.entry_price-pos.stop)==pytest.approx(3.)
    assert pos.stop==pytest.approx(signals.initial_stop_price.iloc[262])
    assert book.open_risk(cfg)<=cfg.starting_usdt*cfg.max_open_risk_pct/100 + 1e-6
    assert pos.quantity*pos.entry_price<=cfg.starting_usdt*cfg.max_symbol_pct/100+1e-5


def test_next_open_gap_below_declared_stop_rejects_trade():
    cfg=Config()
    df=scenario()
    df.loc[263,["open","high","low","close"]]=[90.,91.,89.,90.]
    book=backtest({"BTC/USDT":df},cfg,signal_fn=compression_retest)
    assert not book.positions and not book.events


def test_unchanged_legacy_atr_buy_and_new_price_bracket():
    cfg=Config()
    b=Account.create(cfg)
    b.prices["BTC/USDT"]=100.
    assert b.buy("BTC/USDT","t",100,2,cfg)
    assert b.positions["BTC/USDT"].stop==pytest.approx(100*(1+cfg.slip)-2*cfg.stop_atr)
    empty=Account.create(cfg)
    empty.prices["BTC/USDT"]=100.
    with pytest.raises(ValueError):
        empty.buy("BTC/USDT","t",100,2,cfg,stop_price=95)


def test_bracket_contract_fail_closed():
    cfg=Config()
    df=scenario()
    def bad(x,conf):
        s=compression_retest(x,conf)
        return s.drop(columns="reward_risk")
    with pytest.raises(ValueError,match="both"):
        prepare_signals(df,cfg,bad)
    def nonsensical(x,conf):
        s=compression_retest(x,conf)
        s.loc[262,"initial_stop_price"]=np.inf
        return s
    with pytest.raises(ValueError,match="finite"):
        prepare_signals(df,cfg,nonsensical)


def test_rolling_window_does_not_carry_buy_from_previous_fold():
    cfg=Config()
    df=scenario()
    begin=int(df.iloc[263].open_time_ms)
    end=int(df.iloc[280].open_time_ms)
    book=backtest({"BTC/USDT":df},cfg,signal_fn=compression_retest,
                  start_ms=begin,end_ms=end)
    assert not book.events and not book.positions


def test_paper_uses_same_explicit_bracket_but_is_opt_in_only():
    cfg=Config()
    df=scenario()
    fn=compression_retest
    state=state_for(cfg,"research_strategies.video_six_ma:compression_retest")
    observer=state_for(cfg)
    obs=paper_tick(observer,cfg,{"BTC/USDT":df.iloc[:263]},104,"observe")
    assert obs["account"]["events"]==[]
    # Bootstrap with pre-signal history; process the newly completed signal.
    first=paper_tick(state,cfg,{"BTC/USDT":df.iloc[:262]},105,"a",
                     signal_fn=fn,strategy_ref=state["strategy_ref"])
    second=paper_tick(first,cfg,{"BTC/USDT":df.iloc[:263]},104,"b",
                      signal_fn=fn,strategy_ref=state["strategy_ref"])
    assert len(second["account"]["events"])==1
    pos=second["account"]["positions"]["BTC/USDT"]
    assert (pos["take"]-pos["entry_price"])/(pos["entry_price"]-pos["stop"])==pytest.approx(3)


def test_no_trade_on_constant_prices():
    cfg=Config()
    df=scenario(250)
    for fn in (compression_retest,first_ma20_retest):
        assert not fn(df,cfg).buy_signal.any()


def test_study_rules_are_explicit_and_validated():
    assert ResearchSettings().reward_risk==3
    with pytest.raises(ValueError):
        ResearchSettings(compression_spread_pct=-1)
    with pytest.raises(ValueError):
        ResearchSettings(min_compression_bars=15,zone_lookback_bars=3)


def test_two_video_entry_hypotheses_are_distinct():
    cfg=Config()
    df=scenario()
    # A pullback can reach MA20 without retesting the frozen compression zone.
    df.loc[262,"low"]=100.8
    assert not compression_retest(df,cfg).buy_signal.iloc[262]
    assert first_ma20_retest(df,cfg).buy_signal.iloc[262]


def test_file_loader_supports_dataclass_based_strategy():
    from btc_quant.strategy import resolve_strategy
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/"research_strategies"/"video_six_ma.py"
    fn=resolve_strategy(f"{path}:compression_retest")
    assert fn(scenario(),Config()).buy_signal.iloc[262]


def test_target_and_stop_same_bar_assumes_protective_stop_first():
    cfg=Config()
    df=scenario()
    # Both take and stop are touched on the entry candle, so stop must win.
    df.loc[263,["open","high","low","close"]]=[104,200,90,104]
    result=backtest({"BTC/USDT":df},cfg,signal_fn=compression_retest)
    assert len(result.trades)>=1
    assert result.trades[0]["reason"]=="stop_loss"
    assert result.trades[0]["pnl_usdt"]<0
import json
import math
import pandas as pd
import pytest
from btc_quant.__main__ import synthetic
from btc_quant.config import Config, STEPS
from btc_quant.core import Account, backtest, signals
from btc_quant.data import normalize, validate
from btc_quant.paper import state_for, paper_tick, atomic_save, load_state
from btc_quant.report import metrics


def history(cfg, n=550):
    return {s: validate(synthetic(s, cfg.timeframe, n), cfg.timeframe) for s in cfg.symbols}


def test_config_guards():
    with pytest.raises(ValueError):
        Config(symbols=("BTC/USDT", "BTC/USDT"))
    with pytest.raises(ValueError):
        Config(max_symbol_pct=80)


def test_aligned_two_asset_and_consistent_balance():
    cfg = Config()
    book = backtest(history(cfg), cfg)
    m = metrics(book, cfg)
    assert book.cash >= -1e-8
    assert math.isfinite(m["net_return_pct"])
    assert set(m["pnl_by_symbol"]) == set(cfg.symbols)
    assert all(sum(1 for x in book.positions if x == s) <= 1 for s in cfg.symbols)


def test_no_future_candle_signals():
    cfg = Config(symbols=("BTC/USDT",))
    df = history(cfg)[cfg.symbols[0]]
    sig = signals(df, cfg)
    allowed = {pd.to_datetime(int(df.iloc[i+1].open_time_ms), unit="ms", utc=True).isoformat()
               for i in range(len(df)-1) if bool(sig.iloc[i].buy_signal)}
    book = backtest({cfg.symbols[0]: df}, cfg)
    assert {e["time"] for e in book.events if e["side"] == "BUY"}.issubset(allowed)


def test_shared_cash_and_exposure_limits():
    cfg = Config()
    b = Account.create(cfg)
    b.prices.update({"BTC/USDT": 100, "ETH/USDT": 100})
    assert b.buy("BTC/USDT", "t", 100, 2, cfg)
    assert b.buy("ETH/USDT", "t", 100, 2, cfg)
    assert b.cash >= 0
    assert sum(p.quantity * b.prices[s] for s,p in b.positions.items()) <= cfg.starting_usdt * .5 + 1
    assert b.open_risk(cfg) <= cfg.starting_usdt * .02 + 1e-6


def test_fees_slippage_are_included():
    cfg = Config()
    b = Account.create(cfg)
    b.prices["BTC/USDT"] = 100
    assert b.buy("BTC/USDT", "t", 100, 2, cfg)
    assert b.sell("BTC/USDT", "u", 100, cfg, "close")
    assert b.cash < 1000 and b.fees > 0 and b.slippage > 0
    assert b.trades[0]["pnl_usdt"] < 0


def test_gap_stop_can_exceed_risk_budget():
    cfg = Config(symbols=("BTC/USDT",))
    b = Account.create(cfg)
    b.prices["BTC/USDT"] = 100
    b.buy("BTC/USDT", "t", 100, 2, cfg)
    assert b.protective_exit("BTC/USDT", 60, 120, 60) == (60, "gap_stop")
    b.sell("BTC/USDT", "u", 60, cfg, "gap_stop")
    assert -b.trades[0]["pnl_usdt"] > 10


def test_intrabar_stop_first():
    cfg = Config(symbols=("BTC/USDT",))
    b = Account.create(cfg)
    b.prices["BTC/USDT"] = 100
    b.buy("BTC/USDT", "t", 100, 2, cfg)
    assert b.protective_exit("BTC/USDT", 100, 200, 20)[1] == "stop_loss"


def test_drawdown_halt_does_not_block_exit():
    cfg = Config(symbols=("BTC/USDT",))
    b = Account.create(cfg)
    b.prices["BTC/USDT"] = 100
    b.buy("BTC/USDT", "t", 100, 2, cfg)
    b.prices["BTC/USDT"] = 30
    b.mark("u", cfg)
    assert b.halted
    assert not b.buy("ETH/USDT", "v", 100, 2, cfg)
    assert b.sell("BTC/USDT", "v", 30, cfg, "exit")


def test_reject_gaps():
    cfg = Config(symbols=("BTC/USDT",))
    df = synthetic("BTC/USDT", cfg.timeframe, 400)
    with pytest.raises(ValueError):
        validate(df.drop(index=10), cfg.timeframe)


def test_normalize_drops_unclosed_and_microseconds():
    step = STEPS["4h"]
    opened = 1_700_000_000_000 // step * step
    raw = [[opened * 1000, "100", "105", "95", "101", "7", (opened + step - 1) * 1000],
           [(opened + step) * 1000, "101", "108", "100", "107", "6", (opened + 2*step - 1)*1000]]
    df = normalize(raw, "BTC/USDT", "4h", now_ms=opened+step)
    assert len(df) == 1 and int(df.iloc[0].open_time_ms) == opened


def test_paper_bootstrap_repeated_restart(tmp_path):
    cfg = Config()
    hist = history(cfg)
    state = state_for(cfg)
    state = paper_tick(state, cfg, hist, {s: 100. for s in cfg.symbols}, "initial")
    assert not state["account"]["events"]
    same = paper_tick(state, cfg, hist, {s: 100. for s in cfg.symbols}, "again")
    assert not same["account"]["events"]
    path = tmp_path / "state.json"
    atomic_save(path, same)
    assert load_state(path, cfg) == same
    with pytest.raises(ValueError):
        load_state(path, Config(fee_bps=20))


def test_paper_per_symbol_stale_signals_ignored():
    cfg = Config()
    hist = history(cfg)
    st = state_for(cfg)
    for sym in cfg.symbols:
        st["last_closed_ms"][sym] = int(hist[sym].iloc[-3].open_time_ms)
    st = paper_tick(st, cfg, hist, {s: 100. for s in cfg.symbols}, "late")
    assert not st["account"]["events"]


def test_paper_fails_on_incomplete_snapshot():
    cfg = Config()
    with pytest.raises(ValueError):
        paper_tick(state_for(cfg), cfg, history(cfg), {"BTC/USDT": 100}, "now")


def test_historical_time_alignment():
    cfg = Config()
    hist = history(cfg)
    hist["ETH/USDT"] = hist["ETH/USDT"].iloc[1:]
    book = backtest(hist, cfg)
    assert book.curve  # one-bar offset is handled by intersection
    hist["ETH/USDT"] = hist["ETH/USDT"].drop(index=200)
    with pytest.raises(ValueError):
        backtest(hist, cfg)

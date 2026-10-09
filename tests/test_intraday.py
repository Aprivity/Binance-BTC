"""BTC-only 4h single-factor experiments and walk-forward regressions."""
import json

import numpy as np
import pandas as pd
import pytest

from btc_quant.__main__ import main, synthetic
from btc_quant.config import Config, STEPS
from btc_quant.core import backtest
from btc_quant.intraday import ARMS, arm_signals, intraday_study, run_arm, _windows
from btc_quant.strategies import strategy_signals


def inputs(n=1750):
    cfg = Config(symbols=("BTC/USDT",), timeframe="4h")
    data = synthetic("BTC/USDT", "4h", n=n)
    # Deterministic rising but oscillating market with actual EMA cross-ups.
    idx = np.arange(n)
    price = 100 + .06 * idx + 3 * np.sin(idx / 13)
    data["open"] = price - .2
    data["close"] = price
    data["high"] = price + 1
    data["low"] = price - 1
    return data, cfg


@pytest.mark.parametrize("begin,end", [(None,None),(None,700),(520,700),(600,850)])
def test_a_matches_existing_triple_ema_backtest_exactly(begin, end):
    data, cfg = inputs(1000)
    begin_ms = int(data.iloc[begin].open_time_ms) if begin else None
    end_ms = int(data.iloc[end].open_time_ms) if end else None
    a, detail, at = run_arm(data, cfg, "A_base", start_ms=begin_ms, end_ms=end_ms)
    b = backtest({"BTC/USDT":data}, cfg, strategy="triple_ema", start_ms=begin_ms, end_ms=end_ms)
    assert a.serialize() == b.serialize()
    assert len(detail) == len(b.trades)
    assert (at is not None) == a.halted


@pytest.mark.parametrize("arm", ARMS)
def test_new_signals_do_not_read_future_bars(arm):
    data, cfg = inputs(630)
    short = arm_signals(data.iloc[:450].copy(), cfg, arm)
    long = arm_signals(data, cfg, arm).iloc[:450]
    pd.testing.assert_series_equal(short.buy_signal, long.buy_signal)
    pd.testing.assert_series_equal(short.sell_signal, long.sell_signal)


def test_b_c_only_filter_base_entries_and_keep_exits():
    data, cfg = inputs()
    base = arm_signals(data, cfg, "A_base")
    slope = arm_signals(data, cfg, "B_slope")
    adx = arm_signals(data, cfg, "C_adx")
    for adjusted in [slope, adx]:
        assert (~adjusted.buy_signal | base.buy_signal).all()
        pd.testing.assert_series_equal(adjusted.sell_signal, base.sell_signal)
    assert (slope.ema200.notna()).any()
    assert (adx.adx14.dropna() >= 0).all()
    assert (adx.adx14.dropna() <= 100 + 1e-8).all()


def test_d_changes_only_entry_and_requires_rebound_over_ema21():
    data, cfg = inputs()
    base = arm_signals(data, cfg, "A_base")
    pullback = arm_signals(data, cfg, "D_pullback")
    pd.testing.assert_series_equal(pullback.sell_signal, base.sell_signal)
    ema21 = pullback.close.ewm(span=21, min_periods=21, adjust=False).mean()
    assert (~pullback.buy_signal | (pullback.close > ema21)).all()
    assert (~pullback.buy_signal | (pullback.close > pullback.ema200)).all()
    assert (~pullback.buy_signal | (pullback.close > pullback.open)).all()


def test_e_entry_signal_identical_to_a_and_trail_never_lowers_stop():
    data, cfg = inputs()
    a = arm_signals(data, cfg, "A_base")
    e = arm_signals(data, cfg, "E_trailing")
    pd.testing.assert_series_equal(a.buy_signal, e.buy_signal)
    pd.testing.assert_series_equal(a.sell_signal, e.sell_signal)
    book, details, _ = run_arm(data, cfg, "E_trailing")
    assert all(t["initial_stop"] > 0 for t in details)
    assert all(t["mfe_pct_observed"] >= 0 and t["mae_pct_observed"] <= 0 for t in details)
    assert all(t["hold_hours"] >= 0 for t in details)
    assert book.fees >= 0 and book.slippage >= 0
    if book.positions:
        assert book.positions["BTC/USDT"].stop > 0


def test_trailing_does_not_retroactively_stop_on_signal_bar():
    # The trailing stop is updated only after a full completed candle. Verify
    # both baseline and trailing start from the same book and first signal.
    data, cfg = inputs(670)
    a, _, _ = run_arm(data, cfg, "A_base", end_ms=int(data.iloc[400].open_time_ms))
    e, _, _ = run_arm(data, cfg, "E_trailing", end_ms=int(data.iloc[400].open_time_ms))
    assert a.events and e.events
    assert next(x for x in a.events if x["side"] == "BUY") == next(x for x in e.events if x["side"] == "BUY")


def test_fee_slippage_sensitivity_without_risk_mutation():
    data, cfg = inputs(850)
    cheap = Config(symbols=("BTC/USDT",), fee_bps=0, slippage_bps=0)
    costly = Config(symbols=("BTC/USDT",), fee_bps=50, slippage_bps=25)
    a, _, _ = run_arm(data, cheap, "A_base")
    b, _, _ = run_arm(data, costly, "A_base")
    assert a.fees == 0
    assert b.fees > 0
    assert a.equity() > b.equity()
    assert cfg.max_drawdown_pct == cheap.max_drawdown_pct == costly.max_drawdown_pct


def test_time_windows_sequential_nonoverlap_and_warmup():
    data, _ = inputs(2000)
    windows = _windows(data, train_days=120, test_days=30, step_days=30)
    assert len(windows) >= 2
    for train_start, test_start, test_end in windows:
        assert test_start - train_start == 120 * 86_400_000
        assert test_end - test_start == 30 * 86_400_000
    for left, right in zip(windows, windows[1:]):
        assert left[2] <= right[1]


def test_walkforward_outputs_reset_across_arms_and_times(tmp_path):
    data, cfg = inputs(1700)
    table, summary, meta = intraday_study(data, cfg, arms=("A_base", "B_slope", "E_trailing"),
        train_days=90, test_days=30, step_days=30, output_dir=tmp_path)
    assert meta["version"] == "0.2.2"
    assert meta["folds"] >= 2
    assert len(table) == meta["folds"] * 3 * 2
    assert len(summary) == 3
    assert (tmp_path / "oos_fold_returns.png").exists()
    assert (tmp_path / "trade_diagnostics.csv").exists()
    assert (tmp_path / "fold_equity.csv").exists()
    assert (tmp_path / "methodology.json").exists()
    for _, g in table.groupby(["fold", "arm"]):
        train = g[g.stage == "train"].iloc[0]
        test = g[g.stage == "test"].iloc[0]
        assert train.window_end_exclusive_utc == test.window_start_utc
        assert train.starting_usdt == test.starting_usdt == 1000
    curves = pd.read_csv(tmp_path / "fold_equity.csv")
    assert (curves.groupby(["fold", "stage", "arm"]).size() > 0).all()
    assert "not precision intrabar" in meta["diagnostic_method"]


def test_guardrails_reject_non_btc_non_four_hour_and_overlap(tmp_path):
    data, cfg = inputs()
    with pytest.raises(ValueError):
        intraday_study(data, cfg, step_days=15, test_days=30, output_dir=tmp_path)
    with pytest.raises(ValueError):
        intraday_study(data, cfg, arms=("A_base", "A_base"), output_dir=tmp_path)
    with pytest.raises(ValueError):
        intraday_study(data, cfg, arms=("invalid",), output_dir=tmp_path)
    with pytest.raises(ValueError):
        intraday_study(data, Config(symbols=("BTC/USDT", "ETH/USDT")), output_dir=tmp_path)
    with pytest.raises(ValueError):
        run_arm(data, Config(symbols=("BTC/USDT",), timeframe="1h"), "A_base")


def test_cli_rejects_default_symbols_and_smoke_csv(tmp_path):
    data, _ = inputs(1000)
    src = tmp_path / "data"
    src.mkdir()
    data.to_csv(src / "BTCUSDT_4h.csv", index=False)
    with pytest.raises(SystemExit):
        main(["intraday-study", "--csv-dir", str(src)])
    out = tmp_path / "out"
    main(["intraday-study", "--symbols", "BTC/USDT", "--csv-dir", str(src),
          "--train-days", "70", "--test-days", "20", "--step-days", "20",
          "--arms", "A_base", "B_slope", "--output-dir", str(out)])
    assert json.loads((out / "methodology.json").read_text())["arms"].keys() == {"A_base", "B_slope"}
    assert len(pd.read_csv(out / "oos_summary.csv")) == 2

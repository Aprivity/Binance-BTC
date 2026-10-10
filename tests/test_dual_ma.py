"""Offline regression checks of the optional BTC 200 USDT dual-MA research."""
import json

import pandas as pd
import pytest

from btc_quant.__main__ import synthetic
from btc_quant.config import Config, STEPS
from btc_quant.core import backtest
from btc_quant.data import validate
from btc_quant.strategy import prepare_signals, resolve_strategy
from research.dual_ma.study import COST_CASES, analyze
from research_strategies.dual_ma import CANDIDATES, Parameters, calculate


def test_intraday_timeframes():
    for interval in ("5m", "15m", "1h", "4h"):
        df = validate(synthetic(interval, 560), interval)
        assert df.open_time_ms.iloc[1] - df.open_time_ms.iloc[0] == STEPS[interval]
        assert Config(timeframe=interval).timeframe == interval


@pytest.mark.parametrize("params", [
    Parameters(9, 21),
    Parameters(9, 21, confirm_bars=2),
    Parameters(12, 26),
    Parameters(20, 60, "sma"),
])
def test_future_bars_cannot_change_historical_signals(params):
    frame = synthetic("15m", 560)
    previous = calculate(frame, params)
    altered = frame.copy()
    altered.loc[450:, "close"] += 9000
    modified = calculate(altered, params)
    pd.testing.assert_frame_equal(previous.iloc[:450], modified.iloc[:450])
    assert pd.api.types.is_bool_dtype(previous.buy_signal.dtype)
    assert pd.api.types.is_bool_dtype(previous.sell_signal.dtype)


def test_confirmation_delays_entry():
    frame = pd.DataFrame({"close": [100.0] * 65 + [101.0] * 10 + [100.0] * 30})
    one = calculate(frame, Parameters(9, 21, confirm_bars=1))
    two = calculate(frame, Parameters(9, 21, confirm_bars=2))
    first = one.index[one.buy_signal].tolist()
    delayed = two.index[two.buy_signal].tolist()
    assert first and delayed and delayed[0] == first[0] + 1
    flat = calculate(pd.DataFrame({"close": [100.0] * 100}), Parameters(9, 21))
    assert not flat.buy_signal.any() and not flat.sell_signal.any()


def test_parameter_guards():
    for args in (dict(fast=21, slow=9),
                 dict(fast=9, slow=21, kind="wma"),
                 dict(fast=9, slow=21, confirm_bars=0)):
        with pytest.raises(ValueError):
            Parameters(**args)


def test_explicit_plugins_and_simulated_spot():
    df = validate(synthetic("15m", 560), "15m")
    cfg = Config(timeframe="15m", starting_usdt=200.0)
    for name, fn in CANDIDATES.items():
        aligned = prepare_signals(df, cfg, fn)
        assert not (aligned.buy_signal & aligned.sell_signal).any()
        assert resolve_strategy("research_strategies.dual_ma:" + name) is fn
        result = backtest({"BTC/USDT": df}, cfg, signal_fn=fn)
        assert result.cash >= -1e-7
        assert set(result.positions).issubset({"BTC/USDT"})


def test_research_end_to_end_offline(tmp_path):
    df = validate(synthetic("15m", 1200), "15m")
    results = analyze(df, "15m", tmp_path, days_requested=14)
    assert len(results) == len(CANDIDATES) * len(COST_CASES) * 2
    assert set(results.stage) == {"development", "held_out"}
    assert set(results.cost_case) == set(COST_CASES)
    assert (results.starting_usdt == 200).all()
    assert (tmp_path / "costed_results.csv").exists()
    info = json.loads((tmp_path / "methodology.json").read_text())
    assert info["starting_cash_usdt"] == 200
    assert info["source_fidelity"].startswith("UNVERIFIED")
    assert (tmp_path / "README.md").exists()
    assert len(list((tmp_path / "trades").glob("*.csv"))) == len(results)


def test_research_refuses_missing_candles(tmp_path):
    df = validate(synthetic("15m", 1200), "15m")
    bad = df.drop(index=400).reset_index(drop=True)
    with pytest.raises(ValueError, match="gaps"):
        analyze(bad, "15m", tmp_path)

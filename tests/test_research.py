"""v0.2.1 research regressions; synthetic prices have no investing meaning."""
import json
import numpy as np
import pandas as pd
import pytest

from btc_quant.__main__ import synthetic, main
from btc_quant.config import Config, STEPS
from btc_quant.core import Account, backtest, signals
from btc_quant.research import benchmark_metrics, compare
from btc_quant.strategies import STRATEGIES, strategy_signals, supertrend


def histories(n=640):
    cfg = Config()
    return {s: synthetic(s, cfg.timeframe, n=n) for s in cfg.symbols}


def test_research_ema_identical_to_original_and_legacy_backtest():
    cfg = Config()
    history = histories()
    for s in cfg.symbols:
        a = signals(history[s], cfg)
        b = strategy_signals(history[s], cfg, "ema")
        assert a.equals(b)
    old = backtest(history, cfg)
    new = backtest(history, cfg, strategy="ema")
    assert old.serialize() == new.serialize()


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_no_future_research_signals(strategy):
    cfg = Config(symbols=("BTC/USDT",))
    data = histories()["BTC/USDT"]
    original = strategy_signals(data.iloc[:430], cfg, strategy)
    extended = strategy_signals(data, cfg, strategy).iloc[:430]
    pd.testing.assert_series_equal(original.buy_signal, extended.buy_signal)
    pd.testing.assert_series_equal(original.sell_signal, extended.sell_signal)


def test_triple_ema_restricts_but_never_invents_entries():
    cfg = Config(symbols=("BTC/USDT",))
    x = histories()["BTC/USDT"]
    ema = strategy_signals(x, cfg, "ema")
    triple = strategy_signals(x, cfg, "triple_ema")
    assert (~triple.buy_signal | ema.buy_signal).all()
    assert (~triple.buy_signal | (x.close > triple.ema200)).all()
    assert triple.sell_signal.equals(ema.sell_signal)


def test_donchian_prior_high_prevents_same_bar_channel_cheat():
    cfg = Config(symbols=("BTC/USDT",))
    x = histories()["BTC/USDT"].copy()
    idx = 350
    x.loc[idx, "high"] = 500
    x.loc[idx, "close"] = 450
    x.loc[idx, "open"] = 100
    x.loc[idx, "low"] = 99
    sig = strategy_signals(x, cfg, "donchian")
    assert sig.iloc[idx].buy_signal
    assert float(sig.iloc[idx].channel_high) < 450
    assert not sig.iloc[idx - 1].buy_signal


def test_supertrend_no_lookahead_and_non_mutating():
    cfg = Config(symbols=("BTC/USDT",))
    x = histories()["BTC/USDT"]
    before = x.copy(deep=True)
    sig = strategy_signals(x, cfg, "supertrend_cci")
    assert sig.supertrend_direction.isin([-1, 0, 1]).all()
    assert sig.cci.notna().any()
    pd.testing.assert_frame_equal(x, before)


def test_separate_holdout_accounts_no_leakage_and_expected_rows(tmp_path):
    cfg = Config()
    h = histories()
    rows, m = compare(h, cfg, scope="all", days=None, output_dir=tmp_path, plot=True)
    assert len(rows) == 3 * 4 * 3  # BTC, ETH, portfolio x four strategies x train/test/full
    assert (tmp_path / "compare.csv").exists()
    assert (tmp_path / "methodology.json").exists()
    assert (tmp_path / "oos_comparison.png").exists()
    assert (tmp_path / "oos_equity.png").exists()
    assert set(rows.strategy) == set(STRATEGIES)
    for (scope, strategy), group in rows.groupby(["scope", "strategy"]):
        train = group[group.stage == "train"].iloc[0]
        test = group[group.stage == "test"].iloc[0]
        assert train.last_mark_utc < test.first_mark_utc
        assert train.bars > 150 and test.bars >= 75
        # Cash always starts at the same amount in each window.
        assert -100 <= test.net_return_pct <= 10000
        assert -100 <= train.net_return_pct <= 10000
    assert m["train_ratio"] == 0.7


def test_fee_assumption_changes_strategy_returns(tmp_path):
    h = histories()
    a = Config(fee_bps=0, slippage_bps=0)
    b = Config(fee_bps=50, slippage_bps=25)
    low, _ = compare(h, a, strategies=("ema",), scope="portfolio", output_dir=tmp_path / "low", plot=False)
    high, _ = compare(h, b, strategies=("ema",), scope="portfolio", output_dir=tmp_path / "high", plot=False)
    x = low[low.stage == "full"].iloc[0]
    y = high[high.stage == "full"].iloc[0]
    assert x.closed_trades > 0
    assert x.net_return_pct > y.net_return_pct
    assert x.fees_paid_usdt == 0
    assert y.fees_paid_usdt > 0


def test_benchmark_respects_symbol_cap_and_expost_caveat():
    cfg = Config(symbols=("BTC/USDT",))
    h = {"BTC/USDT": histories()["BTC/USDT"]}
    book = backtest(h, cfg)
    report = benchmark_metrics(h, book, cfg)
    assert report["policy_exposure_cap_pct"] == 30.0
    assert "ex-post" in report["ex_post_comparison_warning"]
    assert 0 <= report["avg_exposure_pct"] <= 30.01


def test_holdout_rejects_bad_split_and_insufficient_history(tmp_path):
    cfg = Config()
    with pytest.raises(ValueError):
        compare(histories(), cfg, train_ratio=0.99, output_dir=tmp_path)
    with pytest.raises(ValueError):
        compare(histories(420), cfg, output_dir=tmp_path)
    with pytest.raises(ValueError):
        compare(histories(), cfg, strategies=("imaginary",), output_dir=tmp_path)


def test_cli_research_offline_uses_read_only_data(tmp_path):
    h = histories()
    source = tmp_path / "data"
    source.mkdir()
    for symbol, frame in h.items():
        frame.to_csv(source / f"{symbol.replace('/', '')}_4h.csv", index=False)
    out = tmp_path / "result"
    main(["compare", "--csv-dir", str(source), "--strategies", "ema", "donchian",
          "--scope", "portfolio", "--output-dir", str(out)])
    data = pd.read_csv(out / "compare.csv")
    assert set(data.strategy) == {"ema", "donchian"}
    assert len(data) == 6
    assert json.loads((out / "methodology.json").read_text())["version"] == "0.2.1"

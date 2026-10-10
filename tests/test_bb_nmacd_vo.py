import numpy as np
import pandas as pd
import pytest
from experiments.bb_nmacd_vo import BAR_MS, indicators, simulate
from experiments.run_bb_nmacd_vo import parse_funding


def candles(n=240, *, start=0):
    t = start + np.arange(n, dtype=np.int64) * BAR_MS
    closes = 100.0 + 0.4 * np.sin(np.arange(n) / 7)
    opens = np.r_[closes[0], closes[:-1]]
    return pd.DataFrame({"open_time_ms": t, "open": opens,
                         "high": np.maximum(opens, closes) + 0.5,
                         "low": np.minimum(opens, closes) - 0.5,
                         "close": closes})


def simulated_frame(n=210):
    x = candles(n)
    x["long_signal"] = False
    x["short_signal"] = False
    return x


def zero_funding():
    return pd.DataFrame({"funding_ms": pd.Series([], dtype="int64"),
                         "rate": pd.Series([], dtype="float64")})


def test_indicators_causal_and_contiguous():
    x = candles(260)
    full = indicators(x)
    assert full.long_signal.dtype == bool
    assert full.short_signal.dtype == bool
    assert not (full.long_signal & full.short_signal).any()
    assert not full.iloc[:200].long_signal.any()
    shortened = indicators(x.iloc[:230])
    for key in ("nmacd", "nmacd_trigger", "bb_middle", "vo_upper"):
        assert np.allclose(full[key].iloc[:230], shortened[key], equal_nan=True)
    gap = x.drop(19)
    with pytest.raises(ValueError, match="contiguous"):
        indicators(gap)


def test_long_stop_first_when_both_touched():
    x = simulated_frame()
    signal = 201
    x.loc[signal, "long_signal"] = True
    x.loc[signal, ["low", "high"]] = [98.0, 102.0]
    x.loc[signal + 1, ["open", "high", "low", "close"]] = [100.0, 104.0, 97.0, 101.0]
    tr, eq, summary = simulate(x, zero_funding(), mode="fixed_1_5")
    assert len(tr) == 1
    assert tr.exit_reason.iloc[0] == "stop"
    assert tr.net_pnl.iloc[0] < 0
    assert summary["gross_leverage_cap"] == 1.0


def test_short_split_and_funding_positive_rate():
    x = simulated_frame()
    i = 201
    x.loc[i, "short_signal"] = True
    x.loc[i, ["high", "low"]] = [102.0, 98.0]
    x.loc[i + 1, ["open", "high", "low", "close"]] = [100.0, 100.5, 99.2, 100.0]
    x.loc[i + 2, ["open", "high", "low", "close"]] = [99.0, 99.5, 95.0, 96.0]
    f = pd.DataFrame({"funding_ms": [int(x.open_time_ms.iloc[i + 2])], "rate": [0.0001]})
    tr, _, summary = simulate(x, f, mode="split", direction="short")
    assert len(tr) == 1
    assert tr.exit_reason.iloc[0] == "take_2R"
    assert tr.net_pnl.iloc[0] > 0
    assert summary["funding_usdt"] > 0  # Short receives positive funding.


def test_next_open_not_signal_close_and_1x_cap():
    x = simulated_frame()
    i = 201
    x.loc[i, "long_signal"] = True
    x.loc[i, "low"] = 98.0
    x.loc[i+1, ["open", "high", "low", "close"]] = [105.0, 105.5, 104.5, 105.0]
    tr, _, _ = simulate(x, zero_funding(), mode="fixed_2")
    assert len(tr) == 1
    assert tr.entry_price.iloc[0] == pytest.approx(105.0 * 1.0002)
    assert tr.quantity.iloc[0] * tr.entry_price.iloc[0] <= 1000


def test_no_trades_no_profit_factor():
    tr, _, summary = simulate(simulated_frame(), zero_funding(), mode="fixed_2")
    assert tr.empty and summary["trades"] == 0
    assert summary["profit_factor"] is None


def test_funding_archive_one_millisecond_timestamp_rounding():
    raw = pd.DataFrame([["BTCUSDT", 1704067200001, 8, 0.0001]])
    result = parse_funding(raw)
    assert int(result.funding_ms.iloc[0]) == 1704067200000
    assert result.rate.iloc[0] == pytest.approx(0.0001)
    raw.iloc[0, 1] = 1704067202000
    with pytest.raises(ValueError, match=">1s"):
        parse_funding(raw)

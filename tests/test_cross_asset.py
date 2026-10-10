"""Anti-leakage and evaluation tests using synthetic data, not profit claims."""
import numpy as np
import pandas as pd
import pytest

from btc_quant.cross_asset import (ASSETS, BTC_FEATURES, bootstrap_mean, build_samples,
    diagnostic_trades, holm, ridge_predict, walk_forward)
from btc_quant.cross_asset_data import available_times, validate_daily


def fixture_data(days=450):
    clock = pd.date_range("2023-01-01", periods=days * 6, freq="4h", tz="UTC")
    rng = np.random.default_rng(50)
    price = 100 * np.exp(np.cumsum(rng.normal(0, .005, len(clock))))
    btc = pd.DataFrame({"open_time_ms": clock.as_unit("ms").asi8,
        "open": price, "high": price * 1.02, "low": price * .98,
        "close": price * 1.001, "base_volume": 1,
        "close_time_ms": clock.as_unit("ms").asi8 + 14_400_000 - 1})
    dates = pd.bdate_range("2022-11-01", clock[-1].tz_localize(None))
    external = {a: pd.DataFrame({"date": dates.strftime("%Y-%m-%d"),
        "close": 100 * np.exp(np.cumsum(rng.normal(0, .005, len(dates))))}) for a in ASSETS}
    return btc, external


def test_availability_dst_and_no_same_day_use():
    got = available_times(["2023-03-11", "2023-03-12", "2023-11-04", "2023-11-05"])
    assert list(got.hour) == [5, 4, 4, 5]
    assert list(got.minute) == [1, 1, 1, 1]
    b, x = fixture_data()
    samples = build_samples(b, x, 24)
    assert samples.entry_utc.dt.hour.eq(8).all()
    assert not samples.entry_utc.duplicated().any()
    assert samples.ndx_session_date.nunique() == len(samples)
    for a in ASSETS:
        assert (samples[f"{a}_available_utc"] < samples.entry_utc).all()
    # A Friday session is used Saturday, but is never repeated Sunday/Monday.
    assert samples.entry_utc.dt.dayofweek.eq(5).any()
    assert not samples.entry_utc.dt.dayofweek.isin([0, 6]).any()


def test_future_market_changes_cannot_change_past_predictors():
    b, x = fixture_data()
    cut = pd.Timestamp("2023-08-01T08:00:00Z")
    before = build_samples(b, x, 24)
    changed = b.copy()
    changed.loc[changed.open_time_ms >= cut.value // 1_000_000, ["open", "high", "low", "close"]] *= 2
    xx = {a: d.copy() for a, d in x.items()}
    for d in xx.values():
        d.loc[available_times(d.date) >= cut, "close"] *= 3
    after = build_samples(changed, xx, 24)
    cols = BTC_FEATURES + [f"{a}_{s}" for a in ASSETS for s in ("r1", "r5", "vol20", "age_hours")]
    # Even the entry candle close must not be used in that entry's forecast.
    pd.testing.assert_frame_equal(before.loc[before.entry_utc <= cut, cols], after.loc[after.entry_utc <= cut, cols])


def test_target_uses_future_opens_and_rejects_missing_bars():
    b, x = fixture_data()
    for h in [4, 24]:
        s = build_samples(b, x, h)
        first = s.iloc[0]
        idx = b.index[b.open_time_ms == first.entry_utc.value // 1_000_000][0]
        assert first.actual_return == pytest.approx(b.open.iloc[idx + h // 4] / b.open.iloc[idx] - 1)
        assert first.exit_utc - first.entry_utc == pd.Timedelta(hours=h)
    with pytest.raises(ValueError, match="Missing candles"):
        build_samples(b.drop(index=90), x, 4)


def test_train_label_purge_and_no_test_target_in_fit():
    b, x = fixture_data()
    s = build_samples(b, x, 24)
    pred, folds = walk_forward(s)
    assert not pred[pred.model == "btc_only"].entry_utc.duplicated().any()
    assert (pd.to_datetime(folds.last_train_label_utc) < pd.to_datetime(folds.test_start_utc)).all()
    cut = pd.Timestamp(folds.test_start_utc.iloc[0])
    train = s[(s.entry_utc < cut) & (s.exit_utc < cut)]
    test = s[s.entry_utc >= cut].copy()
    original = ridge_predict(train, test, BTC_FEATURES)
    test.actual_return = 50
    assert np.array_equal(original, ridge_predict(train, test, BTC_FEATURES))
    # Distant test rows must not affect train scaling or first test prediction.
    assert original[0] == pytest.approx(ridge_predict(train, test.iloc[:1], BTC_FEATURES)[0])


def test_bootstrap_and_multiple_testing():
    a = bootstrap_mean(np.ones(100) * .01, repeats=200)
    assert a["ci_low"] == pytest.approx(.01)
    assert a["p_value"] < .01
    zero = bootstrap_mean(np.zeros(100), repeats=200)
    assert zero["p_value"] == 1
    assert holm([.01, .04, .03]).tolist() == pytest.approx([.03, .06, .06])


def test_roundtrip_costs_and_overlap_rejection():
    dates = pd.date_range("2023-01-01", periods=2, tz="UTC")
    f = pd.DataFrame({"entry_utc": dates, "exit_utc": dates + pd.Timedelta(hours=24),
        "actual_return": [0., 0.], "prediction": [.01, .01]})
    result = diagnostic_trades(f)
    ratio = (.9995 * .999) / (1.0005 * 1.001)
    assert result["toy_net_return_pct"] == pytest.approx(((1 + .3 * (ratio - 1)) ** 2 - 1) * 100)
    assert result["toy_trades"] == 2
    assert result["toy_same_trades_zero_cost_return_pct"] == 0
    f.prediction = .001
    assert diagnostic_trades(f)["toy_trades"] == 0
    f.exit_utc += pd.Timedelta(hours=4)
    with pytest.raises(ValueError, match="Overlapping"):
        diagnostic_trades(f)


@pytest.mark.parametrize("closes", [[100, -1], [100, np.inf], [100, np.nan]])
def test_invalid_daily_prices_rejected(closes):
    with pytest.raises(ValueError):
        validate_daily(pd.DataFrame({"date": ["2023-01-01", "2023-01-02"], "close": closes}))


def test_duplicate_daily_date_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_daily(pd.DataFrame({"date": ["2023-01-01"] * 2, "close": [100, 101]}))

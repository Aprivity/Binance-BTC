"""v0.2.3 robustness: diagnostics, cost scenarios and matched folds."""
from pathlib import Path

import pandas as pd
import pytest

from btc_quant.config import Config
from btc_quant.core import Account
from btc_quant.robustness import (
    cost_scenarios, liquidation_equity, paired_comparison,
    aggregate_paired, robustness_study,
)
from btc_quant.__main__ import synthetic


def cfg(**kwargs):
    return Config(symbols=("BTC/USDT",), timeframe="4h", **kwargs)


def test_cost_scenarios_explicit_assumptions():
    assert cost_scenarios(cfg()) == (("zero_cost_upper_bound", 0., 0.),
                                      ("configured", 10., 5.),
                                      ("stress_2x", 20., 10.))
    with pytest.raises(ValueError, match="exceeds"):
        cost_scenarios(cfg(fee_bps=60))


def test_open_position_liquidation_applies_exit_cost_once_without_mutation():
    c = cfg()
    b = Account.create(c)
    b.prices["BTC/USDT"] = 100
    assert b.buy("BTC/USDT", "t", 100, 2, c)
    b.prices["BTC/USDT"] = 104
    before = b.serialize()
    gross = b.positions["BTC/USDT"].quantity * 104
    expected = b.equity() - gross * (1 - (1 - c.slip)*(1 - c.fee))
    assert liquidation_equity(b, c) == pytest.approx(expected)
    assert b.serialize() == before
    zero = cfg(fee_bps=0, slippage_bps=0)
    assert liquidation_equity(b, zero) == pytest.approx(b.equity())


def test_paired_summary_uses_matching_scenario_and_fold():
    rows = []
    for scenario, fold, base, other in [("configured", 1, -1., -.5),
                                       ("configured", 2, 0., 0.),
                                       ("stress_2x", 1, -2., -3.)]:
        for arm, ret in [("A_base", base), ("E_trailing", other)]:
            rows.append({"scenario": scenario, "fold": fold, "arm": arm,
                         "liquidation_net_return_pct": ret,
                         "net_return_pct": ret, "max_drawdown_pct": -4.,
                         "closed_trades": 1, "halted": False})
    paired, summary = aggregate_paired(pd.DataFrame(rows))
    a = summary[(summary.arm == "E_trailing") & (summary.scenario == "configured")].iloc[0]
    assert a.improved_folds == 1 and a.worse_folds == 0 and a.tied_folds == 1
    assert a.mean_delta_vs_A_pp == pytest.approx(.25)
    stressed = summary[(summary.arm == "E_trailing") & (summary.scenario == "stress_2x")].iloc[0]
    assert stressed.mean_delta_vs_A_pp == pytest.approx(-1.)
    assert (paired[paired.arm == "A_base"].delta_vs_A_pct_points.abs() < 1e-12).all()


def test_paired_rejects_missing_or_duplicate_baseline():
    base = pd.DataFrame([{"scenario": "configured", "fold": 1, "arm": "E_trailing",
                          "liquidation_net_return_pct": -1}])
    with pytest.raises(ValueError, match="A_base"):
        paired_comparison(base)
    with pytest.raises(ValueError, match="unique"):
        paired_comparison(pd.concat([base, base], ignore_index=True))


def test_robustness_integrated_one_real_simulated_fold(tmp_path):
    c = cfg()
    history = synthetic("BTC/USDT", "4h", n=580)
    out = tmp_path / "robust"
    paired, summary, meta = robustness_study(
        history, c, train_days=45, test_days=14, step_days=14, output_dir=out)
    assert meta["folds"] == 1
    assert len(paired) == 15 and len(summary) == 15
    assert set(paired.scenario) == {"zero_cost_upper_bound", "configured", "stress_2x"}
    assert set(paired.arm) == {"A_base", "B_slope", "C_adx", "D_pullback", "E_trailing"}
    assert (paired[paired.arm == "A_base"].delta_vs_A_pct_points.abs() < 1e-10).all()
    assert (paired.mark_to_liquidate_difference_pp >= -1e-10).all()
    for filename in ("cost_fold_comparison.csv", "cost_paired_summary.csv", "methodology.json"):
        assert (out / filename).is_file()
    # No fake claim of trading success from synthetic OHLCV.
    assert meta["study_caveat"].startswith("Exploratory")


def test_rejects_non_btc_and_overlapping_windows(tmp_path):
    history = synthetic("BTC/USDT", "4h", n=580)
    with pytest.raises(ValueError, match="BTC/USDT"):
        robustness_study(history, Config(), train_days=45, test_days=14, step_days=14, output_dir=tmp_path)
    with pytest.raises(ValueError, match="step"):
        robustness_study(history, cfg(), train_days=45, test_days=14, step_days=7, output_dir=tmp_path)
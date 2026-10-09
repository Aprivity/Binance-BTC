"""BTC-only 4h fixed-rule cost robustness and paired fold comparisons.

Research only. Never places orders and never changes the paper engine or arm rules.
Each fold/scenario/arm has an independent fresh simulated account. This is NOT
one continuously investable equity curve and not an untouched OOS study.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import Config, STEPS
from .data import load_csv
from .intraday import ARMS, _windows, _iso, run_arm
from .report import metrics


def cost_scenarios(cfg: Config):
    """Fixed, non-optimized stress assumptions in *per-side* basis points.

    zero_cost is an optimistic mathematical bound, NOT an obtainable exchange
    quote. stress_2x doubles the configured fee and slip, within Config limits.
    """
    if cfg.fee_bps * 2 > 100 or cfg.slippage_bps * 2 > 100:
        raise ValueError("Cost stress 2x exceeds the supported 100 bps per-side model cap")
    return (
        ("zero_cost_upper_bound", 0.0, 0.0),
        ("configured", float(cfg.fee_bps), float(cfg.slippage_bps)),
        ("stress_2x", float(cfg.fee_bps * 2), float(cfg.slippage_bps * 2)),
    )


def liquidation_equity(book, cfg: Config) -> float:
    """Hypothetical immediate liquidation of OPEN positions at the last mark.

    Standard Account.equity marks open positions without hypothetical exit
    fees/slippage. This additional diagnostic subtracts an exit-side cost once,
    without executing an order or modifying the simulated account or P&L.
    It does not model a future spread, gap, liquidity shortfall or price impact.
    """
    equity = float(book.equity())
    for symbol, position in book.positions.items():
        if symbol not in book.prices:
            raise ValueError(f"Missing final price for open position: {symbol}")
        gross = position.quantity * float(book.prices[symbol])
        liquidation_net = gross * (1 - cfg.slip) * (1 - cfg.fee)
        equity -= gross - liquidation_net
    return equity


def paired_comparison(table: pd.DataFrame, *, metric: str = "liquidation_net_return_pct"):
    """Compare each fixed arm against A within matching dates and costs."""
    required = {"scenario", "fold", "arm", metric}
    if not required.issubset(table.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(table.columns))}")
    if table.duplicated(["scenario", "fold", "arm"]).any():
        raise ValueError("Each scenario/fold/arm pair must be unique")
    wide = table.pivot(index=["scenario", "fold"], columns="arm", values=metric)
    if "A_base" not in wide.columns or wide["A_base"].isna().any():
        raise ValueError("A_base required in every paired fold")
    base = wide["A_base"]
    out = table.copy()
    keys = pd.MultiIndex.from_frame(out[["scenario", "fold"]])
    out["baseline_return_pct"] = base.reindex(keys).to_numpy()
    out["delta_vs_A_pct_points"] = out[metric] - out["baseline_return_pct"]
    return out


def aggregate_paired(table: pd.DataFrame):
    paired = paired_comparison(table)
    eps = 1e-9
    group = paired.groupby(["scenario", "arm"], sort=False)
    summary = group.agg(
        folds=("fold", "count"),
        mean_return_pct=("liquidation_net_return_pct", "mean"),
        median_return_pct=("liquidation_net_return_pct", "median"),
        positive_return_folds=("liquidation_net_return_pct", lambda x: int((x > eps).sum())),
        mean_delta_vs_A_pp=("delta_vs_A_pct_points", "mean"),
        median_delta_vs_A_pp=("delta_vs_A_pct_points", "median"),
        improved_folds=("delta_vs_A_pct_points", lambda x: int((x > eps).sum())),
        worse_folds=("delta_vs_A_pct_points", lambda x: int((x < -eps).sum())),
        tied_folds=("delta_vs_A_pct_points", lambda x: int((x.abs() <= eps).sum())),
        worst_fold_pct=("liquidation_net_return_pct", "min"),
        worst_drawdown_pct=("max_drawdown_pct", "min"),
        closed_trades=("closed_trades", "sum"),
        no_closed_trade_folds=("closed_trades", lambda x: int((x == 0).sum())),
        halted_folds=("halted", "sum"),
    ).reset_index()
    return paired, summary


def robustness_study(history: pd.DataFrame, cfg: Config, *,
                     train_days: int = 180, test_days: int = 60,
                     step_days: int = 60,
                     output_dir: str | Path = "outputs/btc-robustness-v023"):
    """Re-run ALL unchanged v0.2.2 arms over matching 60d fold windows.

    This is a sensitivity/diagnostic study, not parameter optimization.
    The 180d training segment is a chronological context ONLY: no parameters
    are fitted; test accounts are reset and cannot trade training signals.
    """
    if cfg.symbols != ("BTC/USDT",) or cfg.timeframe != "4h":
        raise ValueError("This study requires BTC/USDT spot 4h only")
    if train_days < 45 or test_days < 14 or step_days < test_days:
        raise ValueError("Require train >=45d, test >=14d, and step >= test days")
    if len(history) < 220 or not (history.open_time_ms.diff().dropna() == STEPS["4h"]).all():
        raise ValueError("Expected gap-free chronological 4h history with >=220 bars")
    windows = _windows(history, train_days, test_days, step_days)
    rows = []
    for scenario, fee_bps, slip_bps in cost_scenarios(cfg):
        scenario_cfg = replace(cfg, fee_bps=fee_bps, slippage_bps=slip_bps)
        for fold, (_, begin, end) in enumerate(windows, 1):
            for arm in ARMS:
                book, _, halt = run_arm(history, scenario_cfg, arm, start_ms=begin, end_ms=end)
                standard = metrics(book, scenario_cfg)
                liquidated = liquidation_equity(book, scenario_cfg)
                rows.append({
                    "scenario": scenario, "fee_bps": fee_bps, "slippage_bps": slip_bps,
                    "fold": fold, "arm": arm,
                    "window_start_utc": _iso(begin), "window_end_exclusive_utc": _iso(end),
                    "halted_at_utc": halt,
                    "net_return_pct": standard["net_return_pct"],
                    "liquidation_net_return_pct": 100 * (liquidated / cfg.starting_usdt - 1),
                    "mark_to_liquidate_difference_pp": 100 * (book.equity() - liquidated) / cfg.starting_usdt,
                    "open_positions": len(book.positions),
                    "closed_trades": standard["closed_trades"],
                    "fees_paid_usdt": standard["fees_paid_usdt"],
                    "slippage_estimate_usdt": standard["slippage_estimate_usdt"],
                    "max_drawdown_pct": standard["max_drawdown_pct"],
                    "halted": bool(book.halted),
                })
    detail = pd.DataFrame(rows)
    paired, summary = aggregate_paired(detail)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out / "cost_fold_comparison.csv", index=False)
    summary.to_csv(out / "cost_paired_summary.csv", index=False)
    methodology = {
        "version": "0.2.3", "asset": "BTC/USDT", "interval": "4h",
        "strategies": list(ARMS), "folds": len(windows),
        "train_days": train_days, "test_days": test_days, "step_days": step_days,
        "costs": [{"scenario": n, "fee_bps": f, "slippage_bps": s}
                  for n, f, s in cost_scenarios(cfg)],
        "risk_and_size": cfg.to_dict(),
        "baseline": "A_base in the same matching fold and cost scenario; improvement is in percentage POINTS",
        "terminal_equity": "Standard mark-to-market, plus hypothetical final liquidation exit fee/slippage on still open positions ONLY; no forced trades",
        "scope": "All five v0.2.2 strategies held fixed; no tuned parameters or combined new strategy",
        "study_caveat": "Exploratory historical folds previously inspected. No untouched holdout or statistical significance from nine folds; no compounding of reset accounts. Zero-cost case is unrealistically optimistic; slippage is fixed not market impact.",
    }
    (out / "methodology.json").write_text(json.dumps(methodology, ensure_ascii=False, indent=2), encoding="utf-8")
    return paired, summary, methodology


def main(argv=None):
    parser = argparse.ArgumentParser(description="BTC 4h cost robustness research, no trading")
    parser.add_argument("--csv-dir", default="data/market")
    parser.add_argument("--train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=60)
    parser.add_argument("--fee-bps", type=float, default=10)
    parser.add_argument("--slippage-bps", type=float, default=5)
    parser.add_argument("--initial-cash", type=float, default=1000)
    parser.add_argument("--output-dir", default="outputs/btc-robustness-v023")
    args = parser.parse_args(argv)
    cfg = Config(symbols=("BTC/USDT",), timeframe="4h", starting_usdt=args.initial_cash,
                 fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    data = load_csv(Path(args.csv_dir) / "BTCUSDT_4h.csv", "BTC/USDT", "4h")
    _, summary, info = robustness_study(
        data, cfg, train_days=args.train_days, test_days=args.test_days,
        step_days=args.step_days, output_dir=args.output_dir)
    print(f"Fixed-rule, fresh-account {info['folds']} fold cost study (NOT a forecast)")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()

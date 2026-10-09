"""Portfolio performance metrics and CSV/PNG report output."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import STEPS


def metrics(book, cfg):
    curve = pd.DataFrame(book.curve)
    ending = book.equity() if curve.empty else float(curve.equity.iloc[-1])
    pnl = np.asarray([t["pnl_usdt"] for t in book.trades], dtype=float)
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    series = curve.equity.pct_change().dropna() if not curve.empty else pd.Series(dtype=float)
    # Crypto has 365 days/year, no equity-market 252-day assumption.
    periods = 365 * (86_400_000 // STEPS[cfg.timeframe])
    sharpe = None
    if len(series) >= 3 and series.std(ddof=1) > 0:
        sharpe = float(series.mean() / series.std(ddof=1) * np.sqrt(periods))
    contributions = {s: round(sum(t["pnl_usdt"] for t in book.trades if t["symbol"] == s), 4)
                     for s in cfg.symbols}
    return {"starting_usdt": cfg.starting_usdt, "ending_usdt": round(ending, 4),
            "net_return_pct": round((ending / cfg.starting_usdt - 1) * 100, 4),
            "max_drawdown_pct": round(100 * min((x["drawdown"] for x in book.curve), default=0), 4),
            "closed_trades": len(pnl), "win_rate_pct": round(float((pnl > 0).mean()) * 100, 2) if len(pnl) else None,
            "profit_factor": round(float(wins / losses), 4) if losses else None,
            "sharpe": round(sharpe, 4) if sharpe is not None else None,
            "fees_paid_usdt": round(book.fees, 4), "slippage_estimate_usdt": round(book.slippage, 4),
            "realized_pnl_usdt": round(float(pnl.sum()), 4),
            "pnl_by_symbol": contributions,
            "open_quantities": {s: book.positions[s].quantity if s in book.positions else 0.0 for s in cfg.symbols},
            "halted": book.halted,
            "note": "Mark-to-market before hypothetical exit costs; paper results are not live fills."}


def write_report(book, cfg, output_dir, plot=True):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = metrics(book, cfg)
    (out / "metrics.json").write_text(json.dumps({"config": cfg.to_dict(), "metrics": result}, indent=2), encoding="utf-8")
    pd.DataFrame(book.trades).to_csv(out / "trades.csv", index=False)
    pd.DataFrame(book.events).to_csv(out / "events.csv", index=False)
    equity = pd.DataFrame(book.curve)
    equity.to_csv(out / "equity.csv", index=False)
    if plot and not equity.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        t = pd.to_datetime(equity.time, utc=True)
        a.plot(t, equity.equity)
        a.set_ylabel("USDT equity")
        b.plot(t, equity.drawdown * 100)
        b.set_ylabel("Drawdown %")
        fig.tight_layout()
        fig.savefig(out / "equity_drawdown.png", dpi=145)
        plt.close(fig)
    return result

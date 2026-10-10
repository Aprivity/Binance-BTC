"""Strategy-agnostic performance statistics, CSV/PNG reports and BTC baseline."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import STEPS


def metrics(book,cfg):
    curve=pd.DataFrame(book.curve)
    ending=book.equity() if curve.empty else float(curve.equity.iloc[-1])
    pnl=np.asarray([t["pnl_usdt"] for t in book.trades],dtype=float)
    wins=pnl[pnl>0].sum();losses=-pnl[pnl<0].sum()
    returns=curve.equity.pct_change().dropna() if not curve.empty else pd.Series(dtype=float)
    periods=365*(86_400_000//STEPS[cfg.timeframe])
    sharpe=sortino=volatility=None
    if len(returns)>=3:
        std=returns.std(ddof=1)
        if std>0:
            sharpe=float(returns.mean()/std*np.sqrt(periods))
        downside=float(np.sqrt(np.mean(np.square(np.minimum(returns.to_numpy(dtype=float),0)))))
        if downside>0:
            sortino=float(returns.mean()/downside*np.sqrt(periods))
        volatility=float(std*np.sqrt(periods)*100)
    return {"starting_usdt":cfg.starting_usdt,"ending_usdt":round(ending,4),
            "net_return_pct":round((ending/cfg.starting_usdt-1)*100,4),
            "max_drawdown_pct":round(100*min((x["drawdown"] for x in book.curve),default=0),4),
            "closed_trades":len(pnl),
            "win_rate_pct":round(float((pnl>0).mean())*100,2) if len(pnl) else None,
            "profit_factor":round(float(wins/losses),4) if losses else None,
            "sharpe":round(sharpe,4) if sharpe is not None else None,
            "sortino":round(sortino,4) if sortino is not None else None,
            "annualized_volatility_pct":round(volatility,4) if volatility is not None else None,
            "fees_paid_usdt":round(book.fees,4),
            "slippage_estimate_usdt":round(book.slippage,4),
            "realized_pnl_usdt":round(float(pnl.sum()),4),
            "pnl_by_symbol":{"BTC/USDT":round(float(pnl.sum()),4)},
            "open_quantities":{"BTC/USDT":book.positions["BTC/USDT"].quantity if "BTC/USDT" in book.positions else 0.0},
            "halted":book.halted,"halted_at_utc":book.halted_at,
            "note":"Simulated mark-to-market equity; no exit cost until sale, no guaranteed fills."}


def write_report(book,cfg,output_dir,plot=True):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    summary=metrics(book,cfg)
    (out/"metrics.json").write_text(json.dumps({"config":cfg.to_dict(),"metrics":summary},indent=2),encoding="utf-8")
    pd.DataFrame(book.trades).to_csv(out/"trades.csv",index=False)
    pd.DataFrame(book.events).to_csv(out/"events.csv",index=False)
    equity=pd.DataFrame(book.curve);equity.to_csv(out/"equity.csv",index=False)
    if plot and not equity.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,(a,b)=plt.subplots(2,1,figsize=(10,6),sharex=True)
        t=pd.to_datetime(equity.time,utc=True)
        a.plot(t,equity.equity);a.set_ylabel("USDT equity")
        b.plot(t,equity.drawdown*100);b.set_ylabel("Drawdown %")
        fig.tight_layout();fig.savefig(out/"equity_drawdown.png",dpi=145);plt.close(fig)
    return summary


def market_analysis(history,cfg,book):
    """Same BTC-only capped-exposure buy-and-hold baseline, not a default strategy."""
    if not book.curve:
        return {}
    prices=history["BTC/USDT"]
    first=int(pd.Timestamp(book.curve[0]["time"]).timestamp()*1000)
    last=int(pd.Timestamp(book.curve[-1]["time"]).timestamp()*1000)
    frame=prices[(prices.close_time_ms>=first) & (prices.open_time_ms<=last)]
    if frame.empty:
        return {}
    weight=min(cfg.max_total_pct,cfg.max_symbol_pct)/100
    cost=cfg.starting_usdt*weight
    entry=float(frame.iloc[0].open)*(1+cfg.slip)
    qty=cost/(entry*(1+cfg.fee))
    final=cfg.starting_usdt-cost+qty*float(frame.iloc[-1].close)
    return {"benchmark":"BTC-only hold with policy exposure cap, entry fee/slippage, no exit cost",
            "benchmark_exposure_pct":weight*100,
            "benchmark_ending_usdt":round(final,4),
            "benchmark_return_pct":round(100*(final/cfg.starting_usdt-1),4),
            "strategy_minus_benchmark_pct_points":round(100*(book.equity()-final)/cfg.starting_usdt,4)}
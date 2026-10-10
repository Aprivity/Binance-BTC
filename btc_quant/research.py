"""Generic chronological walk-forward validation for explicitly loaded BTC strategies."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from .config import STEPS
from .core import backtest
from .report import metrics


def rolling_validate(history, cfg, *, signal_fn, strategy_ref, train_days=180,
                     test_days=60, step_days=60, output_dir="outputs/rolling-framework", plot=True):
    if signal_fn is None or not strategy_ref:
        raise ValueError("Rolling test requires explicit strategy")
    if train_days<45 or test_days<14 or step_days<test_days:
        raise ValueError("Require >=45d train, >=14d test, and nonoverlapping test folds")
    if set(history)!={"BTC/USDT"}:
        raise ValueError("Only BTC/USDT is supported")
    df=history["BTC/USDT"]
    stamp=df.open_time_ms.astype(int).tolist()
    interval=STEPS[cfg.timeframe]
    if len(stamp)<cfg.warmup_bars+20 or any(b-a!=interval for a,b in zip(stamp,stamp[1:])):
        raise ValueError("Expected contiguous BTC historical data")
    boundary=stamp[-1]+interval
    windows=[]
    anchor=stamp[cfg.warmup_bars]
    while anchor+(train_days+test_days)*86_400_000<=boundary:
        windows.append((anchor,anchor+train_days*86_400_000,
                        anchor+(train_days+test_days)*86_400_000))
        anchor+=step_days*86_400_000
    if not windows:
        raise ValueError("Insufficient data for walk-forward windows")
    rows=[]
    for fold,(first,split,last) in enumerate(windows,1):
        for stage,start,end in (("train",first,split),("test",split,last)):
            book=backtest(history,cfg,signal_fn=signal_fn,start_ms=start,end_ms=end)
            rows.append({"fold":fold,"stage":stage,"start_utc":pd.to_datetime(start,unit="ms",utc=True).isoformat(),
                         "end_exclusive_utc":pd.to_datetime(end,unit="ms",utc=True).isoformat(),
                         "halted_at_utc":book.halted_at,**metrics(book,cfg)})
    result=pd.DataFrame(rows)
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/"rolling_folds.csv",index=False)
    test=result[result.stage=="test"]
    summary={"folds":len(windows),"mean_return_pct":float(test.net_return_pct.mean()),
             "median_return_pct":float(test.net_return_pct.median()),
             "positive_folds":int((test.net_return_pct>0).sum()),
             "worst_return_pct":float(test.net_return_pct.min()),
             "strategy_ref":strategy_ref,"method":"fixed rules; no auto-fitting; fresh simulated accounts"}
    (out/"rolling_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,4))
        ax.bar(test.fold,test.net_return_pct)
        ax.axhline(0,color="gray",linewidth=0.8)
        ax.set(xlabel="Test fold (new account)",ylabel="Net return %")
        fig.tight_layout()
        fig.savefig(out/"rolling_returns.png",dpi=145)
        plt.close(fig)
    return result,summary
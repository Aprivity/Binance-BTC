"""15m BTC USD-M research only: normalized MACD + VO + BB; no exchange actions.

Independent of the repository's spot/long-only account. All entries use the next
bar's open. Futures here means a synthetic 1x-capped P&L ledger, not a margin/
liquidation simulator. No keys or authenticated/execution endpoints.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

BAR_MS = 15 * 60_000
MODES = ("fixed_1_5", "fixed_2", "split")


def indicators(candles: pd.DataFrame) -> pd.DataFrame:
    """TV Pine analogues: SMA20/stdev0; EMA(13,21), 50-range, WMA9; VO100."""
    x = candles.copy().reset_index(drop=True)
    for key in ("open_time_ms", "open", "high", "low", "close"):
        if key not in x:
            raise ValueError(f"Missing {key}")
    if (x[["open", "high", "low", "close"]].isna().any().any()
            or (x.high < x[["open", "close", "low"]].max(axis=1)).any()
            or (x.low > x[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Invalid OHLC")
    if len(x) < 201 or (x.open_time_ms.diff().iloc[1:] != BAR_MS).any():
        raise ValueError("Need >=201 contiguous 15m bars")
    cl = x.close.astype(float)
    op = x.open.astype(float)
    x["bb_middle"] = cl.rolling(20, min_periods=20).mean()
    x["bb_std"] = cl.rolling(20, min_periods=20).std(ddof=0)
    fast = cl.ewm(span=13, adjust=False).mean()
    slow = cl.ewm(span=21, adjust=False).mean()
    ratio = np.minimum(fast, slow) / np.maximum(fast, slow)
    mac = pd.Series(np.where(fast > slow, 2 - ratio, ratio) - 1, index=x.index)
    lo = mac.rolling(50, min_periods=50).min()
    hi = mac.rolling(50, min_periods=50).max()
    x["nmacd"] = (mac - lo) / (hi - lo + 0.000001) * 2 - 1
    weights = np.arange(1.0, 10.0)
    x["nmacd_trigger"] = x.nmacd.rolling(9, min_periods=9).apply(
        lambda a: float(np.dot(a, weights) / weights.sum()), raw=True)
    x["vo_spike"] = cl - op
    x["vo_upper"] = x.vo_spike.rolling(100, min_periods=100).std(ddof=0)
    ready = x.index >= 200
    x["long_signal"] = (ready & (x.nmacd > x.nmacd_trigger)
                         & (x.vo_spike > x.vo_upper)
                         & (op <= x.bb_middle) & (cl > x.bb_middle)).fillna(False)
    x["short_signal"] = (ready & (x.nmacd < x.nmacd_trigger)
                          & (x.vo_spike < -x.vo_upper)
                          & (op >= x.bb_middle) & (cl < x.bb_middle)).fillna(False)
    return x


@dataclass
class Position:
    side: int
    quantity: float
    remaining: float
    entry: float
    stop: float
    r: float
    time: int
    entry_fee: float
    realized: float = 0.0
    funding_cashflow: float = 0.0
    first_taken: bool = False
    exit_reason: str = ""


def simulate(frame: pd.DataFrame, funding: pd.DataFrame, *, mode: str,
             direction: str = "both", initial_cash: float = 1000.0,
             fee_bps: float = 5.0, slippage_bps: float = 2.0,
             risk_pct: float = 1.0, start_ms: int | None = None,
             end_ms: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Sequential 1x-capped isolated *research ledger*; pessimistic stop-first OHLC.

    Funding transfers are approximated from event rates and bar OPEN price.
    Funding event at a bar open applies to positions carried into that bar;
    positions opened at the same timestamp are assumed opened after settlement.
    """
    if mode not in MODES or direction not in ("both", "long", "short"):
        raise ValueError("Unknown mode/direction")
    if not (initial_cash > 0 and fee_bps >= 0 and slippage_bps >= 0
            and 0 < risk_pct <= 1 and fee_bps <= 100 and slippage_bps <= 100):
        raise ValueError("Invalid risk, cash, fee or slippage")
    for col in ("open_time_ms", "open", "high", "low", "close",
                "long_signal", "short_signal"):
        if col not in frame:
            raise ValueError(f"Missing {col}")
    x = frame.reset_index(drop=True)
    ts = x.open_time_ms.to_numpy(dtype=np.int64)
    if len(x) < 2 or np.any(np.diff(ts) != BAR_MS):
        raise ValueError("15m data must be contiguous")
    if not {"funding_ms", "rate"}.issubset(funding.columns):
        raise ValueError("Funding input requires funding_ms,rate")
    if funding.funding_ms.duplicated().any():
        raise ValueError("Duplicate funding timestamp")
    rates = dict(zip(funding.funding_ms.astype(np.int64), funding.rate.astype(float)))
    if any(not np.isfinite(v) for v in rates.values()):
        raise ValueError("Non-finite funding rate")
    if any(int(k) % BAR_MS for k in rates):
        raise ValueError("Funding timestamps must align to 15m candle opens")

    fee = fee_bps / 10000.0
    slip = slippage_bps / 10000.0
    cash = float(initial_cash)
    pos: Position | None = None
    paid_fees = 0.0
    funding_total = 0.0
    trades: list[dict] = []
    curve: list[dict] = []
    peak = cash
    max_dd = 0.0

    def exit_piece(raw: float, quantity: float, reason: str, moment: int) -> None:
        nonlocal cash, pos, paid_fees
        assert pos is not None
        filled = raw * (1 - pos.side * slip)
        commission = quantity * filled * fee
        cash_change = pos.side * quantity * (filled - pos.entry) - commission
        cash += cash_change
        paid_fees += commission
        pos.realized += cash_change
        pos.remaining -= quantity
        pos.exit_reason = reason
        if pos.remaining <= pos.quantity * 1e-10:
            pnl = pos.realized - pos.entry_fee + pos.funding_cashflow
            trades.append({"entry_ms": pos.time, "exit_ms": moment,
                           "side": "long" if pos.side == 1 else "short",
                           "entry_price": pos.entry, "exit_price_last": filled,
                           "stop_initial": pos.stop, "quantity": pos.quantity,
                           "risk_price": pos.r, "net_pnl": pnl,
                           "net_r": pnl / (pos.quantity * pos.r),
                           "funding_pnl": pos.funding_cashflow,
                           "exit_reason": reason})
            pos = None

    for i in range(1, len(x)):
        t = int(ts[i])
        if start_ms is not None and t < start_ms:
            continue
        if end_ms is not None and t >= end_ms:
            break
        bar = x.iloc[i]
        prev = x.iloc[i - 1]
        carried = pos is not None
        # Known settlement at candle open. Never charge a not-yet-opened trade.
        if carried and t in rates:
            flow = -pos.side * pos.remaining * float(bar.open) * rates[t]
            cash += flow
            pos.funding_cashflow += flow
            funding_total += flow
        if not carried and cash > 0:
            side = (1 if bool(prev.long_signal) and direction != "short" else
                    -1 if bool(prev.short_signal) and direction != "long" else 0)
            if side:
                raw_entry = float(bar.open)
                entry = raw_entry * (1 + side * slip)
                stop = float(prev.low if side == 1 else prev.high)
                risk_distance = side * (entry - stop)
                # The modeled stop includes adverse exit slippage and both fees.
                stop_filled = stop * (1 - side * slip)
                unit_loss = side * (entry - stop_filled) + fee * (entry + stop_filled)
                if risk_distance > 0 and unit_loss > 0:
                    qty = min(cash * risk_pct / 100 / unit_loss,
                              cash / (entry * (1 + fee)))
                    if qty * entry >= 10:
                        opening_fee = qty * entry * fee
                        cash -= opening_fee
                        paid_fees += opening_fee
                        pos = Position(side, qty, qty, entry, stop, risk_distance,
                                       t, opening_fee)
        if pos is not None:
            s = pos.side
            op = float(bar.open)
            hi = float(bar.high)
            lo = float(bar.low)
            stop_hit = (op <= pos.stop or lo <= pos.stop) if s == 1 else (
                op >= pos.stop or hi >= pos.stop)
            target_r = 1.5 if mode != "fixed_2" else 2.0
            target_first = pos.entry + s * target_r * pos.r
            first_hit = (op >= target_first or hi >= target_first) if s == 1 else (
                op <= target_first or lo <= target_first)
            # If stop and take are both touched in a bar, assume stop FIRST.
            if stop_hit:
                adverse_gap = (op < pos.stop if s == 1 else op > pos.stop)
                exit_piece(op if adverse_gap else pos.stop, pos.remaining,
                           "gap_stop" if adverse_gap else "stop", t)
            elif mode != "split" and first_hit:
                exit_piece(target_first, pos.remaining, "take", t)
            elif mode == "split":
                if not pos.first_taken and first_hit:
                    part = 0.5 * pos.quantity
                    pos.first_taken = True
                    exit_piece(target_first, part, "take_1_5R", t)
                if pos is not None:
                    target_second = pos.entry + s * 2.0 * pos.r
                    second_hit = (op >= target_second or hi >= target_second) if s == 1 else (
                        op <= target_second or lo <= target_second)
                    if second_hit:
                        exit_piece(target_second, pos.remaining, "take_2R", t)
        unrealized = (pos.side * pos.remaining * (float(bar.close) - pos.entry)
                      if pos is not None else 0.0)
        equity = cash + unrealized
        peak = max(peak, equity)
        dd = 1 - equity / peak if peak > 0 else 1.0
        max_dd = max(max_dd, dd)
        curve.append({"time_ms": t, "equity": equity, "drawdown_pct": dd * 100})

    if pos is not None and curve:
        # End-of-test liquidation; account for exit fee/slippage.
        exit_piece(float(x.loc[x.open_time_ms == curve[-1]["time_ms"], "close"].iloc[0]),
                   pos.remaining, "end_of_test", curve[-1]["time_ms"] + BAR_MS - 1)
        curve[-1]["equity"] = cash
        peak = max(peak, cash)
        max_dd = max(max_dd, (1 - cash / peak) if peak else 1)
        curve[-1]["drawdown_pct"] = (1 - cash / peak) * 100 if peak else 100.0
    tr = pd.DataFrame(trades)
    eq = pd.DataFrame(curve)
    winners = tr.net_pnl[tr.net_pnl > 0] if len(tr) else pd.Series(dtype=float)
    losers = tr.net_pnl[tr.net_pnl < 0] if len(tr) else pd.Series(dtype=float)
    profit_factor = (float(winners.sum() / -losers.sum()) if len(losers)
                     else None)  # No losers => undefined, not a fabricated score
    summary = {"mode": mode, "direction": direction, "initial_usdt": initial_cash,
               "final_usdt": round(cash, 6),
               "return_pct": round(100 * (cash / initial_cash - 1), 4),
               "max_drawdown_pct": round(100 * max_dd, 4),
               "trades": len(tr),
               "win_rate_pct": round(100 * len(winners) / len(tr), 4) if len(tr) else None,
               "profit_factor": round(profit_factor, 5) if profit_factor is not None else None,
               "mean_net_r": round(float(tr.net_r.mean()), 5) if len(tr) else None,
               "fees_usdt": round(paid_fees, 6),
               "funding_usdt": round(funding_total, 6),
               "fee_bps_per_side": fee_bps, "slippage_bps_per_side": slippage_bps,
               "risk_pct": risk_pct, "gross_leverage_cap": 1.0}
    return tr, eq, summary

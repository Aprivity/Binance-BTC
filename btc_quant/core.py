"""Deterministic long-only multi-asset portfolio, indicators and next-open backtest."""
from dataclasses import dataclass, asdict, field
import math
import pandas as pd
from .config import Config


def signals(df, cfg):
    x = df.copy()
    close = x.close
    fast = close.ewm(span=cfg.ema_fast, min_periods=cfg.ema_fast, adjust=False).mean()
    slow = close.ewm(span=cfg.ema_slow, min_periods=cfg.ema_slow, adjust=False).mean()
    prior = close.shift(1)
    tr = pd.concat([x.high - x.low, (x.high - prior).abs(), (x.low - prior).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / cfg.atr_period, min_periods=cfg.atr_period, adjust=False).mean()
    ready = fast.notna() & slow.notna() & fast.shift(1).notna() & slow.shift(1).notna() & atr.notna()
    x["atr"] = atr
    x["buy_signal"] = (ready & (fast > slow) & (fast.shift(1) <= slow.shift(1))).fillna(False)
    x["sell_signal"] = (ready & (fast < slow) & (fast.shift(1) >= slow.shift(1))).fillna(False)
    return x


@dataclass
class Position:
    quantity: float
    entry_price: float
    entry_cost: float
    entry_time: str
    stop: float
    take: float


@dataclass
class Account:
    cash: float
    positions: dict = field(default_factory=dict)
    prices: dict = field(default_factory=dict)
    peak: float = 0.0
    halted: bool = False
    fees: float = 0.0
    slippage: float = 0.0
    trades: list = field(default_factory=list)
    events: list = field(default_factory=list)
    curve: list = field(default_factory=list)

    @classmethod
    def create(cls, cfg):
        return cls(cash=cfg.starting_usdt, peak=cfg.starting_usdt)

    @classmethod
    def restore(cls, record):
        d = record.copy()
        d["positions"] = {s: Position(**p) for s, p in d["positions"].items()}
        return cls(**d)

    def serialize(self):
        return asdict(self)

    def equity(self):
        return self.cash + sum(p.quantity * self.prices[s] for s, p in self.positions.items())

    def mark(self, when, cfg):
        value = self.equity()
        self.peak = max(self.peak, value)
        dd = value / self.peak - 1 if self.peak else 0
        if dd <= -cfg.max_drawdown_pct / 100:
            self.halted = True
        self.curve.append({"time": when, "equity": value, "drawdown": dd})

    def open_risk(self, cfg):
        total = 0.0
        for sym, pos in self.positions.items():
            exit_price = pos.stop * (1 - cfg.slip)
            total += max(0.0, pos.entry_cost - pos.quantity * exit_price * (1 - cfg.fee))
        return total

    def buy(self, symbol, when, raw_price, atr, cfg, *, cash_cap=None):
        if self.halted or symbol in self.positions or not math.isfinite(atr) or atr <= 0:
            return False
        price = raw_price * (1 + cfg.slip)
        stop, take = price - cfg.stop_atr * atr, price + cfg.take_atr * atr
        if stop <= 0:
            return False
        per_unit_risk = price * (1 + cfg.fee) - stop * (1 - cfg.slip) * (1 - cfg.fee)
        if per_unit_risk <= 0:
            return False
        eq = self.equity()
        room = eq * cfg.max_open_risk_pct / 100 - self.open_risk(cfg)
        if room <= 0:
            return False
        exposure = sum(p.quantity * self.prices[s] for s, p in self.positions.items())
        limit = min(eq * cfg.risk_pct / 100 / per_unit_risk,
                    room / per_unit_risk,
                    eq * cfg.max_symbol_pct / 100 / (price * (1 + cfg.fee)),
                    max(0.0, eq * cfg.max_total_pct / 100 - exposure) / (price * (1 + cfg.fee)),
                    min(self.cash, self.cash if cash_cap is None else cash_cap) / (price * (1 + cfg.fee)))
        if limit <= 0 or limit * price < cfg.min_notional:
            return False
        notional, fee = limit * price, limit * price * cfg.fee
        self.cash -= notional + fee
        self.prices[symbol] = raw_price
        self.positions[symbol] = Position(limit, price, notional + fee, when, stop, take)
        self.fees += fee
        self.slippage += limit * raw_price * cfg.slip
        self.events.append({"time": when, "symbol": symbol, "side": "BUY", "price": price,
                            "quantity": limit, "fee": fee, "reason": "ema_cross_up"})
        return True

    def sell(self, symbol, when, raw_price, cfg, reason):
        if symbol not in self.positions:
            return False
        pos = self.positions.pop(symbol)
        fill = raw_price * (1 - cfg.slip)
        fee = pos.quantity * fill * cfg.fee
        pnl = pos.quantity * fill - fee - pos.entry_cost
        self.cash += pos.quantity * fill - fee
        self.fees += fee
        self.slippage += pos.quantity * raw_price * cfg.slip
        self.events.append({"time": when, "symbol": symbol, "side": "SELL", "price": fill,
                            "quantity": pos.quantity, "fee": fee, "reason": reason, "pnl_usdt": pnl})
        self.trades.append({"symbol": symbol, "entry_time": pos.entry_time, "exit_time": when,
                            "entry_price": pos.entry_price, "exit_price": fill,
                            "quantity": pos.quantity, "pnl_usdt": pnl, "reason": reason})
        return True

    def protective_exit(self, sym, opening, high, low):
        if sym not in self.positions:
            return None
        pos = self.positions[sym]
        if opening <= pos.stop:
            return opening, "gap_stop"
        if opening >= pos.take:
            return pos.take, "take_profit"
        if low <= pos.stop:
            return pos.stop, "stop_loss"  # pessimistic on ambiguous bar
        if high >= pos.take:
            return pos.take, "take_profit"
        return None


def backtest(history, cfg, days=None):
    """Shared cash; signals from prior closed bar; no look-ahead or overlapping candles."""
    if set(history) != set(cfg.symbols):
        raise ValueError("History must match symbols")
    enrich = {s: signals(history[s], cfg).set_index("open_time_ms", drop=False) for s in cfg.symbols}
    common = sorted(set.intersection(*(set(x.index) for x in enrich.values())))
    if len(common) < max(220, cfg.ema_slow + 3):
        raise ValueError("Insufficient common, contiguous history")
    from .config import STEPS
    if any(b - a != STEPS[cfg.timeframe] for a, b in zip(common, common[1:])):
        raise ValueError("Misaligned / missing candle across symbols")
    rows = {s: enrich[s].loc[common] for s in cfg.symbols}
    account = Account.create(cfg)
    beginning = common[0] + 200 * STEPS[cfg.timeframe]
    if days:
        beginning = max(beginning, common[-1] - days * 86_400_000)
    for i in range(1, len(common)):
        ts = common[i]
        if ts < beginning:
            continue
        when = pd.to_datetime(ts, unit="ms", utc=True).isoformat()
        for sym in sorted(cfg.symbols):
            account.prices[sym] = float(rows[sym].iloc[i].open)
        if account.equity() <= account.peak * (1 - cfg.max_drawdown_pct / 100):
            account.halted = True
        # Stops from open, then exit signal. Never buy based on current candle.
        exited = set()
        for sym in sorted(cfg.symbols):
            bar, prev = rows[sym].iloc[i], rows[sym].iloc[i-1]
            pos = account.positions.get(sym)
            if pos and float(bar.open) <= pos.stop:
                account.sell(sym, when, float(bar.open), cfg, "gap_stop")
                exited.add(sym)
            elif pos and bool(prev.sell_signal):
                account.sell(sym, when, float(bar.open), cfg, "ema_cross_down")
                exited.add(sym)
        # Allocate equal cash reservation for simultaneous entries; stable symbol order.
        # Same-bar exits must not re-enter, even if an unrelated buy flag exists.
        candidates = [s for s in sorted(cfg.symbols) if s not in account.positions and bool(rows[s].iloc[i-1].buy_signal) and s not in exited]
        budget = account.cash / len(candidates) if candidates else 0
        for sym in candidates:
            account.buy(sym, when, float(rows[sym].iloc[i].open), float(rows[sym].iloc[i-1].atr), cfg, cash_cap=budget)
        for sym in sorted(cfg.symbols):
            bar = rows[sym].iloc[i]
            trigger = account.protective_exit(sym, float(bar.open), float(bar.high), float(bar.low))
            if trigger:
                account.sell(sym, when, trigger[0], cfg, trigger[1])
            account.prices[sym] = float(bar.close)
        account.mark(pd.to_datetime(ts + STEPS[cfg.timeframe] - 1, unit="ms", utc=True).isoformat(), cfg)
    return account

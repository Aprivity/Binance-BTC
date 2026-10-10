"""Deterministic BTC spot simulator: no strategy is bundled or selected by default."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import math
import pandas as pd
from .config import STEPS
from .strategy import prepare_signals


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
    halted_at: str | None = None
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
        data = dict(record)
        data["positions"] = {sym: Position(**pos) for sym, pos in data["positions"].items()}
        return cls(**data)

    def serialize(self):
        return asdict(self)

    def equity(self):
        return self.cash + sum(p.quantity * self.prices[s] for s, p in self.positions.items())

    def check_halt(self, when, cfg):
        if self.equity() <= self.peak * (1 - cfg.max_drawdown_pct/100):
            if not self.halted:
                self.halted_at = when
            self.halted = True

    def mark(self, when, cfg):
        eq = self.equity()
        self.peak = max(self.peak, eq)
        drawdown = eq / self.peak - 1 if self.peak else 0
        self.check_halt(when, cfg)
        self.curve.append({"time": when, "equity": eq, "drawdown": drawdown})

    def open_risk(self, cfg):
        total = 0.0
        for sym, pos in self.positions.items():
            exit_price = pos.stop * (1-cfg.slip)
            total += max(0.0, pos.entry_cost - pos.quantity * exit_price * (1-cfg.fee))
        return total

    def buy(self, symbol, when, raw_price, atr, cfg, *, cash_cap=None, reason="plugin_buy"):
        if (symbol not in cfg.symbols or self.halted or symbol in self.positions or
                not math.isfinite(atr) or atr <= 0 or not math.isfinite(raw_price) or raw_price <= 0):
            return False
        price = raw_price * (1+cfg.slip)
        stop, take = price-cfg.stop_atr*atr, price+cfg.take_atr*atr
        if stop <= 0:
            return False
        per_unit_risk = price*(1+cfg.fee) - stop*(1-cfg.slip)*(1-cfg.fee)
        if per_unit_risk <= 0:
            return False
        eq = self.equity()
        room = eq*cfg.max_open_risk_pct/100 - self.open_risk(cfg)
        if room <= 0:
            return False
        exposure = sum(p.quantity*self.prices[s] for s,p in self.positions.items())
        limit = min(eq*cfg.risk_pct/100/per_unit_risk,
                    room/per_unit_risk,
                    eq*cfg.max_symbol_pct/100/(price*(1+cfg.fee)),
                    max(0.0,eq*cfg.max_total_pct/100-exposure)/(price*(1+cfg.fee)),
                    min(self.cash, self.cash if cash_cap is None else cash_cap)/(price*(1+cfg.fee)))
        if limit <= 0 or limit*price < cfg.min_notional:
            return False
        notional, fee = limit*price, limit*price*cfg.fee
        self.cash -= notional + fee
        self.prices[symbol] = raw_price
        self.positions[symbol] = Position(limit, price, notional+fee, when, stop, take)
        self.fees += fee
        self.slippage += limit*raw_price*cfg.slip
        self.events.append({"time": when,"symbol":symbol,"side":"BUY","price":price,
                            "quantity":limit,"fee":fee,"reason":reason})
        return True

    def sell(self, symbol, when, raw_price, cfg, reason):
        if symbol not in self.positions:
            return False
        pos = self.positions.pop(symbol)
        fill = raw_price*(1-cfg.slip)
        fee = pos.quantity*fill*cfg.fee
        pnl = pos.quantity*fill-fee-pos.entry_cost
        self.cash += pos.quantity*fill-fee
        self.fees += fee
        self.slippage += pos.quantity*raw_price*cfg.slip
        self.events.append({"time":when,"symbol":symbol,"side":"SELL","price":fill,
                            "quantity":pos.quantity,"fee":fee,"reason":reason,"pnl_usdt":pnl})
        self.trades.append({"symbol":symbol,"entry_time":pos.entry_time,"exit_time":when,
                            "entry_price":pos.entry_price,"exit_price":fill,"quantity":pos.quantity,
                            "pnl_usdt":pnl,"reason":reason})
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
            return pos.stop, "stop_loss"  # pessimistic when OHLC path ambiguous
        if high >= pos.take:
            return pos.take, "take_profit"
        return None


def backtest(history, cfg, *, signal_fn=None, days=None, start_ms=None, end_ms=None):
    """Explicit plugin only; signal at closed candle executes next-bar open.

    Start/end are [start_ms, end_ms), with fresh accounts and no carry-in signal.
    """
    if signal_fn is None:
        raise ValueError("Backtest requires an explicit strategy plugin; none is installed by default")
    if set(history) != {"BTC/USDT"}:
        raise ValueError("BTC-only history required")
    x = prepare_signals(history["BTC/USDT"], cfg, signal_fn)
    timestamps = x.open_time_ms.astype(int).to_list()
    step = STEPS[cfg.timeframe]
    if len(timestamps) < cfg.warmup_bars + 20 or any(b-a!=step for a,b in zip(timestamps,timestamps[1:])):
        raise ValueError("Need aligned, contiguous closed BTC OHLCV and warmup")
    beginning = timestamps[cfg.warmup_bars]
    if days is not None:
        beginning = max(beginning, timestamps[-1] - days*86_400_000)
    if start_ms is not None:
        beginning = max(beginning, int(start_ms))
    if end_ms is not None and int(end_ms) <= beginning:
        raise ValueError("Empty backtest window")
    book = Account.create(cfg)
    symbol = "BTC/USDT"
    for i in range(cfg.warmup_bars,len(timestamps)):
        ts = timestamps[i]
        if ts < beginning:
            continue
        if end_ms is not None and ts >= int(end_ms):
            break
        bar, prev = x.iloc[i],x.iloc[i-1]
        when = pd.to_datetime(ts, unit="ms", utc=True).isoformat()
        book.prices[symbol] = float(bar.open)
        book.check_halt(when,cfg)
        exited = False
        if symbol in book.positions:
            pos=book.positions[symbol]
            if float(bar.open) <= pos.stop:
                book.sell(symbol,when,float(bar.open),cfg,"gap_stop")
                exited=True
            elif bool(prev.sell_signal):
                book.sell(symbol,when,float(bar.open),cfg,"plugin_exit")
                exited=True
        # Skip first candle of freshly reset fold; previous signal is outside window.
        prev_inside = timestamps[i-1]>=beginning if start_ms is not None else True
        if (not exited and prev_inside and bool(prev.buy_signal) and symbol not in book.positions):
            book.buy(symbol,when,float(bar.open),float(prev.atr),cfg)
        trigger=book.protective_exit(symbol,float(bar.open),float(bar.high),float(bar.low))
        if trigger:
            book.sell(symbol,when,trigger[0],cfg,trigger[1])
        book.prices[symbol]=float(bar.close)
        book.mark(pd.to_datetime(ts+step-1,unit="ms",utc=True).isoformat(),cfg)
    return book
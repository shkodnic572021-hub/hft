"""
Контроль риска и качества исполнений.

MarkoutTracker — главный инструмент оценки прибыльности маркет-мейкера.
Для каждого исполнения смотрим, где mid через 1/5/30 секунд:
  markout_bps = знак · (mid_через_h − цена_исполнения) / цена · 1e4
  net_bps     = markout_bps − комиссия
Положительный net — заработали спред, отрицательный — нас «переехал»
информированный поток (adverse selection). По скользящему среднему net
решаем, токсична ли монета, а измеренный adverse расширяет спред.
"""

import csv
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class PendingFill:
    ts: float
    symbol: str
    side: str          # BUY / SELL
    price: float
    qty: float
    fee_bps: float
    mid_at_fill: float
    marks: Dict[float, float] = field(default_factory=dict)


class MarkoutTracker:
    def __init__(self, horizons: List[float], toxic_horizon: float,
                 window: int, adverse_halflife_fills: float,
                 csv_path: Optional[str] = None):
        self.horizons = sorted(horizons)
        if toxic_horizon not in self.horizons:
            self.horizons = sorted(self.horizons + [toxic_horizon])
        self.toxic_h = toxic_horizon
        self.pending: Deque[PendingFill] = deque()
        self.net: Dict[str, Deque[float]] = {}
        self.adverse: Dict[str, float] = {}
        self.n_fills: Dict[str, int] = {}
        self._adv_alpha = 1 - 0.5 ** (1 / max(adverse_halflife_fills, 1))
        self.window = window
        self.csv_path = csv_path
        if csv_path and not os.path.exists(csv_path):
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["ts", "symbol", "side", "price", "qty", "fee_bps", "mid_at_fill"]
                    + [f"markout_{h:g}s" for h in self.horizons]
                )

    def on_fill(self, symbol: str, side: str, price: float, qty: float,
                fee_bps: float, mid: float, ts: Optional[float] = None):
        self.pending.append(PendingFill(
            time.time() if ts is None else ts, symbol, side, price, qty, fee_bps, mid
        ))

    def update(self, mids: Dict[str, float], now: Optional[float] = None):
        """Вызывать регулярно: проставляет markout для созревших горизонтов."""
        now = time.time() if now is None else now
        done = []
        for pf in self.pending:
            mid = mids.get(pf.symbol, 0.0)
            if mid <= 0:
                if now - pf.ts > self.horizons[-1] * 3:
                    done.append(pf)   # монету убрали — не копим вечно
                continue
            sign = 1.0 if pf.side == "BUY" else -1.0
            for h in self.horizons:
                if h not in pf.marks and now - pf.ts >= h:
                    pf.marks[h] = sign * (mid - pf.price) / pf.price * 1e4
                    if h == self.toxic_h:
                        self._record(pf, mid, sign)
            if len(pf.marks) == len(self.horizons):
                done.append(pf)
        for pf in done:
            try:
                self.pending.remove(pf)
            except ValueError:
                pass
            self._write(pf)

    def _record(self, pf: PendingFill, mid_h: float, sign: float):
        net = pf.marks[self.toxic_h] - pf.fee_bps
        dq = self.net.setdefault(pf.symbol, deque(maxlen=self.window))
        dq.append(net)
        self.n_fills[pf.symbol] = self.n_fills.get(pf.symbol, 0) + 1
        # adverse: насколько mid ушёл против нас после исполнения (≥ 0)
        if pf.mid_at_fill > 0:
            move = sign * (mid_h - pf.mid_at_fill) / pf.mid_at_fill * 1e4
            adv = max(0.0, -move)
            prev = self.adverse.get(pf.symbol)
            self.adverse[pf.symbol] = adv if prev is None else \
                (1 - self._adv_alpha) * prev + self._adv_alpha * adv

    def _write(self, pf: PendingFill):
        if not self.csv_path:
            return
        try:
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [f"{pf.ts:.3f}", pf.symbol, pf.side, pf.price, pf.qty,
                     f"{pf.fee_bps:.2f}", pf.mid_at_fill]
                    + [f"{pf.marks[h]:.2f}" if h in pf.marks else "" for h in self.horizons]
                )
        except OSError:
            pass

    def avg_net(self, symbol: str) -> float:
        dq = self.net.get(symbol)
        return sum(dq) / len(dq) if dq else 0.0

    def count(self, symbol: str) -> int:
        dq = self.net.get(symbol)
        return len(dq) if dq else 0

    def adverse_bps(self, symbol: str) -> float:
        return self.adverse.get(symbol, 0.0)

    def reset(self, symbol: str):
        self.net.pop(symbol, None)
        self.adverse.pop(symbol, None)


class BtcShock:
    """Резкое движение BTC за короткое окно — пауза по всем монетам."""

    def __init__(self, move_bps: float, window_s: float):
        self.move_bps = move_bps
        self.window_s = window_s
        self.hist: Deque = deque()

    def on_mid(self, mid: float, now: float) -> bool:
        # bookTicker BTC приходит сотни раз в секунду — сэмплируем по 50 мс
        if not self.hist or now - self.hist[-1][0] >= 0.05:
            self.hist.append((now, mid))
        while self.hist and now - self.hist[0][0] > self.window_s:
            self.hist.popleft()
        lo = min(m for _, m in self.hist)
        hi = max(m for _, m in self.hist)
        return lo > 0 and (hi - lo) / lo * 1e4 >= self.move_bps


def symbol_shock(spread_bps: float, median_spread_bps: float,
                 sigma_fast: float, sigma_slow: float,
                 spread_mult: float, vol_mult: float) -> Optional[str]:
    if median_spread_bps > 0 and spread_bps > spread_mult * median_spread_bps:
        return f"spread {spread_bps:.1f}bps > {spread_mult:g}× медианы {median_spread_bps:.1f}"
    if sigma_slow > 0 and sigma_fast > vol_mult * sigma_slow:
        return f"σ {sigma_fast:.1f}bps > {vol_mult:g}× базовой {sigma_slow:.1f}"
    return None


class RateLimiter:
    """Скользящее окно 10 с для числа ордерных действий."""

    def __init__(self, max_per_10s: int):
        self.max = max_per_10s
        self.events: Deque[float] = deque()

    def allow(self, now: Optional[float] = None, reserve: int = 0) -> bool:
        now = time.time() if now is None else now
        while self.events and now - self.events[0] > 10.0:
            self.events.popleft()
        return len(self.events) + reserve < self.max

    def hit(self, now: Optional[float] = None):
        self.events.append(time.time() if now is None else now)


def unrealized_bps(position: float, avg_entry: float, mid: float) -> float:
    if position == 0 or avg_entry <= 0 or mid <= 0:
        return 0.0
    return math.copysign(1.0, position) * (mid - avg_entry) / avg_entry * 1e4

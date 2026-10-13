"""
Рыночные данные и сигналы.

Book — состояние одной монеты: лучшие цены, дисбаланс стакана, поток сделок,
волатильность, медианный спред. MarketData — WebSocket-подключения:
  /public  : <sym>@bookTicker (каждое изменение лучших цен), <sym>@depth5@100ms
  /market  : <sym>@aggTrade
"""

import asyncio
import json
import logging
import math
import time
from collections import deque
from statistics import median
from typing import Awaitable, Callable, Dict, List, Optional

import websockets

logger = logging.getLogger(__name__)


class Ewma:
    """Экспоненциальное среднее с полураспадом по времени."""

    def __init__(self, halflife_s: float):
        self.halflife = max(halflife_s, 1e-6)
        self.value = 0.0
        self.ts: Optional[float] = None

    def decay_to(self, now: float):
        if self.ts is not None and now > self.ts:
            self.value *= 0.5 ** ((now - self.ts) / self.halflife)
        self.ts = now

    def add(self, x: float, now: float):
        self.decay_to(now)
        self.value += x

    def get(self, now: float) -> float:
        self.decay_to(now)
        return self.value


class VolEstimator:
    """
    σ лог-доходностей mid за 1 секунду, в bps. Mid сэмплируется раз в секунду;
    EWMA дисперсии с полураспадом halflife_s (в секундах).
    """

    def __init__(self, halflife_s: float):
        self.alpha = 1.0 - 0.5 ** (1.0 / max(halflife_s, 1.0))
        self.var = 0.0
        self.n = 0
        self.last_mid = 0.0
        self.last_sec = 0

    def on_mid(self, mid: float, now: float):
        sec = int(now)
        if self.last_mid <= 0:
            self.last_mid, self.last_sec = mid, sec
            return
        if sec <= self.last_sec:
            return
        gap = sec - self.last_sec
        r = math.log(mid / self.last_mid) * 1e4
        # Если пропустили секунды — нормируем на число секунд
        r2 = (r * r) / gap
        self.var = r2 if self.n == 0 else (1 - self.alpha) * self.var + self.alpha * r2
        self.n += 1
        self.last_mid, self.last_sec = mid, sec

    @property
    def sigma_bps(self) -> float:
        return math.sqrt(self.var)

    @property
    def ready(self) -> bool:
        return self.n >= 30


class Book:
    def __init__(self, symbol: str, flow_halflife_s: float = 2.0):
        self.symbol = symbol
        self.bid = 0.0
        self.ask = 0.0
        self.bid_qty = 0.0
        self.ask_qty = 0.0
        self.ts = 0.0               # время последнего обновления bookTicker
        self.depth_imb = 0.0        # [-1..1], >0 — больше покупателей
        self.last_trade_px = 0.0
        self.buy_flow = Ewma(flow_halflife_s)
        self.sell_flow = Ewma(flow_halflife_s)
        self.vol_fast = VolEstimator(30)
        self.vol_slow = VolEstimator(600)
        self.spreads_bps: deque = deque(maxlen=300)   # 1 сэмпл/сек
        self._last_spread_sec = 0
        self.updated = asyncio.Event()

    @property
    def ready(self) -> bool:
        return self.bid > 0 and self.ask > 0 and self.ask > self.bid

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_bps(self) -> float:
        m = self.mid
        return (self.ask - self.bid) / m * 1e4 if m > 0 else 0.0

    @property
    def micro(self) -> float:
        """Microprice: mid, смещённый к стороне с меньшим объёмом на лучшем уровне."""
        tot = self.bid_qty + self.ask_qty
        if tot <= 0:
            return self.mid
        return (self.bid * self.ask_qty + self.ask * self.bid_qty) / tot

    def flow_imb(self, now: float) -> float:
        b = self.buy_flow.get(now)
        s = self.sell_flow.get(now)
        tot = b + s
        return (b - s) / tot if tot > 0 else 0.0

    @property
    def median_spread_bps(self) -> float:
        return median(self.spreads_bps) if len(self.spreads_bps) >= 30 else 0.0

    # --- обработчики сообщений ---

    def on_book_ticker(self, bid: float, bid_qty: float, ask: float, ask_qty: float, now: float):
        if bid <= 0 or ask <= 0:
            return
        self.bid, self.bid_qty, self.ask, self.ask_qty = bid, bid_qty, ask, ask_qty
        self.ts = now
        mid = self.mid
        self.vol_fast.on_mid(mid, now)
        self.vol_slow.on_mid(mid, now)
        sec = int(now)
        if sec != self._last_spread_sec:
            self._last_spread_sec = sec
            self.spreads_bps.append(self.spread_bps)
        self.updated.set()

    def on_depth(self, bids: List, asks: List):
        try:
            bv = sum(float(q) for _, q in bids[:5])
            av = sum(float(q) for _, q in asks[:5])
        except (ValueError, TypeError):
            return
        tot = bv + av
        self.depth_imb = (bv - av) / tot if tot > 0 else 0.0

    def on_trade(self, price: float, qty: float, buyer_is_maker: bool, now: float):
        self.last_trade_px = price
        notional = price * qty
        # buyer_is_maker=True → агрессор продавец
        if buyer_is_maker:
            self.sell_flow.add(notional, now)
        else:
            self.buy_flow.add(notional, now)


TradeCallback = Callable[[str, float, float, bool, float], Awaitable[None]]
BookCallback = Callable[[str], Awaitable[None]]


class MarketData:
    def __init__(self, public_url: str, market_url: str, flow_halflife_s: float = 2.0):
        self.public_url = public_url.rstrip("/")
        self.market_url = market_url.rstrip("/")
        self.flow_halflife_s = flow_halflife_s
        self.books: Dict[str, Book] = {}
        self._tasks: Dict[str, List[asyncio.Task]] = {}
        self.trade_listeners: List[TradeCallback] = []
        self.book_listeners: List[BookCallback] = []
        self.running = True

    def book(self, symbol: str) -> Book:
        if symbol not in self.books:
            self.books[symbol] = Book(symbol, self.flow_halflife_s)
        return self.books[symbol]

    def subscribe(self, symbol: str, trades: bool = True, depth: bool = True):
        if symbol in self._tasks:
            return
        self.book(symbol)
        s = symbol.lower()
        public = [f"{s}@bookTicker"] + ([f"{s}@depth5@100ms"] if depth else [])
        tasks = [asyncio.create_task(
            self._run(f"{self.public_url}/stream?streams={'/'.join(public)}", symbol)
        )]
        if trades:
            tasks.append(asyncio.create_task(
                self._run(f"{self.market_url}/stream?streams={s}@aggTrade", symbol)
            ))
        self._tasks[symbol] = tasks

    def unsubscribe(self, symbol: str):
        for t in self._tasks.pop(symbol, []):
            t.cancel()
        self.books.pop(symbol, None)

    async def close(self):
        self.running = False
        for sym in list(self._tasks):
            self.unsubscribe(sym)

    async def _run(self, url: str, symbol: str):
        while self.running:
            try:
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=20, close_timeout=3,
                    max_queue=None,
                ) as ws:
                    logger.info(f"📡 {symbol}: WS {url.split('?')[0]}")
                    async for raw in ws:
                        await self._dispatch(raw)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(f"{symbol} WS: {e!r}. Переподключение через 2 с")
                await asyncio.sleep(2)

    async def _dispatch(self, raw):
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        data = msg.get("data", msg)
        ev = data.get("e")
        now = time.time()
        if ev == "bookTicker":
            sym = data.get("s", "")
            book = self.books.get(sym)
            if book is None:
                return
            try:
                book.on_book_ticker(
                    float(data["b"]), float(data["B"]),
                    float(data["a"]), float(data["A"]), now,
                )
            except (KeyError, ValueError):
                return
            for cb in self.book_listeners:
                await cb(sym)
        elif ev == "depthUpdate":
            book = self.books.get(data.get("s", ""))
            if book is not None:
                book.on_depth(data.get("b") or [], data.get("a") or [])
        elif ev == "aggTrade":
            sym = data.get("s", "")
            book = self.books.get(sym)
            if book is None:
                return
            try:
                px, qty, bm = float(data["p"]), float(data["q"]), bool(data["m"])
            except (KeyError, ValueError):
                return
            book.on_trade(px, qty, bm, now)
            for cb in self.trade_listeners:
                await cb(sym, px, qty, bm, now)

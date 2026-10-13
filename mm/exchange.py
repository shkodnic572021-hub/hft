"""
Слой биржи: LiveExchange (Binance USDⓈ-M) и PaperExchange (симуляция).
Интерфейс одинаковый, поэтому стратегия не знает, в каком режиме работает.
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional, Tuple

import websockets
from binance import AsyncClient
from binance.exceptions import BinanceAPIException

from .config import BotConfig
from .strategy import SymbolFilters

logger = logging.getLogger(__name__)


@dataclass
class Fill:
    symbol: str
    side: str          # BUY / SELL
    qty: float
    price: float
    fee_usdt: float
    cid: str
    maker: bool
    ts: float


@dataclass
class PlaceResult:
    ok: bool
    code: int = 0
    msg: str = ""


FillCb = Callable[[Fill], Awaitable[None]]
StatusCb = Callable[[str, str, str], Awaitable[None]]   # symbol, cid, status

# Коды Binance
ERR_UNKNOWN_ORDER = -2011
ERR_POST_ONLY = -5022
ERR_REDUCE_ONLY = -2022
ERR_MAX_POSITION = -2027


class BaseExchange:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.client: Optional[AsyncClient] = None
        self.on_fill: Optional[FillCb] = None
        self.on_status: Optional[StatusCb] = None
        self._filters: Dict[str, SymbolFilters] = {}
        self.latency_ms = 0.0     # EWMA времени ответа на ордерные запросы

    def _track_latency(self, t0: float):
        ms = (time.perf_counter() - t0) * 1000
        self.latency_ms = ms if self.latency_ms == 0 else 0.8 * self.latency_ms + 0.2 * ms

    async def load_filters(self) -> Dict[str, SymbolFilters]:
        info = await self.client.futures_exchange_info()
        out = {}
        for s in info["symbols"]:
            if s.get("status") != "TRADING" or s.get("contractType") != "PERPETUAL":
                continue
            tick = step = min_qty = min_notional = 0.0
            for f in s["filters"]:
                t = f["filterType"]
                if t == "PRICE_FILTER":
                    tick = float(f["tickSize"])
                elif t == "LOT_SIZE":
                    step = float(f["stepSize"])
                    min_qty = float(f["minQty"])
                elif t == "MIN_NOTIONAL":
                    min_notional = float(f.get("notional", 0) or 0)
            if tick > 0 and step > 0:
                out[s["symbol"]] = SymbolFilters(tick, step, min_qty, min_notional)
        self._filters = out
        return out

    def filters(self, symbol: str) -> Optional[SymbolFilters]:
        return self._filters.get(symbol)

    async def load_fees(self):
        """Фактические комиссии аккаунта (только live, нужен ключ)."""
        return

    async def funding_info(self, symbol: str) -> Tuple[float, float]:
        """(nextFundingTime в секундах, lastFundingRate)."""
        try:
            d = await self.client.futures_mark_price(symbol=symbol)
            return float(d.get("nextFundingTime", 0)) / 1000.0, float(d.get("lastFundingRate", 0))
        except Exception as e:
            logger.debug(f"{symbol}: funding_info {e}")
            return 0.0, 0.0


# ---------------------------------------------------------------- live


class LiveExchange(BaseExchange):
    def __init__(self, cfg: BotConfig):
        super().__init__(cfg)
        self._user_task: Optional[asyncio.Task] = None
        self._listen_key = ""
        self.running = False

    async def start(self):
        e = self.cfg.exchange
        self.client = await AsyncClient.create(
            api_key=e.api_key, api_secret=e.api_secret, testnet=e.testnet,
        )
        await self.load_filters()
        try:
            await self.client.futures_change_position_mode(dualSidePosition="false")
        except BinanceAPIException as ex:
            if ex.code != -4059:     # уже one-way
                logger.warning(f"position mode: {ex}")
        self.running = True
        self._user_task = asyncio.create_task(self._user_stream())

    async def close(self):
        self.running = False
        if self._user_task:
            self._user_task.cancel()
        if self.client:
            await self.client.close_connection()

    async def balance(self) -> float:
        acc = await self.client.futures_account()
        return float(acc["totalWalletBalance"])

    async def load_fees(self):
        try:
            r = await self.client.futures_commission_rate(symbol=self.cfg.risk.btc_symbol)
            e = self.cfg.exchange
            e.maker_fee_bps = float(r["makerCommissionRate"]) * 1e4
            e.taker_fee_bps = float(r["takerCommissionRate"]) * 1e4
        except Exception as ex:
            logger.warning(f"Комиссии не получены, беру из конфига: {ex}")

    async def setup_symbol(self, symbol: str) -> int:
        e = self.cfg.exchange
        lev = e.leverage
        try:
            br = await self.client.futures_leverage_bracket(symbol=symbol)
            if isinstance(br, dict):
                br = [br]
            for item in br or []:
                if item.get("symbol", symbol) == symbol and item.get("brackets"):
                    lev = min(lev, int(item["brackets"][0]["initialLeverage"]))
        except Exception as ex:
            logger.warning(f"{symbol}: leverage bracket {ex}")
        try:
            await self.client.futures_change_margin_type(symbol=symbol, marginType=e.margin_type)
        except BinanceAPIException as ex:
            if ex.code != -4046:     # уже такой тип
                logger.warning(f"{symbol}: margin type {ex}")
        await self.client.futures_change_leverage(symbol=symbol, leverage=lev)
        return lev

    async def place_limit(self, symbol: str, side: str, price: str, qty: str,
                          cid: str, reduce_only: bool) -> PlaceResult:
        params = dict(symbol=symbol, side=side, type="LIMIT", timeInForce="GTX",
                      price=price, quantity=qty, newClientOrderId=cid)
        if reduce_only:
            params["reduceOnly"] = "true"
        t0 = time.perf_counter()
        try:
            await self.client.futures_create_order(**params)
            self._track_latency(t0)
            return PlaceResult(True)
        except BinanceAPIException as ex:
            self._track_latency(t0)
            return PlaceResult(False, ex.code, ex.message)
        except Exception as ex:
            return PlaceResult(False, 0, repr(ex))

    async def cancel(self, symbol: str, cid: str) -> bool:
        """True — ордера больше нет на бирже (отменён или уже исполнен)."""
        t0 = time.perf_counter()
        try:
            await self.client.futures_cancel_order(symbol=symbol, origClientOrderId=cid)
            self._track_latency(t0)
            return True
        except BinanceAPIException as ex:
            self._track_latency(t0)
            return ex.code == ERR_UNKNOWN_ORDER
        except Exception:
            return False

    async def cancel_all(self, symbol: str):
        try:
            await self.client.futures_cancel_all_open_orders(symbol=symbol)
        except Exception as ex:
            logger.warning(f"{symbol}: cancel_all {ex}")

    async def market(self, symbol: str, side: str, qty: str, reduce_only: bool = True) -> PlaceResult:
        params = dict(symbol=symbol, side=side, type="MARKET", quantity=qty)
        if reduce_only:
            params["reduceOnly"] = "true"
        try:
            await self.client.futures_create_order(**params)
            return PlaceResult(True)
        except BinanceAPIException as ex:
            return PlaceResult(False, ex.code, ex.message)

    async def position(self, symbol: str) -> Tuple[float, float]:
        for p in await self.client.futures_position_information(symbol=symbol):
            if p["symbol"] == symbol:
                return float(p["positionAmt"]), float(p.get("entryPrice", 0) or 0)
        return 0.0, 0.0

    # --- user data stream (/private) ---

    async def _user_stream(self):
        _, _, private = self.cfg.exchange.ws_urls()
        while self.running:
            keepalive = None
            try:
                self._listen_key = await self.client.futures_stream_get_listen_key()
                url = (f"{private.rstrip('/')}/ws?listenKey={self._listen_key}"
                       f"&events=ORDER_TRADE_UPDATE/ACCOUNT_UPDATE")
                keepalive = asyncio.create_task(self._keepalive())
                async with websockets.connect(url, ping_interval=20, ping_timeout=20,
                                              max_queue=None) as ws:
                    logger.info("📡 User data stream подключён")
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        msg = msg.get("data", msg)
                        ev = msg.get("e")
                        if ev == "ORDER_TRADE_UPDATE":
                            await self._on_order_update(msg.get("o", {}))
                        elif ev == "listenKeyExpired":
                            logger.warning("listenKey истёк — переподключение")
                            break
            except asyncio.CancelledError:
                return
            except Exception as ex:
                logger.error(f"User stream: {ex!r}. Переподключение через 2 с")
                await asyncio.sleep(2)
            finally:
                if keepalive:
                    keepalive.cancel()

    async def _keepalive(self):
        while True:
            await asyncio.sleep(25 * 60)
            try:
                await self.client.futures_stream_keepalive(listenKey=self._listen_key)
            except Exception as ex:
                logger.warning(f"listenKey keepalive: {ex}")

    async def _on_order_update(self, o: dict):
        symbol, cid, status = o.get("s", ""), o.get("c", ""), o.get("X", "")
        if o.get("x") == "TRADE":
            try:
                qty, px = float(o.get("l", 0)), float(o.get("L", 0))
            except (TypeError, ValueError):
                qty = px = 0.0
            if qty > 0 and px > 0 and self.on_fill:
                maker = bool(o.get("m", False))
                fee = None
                if o.get("N") == self.cfg.scanner.quote_asset:
                    try:
                        fee = float(o.get("n", 0))
                    except (TypeError, ValueError):
                        fee = None
                if fee is None:
                    e = self.cfg.exchange
                    fee = qty * px * (e.maker_fee_bps if maker else e.taker_fee_bps) / 1e4
                await self.on_fill(Fill(symbol, o.get("S", ""), qty, px, fee, cid, maker, time.time()))
        if self.on_status and status:
            await self.on_status(symbol, cid, status)


# ---------------------------------------------------------------- paper


@dataclass
class PaperOrder:
    symbol: str
    side: str
    price: float
    qty: float
    reduce_only: bool
    cid: str


class PaperExchange(BaseExchange):
    """
    Бумажная торговля на живых данных Binance.
    - Каждый запрос «идёт» latency_ms (как с VPS в Гонконге).
    - Лимитная заявка исполняется, только если сделка прошла СКВОЗЬ её цену
      или противоположная лучшая цена пересекла её. На своей цене мы считаем
      себя в конце очереди (touch_fill_prob = 0) — оценка консервативная.
    - Рыночный ордер исполняется по лучшей цене с комиссией тейкера.
    """

    def __init__(self, cfg: BotConfig, books):
        super().__init__(cfg)
        self.books = books               # MarketData.books
        self.orders: Dict[str, PaperOrder] = {}
        self.positions: Dict[str, Tuple[float, float]] = {}
        self.cash = cfg.paper.balance_usdt
        self.rng = random.Random(42)

    async def start(self):
        self.client = await AsyncClient.create(testnet=self.cfg.exchange.testnet)
        await self.load_filters()

    async def close(self):
        if self.client:
            await self.client.close_connection()

    async def balance(self) -> float:
        return self.cash

    async def setup_symbol(self, symbol: str) -> int:
        return self.cfg.exchange.leverage

    async def _lat(self, frac: float = 1.0):
        await asyncio.sleep(self.cfg.paper.latency_ms / 1000.0 * frac)

    async def place_limit(self, symbol, side, price, qty, cid, reduce_only) -> PlaceResult:
        t0 = time.perf_counter()
        await self._lat(0.5)                       # запрос дошёл до биржи
        px, q = float(price), float(qty)
        book = self.books.get(symbol)
        if book is None or not book.ready:
            return PlaceResult(False, -1, "no book")
        if (side == "BUY" and px >= book.ask) or (side == "SELL" and px <= book.bid):
            await self._lat(0.5)
            self._track_latency(t0)
            return PlaceResult(False, ERR_POST_ONLY, "post only rejected")
        pos, _ = self.positions.get(symbol, (0.0, 0.0))
        if reduce_only and (pos == 0 or (side == "BUY") == (pos > 0)):
            await self._lat(0.5)
            return PlaceResult(False, ERR_REDUCE_ONLY, "reduce only rejected")
        self.orders[cid] = PaperOrder(symbol, side, px, q, reduce_only, cid)
        await self._lat(0.5)                       # ответ вернулся
        self._track_latency(t0)
        return PlaceResult(True)

    async def cancel(self, symbol, cid) -> bool:
        t0 = time.perf_counter()
        await self._lat(0.5)                       # пока запрос летит, заявка может исполниться
        o = self.orders.pop(cid, None)
        await self._lat(0.5)
        self._track_latency(t0)
        if o and self.on_status:
            await self.on_status(symbol, cid, "CANCELED")
        return True

    async def cancel_all(self, symbol):
        for cid in [c for c, o in self.orders.items() if o.symbol == symbol]:
            await self.cancel(symbol, cid)

    async def market(self, symbol, side, qty, reduce_only=True) -> PlaceResult:
        await self._lat(0.5)
        book = self.books.get(symbol)
        if book is None or not book.ready:
            return PlaceResult(False, -1, "no book")
        q = float(qty)
        pos, _ = self.positions.get(symbol, (0.0, 0.0))
        if reduce_only:
            q = min(q, abs(pos)) if pos != 0 and (side == "BUY") != (pos > 0) else 0.0
        if q <= 0:
            return PlaceResult(False, ERR_REDUCE_ONLY, "reduce only rejected")
        px = book.ask if side == "BUY" else book.bid
        fee = q * px * self.cfg.exchange.taker_fee_bps / 1e4
        await self._apply_fill(symbol, side, q, px, fee, f"m_{int(time.time()*1e3)}", maker=False)
        await self._lat(0.5)
        return PlaceResult(True)

    async def position(self, symbol) -> Tuple[float, float]:
        return self.positions.get(symbol, (0.0, 0.0))

    # --- исполнение по рыночным данным ---

    async def on_trade(self, symbol: str, price: float, qty: float, buyer_is_maker: bool, now: float):
        p = self.cfg.paper.touch_fill_prob
        for o in list(self.orders.values()):
            if o.symbol != symbol:
                continue
            if buyer_is_maker and o.side == "BUY":       # агрессивный продавец
                hit = price < o.price or (price == o.price and p > 0 and self.rng.random() < p)
            elif not buyer_is_maker and o.side == "SELL":
                hit = price > o.price or (price == o.price and p > 0 and self.rng.random() < p)
            else:
                hit = False
            if hit:
                await self._fill_order(o)

    async def on_book(self, symbol: str):
        book = self.books.get(symbol)
        if book is None or not book.ready:
            return
        for o in list(self.orders.values()):
            if o.symbol != symbol:
                continue
            if (o.side == "BUY" and book.ask <= o.price) or (o.side == "SELL" and book.bid >= o.price):
                await self._fill_order(o)

    async def _fill_order(self, o: PaperOrder):
        if self.orders.pop(o.cid, None) is None:
            return
        q = o.qty
        if o.reduce_only:
            pos, _ = self.positions.get(o.symbol, (0.0, 0.0))
            if pos == 0 or (o.side == "BUY") == (pos > 0):
                if self.on_status:
                    await self.on_status(o.symbol, o.cid, "EXPIRED")
                return
            q = min(q, abs(pos))
        fee = q * o.price * self.cfg.exchange.maker_fee_bps / 1e4
        await self._apply_fill(o.symbol, o.side, q, o.price, fee, o.cid, maker=True)
        if self.on_status:
            await self.on_status(o.symbol, o.cid, "FILLED")

    async def _apply_fill(self, symbol, side, qty, price, fee, cid, maker):
        pos, entry = self.positions.get(symbol, (0.0, 0.0))
        signed = qty if side == "BUY" else -qty
        new = pos + signed
        if abs(new) < 1e-12:
            new = 0.0
        if pos == 0 or (pos > 0) == (signed > 0):
            entry = (entry * abs(pos) + price * qty) / (abs(pos) + qty)
        else:
            closed = min(abs(pos), qty)
            self.cash += closed * (price - entry) * (1 if pos > 0 else -1)
            if abs(signed) > abs(pos):
                entry = price
        if new == 0:
            entry = 0.0
        self.cash -= fee
        self.positions[symbol] = (new, entry)
        if self.on_fill:
            await self.on_fill(Fill(symbol, side, qty, price, fee, cid, maker, time.time()))

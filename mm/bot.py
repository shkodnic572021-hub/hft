"""
Оркестрация: SymbolTrader торгует одной монетой, Bot управляет набором
монет, общими рисками, ротацией и остановкой.
"""

import asyncio
import logging
import os
import secrets
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional

from .config import BotConfig
from .exchange import (
    ERR_MAX_POSITION, ERR_POST_ONLY, ERR_REDUCE_ONLY,
    BaseExchange, Fill, LiveExchange, PaperExchange,
)
from .marketdata import Book, MarketData
from .pnl import PnLTracker
from .risk import BtcShock, MarkoutTracker, RateLimiter, symbol_shock, unrealized_bps
from .scanner import Scanner
from .strategy import QuoteInputs, Side, SymbolFilters, compute_quote, fmt, needs_requote

logger = logging.getLogger(__name__)

ERR_MIN_NOTIONAL = -4164
FINAL_STATUSES = ("FILLED", "CANCELED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH")


@dataclass
class LiveOrder:
    cid: str
    side: str
    price: float
    qty: float
    reduce_only: bool
    state: str = "pending"      # pending → live → cancelling


class SymbolTrader:
    def __init__(self, bot: "Bot", symbol: str, filters: SymbolFilters, book: Book):
        self.bot = bot
        self.cfg = bot.cfg
        self.ex: BaseExchange = bot.ex
        self.symbol = symbol
        self.f = filters
        self.book = book
        self.orders: Dict[str, Optional[LiveOrder]] = {"BUY": None, "SELL": None}

        self.pos_opened_at = 0.0
        self.pause_until = 0.0
        self.pause_reason = ""
        self.cooldown_until = 0.0
        self.no_open_until = 0.0
        self.funding_time = 0.0
        self.toxic = False
        self.rotate_requested = False
        self.stopping = False
        self.running = True
        self.fills = 0
        self.last_quote = None
        self._last_step = 0.0
        self._mismatch = 0
        self._task: Optional[asyncio.Task] = None

    # --- состояние позиции (источник — P&L-трекер, сверяется с биржей) ---

    @property
    def pnl_state(self):
        return self.bot.pnl.get_state(self.symbol)

    @property
    def position(self) -> float:
        return self.pnl_state.position_qty

    def start(self):
        self._task = asyncio.create_task(self._loop(), name=f"trader_{self.symbol}")

    async def stop(self, flatten: bool):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        await self.ex.cancel_all(self.symbol)
        self.orders = {"BUY": None, "SELL": None}
        if flatten:
            await self._market_flatten("остановка")

    # --- события биржи ---

    async def on_fill(self, fill: Fill):
        old = self.position
        self.bot.pnl.on_fill(self.symbol, fill.side, fill.qty, fill.price, fee=fill.fee_usdt)
        new = self.position
        now = time.time()
        if old == 0 or (old > 0) != (new > 0):
            self.pos_opened_at = now if new != 0 else 0.0
        elif new == 0:
            self.pos_opened_at = 0.0
        self.fills += 1
        notional = fill.qty * fill.price
        fee_bps = fill.fee_usdt / notional * 1e4 if notional > 0 else 0.0
        self.bot.markouts.on_fill(self.symbol, fill.side, fill.price, fill.qty,
                                  fee_bps, self.book.mid if self.book.ready else fill.price, now)
        for side, lo in self.orders.items():
            if lo is not None and lo.cid == fill.cid:
                lo.qty -= fill.qty
                if lo.qty <= self.f.step / 2:
                    self.orders[side] = None
        logger.info(
            f"💥 {self.symbol} {fill.side} {fill.qty:g} @ {fill.price:g} "
            f"{'maker' if fill.maker else 'TAKER'} fee={fill.fee_usdt:.4f} | "
            f"pos={new:g} avg={self.pnl_state.avg_entry:g} "
            f"realized={self.pnl_state.realized_pnl:+.4f}"
        )
        self.book.updated.set()

    async def on_status(self, cid: str, status: str):
        if status not in FINAL_STATUSES:
            return
        for side, lo in self.orders.items():
            if lo is not None and lo.cid == cid:
                self.orders[side] = None
                self.book.updated.set()

    # --- главный цикл ---

    async def _loop(self):
        q = self.cfg.quote
        while self.running:
            try:
                await asyncio.wait_for(self.book.updated.wait(), q.loop_timeout_ms / 1000)
            except asyncio.TimeoutError:
                pass
            self.book.updated.clear()
            wait = q.requote_min_interval_ms / 1000 - (time.time() - self._last_step)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_step = time.time()
            try:
                await self._step()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(f"{self.symbol}: step error {e}")
                await asyncio.sleep(1)

    def _pause(self, seconds: float, reason: str):
        until = time.time() + seconds
        if until > self.pause_until:
            if time.time() >= self.pause_until:
                logger.warning(f"⛈️  {self.symbol}: пауза входов {seconds:.0f}с — {reason}")
            self.pause_until = until
            self.pause_reason = reason

    async def _step(self):
        if self.stopping:
            return
        now = time.time()
        book = self.book
        r = self.cfg.risk
        if not book.ready or now - book.ts > 5.0:
            # Нет свежих данных — не держим заявки вслепую
            await self._cancel_side("BUY")
            await self._cancel_side("SELL")
            return

        mid = book.mid
        pos = self.position
        pnl = self.pnl_state
        self.bot.pnl.update_unrealized(self.symbol, mid)
        hold_s = now - self.pos_opened_at if pos != 0 and self.pos_opened_at > 0 else 0.0

        # 1. Стоп по цене и по времени удержания — выход по рынку
        if pos != 0:
            ubps = unrealized_bps(pos, pnl.avg_entry, mid)
            if ubps <= -r.stop_loss_bps:
                await self._flatten(f"STOP {ubps:.1f}bps от входа {pnl.avg_entry:g}")
                return
            if hold_s >= r.max_hold_s:
                await self._flatten(f"MAX HOLD {hold_s:.0f}с (P&L {ubps:+.1f}bps)")
                return

        # 2. Шоки
        shock = symbol_shock(book.spread_bps, book.median_spread_bps,
                             book.vol_fast.sigma_bps if book.vol_fast.ready else 0.0,
                             book.vol_slow.sigma_bps if book.vol_slow.ready else 0.0,
                             r.spread_shock_mult, r.vol_shock_mult)
        if shock:
            self._pause(r.shock_pause_s, shock)
        if self.bot.btc_pause_until > now:
            self._pause(self.bot.btc_pause_until - now, "BTC shock")

        # 3. Токсичность потока по markout
        mk = self.bot.markouts
        if mk.count(self.symbol) >= r.toxic_min_fills and mk.avg_net(self.symbol) < r.toxic_net_bps:
            if not self.toxic:
                self.toxic = True
                self.rotate_requested = True
                logger.warning(
                    f"☠️  {self.symbol}: токсичный поток — средний net markout "
                    f"{mk.avg_net(self.symbol):+.2f}bps за {mk.count(self.symbol)} fills"
                )
                self._pause(r.toxic_pause_s, "toxic flow")

        # 4. Убыток по монете
        if pnl.net_pnl < -r.symbol_max_loss_usdt and not self.rotate_requested:
            self.rotate_requested = True
            logger.warning(f"📉 {self.symbol}: убыток {pnl.net_pnl:+.2f} USDT — запрошена ротация")

        funding_block = (self.funding_time > 0
                         and -5 < self.funding_time - now < r.funding_blackout_s)

        allow_open = (
            self.running and not self.bot.stopping
            and now >= self.pause_until and now >= self.cooldown_until
            and now >= self.no_open_until
            and not funding_block and not self.rotate_requested
            and book.vol_fast.ready
        )

        e = self.cfg.exchange
        quote = compute_quote(QuoteInputs(
            bid=book.bid, ask=book.ask, bid_qty=book.bid_qty, ask_qty=book.ask_qty,
            depth_imb=book.depth_imb, flow_imb=book.flow_imb(now),
            sigma_1s_bps=book.vol_fast.sigma_bps,
            adverse_bps=mk.adverse_bps(self.symbol),
            position=pos, avg_entry=pnl.avg_entry, hold_s=hold_s,
            allow_open=allow_open,
            latency_bad=self.ex.latency_ms > self.cfg.quote.latency_bad_ms,
            maker_fee_bps=e.maker_fee_bps,
            max_inventory_usdt=r.max_inventory_usdt,
        ), self.f, self.cfg.quote)
        if quote is None:
            return
        self.last_quote = quote
        await asyncio.gather(
            self._reconcile("BUY", quote.bid),
            self._reconcile("SELL", quote.ask),
        )

    # --- управление заявками ---

    async def _reconcile(self, side: str, want: Optional[Side]):
        cur = self.orders[side]
        if cur is not None and cur.state != "live":
            return                               # запрос ещё в пути
        if want is None:
            if cur is not None:
                await self._cancel_side(side)
            return
        if cur is None:
            await self._place(side, want)
            return
        if needs_requote(cur.price, cur.qty, cur.reduce_only, want,
                         self.f.tick, self.cfg.quote.requote_bps):
            if not self.bot.rate.allow(reserve=2) and not want.reduce_only:
                return
            if await self._cancel_side(side):
                await self._place(side, want)

    async def _place(self, side: str, want: Side):
        if not self.bot.rate.allow(reserve=0 if want.reduce_only else 10):
            return
        cid = f"mm{side[0].lower()}{secrets.token_hex(8)}"
        lo = LiveOrder(cid, side, want.price, want.qty, want.reduce_only)
        self.orders[side] = lo
        self.bot.rate.hit()
        res = await self.ex.place_limit(
            self.symbol, side, fmt(want.price, self.f.tick), fmt(want.qty, self.f.step),
            cid, want.reduce_only,
        )
        if res.ok:
            if self.orders[side] is lo:
                lo.state = "live"
            return
        if self.orders[side] is lo:
            self.orders[side] = None
        if res.code == ERR_POST_ONLY:
            return                               # рынок сдвинулся, перекотируем
        if res.code == ERR_REDUCE_ONLY:
            await self.sync_position(force=True)
            return
        if res.code == ERR_MIN_NOTIONAL and want.reduce_only:
            await self._market_flatten("остаток меньше минимального номинала")
            return
        if res.code == ERR_MAX_POSITION:
            self.no_open_until = time.time() + 60
        logger.error(f"❌ {self.symbol} {side} {want.qty:g} @ {want.price:g}: {res.code} {res.msg}")
        self.no_open_until = max(self.no_open_until, time.time() + 2)

    async def _cancel_side(self, side: str) -> bool:
        lo = self.orders[side]
        if lo is None:
            return True
        if lo.state == "cancelling":
            return False
        lo.state = "cancelling"
        self.bot.rate.hit()
        ok = await self.ex.cancel(self.symbol, lo.cid)
        if ok:
            if self.orders[side] is lo:
                self.orders[side] = None
            return True
        if self.orders[side] is lo:
            lo.state = "live"
        return False

    async def _flatten(self, reason: str):
        logger.critical(f"🛑 {self.symbol}: {reason} → выход по рынку")
        self.cooldown_until = time.time() + self.cfg.risk.reentry_cooldown_s
        self.stopping = True
        try:
            await asyncio.gather(self._cancel_side("BUY"), self._cancel_side("SELL"))
            await self._market_flatten(reason)
        finally:
            self.stopping = False

    async def _market_flatten(self, reason: str):
        try:
            qty, _ = await self.ex.position(self.symbol)
        except Exception as e:
            logger.error(f"{self.symbol}: позиция не получена: {e}")
            qty = self.position
        if qty == 0:
            return
        side = "SELL" if qty > 0 else "BUY"
        res = await self.ex.market(self.symbol, side, fmt(abs(qty), self.f.step), reduce_only=True)
        if res.ok:
            logger.warning(f"🔒 {self.symbol}: MARKET {side} {abs(qty):g} ({reason})")
        else:
            logger.error(f"❌ {self.symbol}: market close {res.code} {res.msg}")

    async def sync_position(self, force: bool = False):
        """Сверка с биржей; правим только устойчивое (2 проверки) расхождение."""
        try:
            qty, entry = await self.ex.position(self.symbol)
        except Exception as e:
            logger.debug(f"{self.symbol}: sync {e}")
            return
        if abs(qty - self.position) <= self.f.step / 2:
            self._mismatch = 0
            return
        self._mismatch += 1
        if self._mismatch >= 2 or force:
            logger.warning(f"🔄 {self.symbol}: позиция {self.position:g} → {qty:g} (биржа)")
            if self.position == 0 and qty != 0:
                self.pos_opened_at = time.time()
            self.bot.pnl.sync_position(self.symbol, qty, entry)
            if qty == 0:
                self.pos_opened_at = 0.0
            self._mismatch = 0


class Bot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        pub, mkt, _ = cfg.exchange.ws_urls()
        self.md = MarketData(pub, mkt, cfg.quote.flow_halflife_s)
        self.ex: BaseExchange = (PaperExchange(cfg, self.md.books) if cfg.paper_mode
                                 else LiveExchange(cfg))
        os.makedirs(cfg.log.dir, exist_ok=True)
        mode = "paper" if cfg.paper_mode else "live"
        self.pnl = PnLTracker(SimpleNamespace(
            csv_path=os.path.join(cfg.log.dir, f"pnl_{mode}.csv"),
            log_interval_sec=cfg.log.status_interval_s,
        ), maker_fee_pct=cfg.exchange.maker_fee_bps / 100)
        r = cfg.risk
        self.markouts = MarkoutTracker(
            r.markout_horizons_s, r.toxic_horizon_s, r.toxic_window,
            r.adverse_halflife_fills, os.path.join(cfg.log.dir, f"fills_{mode}.csv"),
        )
        self.rate = RateLimiter(r.max_orders_per_10s)
        self.btc = BtcShock(r.btc_move_bps, r.btc_window_s)
        self.btc_pause_until = 0.0
        self.traders: Dict[str, SymbolTrader] = {}
        self.scanner: Optional[Scanner] = None
        self.initial_balance = 0.0
        self.stopping = False
        self.running = False
        self._tasks: List[asyncio.Task] = []
        self._rotate_lock = asyncio.Lock()

    # --- события ---

    async def _on_fill(self, fill: Fill):
        t = self.traders.get(fill.symbol)
        if t is not None:
            await t.on_fill(fill)
        else:
            # Монета уже снята — сделку всё равно учитываем в P&L
            self.pnl.on_fill(fill.symbol, fill.side, fill.qty, fill.price, fee=fill.fee_usdt)

    async def _on_status(self, symbol: str, cid: str, status: str):
        t = self.traders.get(symbol)
        if t is not None:
            await t.on_status(cid, status)

    async def _on_book(self, symbol: str):
        if symbol == self.cfg.risk.btc_symbol:
            b = self.md.books.get(symbol)
            if b and b.ready and self.btc.on_mid(b.mid, b.ts):
                if self.btc_pause_until < time.time():
                    logger.warning("⛈️  BTC shock — пауза входов по всем монетам")
                self.btc_pause_until = time.time() + self.cfg.risk.shock_pause_s
        if isinstance(self.ex, PaperExchange):
            await self.ex.on_book(symbol)

    async def _on_trade(self, symbol, px, qty, bm, now):
        if isinstance(self.ex, PaperExchange):
            await self.ex.on_trade(symbol, px, qty, bm, now)

    # --- монеты ---

    async def add_symbol(self, symbol: str) -> bool:
        f = self.ex.filters(symbol)
        if f is None:
            logger.error(f"❌ {symbol}: нет фильтров (не торгуется?)")
            return False
        try:
            lev = await self.ex.setup_symbol(symbol)
            await self.ex.cancel_all(symbol)
        except Exception as e:
            logger.error(f"❌ {symbol}: настройка не удалась: {e}")
            return False
        self.md.subscribe(symbol)
        t = SymbolTrader(self, symbol, f, self.md.book(symbol))
        qty, entry = await self.ex.position(symbol)
        if qty != 0:
            logger.warning(f"⚠️  {symbol}: на бирже уже есть позиция {qty:g} @ {entry:g}")
            self.pnl.sync_position(symbol, qty, entry)
            t.pos_opened_at = time.time()
        t.funding_time, _ = await self.ex.funding_info(symbol)
        self.traders[symbol] = t
        t.start()
        logger.info(f"✅ {symbol}: торгуем | tick={f.tick:g} step={f.step:g} "
                    f"minNotional={f.min_notional:g} leverage={lev}x")
        return True

    async def remove_symbol(self, symbol: str):
        t = self.traders.pop(symbol, None)
        if t is None:
            return
        await t.stop(flatten=True)
        if symbol != self.cfg.risk.btc_symbol:
            self.md.unsubscribe(symbol)
        self.markouts.reset(symbol)
        logger.info(f"❎ {symbol}: снят с торговли")

    # --- фоновые задачи ---

    async def _markout_loop(self):
        while self.running:
            await asyncio.sleep(0.25)
            mids = {s: b.mid for s, b in self.md.books.items() if b.ready}
            self.markouts.update(mids)

    async def _sync_loop(self):
        while self.running:
            await asyncio.sleep(self.cfg.risk.position_sync_s)
            for t in list(self.traders.values()):
                await t.sync_position()

    async def _funding_loop(self):
        while self.running:
            await asyncio.sleep(60)
            for t in list(self.traders.values()):
                t.funding_time, _ = await self.ex.funding_info(t.symbol)

    async def _status_loop(self):
        while self.running:
            await asyncio.sleep(self.cfg.log.status_interval_s)
            prices = {}
            for s, t in self.traders.items():
                if not t.book.ready:
                    continue
                prices[s] = t.book.mid
                q = t.last_quote
                mk = self.markouts
                logger.info(
                    f"📈 {s} pos={t.position:g} fills={t.fills} "
                    f"markout5s={mk.avg_net(s):+.2f}bps (n={mk.count(s)}) "
                    f"adverse={mk.adverse_bps(s):.2f}bps "
                    f"spread_mkt={t.book.spread_bps:.1f}bps "
                    + (f"half={q.half_bps:.1f} alpha={q.alpha_bps:+.1f} skew={q.skew_bps:+.1f}bps "
                       if q else "")
                    + f"σ1s={t.book.vol_fast.sigma_bps:.1f}bps latency={self.ex.latency_ms:.0f}ms"
                    + (f" PAUSE({t.pause_reason})" if t.pause_until > time.time() else "")
                )
            if prices:
                self.pnl.log_if_due(prices)
            await self._check_session_loss()

    async def _check_session_loss(self):
        for s, t in self.traders.items():
            if t.book.ready:
                self.pnl.update_unrealized(s, t.book.mid)
        net = self.pnl.session_net_pnl()
        limit = self.initial_balance * self.cfg.risk.session_max_loss_pct
        if self.initial_balance > 0 and net < -limit:
            logger.critical(f"🛑 Убыток сессии {net:+.2f} USDT превысил {limit:.2f} — остановка")
            self.running = False

    async def _rotation_loop(self):
        if self.cfg.symbols:
            # Ручной список: только снимаем токсичные монеты
            while self.running:
                await asyncio.sleep(10)
                for s, t in list(self.traders.items()):
                    if t.rotate_requested and t.position == 0 and t.pause_until <= time.time():
                        t.rotate_requested = False
                        t.toxic = False
                        self.markouts.reset(s)
            return
        last_scan = time.time()
        while self.running:
            await asyncio.sleep(10)
            since = time.time() - last_scan
            need = any(t.rotate_requested for t in self.traders.values()) \
                or since >= self.cfg.scanner.rescan_interval_sec \
                or (len(self.traders) < self.cfg.max_symbols
                    and since >= self.cfg.scanner.empty_rescan_sec)
            if not need:
                continue
            last_scan = time.time()
            try:
                await self._rotate()
            except Exception as e:
                logger.exception(f"rotation: {e}")

    async def _rotate(self):
        async with self._rotate_lock:
            cands = await self.scanner.scan()
            by_sym = {c.symbol: c for c in cands}
            # 1. снять монеты, которые просили ротацию (и уже без позиции)
            for s, t in list(self.traders.items()):
                if t.rotate_requested and (t.position == 0 or t.toxic):
                    await self.remove_symbol(s)
            # 2. заменить монету, если есть заметно лучшая (только без позиции)
            mult = self.cfg.scanner.rotate_score_mult
            for s, t in list(self.traders.items()):
                cur = by_sym.get(s)
                best = next((c for c in cands if c.symbol not in self.traders), None)
                if best is None or t.position != 0:
                    continue
                if cur is None or best.score > cur.score * mult:
                    logger.warning(f"🔄 Ротация {s} → {best.symbol} (score {best.score:.2f})")
                    await self.remove_symbol(s)
            # 3. добрать до max_symbols
            for c in cands:
                if len(self.traders) >= self.cfg.max_symbols:
                    break
                if c.symbol not in self.traders:
                    await self.add_symbol(c.symbol)

    # --- запуск / остановка ---

    async def run(self):
        self.ex.on_fill = self._on_fill
        self.ex.on_status = self._on_status
        self.md.book_listeners.append(self._on_book)
        self.md.trade_listeners.append(self._on_trade)
        await self.ex.start()
        self.running = True
        self.initial_balance = await self.ex.balance()
        logger.info(f"💰 Баланс: {self.initial_balance:.2f} USDT")
        await self.ex.load_fees()
        e = self.cfg.exchange
        logger.info(f"💸 Комиссии: maker {e.maker_fee_bps:.2f}bps, taker {e.taker_fee_bps:.2f}bps")
        self.scanner = Scanner(self.ex.client, self.cfg, self.ex.filters)
        self.md.subscribe(self.cfg.risk.btc_symbol, trades=False, depth=False)

        try:
            if self.cfg.symbols:
                for s in self.cfg.symbols[: self.cfg.max_symbols]:
                    await self.add_symbol(s.upper())
            else:
                await self._rotate()
            self._tasks = [asyncio.create_task(c) for c in (
                self._markout_loop(), self._sync_loop(), self._funding_loop(),
                self._status_loop(), self._rotation_loop(),
            )]
            while self.running:
                await asyncio.sleep(1)
        finally:
            await self.shutdown()

    async def shutdown(self):
        if self.stopping:
            return
        self.stopping = True
        self.running = False
        logger.info("🔻 Остановка: снимаю заявки и закрываю позиции...")
        for t in self._tasks:
            t.cancel()
        for s in list(self.traders):
            try:
                t = self.traders.pop(s)
                await asyncio.wait_for(t.stop(flatten=self.cfg.flatten_on_exit), 15)
            except Exception as e:
                logger.error(f"{s}: остановка — {e}")
        await asyncio.sleep(0.5)    # дождаться событий последних сделок
        mids = {s: b.mid for s, b in self.md.books.items() if b.ready}
        for s in list(self.pnl.states):
            if s in mids:
                self.pnl.update_unrealized(s, mids[s])
        self.markouts.update(mids, now=time.time() + 3600)
        self.pnl.print_summary()
        if isinstance(self.ex, PaperExchange):
            logger.info(f"📄 Бумажный баланс: {self.ex.cash:.4f} USDT "
                        f"(старт {self.cfg.paper.balance_usdt:.2f})")
        await self.md.close()
        await self.ex.close()
        logger.info("✅ Остановлено")

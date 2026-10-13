"""
HFT маркет-мейкер с Storm Mode (v6).

Изменения v5:
- Динамическое плечо (min(requested, symbol_max))
- Обработка -2027 → hard_kill символа (не спамить)

Изменения v6 (исправления ошибок):
- Позиция и P&L считаются по последней сделке (l/L), а не по
  накопленному объёму ордера (z/ap) — раньше при частичных исполнениях
  позиция задваивалась
- Цены и объёмы отправляются в десятичной записи (без 1e-05)
- Синхронизация позиции с биржей обновляет и P&L-трекер
- Защиты (loss stop, max drawdown) работают по актуальному unrealized
- Ротации сериализованы, HARD KILL закрывает позицию
"""

import asyncio
import json
import logging
import math
import secrets
import signal
import time
from collections import deque
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Optional, Set, Tuple

import websockets
import numpy as np

from binance import AsyncClient, BinanceSocketManager
from binance.enums import (
    SIDE_BUY, SIDE_SELL, ORDER_TYPE_LIMIT, TIME_IN_FORCE_GTX,
    TIME_IN_FORCE_GTC,
)
from binance.exceptions import BinanceAPIException

from config import BotConfig
from calibrator import MarketCalibrator
from scanner import SymbolScanner
from pnl_tracker import PnLTracker

logger = logging.getLogger(__name__)


class AvellanedaStoikovModel:
    def __init__(self, gamma: float, kappa: float, time_horizon: float):
        self.gamma = gamma
        self.kappa = kappa
        self.T = time_horizon
        self.sigma_log = 0.0

    def update_sigma(self, sigma_log: float):
        self.sigma_log = sigma_log

    def update_kappa(self, kappa: float):
        self.kappa = max(kappa, 1e-6)

    def _sigma_price(self, mid: float) -> float:
        return self.sigma_log * mid

    def reservation_price(self, mid: float, inv: float, t_left: float) -> float:
        if self.sigma_log <= 0 or t_left <= 0:
            return mid
        s_price = self._sigma_price(mid)
        risk_term = inv * self.gamma * (s_price ** 2) * t_left
        return mid - risk_term

    def optimal_spread(self, mid: float, t_left: float) -> float:
        # kappa — в относительных единицах (1 / spread_pct), см. calibrator
        if self.sigma_log <= 0 or self.kappa <= 1e-6 or t_left <= 0:
            return 0.0
        s_price = self._sigma_price(mid)
        risk_part = self.gamma * (s_price ** 2) * t_left
        liquidity_part = (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        return risk_part + liquidity_part * mid


class SymbolState:
    def __init__(self, symbol: str, tick_size: float, lot_size: float, cfg):
        self.symbol = symbol
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.cfg = cfg
        self.actual_leverage = 0          # ← НОВОЕ: реальное плечо на бирже

        self.mid_price = 0.0
        self.best_bid = 0.0
        self.best_ask = 0.0
        self._mid_logged = False

        self.position_qty = 0.0
        self.position_usdt = 0.0

        self.grid_order_ids: Set[str] = set()
        self.tp_order_ids: Dict[str, dict] = {}

        self.orders_placed_total = 0
        self.orders_rejected_gtx = 0

        self.calibrator = MarketCalibrator(window_size=cfg.calibration_window)
        self.model = AvellanedaStoikovModel(
            gamma=cfg.gamma, kappa=cfg.kappa, time_horizon=cfg.time_horizon,
        )

        self.last_latency_ms = 0.0
        self.kill_switch = False
        self.reduce_mode = False
        self.hard_kill = False
        self.total_fills = 0
        self.added_at = time.time()

        self.fading = False
        self.rotating = False
        self.last_natr = 0.0
        self.last_natr_update = 0.0
        self.natr_history = deque(maxlen=60)
        self.last_rotation_at = 0.0

        # --- Storm Mode ---
        self.spread_history = deque(maxlen=200)
        self.bid_volume = 0.0
        self.ask_volume = 0.0
        self.storm_mode = False
        self.storm_until = 0.0
        self.storm_reason = ""
        self.imbalance_count = 0
        self.spread_count = 0
        self.btc_count = 0

        # --- Sync с биржей ---
        self.pos_mismatch_count = 0

        # --- Locks ---
        self.grid_lock = asyncio.Lock()
        self.tp_lock = asyncio.Lock()

    @staticmethod
    def _round_to_step(value: float, step_f: float, rounding=ROUND_DOWN) -> Decimal:
        step = Decimal(str(step_f))
        if step <= 0:
            return Decimal(str(value))
        steps = (Decimal(str(value)) / step).quantize(
            Decimal("1"), rounding=rounding
        )
        return steps * step

    @staticmethod
    def _fmt(d: Decimal) -> str:
        # Всегда обычная десятичная запись: str(1e-05) == '1e-05',
        # такой формат Binance отклоняет.
        s = format(d, "f")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s or "0"

    def round_price(self, price: float) -> float:
        return float(self._round_to_step(price, self.tick_size))

    def round_qty(self, qty: float) -> float:
        return float(self._round_to_step(qty, self.lot_size))

    def fmt_price(self, price: float) -> str:
        return self._fmt(self._round_to_step(price, self.tick_size))

    def fmt_qty(self, qty: float) -> str:
        return self._fmt(self._round_to_step(qty, self.lot_size))

    @staticmethod
    def make_grid_cid() -> str:
        return f"g_{secrets.token_hex(10)}"

    @staticmethod
    def make_tp_cid() -> str:
        return f"t_{secrets.token_hex(10)}"

    def tp_covered_qty(self) -> float:
        return sum(t.get("qty", 0.0) for t in self.tp_order_ids.values())


class MultiMarketMakerBot:
    def __init__(self, config: BotConfig):
        self.cfg = config
        self.client: Optional[AsyncClient] = None
        self.bm: Optional[BinanceSocketManager] = None
        self.scanner: Optional[SymbolScanner] = None

        self.states: Dict[str, SymbolState] = {}
        self._state_lock = asyncio.Lock()

        self.pnl = PnLTracker(config.pnl, maker_fee_pct=0.02)

        self.btc_price_history = deque(maxlen=300)

        # Ротации (scanner_loop и fading_monitor) не должны идти параллельно:
        # бот рассчитан на одну монету одновременно.
        self._rotation_lock = asyncio.Lock()
        # Ссылки на фоновые задачи, чтобы их не собрал GC.
        self._bg_tasks: Set[asyncio.Task] = set()

        self.is_running = False
        self.initial_balance = 0.0
        self.drawdown_triggered = False

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    async def initialize(self):
        self.client = await AsyncClient.create(
            api_key=self.cfg.exchange.api_key,
            api_secret=self.cfg.exchange.api_secret,
            testnet=self.cfg.exchange.testnet,
        )
        self.bm = BinanceSocketManager(self.client)
        self.scanner = SymbolScanner(self.client, self.cfg)

        account = await self.client.futures_account()
        self.initial_balance = float(account["totalWalletBalance"])
        logger.info(f"💰 Баланс: ${self.initial_balance:.2f}")

    async def _get_symbol_max_leverage(self, symbol: str) -> int:
        """
        Возвращает максимально допустимое плечо для символа на Binance.
        Если не удалось получить — возвращает 1 (самое безопасное).
        """
        try:
            brackets = await self.client.futures_leverage_bracket(symbol=symbol)
            # С параметром symbol Binance может вернуть объект, а не список
            if isinstance(brackets, dict):
                brackets = [brackets]
            for item in brackets or []:
                if item.get("symbol", symbol) != symbol:
                    continue
                b = item.get("brackets", [])
                if b:
                    return int(b[0].get("initialLeverage", 1))
        except BinanceAPIException as e:
            logger.warning(f"{symbol}: leverage_bracket — {e}")
        except Exception as e:
            logger.debug(f"{symbol}: leverage_bracket — {e}")
        return 1

    async def _prepare_symbol(self, symbol: str) -> Optional[SymbolState]:
        info = await self.client.futures_exchange_info()
        tick_size = 0.0
        lot_size = 0.0
        min_notional = 0.0
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                if s.get("status", "TRADING") != "TRADING":
                    logger.error(f"❌ {symbol}: статус {s.get('status')}, не торгуется")
                    return None
                if s.get("contractType", "PERPETUAL") != "PERPETUAL":
                    logger.error(f"❌ {symbol}: не бессрочный контракт")
                    return None
                for f in s["filters"]:
                    if f["filterType"] == "PRICE_FILTER":
                        tick_size = float(f["tickSize"])
                    elif f["filterType"] == "LOT_SIZE":
                        lot_size = float(f["stepSize"])
                    elif f["filterType"] == "MIN_NOTIONAL":
                        min_notional = float(f["notional"])
                break

        if tick_size <= 0 or lot_size <= 0:
            logger.error(f"❌ {symbol}: нет tick/lot size")
            return None

        if min_notional > 0 and self.cfg.strategy.order_size_usdt < min_notional:
            logger.warning(
                f"⚠️  {symbol}: MIN_NOTIONAL={min_notional}$ > "
                f"order_size={self.cfg.strategy.order_size_usdt}$"
            )

        # --- Динамическое плечо ---
        requested_lev = self.cfg.exchange.leverage
        symbol_max_lev = await self._get_symbol_max_leverage(symbol)
        actual_lev = min(requested_lev, symbol_max_lev)

        if actual_lev != requested_lev:
            logger.info(
                f"{symbol}: max leverage = {symbol_max_lev}x "
                f"(requested {requested_lev}x → using {actual_lev}x)"
            )

        try:
            await self.client.futures_change_margin_type(
                symbol=symbol, marginType=self.cfg.exchange.margin_type
            )
        except BinanceAPIException as e:
            err = str(e)
            if "-4046" not in err and "No need to change" not in err:
                logger.warning(f"{symbol}: margin — {err}")

        try:
            await self.client.futures_change_leverage(
                symbol=symbol, leverage=actual_lev
            )
        except BinanceAPIException as e:
            err = str(e)
            if "No need to change" not in err:
                logger.warning(f"{symbol}: leverage — {err}")
                # Попробуем еще раз с меньшим плечом
                if actual_lev > 1:
                    try:
                        actual_lev = max(1, actual_lev // 2)
                        await self.client.futures_change_leverage(
                            symbol=symbol, leverage=actual_lev
                        )
                        logger.warning(
                            f"{symbol}: leverage retry at {actual_lev}x"
                        )
                    except BinanceAPIException as e2:
                        logger.error(f"{symbol}: leverage retry failed — {e2}")
                        return None

        try:
            await self.client.futures_change_position_mode(dualSidePosition=False)
        except BinanceAPIException as e:
            if "No need to change" not in str(e):
                logger.warning(f"{symbol}: pos mode — {e}")

        state = SymbolState(symbol, tick_size, lot_size, self.cfg.strategy)
        state.actual_leverage = actual_lev

        positions = await self.client.futures_position_information(symbol=symbol)
        for p in positions:
            if p["symbol"] == symbol:
                state.position_qty = float(p["positionAmt"])
                state.position_usdt = abs(state.position_qty) * float(p["markPrice"])
                # Позиция могла остаться с прошлого запуска — без цены входа
                # TP не выставится, а стопы не сработают.
                self.pnl.sync_position(
                    symbol, state.position_qty, float(p.get("entryPrice", 0) or 0)
                )
                break

        try:
            await self.client.futures_cancel_all_open_orders(symbol=symbol)
        except BinanceAPIException:
            pass

        logger.info(
            f"✅ {symbol} подготовлен | tick={tick_size} | lot={lot_size} | "
            f"minNotional={min_notional}$ | leverage={actual_lev}x "
            f"(max {symbol_max_lev}x) | pos={state.position_qty}"
        )
        return state

    def _calculate_half_spread_pct(self, state: SymbolState) -> float:
        cfg = self.cfg.strategy
        if cfg.spread_mode == "static":
            return cfg.min_half_spread_pct

        base = cfg.spread_base_half_pct
        natr = state.last_natr if state.last_natr > 0 else 1.5
        vol_component = cfg.spread_k_vol * natr

        market_component = 0.0
        if state.best_bid > 0 and state.best_ask > 0 and state.mid_price > 0:
            market_spread_pct = (
                (state.best_ask - state.best_bid) / state.mid_price * 100.0
            )
            market_component = market_spread_pct * cfg.spread_k_market

        half_pct = max(base, vol_component, market_component)
        return min(half_pct, cfg.spread_max_half_pct)

    async def _remove_symbol(self, symbol: str):
        async with self._state_lock:
            state = self.states.pop(symbol, None)
        if state is None:
            return
        try:
            await self.client.futures_cancel_all_open_orders(symbol=symbol)
        except BinanceAPIException as e:
            logger.warning(f"{symbol}: cancel — {e}")
        if state.mid_price > 0:
            self.pnl.force_log(symbol, state.mid_price)
        logger.info(f"❌ {symbol} убран из ротации")

    async def _close_position_market(self, symbol: str, position_qty: float):
        if position_qty == 0:
            return
        try:
            positions = await self.client.futures_position_information(symbol=symbol)
            for p in positions:
                if p["symbol"] == symbol:
                    # Биржа — источник истины: если там 0, закрывать нечего
                    # (иначе reduceOnly-ордер получит отказ -2022).
                    position_qty = float(p["positionAmt"])
                    break
        except Exception as e:
            logger.warning(f"{symbol}: close — позиция не получена ({e})")

        if position_qty == 0:
            return

        state = self.states.get(symbol)
        side = SIDE_SELL if position_qty > 0 else SIDE_BUY
        qty = abs(position_qty)
        qty_str = (
            state.fmt_qty(qty) if state is not None
            else SymbolState._fmt(Decimal(repr(qty)))
        )
        if float(qty_str) <= 0:
            return
        try:
            await self.client.futures_create_order(
                symbol=symbol, side=side, type="MARKET",
                quantity=qty_str, reduceOnly="true",
            )
            logger.info(f"🔒 {symbol}: MARKET {side} {qty_str}")
        except BinanceAPIException as e:
            logger.error(f"❌ {symbol}: close — {e}")

    async def _cancel_grid_orders(self, state: SymbolState):
        if not state.grid_order_ids:
            return
        cids = list(state.grid_order_ids)
        await asyncio.gather(*[
            self.client.futures_cancel_order(
                symbol=state.symbol, origClientOrderId=cid
            ) for cid in cids
        ], return_exceptions=True)
        state.grid_order_ids.clear()

    async def _cancel_all_tps(self, state: SymbolState, reason: str = ""):
        if not state.tp_order_ids:
            return
        cids = list(state.tp_order_ids.keys())
        await asyncio.gather(*[
            self.client.futures_cancel_order(
                symbol=state.symbol, origClientOrderId=cid
            ) for cid in cids
        ], return_exceptions=True)
        state.tp_order_ids.clear()
        logger.info(f"🚫 {state.symbol}: TP отменены ({reason})")

    async def stream_order_books(self):
        active: Dict[str, asyncio.Task] = {}
        try:
            while self.is_running:
                for sym in list(self.states.keys()):
                    if sym not in active or active[sym].done():
                        active[sym] = asyncio.create_task(
                            self._stream_single_depth(sym)
                        )
                for sym in list(active.keys()):
                    if sym not in self.states:
                        active[sym].cancel()
                        del active[sym]
                await asyncio.sleep(1)
        finally:
            # При остановке дочерние WS-задачи иначе остались бы висеть
            for t in active.values():
                t.cancel()

    async def _stream_single_depth(self, symbol: str):
        base_url = ("wss://stream.binancefuture.com/ws"
                    if self.cfg.exchange.testnet
                    else "wss://fstream.binance.com/ws")
        url = f"{base_url}/{symbol.lower()}@depth10@100ms"

        while self.is_running:
            if symbol not in self.states:
                return
            try:
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=10,
                    close_timeout=5, max_queue=None,
                ) as ws:
                    logger.info(f"📡 {symbol} depth WS подключён")
                    async for raw in ws:
                        if not self.is_running or symbol not in self.states:
                            break
                        try:
                            msg = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        await self._process_depth_msg(symbol, msg)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"{symbol} depth WS: {e}. Reconnect in 3s...")
                await asyncio.sleep(3)

    async def _process_depth_msg(self, symbol: str, msg: dict):
        state = self.states.get(symbol)
        if state is None:
            return
        bids = msg.get("b") or []
        asks = msg.get("a") or []
        if not bids or not asks:
            return
        try:
            bb = float(bids[0][0])
            ba = float(asks[0][0])
        except (IndexError, ValueError, TypeError):
            return
        if bb <= 0 or ba <= 0:
            return

        now = time.time()
        state.best_bid = bb
        state.best_ask = ba
        state.mid_price = (bb + ba) / 2.0
        spread = ba - bb

        state.spread_history.append((now, spread))

        n = self.cfg.storm.imbalance_levels
        try:
            state.bid_volume = sum(float(b[1]) for b in bids[:n])
            state.ask_volume = sum(float(a[1]) for a in asks[:n])
        except (IndexError, ValueError):
            pass

        state.calibrator.update(state.mid_price, spread)
        if state.calibrator.ready:
            sigma, kappa = state.calibrator.get_params()
            state.model.update_sigma(sigma)
            state.model.update_kappa(kappa)

        if not state._mid_logged:
            logger.info(f"📊 {symbol}: mid={state.mid_price:.6f} spread={spread:.6f}")
            state._mid_logged = True

        if self.cfg.storm.enabled and not state.storm_mode and not state.rotating:
            await self._check_predictive_triggers(state, now)

    async def natr_updater(self):
        while self.is_running:
            await asyncio.sleep(self.cfg.strategy.spread_natr_refresh_sec)
            if not self.is_running:
                break

            for symbol in list(self.states.keys()):
                state = self.states.get(symbol)
                if state is None:
                    continue
                try:
                    k5m = await self.client.futures_klines(
                        symbol=symbol, interval="5m", limit=20
                    )
                    if len(k5m) < 15:
                        continue
                    completed = k5m[:-1]
                    natr = SymbolScanner.compute_natr_5m(completed)
                    if natr > 0:
                        state.last_natr = natr
                        state.last_natr_update = time.time()
                        logger.debug(f"{symbol}: NATR updated = {natr:.2f}%")
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    logger.debug(f"{symbol}: natr_updater: {e}")

    async def btc_monitor(self):
        if not self.cfg.storm.enabled:
            return
        btc = self.cfg.storm.btc_symbol.lower()
        base_url = ("wss://stream.binancefuture.com/ws"
                    if self.cfg.exchange.testnet
                    else "wss://fstream.binance.com/ws")
        url = f"{base_url}/{btc}@depth5@100ms"

        while self.is_running:
            try:
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=10, max_queue=None,
                ) as ws:
                    logger.info(f"📡 BTC monitor WS подключён ({btc.upper()})")
                    async for raw in ws:
                        if not self.is_running:
                            break
                        try:
                            msg = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        bids = msg.get("b") or []
                        asks = msg.get("a") or []
                        if not bids or not asks:
                            continue
                        try:
                            bb = float(bids[0][0])
                            ba = float(asks[0][0])
                        except (IndexError, ValueError):
                            continue
                        if bb > 0 and ba > 0:
                            self.btc_price_history.append(
                                (time.time(), (bb + ba) / 2.0)
                            )
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"BTC monitor: {e}. Reconnect in 3s...")
                await asyncio.sleep(3)

    async def _check_predictive_triggers(self, state: SymbolState, now: float):
        cfg = self.cfg.storm
        trigger_reason = None

        # --- 1. Дисбаланс стакана ---
        total = state.bid_volume + state.ask_volume
        if total > 0:
            imbalance = state.bid_volume / total
            bearish = imbalance < cfg.imbalance_threshold
            bullish = imbalance > (1.0 - cfg.imbalance_threshold)
            if bearish or bullish:
                state.imbalance_count += 1
            else:
                state.imbalance_count = 0

            if state.imbalance_count >= cfg.imbalance_consecutive:
                side = "asks dominate" if bearish else "bids dominate"
                trigger_reason = (
                    f"imbalance {imbalance:.2f} "
                    f"({side}, {state.imbalance_count}x)"
                )
        else:
            state.imbalance_count = 0

        # --- 2. Расширение спреда ---
        baseline_cut = now - cfg.spread_baseline_sec
        baseline_spreads = [s for t, s in state.spread_history if t >= baseline_cut]
        if len(baseline_spreads) >= 20:
            baseline = float(np.median(baseline_spreads))
            current = state.spread_history[-1][1] if state.spread_history else 0
            if baseline > 0 and current > baseline * cfg.spread_mult:
                state.spread_count += 1
            else:
                state.spread_count = 0

            if state.spread_count >= cfg.spread_consecutive:
                trigger_reason = (
                    trigger_reason or
                    f"spread {current:.6f} > {cfg.spread_mult}x baseline "
                    f"({state.spread_count}x)"
                )
        else:
            state.spread_count = 0

        # --- 3. BTC-дёргание ---
        btc_cut = now - cfg.btc_window_sec
        btc_recent = [(t, p) for t, p in self.btc_price_history if t >= btc_cut]
        if len(btc_recent) >= 5:
            first_p = btc_recent[0][1]
            last_p = btc_recent[-1][1]
            if first_p > 0:
                btc_change = (last_p - first_p) / first_p * 100.0
                if abs(btc_change) > cfg.btc_drop_pct:
                    state.btc_count += 1
                else:
                    state.btc_count = 0
                if state.btc_count >= cfg.btc_consecutive:
                    trigger_reason = (
                        trigger_reason or
                        f"BTC {btc_change:+.2f}% in {cfg.btc_window_sec}s "
                        f"({state.btc_count}x)"
                    )

        if trigger_reason:
            await self._enter_storm_mode(state, trigger_reason)

    async def _enter_storm_mode(self, state: SymbolState, reason: str):
        cfg = self.cfg.storm
        now = time.time()
        state.storm_mode = True
        state.storm_until = now + cfg.duration_sec
        state.storm_reason = reason
        state.imbalance_count = 0
        state.spread_count = 0
        state.btc_count = 0

        logger.warning(
            f"⛈️  {state.symbol}: STORM MODE ON ({reason}) "
            f"→ блок на {cfg.duration_sec:.0f}s"
        )

        async with state.grid_lock:
            await self._cancel_grid_orders(state)

        if cfg.close_position and abs(state.position_qty) > 0:
            try:
                positions = await self.client.futures_position_information(
                    symbol=state.symbol
                )
                for p in positions:
                    if p["symbol"] == state.symbol:
                        real_qty = float(p["positionAmt"])
                        if abs(real_qty) > 0:
                            logger.warning(
                                f"⛈️  {state.symbol}: closing position "
                                f"{real_qty:.4f} "
                                f"(${abs(real_qty)*state.mid_price:.2f})"
                            )
                            await self._close_position_market(
                                state.symbol, real_qty
                            )
                            await self._cancel_all_tps(state, "storm close")
                        break
            except Exception as e:
                logger.error(f"{state.symbol}: storm close error — {e}")

    async def storm_guard(self):
        if not self.cfg.storm.enabled:
            logger.info("⏸️  Storm mode отключён в конфиге")
            return
        logger.info("🛡 Storm Guard активен")
        while self.is_running:
            await asyncio.sleep(0.2)
            if not self.is_running:
                break
            now = time.time()
            for symbol in list(self.states.keys()):
                state = self.states.get(symbol)
                if state is None or state.rotating:
                    continue

                if state.storm_mode and now >= state.storm_until:
                    state.storm_mode = False
                    logger.info(
                        f"🌤️  {symbol}: STORM MODE OFF "
                        f"(было: {state.storm_reason})"
                    )

    async def stream_user_data(self):
        while self.is_running:
            try:
                socket = self.bm.futures_user_socket()
                logger.info("📡 UserData WS подключён")
                async with socket as stream:
                    while self.is_running:
                        msg = await stream.recv()
                        if not isinstance(msg, dict):
                            continue
                        event = msg.get("e")
                        if event == "error":
                            # python-binance не бросает исключение, а шлёт
                            # такое сообщение — без переподключения поток
                            # молча умрёт и бот перестанет видеть fills.
                            raise RuntimeError(msg.get("m", "user stream error"))
                        if event == "ORDER_TRADE_UPDATE":
                            await self._handle_order_update(msg)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"UserData WS: {e}. Reconnect in 3s...")
                await asyncio.sleep(3)

    async def maintain_listen_key(self):
        while self.is_running:
            await asyncio.sleep(1800)
            try:
                listen_key = await self.client.futures_stream_get_listen_key()
                await self.client.futures_stream_keepalive(listenKey=listen_key)
                logger.debug("Listen key extended")
            except Exception as e:
                logger.debug(f"Listen key: {e}")

    async def _handle_order_update(self, msg: dict):
        order = msg.get("o", {})
        status = order.get("X", "")
        exec_type = order.get("x", "")
        client_id = order.get("c", "")
        symbol = order.get("s", "")

        # Объём и цена ИМЕННО этой сделки. z/ap — накопленный объём и
        # средняя цена всего ордера: при частичных исполнениях их нельзя
        # прибавлять к позиции, иначе она задваивается.
        try:
            last_qty = float(order.get("l", 0) or 0)
            last_price = float(order.get("L", 0) or 0)
        except (TypeError, ValueError):
            last_qty, last_price = 0.0, 0.0
        side = order.get("S", "")
        is_trade = exec_type == "TRADE" and last_qty > 0 and last_price > 0

        fee = None
        if is_trade:
            try:
                if order.get("N") == self.cfg.scanner.quote_asset:
                    fee = float(order.get("n", 0) or 0)
            except (TypeError, ValueError):
                fee = None

        state = self.states.get(symbol)
        if state is None:
            # Символ уже убран (ротация), но закрывающие сделки всё равно
            # должны попасть в P&L, иначе там останется «висящая» позиция.
            if is_trade:
                self.pnl.on_fill(
                    symbol=symbol, side="BUY" if side == SIDE_BUY else "SELL",
                    fill_qty=last_qty, fill_price=last_price, fee=fee,
                )
            return

        if is_trade:
            async with self._state_lock:
                old_pos = state.position_qty
                delta = last_qty if side == SIDE_BUY else -last_qty
                new_pos = round(old_pos + delta, 10)
                if abs(new_pos) < 1e-9:
                    new_pos = 0.0
                state.position_qty = new_pos
                ref_price = state.mid_price if state.mid_price > 0 else last_price
                state.position_usdt = abs(new_pos) * ref_price
                state.total_fills += 1

            realized = self.pnl.on_fill(
                symbol=symbol,
                side="BUY" if side == SIDE_BUY else "SELL",
                fill_qty=last_qty, fill_price=last_price, fee=fee,
            )
            is_opening = self._is_opening_fill(old_pos, new_pos, side)
            is_tp = (
                client_id in state.tp_order_ids
                or client_id.startswith("t_")
            )

            logger.info(
                f"💥 {symbol} FILL: {side} {last_qty} @ {last_price} | "
                f"Pos: {new_pos:.4f} (${state.position_usdt:.2f}) | "
                f"real={realized:+.4f}"
                + (" [OPENING]" if is_opening else " [CLOSING]")
                + (" [TP]" if is_tp else "")
            )

            if is_opening and not is_tp and not state.hard_kill:
                await self._place_tp_for_fill(
                    state, side, last_qty, last_price
                )

            if new_pos == 0 and state.tp_order_ids:
                await self._cancel_all_tps(state, "pos = 0")

        if status == "PARTIALLY_FILLED":
            if client_id in state.grid_order_ids:
                try:
                    await self.client.futures_cancel_order(
                        symbol=symbol, origClientOrderId=client_id
                    )
                except Exception:
                    pass

        if status in ("CANCELED", "EXPIRED", "FILLED", "REJECTED"):
            state.grid_order_ids.discard(client_id)
            state.tp_order_ids.pop(client_id, None)

    @staticmethod
    def _is_opening_fill(old_pos: float, new_pos: float, side: str) -> bool:
        if side == SIDE_BUY:
            return new_pos > 0 and (old_pos <= 0 or new_pos > old_pos)
        return new_pos < 0 and (old_pos >= 0 or new_pos < old_pos)

    async def _place_tp_for_fill(self, state, fill_side, fill_qty, fill_price):
        # Лок: обработчик fill и tp_watchdog могут выставлять TP
        # одновременно — без него на бирже появлялись дубли TP.
        async with state.tp_lock:
            await self._place_tp_locked(state)

    async def _place_tp_locked(self, state):
        tp_pct = self.cfg.strategy.tp_offset_pct / 100.0
        pnl_state = self.pnl.get_state(state.symbol)
        if pnl_state.avg_entry <= 0:
            return

        await self._cancel_all_tps(state, "re-place from avg")

        entry = pnl_state.avg_entry
        if state.position_qty > 0:
            tp_side = SIDE_SELL
            tp_price = entry * (1.0 + tp_pct)
        elif state.position_qty < 0:
            tp_side = SIDE_BUY
            tp_price = entry * (1.0 - tp_pct)
        else:
            return

        price_str = state.fmt_price(tp_price)
        qty_str = state.fmt_qty(abs(state.position_qty))
        tp_price = float(price_str)
        tp_qty = float(qty_str)
        if tp_price <= 0 or tp_qty <= 0:
            return

        cid = SymbolState.make_tp_cid()
        try:
            order = await self.client.futures_create_order(
                symbol=state.symbol, side=tp_side,
                type=ORDER_TYPE_LIMIT, timeInForce=TIME_IN_FORCE_GTC,
                price=price_str, quantity=qty_str,
                reduceOnly="true", newClientOrderId=cid,
            )
            real_cid = order.get("clientOrderId", cid)
            state.tp_order_ids[real_cid] = {
                "side": tp_side, "qty": tp_qty,
                "price": tp_price, "entry": entry,
            }
            logger.info(
                f"🎯 {state.symbol} TP {tp_side} {qty_str} @ {price_str} "
                f"(avg={entry:.6f}, +{self.cfg.strategy.tp_offset_pct}%)"
            )
        except BinanceAPIException as e:
            logger.error(f"❌ TP error: {str(e)[:150]}")

    async def tp_watchdog(self):
        while self.is_running:
            await asyncio.sleep(2)
            if not self.is_running:
                break
            for symbol in list(self.states.keys()):
                state = self.states.get(symbol)
                if state is None or state.mid_price <= 0:
                    continue
                if state.storm_mode:
                    continue
                try:
                    pos_info = await self.client.futures_position_information(
                        symbol=symbol
                    )
                    real_pos = 0.0
                    real_entry = 0.0
                    for p in pos_info:
                        if p["symbol"] == symbol:
                            real_pos = float(p["positionAmt"])
                            real_entry = float(p.get("entryPrice", 0) or 0)
                            break

                    # Расхождение должно продержаться две проверки подряд:
                    # REST может показать сделку раньше, чем придёт событие
                    # по WS, и тогда мгновенная «синхронизация» + fill
                    # из WS посчитали бы её дважды.
                    if abs(real_pos - state.position_qty) > 1e-9:
                        state.pos_mismatch_count += 1
                    else:
                        state.pos_mismatch_count = 0

                    if state.pos_mismatch_count >= 2:
                        logger.info(
                            f"🔄 {symbol}: sync pos "
                            f"{state.position_qty:.4f} → {real_pos:.4f}"
                        )
                        state.position_qty = real_pos
                        state.position_usdt = abs(real_pos) * state.mid_price
                        state.pos_mismatch_count = 0
                        self.pnl.sync_position(symbol, real_pos, real_entry)

                    open_orders = await self.client.futures_get_open_orders(
                        symbol=symbol
                    )
                    open_cids = {o.get("clientOrderId", "") for o in open_orders}
                    # Только наши TP: reduceOnly-ордера сетки в reduce mode
                    # тоже reduceOnly, но это не тейк-профит.
                    tp_on_exchange = [
                        o for o in open_orders
                        if o.get("clientOrderId", "").startswith("t_")
                    ]

                    for c in [c for c in list(state.tp_order_ids.keys())
                              if c not in open_cids]:
                        state.tp_order_ids.pop(c, None)

                    if (abs(real_pos) > 0
                            and abs(real_pos - state.position_qty) <= 1e-9
                            and not tp_on_exchange
                            and not state.hard_kill):
                        pnl_state = self.pnl.get_state(symbol)
                        if pnl_state.avg_entry > 0:
                            logger.warning(
                                f"🔄 {symbol}: TP missing, restore from "
                                f"avg={pnl_state.avg_entry:.6f}"
                            )
                            opening_side = (
                                SIDE_BUY if real_pos > 0 else SIDE_SELL
                            )
                            await self._place_tp_for_fill(
                                state, opening_side, abs(real_pos),
                                pnl_state.avg_entry,
                            )

                    if real_pos == 0 and state.tp_order_ids:
                        await self._cancel_all_tps(state, "pos = 0")
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    logger.debug(f"{symbol}: watchdog: {e}")

    async def _update_symbol_quotes(self, state: SymbolState):
        if state.hard_kill or state.rotating or state.mid_price <= 0:
            return
        if state.fading:
            return
        if state.storm_mode:
            return

        cfg = self.cfg.strategy
        t_left = max(cfg.time_horizon, 0.01)

        r_price = state.model.reservation_price(
            state.mid_price, state.position_qty, t_left
        )
        spread = state.model.optimal_spread(state.mid_price, t_left)

        half_pct = self._calculate_half_spread_pct(state)
        min_half_spread = state.mid_price * (half_pct / 100.0)
        max_half_spread = state.mid_price * (cfg.spread_max_half_pct / 100.0)
        # A-S часть ограничиваем сверху так же, как динамический спред
        half_spread = max(min(spread / 2.0, max_half_spread), min_half_spread)

        logger.debug(
            f"{state.symbol}: half_spread={half_pct:.3f}% "
            f"(NATR={state.last_natr:.2f}%, "
            f"market="
            f"{(state.best_ask - state.best_bid) / state.mid_price * 100:.3f}%)"
        )

        if state.last_latency_ms > cfg.max_latency_ms:
            logger.warning(
                f"⚠️  {state.symbol}: HIGH LATENCY "
                f"({state.last_latency_ms:.0f}ms)"
            )
            half_spread *= 2.0

        spacing_by_pct = state.mid_price * (cfg.level_spacing_pct / 100.0)
        spacing_by_ticks = cfg.level_spacing_ticks * state.tick_size
        spacing = max(spacing_by_pct, spacing_by_ticks)

        bid_base = r_price - half_spread
        ask_base = r_price + half_spread

        inv_limit = cfg.max_inventory_usdt
        inv_abs = abs(state.position_usdt)

        if inv_abs >= inv_limit * 1.5:
            await self._hard_kill(state, f"inventory ${inv_abs:.2f}")
            return

        prev_reduce = state.reduce_mode
        state.reduce_mode = inv_abs >= inv_limit * 0.9
        if state.reduce_mode != prev_reduce:
            if state.reduce_mode:
                logger.warning(
                    f"⚙️  {state.symbol}: REDUCE MODE ON (${inv_abs:.2f})"
                )
            else:
                logger.info(f"✅ {state.symbol}: REDUCE MODE OFF")

        if inv_abs > inv_limit * 0.8 and not state.reduce_mode:
            # Знак важен: position_usdt — модуль, поэтому раньше и лонг,
            # и шорт сдвигали котировки вниз. Шорт должен сдвигать вверх.
            signed_usdt = math.copysign(inv_abs, state.position_qty)
            skew = (signed_usdt / inv_limit) * state.tick_size * 10
            bid_base -= skew
            ask_base -= skew

        place_buys = True
        place_sells = True
        reduce_only = False

        if state.reduce_mode:
            reduce_only = True
            if state.position_qty > 0:
                place_buys = False
            elif state.position_qty < 0:
                place_sells = False
            tp_covered = state.tp_covered_qty()
            if max(0.0, abs(state.position_qty) - tp_covered) <= 0:
                place_buys = False
                place_sells = False

        if reduce_only:
            pnl_state = self.pnl.get_state(state.symbol)
            avg = pnl_state.avg_entry
            if place_sells and state.position_qty > 0 and avg > 0:
                if ask_base < avg:
                    place_sells = False
            if place_buys and state.position_qty < 0 and avg > 0:
                if bid_base > avg:
                    place_buys = False

        async with state.grid_lock:
            if state.storm_mode or state.hard_kill or state.rotating:
                return

            await self._cancel_grid_orders(state)

            if state.storm_mode or state.hard_kill or state.rotating:
                return

            if not (place_buys or place_sells):
                return

            qty = state.round_qty(cfg.order_size_usdt / state.mid_price)
            if qty <= 0:
                return

            tasks = []
            if place_buys:
                for i in range(cfg.num_levels):
                    p = state.round_price(bid_base - i * spacing)
                    if p > 0:
                        tasks.append(
                            self._place_grid(state, SIDE_BUY, p, qty, reduce_only)
                        )
            if place_sells:
                for i in range(cfg.num_levels):
                    p = state.round_price(ask_base + i * spacing)
                    if p > 0:
                        tasks.append(
                            self._place_grid(state, SIDE_SELL, p, qty, reduce_only)
                        )

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            if state.storm_mode or state.hard_kill or state.rotating:
                if state.grid_order_ids:
                    logger.warning(
                        f"⚠️  {state.symbol}: storm fired during placement "
                        f"— cancelling {len(state.grid_order_ids)} fresh orders"
                    )
                    await self._cancel_grid_orders(state)

    async def _hard_kill(self, state: SymbolState, reason: str, force: bool = False):
        """
        Останавливает торговлю символом: снимает сетку, TP и закрывает
        позицию по рынку. Раньше HARD KILL по инвентарю только снимал
        сетку, а позиция оставалась открытой без восстановления TP.
        Символ остаётся выключенным до ротации или перезапуска.
        """
        if state.hard_kill and not force:
            return
        state.hard_kill = True
        state.reduce_mode = True
        logger.critical(f"🚨 {state.symbol}: HARD KILL ({reason})")
        async with state.grid_lock:
            await self._cancel_grid_orders(state)
        await self._close_position_market(state.symbol, state.position_qty)
        await self._cancel_all_tps(state, "hard kill")
        if state.mid_price > 0:
            self.pnl.force_log(state.symbol, state.mid_price)

    async def _place_grid(self, state, side, price, qty, reduce_only=False) -> bool:
        if price <= 0 or qty <= 0:
            return False

        cid = SymbolState.make_grid_cid()
        t0 = time.perf_counter()
        try:
            kwargs = dict(
                symbol=state.symbol, side=side,
                type=ORDER_TYPE_LIMIT, timeInForce=TIME_IN_FORCE_GTX,
                price=state.fmt_price(price), quantity=state.fmt_qty(qty),
                newClientOrderId=cid,
            )
            if reduce_only:
                kwargs["reduceOnly"] = "true"

            order = await self.client.futures_create_order(**kwargs)
            state.last_latency_ms = (time.perf_counter() - t0) * 1000
            state.grid_order_ids.add(order.get("clientOrderId", cid))
            state.orders_placed_total += 1

            logger.info(
                f"📝 {state.symbol} {side} {qty} @ {price} "
                f"{'[REDUCE]' if reduce_only else ''} "
                f"| latency={state.last_latency_ms:.0f}ms"
            )
            return True

        except BinanceAPIException as e:
            err = str(e)
            err_lower = err.lower()
            code = getattr(e, "code", None)

            # -2027: превышен максимум позиции — стоп спам
            if code == -2027 or "maximum allowable position" in err_lower:
                if not state.hard_kill:
                    # Вызов идёт изнутри grid_lock — сам hard kill
                    # запускаем фоном, иначе будет deadlock на локе.
                    state.hard_kill = True
                    self._spawn(self._hard_kill(
                        state, "-2027 max position exceeded", force=True
                    ))
                return False

            # -5022 GTX post-only
            if "GTX" in err or "post-only" in err_lower:
                state.orders_rejected_gtx += 1
                logger.debug(
                    f"{state.symbol} GTX reject {side} @ {price}: {err[:100]}"
                )
            else:
                logger.error(
                    f"❌ {state.symbol} {side} {qty} @ {price} | "
                    f"code={code} msg={err}"
                )
            return False
        except Exception as e:
            logger.error(f"❌ {state.symbol} {side} {qty} @ {price} | {e}")
            return False

    async def _check_fading(self, state: SymbolState) -> Tuple[bool, str]:
        symbol = state.symbol
        cfg = self.cfg.scanner

        warmup_sec = 600
        if time.time() - state.added_at < warmup_sec:
            return False, ""

        try:
            k1m = await self.client.futures_klines(
                symbol=symbol, interval="1m", limit=61
            )
        except Exception:
            return False, ""

        k1m = k1m[:-1]
        if len(k1m) < 30:
            return False, ""

        volumes = [float(x[7]) for x in k1m]
        recent_avg = float(np.mean(volumes[-5:]))
        baseline_window = (
            volumes[-35:-5] if len(volumes) >= 35 else volumes[:-5]
        )
        if not baseline_window:
            return False, ""
        baseline_median = float(np.median(baseline_window))

        if baseline_median > 0:
            drop_ratio = 1.0 - (recent_avg / baseline_median)
            if drop_ratio > cfg.volume_drop_pct / 100.0:
                return True, f"volume drop {drop_ratio*100:.1f}%"

        try:
            k5m = await self.client.futures_klines(
                symbol=symbol, interval="5m", limit=20
            )
        except Exception:
            return False, ""

        k5m = k5m[:-1]
        current_natr = SymbolScanner.compute_natr_5m(k5m)
        state.last_natr = current_natr
        state.natr_history.append((time.time(), current_natr))

        if len(state.natr_history) >= 10:
            local_max = max(v for _, v in state.natr_history)
            if local_max > 0 and current_natr < local_max / cfg.natr_drop_factor:
                return True, (
                    f"NATR drop: {current_natr:.2f}% < "
                    f"{local_max:.2f}%/{cfg.natr_drop_factor:g}"
                )

        return False, ""

    async def _find_better_symbol(
        self, exclude: str, current_natr: float
    ) -> Optional[str]:
        top = await self.scanner.scan()
        mult = self.cfg.scanner.better_coin_natr_mult
        for c in top:
            if c.symbol == exclude:
                continue
            if c.natr_5m > current_natr * mult:
                return c.symbol
        return None

    async def _rotate_to_symbol(self, old_symbol: str, new_symbol: str):
        async with self._rotation_lock:
            if old_symbol not in self.states:
                # Пока ждали лок, монету уже сменила другая ротация
                logger.info(f"🔄 {old_symbol}: ротация уже выполнена, пропуск")
                return
            if new_symbol in self.states:
                return
            await self._rotate_locked(old_symbol, new_symbol)

    async def _rotate_locked(self, old_symbol: str, new_symbol: str):
        logger.warning(f"🔄 ROTATION: {old_symbol} → {new_symbol}")
        state = self.states.get(old_symbol)
        if state:
            state.rotating = True
            state.hard_kill = True
            state.last_rotation_at = time.time()
            try:
                await self.client.futures_cancel_all_open_orders(
                    symbol=old_symbol
                )
            except Exception:
                pass
            try:
                positions = await self.client.futures_position_information(
                    symbol=old_symbol
                )
                for p in positions:
                    if p["symbol"] == old_symbol:
                        real_qty = float(p["positionAmt"])
                        if abs(real_qty) > 0:
                            await self._close_position_market(
                                old_symbol, real_qty
                            )
                        break
            except Exception as e:
                logger.error(f"close: {e}")
            async with self._state_lock:
                self.states.pop(old_symbol, None)

        new_state = await self._prepare_symbol(new_symbol)
        if new_state:
            async with self._state_lock:
                self.states[new_symbol] = new_state
            logger.info(f"✅ Ротация завершена, торгуем {new_symbol}")
        else:
            logger.error(f"❌ Не удалось подготовить {new_symbol}")

    async def fading_monitor(self):
        if self.cfg.manual_symbol.strip():
            return
        cooldown_sec = 300
        while self.is_running:
            await asyncio.sleep(self.cfg.scanner.fading_check_interval_sec)
            if not self.is_running:
                break
            for symbol in list(self.states.keys()):
                state = self.states.get(symbol)
                if state is None or state.rotating:
                    continue
                if state.mid_price <= 0 or state.storm_mode:
                    continue
                if time.time() - state.last_rotation_at < cooldown_sec:
                    continue
                try:
                    fading, reason = await self._check_fading(state)
                    if not fading and state.fading:
                        # Раньше флаг не сбрасывался никогда: если замену
                        # не нашли, монета навсегда оставалась без котировок.
                        logger.info(f"🌤️  {symbol}: активность восстановилась")
                        state.fading = False
                    elif fading and not state.fading:
                        logger.warning(f"🌫️  {symbol}: FADING — {reason}")
                        state.fading = True
                        new_sym = await self._find_better_symbol(
                            exclude=symbol, current_natr=state.last_natr,
                        )
                        if new_sym:
                            await self._rotate_to_symbol(symbol, new_sym)
                        else:
                            logger.warning(
                                f"🌫️  {symbol}: замена не найдена"
                            )
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    logger.debug(f"{symbol}: fading_monitor: {e}")

    async def scanner_loop(self):
        if self.cfg.manual_symbol.strip():
            return
        while self.is_running:
            await asyncio.sleep(self.cfg.scanner.scan_interval_sec)
            if not self.is_running:
                break
            try:
                top = await self.scanner.scan()
                if not top:
                    logger.warning("🔍 Сканер: подходящих монет не найдено")
                    continue
                best = top[0]
                top_symbols = {c.symbol for c in top}

                if not self.states:
                    async with self._rotation_lock:
                        if not self.states:
                            new_state = await self._prepare_symbol(best.symbol)
                            if new_state:
                                async with self._state_lock:
                                    self.states[best.symbol] = new_state
                                logger.info(f"✅ Приняли {best.symbol} в работу")
                    continue

                current_sym = list(self.states.keys())[0]
                if current_sym == best.symbol:
                    continue
                current_state = self.states.get(current_sym)
                if current_state is None:
                    continue

                if time.time() - current_state.last_rotation_at < 300:
                    continue

                if current_sym not in top_symbols:
                    if abs(current_state.position_usdt) > 0.5:
                        continue
                    logger.warning(
                        f"🔄 {current_sym} выпал из топа → {best.symbol}"
                    )
                    await self._rotate_to_symbol(current_sym, best.symbol)
                    continue

                current_natr = current_state.last_natr
                if current_natr <= 0:
                    for c in top:
                        if c.symbol == current_sym:
                            current_natr = c.natr_5m
                            break
                    if current_natr <= 0:
                        current_natr = 0.1

                if best.natr_5m > current_natr * self.cfg.scanner.better_coin_natr_mult:
                    logger.warning(
                        f"⭐ Лучшая монета: {best.symbol} "
                        f"NATR={best.natr_5m:.2f}%"
                    )
                    await self._rotate_to_symbol(current_sym, best.symbol)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"scanner_loop: {e}", exc_info=True)

    async def trading_loop(self):
        while self.is_running:
            symbols = list(self.states.keys())
            prices = {}
            for sym in symbols:
                st = self.states.get(sym)
                if st and st.mid_price > 0:
                    prices[sym] = st.mid_price
            # unrealized и position_usdt пересчитываем каждый цикл: раньше
            # unrealized обновлялся раз в 60 с (только для лога), и loss stop
            # смотрел на устаревшие данные.
            for sym, mid in prices.items():
                self.pnl.update_unrealized(sym, mid)
                st = self.states.get(sym)
                if st is not None:
                    st.position_usdt = abs(st.position_qty) * mid
            if prices:
                self.pnl.log_if_due(prices)

            if await self._check_max_drawdown():
                return

            if self.cfg.storm.enabled:
                pos_stop = self.cfg.storm.position_stop_pct / 100.0
                for symbol in list(self.states.keys()):
                    state = self.states.get(symbol)
                    if state is None or state.rotating or state.storm_mode:
                        continue
                    if abs(state.position_qty) < 1e-9 or state.mid_price <= 0:
                        continue
                    pnl_state = self.pnl.get_state(symbol)
                    avg = pnl_state.avg_entry
                    if avg <= 0:
                        continue
                    if state.position_qty > 0:
                        deviation = (state.mid_price - avg) / avg
                    else:
                        deviation = (avg - state.mid_price) / avg
                    if deviation < -pos_stop:
                        logger.critical(
                            f"🛑 {symbol}: POSITION STOP "
                            f"(deviation {deviation*100:.2f}%, "
                            f"avg={avg:.6f}, mid={state.mid_price:.6f})"
                        )
                        await self._cancel_grid_orders(state)
                        await self._close_position_market(
                            symbol, state.position_qty
                        )
                        await self._cancel_all_tps(state, "position stop")
                        state.storm_mode = True
                        state.storm_until = time.time() + 15.0
                        state.storm_reason = (
                            f"position stop {deviation*100:.2f}%"
                        )

            if self.cfg.storm.enabled and self.initial_balance > 0:
                threshold = (
                    -self.initial_balance * self.cfg.storm.loss_stop_pct
                )
                for symbol in list(self.states.keys()):
                    state = self.states.get(symbol)
                    if (state is None or state.rotating or state.storm_mode
                            or state.hard_kill):
                        continue
                    pnl_state = self.pnl.get_state(symbol)
                    unreal = pnl_state.unrealized_pnl
                    if unreal < threshold and abs(state.position_usdt) > 0:
                        await self._hard_kill(
                            state,
                            f"LOSS STOP: unrealized ${unreal:.2f} < "
                            f"${threshold:.2f}",
                        )

            tasks = []
            for symbol in symbols:
                state = self.states.get(symbol)
                if state is None or state.rotating:
                    continue
                if (abs(state.position_usdt)
                        >= self.cfg.strategy.max_inventory_usdt * 1.5):
                    if not state.hard_kill:
                        await self._hard_kill(
                            state, f"inventory ${state.position_usdt:.2f}"
                        )
                    continue
                tasks.append(self._update_symbol_quotes(state))

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(self.cfg.strategy.min_order_lifetime_sec)

    async def _check_max_drawdown(self) -> bool:
        """
        Общий стоп сессии по strategy.max_drawdown_pct (доля баланса,
        0.10 = 10%). Раньше параметр был в конфиге, но нигде не
        использовался. При срабатывании бот останавливается, позиции
        закрываются в _shutdown_safe.
        """
        limit = self.cfg.strategy.max_drawdown_pct
        if self.drawdown_triggered or limit <= 0 or self.initial_balance <= 0:
            return self.drawdown_triggered
        net = self.pnl.session_net_pnl()
        if net < -self.initial_balance * limit:
            self.drawdown_triggered = True
            logger.critical(
                f"🛑 MAX DRAWDOWN: P&L сессии {net:+.2f} USDT "
                f"< -{limit*100:.1f}% баланса — останавливаю бота"
            )
            self.is_running = False
            return True
        return False

    async def _shutdown_safe(self, tasks: list):
        logger.info("=" * 60)
        logger.info("🔻 GRACEFUL SHUTDOWN")
        logger.info("=" * 60)
        self.is_running = False

        tasks = list(tasks) + list(self._bg_tasks)
        for t in tasks:
            t.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=5.0,
            )
            logger.info("   ✅ Задачи остановлены")
        except asyncio.TimeoutError:
            logger.warning("   ⚠️  Timeout 5 сек")

        logger.info("🚫 Отменяю ордера...")
        for symbol in list(self.states.keys()):
            try:
                await asyncio.wait_for(
                    self.client.futures_cancel_all_open_orders(symbol=symbol),
                    timeout=5.0,
                )
                state = self.states.get(symbol)
                if state:
                    state.grid_order_ids.clear()
                    state.tp_order_ids.clear()
                logger.info(f"   ✅ {symbol}: ордера отменены")
            except Exception as e:
                logger.warning(f"   ⚠️  {symbol}: {e}")

        logger.info("🔒 Закрываю позиции...")
        for symbol in list(self.states.keys()):
            try:
                positions = await asyncio.wait_for(
                    self.client.futures_position_information(symbol=symbol),
                    timeout=5.0,
                )
                for p in positions:
                    if p["symbol"] == symbol:
                        q = float(p["positionAmt"])
                        if q != 0:
                            await asyncio.wait_for(
                                self._close_position_market(symbol, q),
                                timeout=10.0,
                            )
                        else:
                            logger.info(f"   ✅ {symbol}: позиция 0")
                        break
            except Exception as e:
                logger.error(f"   ❌ {symbol}: {e}")

        if self.cfg.pnl.summary_on_exit:
            try:
                self.pnl.print_summary()
            except Exception as e:
                logger.error(f"P&L summary: {e}")

        try:
            await asyncio.wait_for(self.client.close_connection(), timeout=5.0)
        except Exception:
            pass
        logger.info("✅ Shutdown завершён.")

    async def run(self):
        self.is_running = True
        # SIGTERM (systemd, docker stop, kill) раньше убивал процесс без
        # graceful shutdown — ордера и позиции оставались на бирже.
        try:
            asyncio.get_running_loop().add_signal_handler(
                signal.SIGTERM, self._on_sigterm
            )
        except (NotImplementedError, RuntimeError):
            pass  # Windows: сигналы в event loop не поддерживаются
        await self.initialize()

        manual = self.cfg.manual_symbol.strip().upper()
        if manual:
            logger.info(f"⚙️  Ручной режим: {manual}")
            state = await self._prepare_symbol(manual)
            if state is None:
                logger.critical(f"❌ Не удалось подготовить {manual}")
                return
            async with self._state_lock:
                self.states[manual] = state
        else:
            logger.info("🔍 Авто-режим: сканер + детектор затухания")
            top = await self.scanner.scan()
            if top:
                new_state = await self._prepare_symbol(top[0].symbol)
                if new_state:
                    async with self._state_lock:
                        self.states[top[0].symbol] = new_state
                    logger.info(f"✅ Начальная монета: {top[0].symbol}")

        tasks = [
            asyncio.create_task(self.stream_order_books(), name="orderbook_ws"),
            asyncio.create_task(self.stream_user_data(), name="userdata_ws"),
            asyncio.create_task(self.maintain_listen_key(), name="listen_key"),
            asyncio.create_task(self.trading_loop(), name="trading_loop"),
            asyncio.create_task(self.tp_watchdog(), name="tp_watchdog"),
            asyncio.create_task(self.storm_guard(), name="storm_guard"),
            asyncio.create_task(self.btc_monitor(), name="btc_monitor"),
            asyncio.create_task(self.natr_updater(), name="natr_updater"),
        ]

        if not manual:
            tasks.append(
                asyncio.create_task(self.scanner_loop(), name="scanner_loop")
            )
            tasks.append(
                asyncio.create_task(self.fading_monitor(), name="fading_monitor")
            )

        try:
            while self.is_running:
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("🛑 Ctrl+C")
            raise
        finally:
            try:
                await self._shutdown_safe(tasks)
            except Exception as e:
                logger.exception(f"shutdown error: {e}")

    def _on_sigterm(self):
        logger.warning("🛑 SIGTERM — останавливаюсь")
        self.is_running = False

    async def stop(self):
        self.is_running = False


MarketMakerBot = MultiMarketMakerBot

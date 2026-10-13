"""
Офлайн-тесты нового движка (mm/). Сеть не нужна.
Запуск: python -m pytest tests -q
"""
import asyncio
import os
import random
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm.bot import Bot, SymbolTrader  # noqa: E402
from mm.config import BotConfig, QuoteConfig  # noqa: E402
from mm.exchange import ERR_POST_ONLY, PaperExchange  # noqa: E402
from mm.marketdata import Book  # noqa: E402
from mm.risk import BtcShock, MarkoutTracker, RateLimiter, symbol_shock  # noqa: E402
from mm.scanner import Candidate, Scanner, kline_stats, score_candidate  # noqa: E402
from mm.strategy import (  # noqa: E402
    QuoteInputs, Side, SymbolFilters, compute_quote, fmt, needs_requote, round_to,
)

F = SymbolFilters(tick=0.0001, step=1.0, min_qty=1.0, min_notional=5.0)


def inputs(**kw):
    d = dict(bid=0.9999, ask=1.0001, bid_qty=1000, ask_qty=1000, depth_imb=0.0,
             flow_imb=0.0, sigma_1s_bps=0.5, adverse_bps=0.0, position=0.0,
             avg_entry=0.0, hold_s=0.0, allow_open=True, latency_bad=False,
             maker_fee_bps=2.0, max_inventory_usdt=18.0)
    d.update(kw)
    return QuoteInputs(**d)


# ---------------------------------------------------------------- strategy

def test_flat_quote_symmetric_and_covers_fees():
    q = compute_quote(inputs(), F, QuoteConfig())
    assert q.bid and q.ask and not q.bid.reduce_only and not q.ask.reduce_only
    mid = 1.0
    assert (mid - q.bid.price) / mid * 1e4 >= 3.0 - 1     # fee 2 + edge 1 (минус тик)
    assert (q.ask.price - mid) / mid * 1e4 >= 3.0 - 1
    assert abs((mid - q.bid.price) - (q.ask.price - mid)) <= F.tick * 1.01


def test_long_position_skews_down_and_reduce_only_full_size():
    flat = compute_quote(inputs(), F, QuoteConfig())
    q = compute_quote(inputs(position=12.0, avg_entry=1.0, hold_s=60), F, QuoteConfig())
    assert q.res < flat.res
    assert q.ask.reduce_only and q.ask.qty == 12.0
    assert q.bid is None or q.bid.qty < flat.bid.qty


def test_full_inventory_blocks_same_side():
    q = compute_quote(inputs(position=18.0, avg_entry=1.0, hold_s=60), F, QuoteConfig())
    assert q.bid is None
    assert q.ask.reduce_only


def test_short_position_mirrors():
    q = compute_quote(inputs(position=-12.0, avg_entry=1.0, hold_s=60), F, QuoteConfig())
    flat = compute_quote(inputs(), F, QuoteConfig())
    assert q.res > flat.res
    assert q.bid.reduce_only and q.bid.qty == 12.0


def test_exit_patience_protects_entry():
    cfg = QuoteConfig()
    q = compute_quote(inputs(position=6.0, avg_entry=1.002, hold_s=5), F, cfg)
    assert q.ask.price >= 1.002 * (1 + 5 / 1e4) - 1e-12      # вход + 2·fee + edge
    q2 = compute_quote(inputs(position=6.0, avg_entry=1.002, hold_s=cfg.exit_patience_s + 1), F, cfg)
    assert q2.ask.price < q.ask.price


def test_no_open_only_reduces():
    q = compute_quote(inputs(allow_open=False), F, QuoteConfig())
    assert q.bid is None and q.ask is None
    q = compute_quote(inputs(allow_open=False, position=6.0, avg_entry=1.0), F, QuoteConfig())
    assert q.bid is None and q.ask.reduce_only


def test_post_only_never_crosses():
    # сильный alpha вверх не должен сделать bid ≥ ask
    q = compute_quote(inputs(flow_imb=1.0, depth_imb=1.0, bid_qty=1, ask_qty=1e6), F, QuoteConfig())
    assert q.bid.price <= 1.0001 - F.tick + 1e-12
    assert q.ask.price >= 0.9999 + F.tick - 1e-12


def test_volatility_and_adverse_widen():
    base = compute_quote(inputs(), F, QuoteConfig()).half_bps
    assert compute_quote(inputs(sigma_1s_bps=10), F, QuoteConfig()).half_bps > base
    assert compute_quote(inputs(adverse_bps=4), F, QuoteConfig()).half_bps > base
    assert compute_quote(inputs(sigma_1s_bps=1000), F, QuoteConfig()).half_bps == QuoteConfig().max_half_spread_bps


def test_min_notional_respected():
    q = compute_quote(inputs(), SymbolFilters(0.0001, 1.0, 1.0, 50.0), QuoteConfig())
    assert q.bid is None and q.ask is None


def test_fmt_and_round():
    assert fmt(0.00001234, 0.0000001) == "0.0000123"
    assert fmt(12.0, 1.0) == "12"
    assert fmt(101.23, 0.01) == "101.23"
    assert round_to(1.00019, 0.0001, up=False) == pytest.approx(1.0001)
    assert round_to(1.00011, 0.0001, up=True) == pytest.approx(1.0002)


def test_needs_requote_threshold():
    s = Side(1.0, 6, False)
    assert not needs_requote(1.00005, 6, False, s, 0.00001, 1.0)
    assert needs_requote(1.0002, 6, False, s, 0.00001, 1.0)
    assert needs_requote(1.0, 7, False, s, 0.00001, 1.0)


# ---------------------------------------------------------------- risk

def test_markout_positive_when_spread_earned(tmp_path):
    m = MarkoutTracker([1, 5], 5, 10, 5, str(tmp_path / "f.csv"))
    m.on_fill("X", "BUY", 0.9997, 10, 2.0, 1.0, ts=0)
    m.update({"X": 1.0}, now=5)
    assert m.avg_net("X") == pytest.approx(3.0 - 2.0, abs=0.01)
    assert m.adverse_bps("X") == 0.0
    m.on_fill("X", "BUY", 0.9997, 10, 2.0, 1.0, ts=10)
    m.update({"X": 0.999}, now=15)            # цена ушла против нас
    assert m.adverse_bps("X") > 0
    assert len(open(tmp_path / "f.csv").read().splitlines()) == 3


def test_shocks_and_rate_limit():
    assert symbol_shock(20, 4, 1, 1, 4, 4)
    assert symbol_shock(5, 4, 9, 2, 4, 4)
    assert symbol_shock(5, 4, 2, 2, 4, 4) is None
    b = BtcShock(40, 3)
    assert not b.on_mid(100, 0)
    assert b.on_mid(100.5, 1)
    r = RateLimiter(3)
    for _ in range(3):
        r.hit(0)
    assert not r.allow(1)
    assert r.allow(11)


# ---------------------------------------------------------------- scanner

def test_kline_stats_trend_and_sigma():
    flat = [[0, 0, 0, 0, str(100 + (i % 2) * 0.1), 0, 0, 0, "100"] for i in range(61)]
    up = [[0, 0, 0, 0, str(100 * 1.001 ** i), 0, 0, 0, "50"] for i in range(61)]
    assert kline_stats(flat)["trend"] < 0.1
    assert kline_stats(up)["trend"] > 0.9
    assert kline_stats(flat)["tpm"] == 100


def test_score_prefers_wide_spread_low_vol():
    a = Candidate("A", spread_bps=10, trades_per_min=100, sigma_1m_bps=10)
    b = Candidate("B", spread_bps=4, trades_per_min=100, sigma_1m_bps=10)
    c = Candidate("C", spread_bps=10, trades_per_min=100, sigma_1m_bps=60)
    for x in (a, b, c):
        score_candidate(x, 2.0, 0.1)
    assert a.score > b.score and a.score > c.score


class FakeScanClient:
    async def futures_ticker(self):
        return [
            {"symbol": "GOODUSDT", "quoteVolume": "1e8", "priceChangePercent": "3"},
            {"symbol": "PUMPUSDT", "quoteVolume": "1e8", "priceChangePercent": "60"},
            {"symbol": "BTCUSDT", "quoteVolume": "1e10", "priceChangePercent": "1"},
            {"symbol": "TIGHTUSDT", "quoteVolume": "1e8", "priceChangePercent": "2"},
        ]

    async def futures_mark_price(self):
        return [{"symbol": s, "lastFundingRate": "0.0001"}
                for s in ("GOODUSDT", "PUMPUSDT", "BTCUSDT", "TIGHTUSDT")]

    async def futures_orderbook_ticker(self):
        return [
            {"symbol": "GOODUSDT", "bidPrice": "0.9995", "askPrice": "1.0005"},
            {"symbol": "TIGHTUSDT", "bidPrice": "0.99999", "askPrice": "1.00001"},
            {"symbol": "PUMPUSDT", "bidPrice": "0.9995", "askPrice": "1.0005"},
        ]

    async def futures_klines(self, symbol, interval, limit):
        return [[0, 0, 0, 0, str(1 + (i % 2) * 0.0005), 0, 0, 0, "120"] for i in range(limit)]


def test_scanner_filters_pumps_top_coins_and_tight_spreads():
    cfg = BotConfig()
    cfg.scanner.spread_samples = 1
    sc = Scanner(FakeScanClient(), cfg, lambda s: SymbolFilters(0.00001, 1, 1, 5))
    out = asyncio.run(sc.scan())
    assert [c.symbol for c in out] == ["GOODUSDT"]


# ---------------------------------------------------------------- paper exchange

def make_paper(latency_ms=0.0):
    cfg = BotConfig()
    cfg.paper.latency_ms = latency_ms
    books = {}
    ex = PaperExchange(cfg, books)
    b = Book("XUSDT")
    b.on_book_ticker(0.9999, 1000, 1.0001, 1000, time.time())
    books["XUSDT"] = b
    fills, statuses = [], []

    async def on_fill(f):
        fills.append(f)

    async def on_status(s, c, st):
        statuses.append((c, st))
    ex.on_fill, ex.on_status = on_fill, on_status
    return ex, b, fills, statuses


def test_paper_trade_through_only():
    ex, b, fills, _ = make_paper()

    async def go():
        assert (await ex.place_limit("XUSDT", "BUY", "0.9998", "10", "c1", False)).ok
        await ex.on_trade("XUSDT", 0.9998, 5, True, time.time())   # на нашей цене — очередь
        assert not fills
        await ex.on_trade("XUSDT", 0.9997, 5, True, time.time())   # сквозь — исполнение
        assert len(fills) == 1 and fills[0].maker and fills[0].qty == 10
    asyncio.run(go())
    assert ex.positions["XUSDT"][0] == 10


def test_paper_post_only_and_reduce_only():
    ex, b, fills, _ = make_paper()

    async def go():
        r = await ex.place_limit("XUSDT", "BUY", "1.0001", "10", "c1", False)
        assert not r.ok and r.code == ERR_POST_ONLY
        r = await ex.place_limit("XUSDT", "SELL", "1.0003", "10", "c2", True)
        assert not r.ok                                           # нет позиции
    asyncio.run(go())


def test_paper_cash_matches_pnl():
    ex, b, fills, _ = make_paper()

    async def go():
        await ex.place_limit("XUSDT", "BUY", "0.9998", "10", "c1", False)
        await ex.on_trade("XUSDT", 0.9990, 50, True, time.time())
        b.on_book_ticker(1.0004, 1000, 1.0006, 1000, time.time())
        await ex.place_limit("XUSDT", "SELL", "1.0008", "10", "c2", True)
        await ex.on_trade("XUSDT", 1.0010, 50, False, time.time())
    asyncio.run(go())
    fees = sum(f.fee_usdt for f in fills)
    assert ex.positions["XUSDT"][0] == 0
    assert ex.cash - 100 == pytest.approx(10 * (1.0008 - 0.9998) - fees)


# ---------------------------------------------------------------- end-to-end

class SimMarket:
    """Синтетический рынок: случайное блуждание + поток сделок."""

    def __init__(self, ex, book, seed=1, drift_bps=0.0):
        self.ex, self.book = ex, book
        self.rng = random.Random(seed)
        self.mid = 1.0
        self.drift = drift_bps

    async def tick(self):
        self.mid *= 1 + (self.rng.gauss(0, 1.0) + self.drift) / 1e4
        half = 0.0003
        bid = round(self.mid - half, 4)
        ask = round(self.mid + half, 4)
        now = time.time()
        self.book.on_book_ticker(bid, 1000, ask, 1000, now)
        await self.ex.on_book("XUSDT")
        side = self.rng.random() < 0.5
        px = round(self.mid + (-1 if side else 1) * half * self.rng.uniform(0, 3), 4)
        self.book.on_trade(px, 10, side, now)
        await self.ex.on_trade("XUSDT", px, 10, side, now)


def build_bot(tmp_path, latency_ms=2.0):
    cfg = BotConfig()
    cfg.log.dir = str(tmp_path)
    cfg.paper.latency_ms = latency_ms
    cfg.quote.requote_min_interval_ms = 0
    cfg.quote.loop_timeout_ms = 20
    bot = Bot(cfg)
    bot.ex._filters["XUSDT"] = F
    bot.ex.on_fill = bot._on_fill
    bot.ex.on_status = bot._on_status
    bot.running = True
    book = bot.md.book("XUSDT")
    # прогреть волатильность, чтобы разрешить входы
    for i in range(40):
        book.vol_fast.on_mid(1.0 + (i % 2) * 1e-4, 1000 + i)
    trader = SymbolTrader(bot, "XUSDT", F, book)
    bot.traders["XUSDT"] = trader
    return bot, trader, book


def test_end_to_end_paper_consistency(tmp_path):
    bot, trader, book = build_bot(tmp_path)

    async def go():
        sim = SimMarket(bot.ex, book, seed=3)
        await sim.tick()
        trader.start()
        for _ in range(400):
            await sim.tick()
            await asyncio.sleep(0.003)
        await trader.stop(flatten=True)
    asyncio.run(go())
    paper_pos = bot.ex.positions.get("XUSDT", (0.0, 0.0))[0]
    assert trader.fills > 0
    assert paper_pos == 0
    assert bot.pnl.get_state("XUSDT").position_qty == pytest.approx(paper_pos)
    s = bot.pnl.get_state("XUSDT")
    # Кэш бумажной биржи = realized − комиссии по трекеру
    assert bot.ex.cash - 100 == pytest.approx(s.realized_pnl - s.total_fees, abs=1e-9)
    # инвентарь не выходил за лимит заметно
    assert abs(paper_pos) * 1.0 <= bot.cfg.risk.max_inventory_usdt * 1.5


def test_stop_loss_flattens_on_crash(tmp_path):
    bot, trader, book = build_bot(tmp_path)
    bot.cfg.risk.stop_loss_bps = 10

    async def go():
        book.on_book_ticker(0.9999, 1000, 1.0001, 1000, time.time())
        # вручную открыть лонг
        await bot.ex.place_limit("XUSDT", "BUY", "0.9998", "12", "x1", False)
        await bot.ex.on_trade("XUSDT", 0.9990, 50, True, time.time())
        assert trader.position == 12
        book.on_book_ticker(0.9950, 1000, 0.9952, 1000, time.time())   # −50 bps
        await trader._step()
    asyncio.run(go())
    assert bot.ex.positions["XUSDT"][0] == 0
    assert trader.position == 0
    assert trader.cooldown_until > time.time()


def test_bot_run_and_shutdown_offline(tmp_path):
    """Полный цикл Bot.run(): сканер → монета → фоновые задачи → остановка."""
    cfg = BotConfig()
    cfg.log.dir = str(tmp_path)
    cfg.log.status_interval_s = 0.2
    cfg.scanner.spread_samples = 1
    cfg.paper.latency_ms = 1
    bot = Bot(cfg)

    async def fake_start():
        bot.ex.client = FakeScanClient()
        bot.ex._filters = {s: SymbolFilters(0.00001, 1, 1, 5)
                           for s in ("GOODUSDT", "PUMPUSDT", "TIGHTUSDT", "BTCUSDT")}

    async def fake_close():
        pass

    async def funding(symbol):
        return 0.0, 0.0
    bot.ex.start, bot.ex.close, bot.ex.funding_info = fake_start, fake_close, funding
    subscribed = []
    bot.md.subscribe = lambda s, trades=True, depth=True: (subscribed.append(s), bot.md.book(s))

    async def go():
        async def feed():
            sim = SimMarket(bot.ex, bot.md.book("GOODUSDT"), seed=5)
            for _ in range(200):
                if "GOODUSDT" in bot.traders:
                    sim.ex = bot.ex
                    await sim.tick_symbol("GOODUSDT")
                await asyncio.sleep(0.005)
            bot.running = False
        await asyncio.gather(bot.run(), feed())
    SimMarket.tick_symbol = _tick_symbol
    asyncio.run(go())
    assert "GOODUSDT" in subscribed and "BTCUSDT" in subscribed
    assert bot.stopping and not bot.traders
    assert bot.ex.positions.get("GOODUSDT", (0.0, 0.0))[0] == 0


async def _tick_symbol(self, symbol):
    self.mid *= 1 + self.rng.gauss(0, 1.0) / 1e4
    bid, ask = round(self.mid - 0.0003, 5), round(self.mid + 0.0003, 5)
    now = time.time()
    self.book.on_book_ticker(bid, 1000, ask, 1000, now)
    await self.ex.on_book(symbol)
    side = self.rng.random() < 0.5
    px = round(self.mid + (-1 if side else 1) * 0.0003 * self.rng.uniform(0, 3), 5)
    self.book.on_trade(px, 10, side, now)
    await self.ex.on_trade(symbol, px, 10, side, now)

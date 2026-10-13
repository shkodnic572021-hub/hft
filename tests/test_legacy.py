"""
Офлайн-тесты: биржа заменена фейковым клиентом, сеть не нужна.
Запуск: python -m pytest tests -q
"""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "legacy"))

from calibrator import MarketCalibrator  # noqa: E402
from config import BotConfig  # noqa: E402
from market_maker import MultiMarketMakerBot, SymbolState  # noqa: E402
from pnl_tracker import PnLTracker  # noqa: E402


class FakeClient:
    def __init__(self, position=0.0, entry=0.0):
        self.position = position
        self.entry = entry
        self.orders = []
        self.cancels = []

    async def futures_position_information(self, symbol):
        return [{"symbol": symbol, "positionAmt": str(self.position),
                 "entryPrice": str(self.entry), "markPrice": "1"}]

    async def futures_create_order(self, **kw):
        self.orders.append(kw)
        return {"clientOrderId": kw.get("newClientOrderId", "x")}

    async def futures_cancel_order(self, **kw):
        self.cancels.append(kw)

    async def futures_get_open_orders(self, symbol):
        return []


def make_bot(tmp_path, client=None):
    cfg = BotConfig()
    cfg.pnl.csv_path = str(tmp_path / "pnl.csv")
    bot = MultiMarketMakerBot(cfg)
    bot.client = client or FakeClient()
    st = SymbolState("XUSDT", 0.0001, 1.0, cfg.strategy)
    st.mid_price = 0.5
    bot.states["XUSDT"] = st
    return bot, st


def trade(side, l, L, z, X, c="g_1", n="0.0001"):
    return {"e": "ORDER_TRADE_UPDATE", "o": {
        "s": "XUSDT", "S": side, "x": "TRADE", "X": X, "c": c,
        "l": str(l), "L": str(L), "z": str(z), "ap": str(L),
        "N": "USDT", "n": n,
    }}


def test_partial_fills_do_not_double_position(tmp_path):
    bot, st = make_bot(tmp_path)

    async def go():
        await bot._handle_order_update(trade("BUY", 4, 0.5, 4, "PARTIALLY_FILLED"))
        await bot._handle_order_update(trade("BUY", 6, 0.5, 10, "FILLED"))
    asyncio.run(go())
    # Раньше: 4 + 10 = 14
    assert st.position_qty == 10
    assert bot.pnl.get_state("XUSDT").position_qty == 10


def test_pnl_uses_trade_price_and_real_fee(tmp_path):
    bot, st = make_bot(tmp_path)

    async def go():
        await bot._handle_order_update(trade("BUY", 10, 0.50, 10, "FILLED", n="0.001"))
        await bot._handle_order_update(trade("SELL", 10, 0.51, 10, "FILLED", c="t_1", n="0.001"))
    asyncio.run(go())
    s = bot.pnl.get_state("XUSDT")
    assert s.realized_pnl == pytest.approx(0.1)
    assert s.total_fees == pytest.approx(0.002)
    assert s.position_qty == 0 and s.avg_entry == 0
    assert st.position_qty == 0


def test_fill_after_rotation_still_counted(tmp_path):
    bot, st = make_bot(tmp_path)
    asyncio.run(bot._handle_order_update(trade("BUY", 10, 0.5, 10, "FILLED")))
    bot.states.clear()
    asyncio.run(bot._handle_order_update(trade("SELL", 10, 0.6, 10, "FILLED")))
    s = bot.pnl.get_state("XUSDT")
    assert s.position_qty == 0
    assert s.realized_pnl == pytest.approx(1.0)


def test_float_dust_resets_position(tmp_path):
    t = PnLTracker(BotConfig().pnl.__class__(csv_path=str(tmp_path / "p.csv")))
    for _ in range(3):
        t.on_fill("A", "BUY", 0.1, 1.0)
    t.on_fill("A", "SELL", 0.3, 1.0)
    s = t.get_state("A")
    assert s.position_qty == 0 and s.avg_entry == 0


def test_csv_mid_price_column(tmp_path):
    t = PnLTracker(BotConfig().pnl.__class__(csv_path=str(tmp_path / "p.csv")))
    t.on_fill("A", "BUY", 1, 100.0)
    t.force_log("A", 101.0)
    row = open(tmp_path / "p.csv").read().splitlines()[-1].split(",")
    assert float(row[4]) == pytest.approx(101.0)   # mid_price
    assert float(row[6]) == pytest.approx(1.0)     # unrealized


def test_price_qty_formatting_no_exponent():
    st = SymbolState("X", 0.0000001, 1.0, BotConfig().strategy)
    assert st.fmt_price(0.00001234) == "0.0000123"
    assert "e" not in st.fmt_price(0.00005)
    st2 = SymbolState("Y", 0.01, 0.001, BotConfig().strategy)
    assert st2.fmt_price(101.239) == "101.23"
    assert st2.fmt_qty(0.0599) == "0.059"


def test_close_market_skips_when_exchange_flat(tmp_path):
    bot, st = make_bot(tmp_path, FakeClient(position=0.0))
    asyncio.run(bot._close_position_market("XUSDT", 10))
    assert bot.client.orders == []


def test_close_market_uses_exchange_qty(tmp_path):
    bot, st = make_bot(tmp_path, FakeClient(position=-7.0))
    asyncio.run(bot._close_position_market("XUSDT", 10))
    o = bot.client.orders[0]
    assert o["side"] == "BUY" and o["quantity"] == "7" and o["reduceOnly"] == "true"


def test_short_skew_moves_quotes_up(tmp_path):
    bot, st = make_bot(tmp_path)
    st.position_qty = -20.0         # шорт $10 при лимите $12 (> 80%, < 90%)
    st.position_usdt = 10.0
    st.best_bid, st.best_ask = 0.4999, 0.5001
    asyncio.run(bot._update_symbol_quotes(st))
    bot2, st_long = make_bot(tmp_path)
    st_long.position_qty = 20.0
    st_long.position_usdt = 10.0
    st_long.best_bid, st_long.best_ask = 0.4999, 0.5001
    asyncio.run(bot2._update_symbol_quotes(st_long))
    buy_short = [o for o in bot.client.orders if o["side"] == "BUY"][0]
    buy_long = [o for o in bot2.client.orders if o["side"] == "BUY"][0]
    assert float(buy_short["price"]) > float(buy_long["price"])


def test_hard_kill_closes_position(tmp_path):
    bot, st = make_bot(tmp_path, FakeClient(position=40.0))
    st.position_qty = 40.0
    st.position_usdt = 20.0          # > 1.5 * 12
    asyncio.run(bot._update_symbol_quotes(st))
    assert st.hard_kill
    assert any(o["type"] == "MARKET" and o["side"] == "SELL" for o in bot.client.orders)


def test_calibrator_kappa_is_price_invariant():
    def kappa_for(price):
        c = MarketCalibrator(window_size=500)
        for i in range(200):
            p = price * (1 + 0.0001 * ((-1) ** i))
            c.update(p, price * 0.0005)
        return c.kappa
    assert kappa_for(50000) == pytest.approx(kappa_for(0.01), rel=1e-6)
    assert kappa_for(1.0) == pytest.approx(2000, rel=1e-3)


def test_leverage_bracket_dict_response(tmp_path):
    bot, _ = make_bot(tmp_path)

    async def lb(symbol):
        return {"symbol": symbol, "brackets": [{"initialLeverage": 20}]}
    bot.client.futures_leverage_bracket = lb
    assert asyncio.run(bot._get_symbol_max_leverage("XUSDT")) == 20


def test_rotation_is_serialized(tmp_path):
    bot, st = make_bot(tmp_path, FakeClient(position=0.0))
    prepared = []

    async def fake_prepare(sym):
        prepared.append(sym)
        await asyncio.sleep(0.01)
        return SymbolState(sym, 0.0001, 1.0, bot.cfg.strategy)
    bot._prepare_symbol = fake_prepare

    async def cancel_all(symbol):
        return None
    bot.client.futures_cancel_all_open_orders = cancel_all

    async def go():
        await asyncio.gather(
            bot._rotate_to_symbol("XUSDT", "AUSDT"),
            bot._rotate_to_symbol("XUSDT", "BUSDT"),
        )
    asyncio.run(go())
    assert list(bot.states) == ["AUSDT"]
    assert prepared == ["AUSDT"]

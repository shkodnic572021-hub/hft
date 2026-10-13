import asyncio
import logging
from config import BotConfig
from market_maker import MultiMarketMakerBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def main():
    config = BotConfig()

    if not config.exchange.api_key or not config.exchange.api_secret:
        logger.error("❌ API ключи отсутствуют в key.env")
        return

    logger.info("=" * 60)
    logger.info("MULTI-SYMBOL HFT MARKET MAKER")
    logger.info(f"Testnet: {config.exchange.testnet} | Dry: {config.dry_run}")
    logger.info(f"Leverage: {config.exchange.leverage}x")

    manual = config.manual_symbol.strip().upper()
    if manual:
        logger.info(f"⚙️  РЕЖИМ: ручной | Symbol: {manual}")
    else:
        logger.info("🔍 РЕЖИМ: авто (сканер + детектор затухания)")

    logger.info(f"Levels:   {config.strategy.num_levels} per side")
    logger.info(
        f"Order:    ${config.strategy.order_size_usdt} notional "
        f"(margin ${config.strategy.order_size_usdt / config.exchange.leverage:.2f})"
    )
    logger.info(f"Max inv:  ${config.strategy.max_inventory_usdt}")
    logger.info(
        f"Spread:   {config.strategy.min_half_spread_pct * 2:.2f}% "
        f"between best bid/ask"
    )
    logger.info(
        f"P&L log:  {config.pnl.csv_path} every {config.pnl.log_interval_sec}s"
    )
    logger.info("=" * 60)

    bot = MultiMarketMakerBot(config)
    try:
        await bot.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Бот остановлен пользователем.")

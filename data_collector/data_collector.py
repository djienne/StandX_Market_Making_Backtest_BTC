"""
StandX Data Collector
Collects orderbook and trade data from WebSocket, saves to Parquet files.
"""

import asyncio
import signal
from datetime import datetime, timezone
from typing import Dict, List, Optional

from standx_common import (
    BaseWebSocketClient,
    Config,
    CHANNEL_DEPTH_BOOK,
    CHANNEL_PUBLIC_TRADE,
    flush_buffer_to_parquet,
    parse_timestamp,
    sort_orderbook,
    validate_orderbook,
    calculate_spread,
    calculate_mid_price,
    logger,
)


class DataBuffer:
    """Thread-safe buffer for market data with Parquet flush capability."""

    def __init__(self, config: Config):
        self.config = config
        self.orderbook_buffer: List[Dict] = []
        self.trades_buffer: List[Dict] = []
        self._lock = asyncio.Lock()
        # Track recent trades to detect duplicates (key: (timestamp, price, qty, side))
        self._recent_trade_keys: set = set()

    async def add_orderbook(self, data: Dict):
        """Add orderbook snapshot to buffer."""
        async with self._lock:
            self.orderbook_buffer.append(data)

    async def add_trade(self, data: Dict):
        """Add trade to buffer, skipping exact duplicates."""
        async with self._lock:
            # Create unique key for deduplication
            trade_key = (
                str(data.get("timestamp")),
                data.get("price"),
                data.get("quantity"),
                data.get("side")
            )

            if trade_key in self._recent_trade_keys:
                # Skip duplicate
                return False

            self._recent_trade_keys.add(trade_key)
            self.trades_buffer.append(data)
            return True

    async def flush(self):
        """Flush all buffers to Parquet files."""
        async with self._lock:
            if self.orderbook_buffer:
                flush_buffer_to_parquet(
                    self.orderbook_buffer,
                    self.config.output_dir,
                    "orderbook"
                )
                self.orderbook_buffer.clear()

            if self.trades_buffer:
                flush_buffer_to_parquet(
                    self.trades_buffer,
                    self.config.output_dir,
                    "trades",
                    dedupe_columns=["timestamp", "price", "quantity", "side"]
                )
                self.trades_buffer.clear()
                # Clear dedup set after successful flush
                self._recent_trade_keys.clear()


class DataCollector(BaseWebSocketClient):
    """WebSocket collector for orderbook and trade data."""

    def __init__(self, symbol: str, config: Config, buffer: DataBuffer):
        super().__init__()
        self.symbol = symbol
        self.config = config
        self.buffer = buffer
        self.orderbook_count = 0
        self.trade_count = 0

    async def subscribe(self):
        """Subscribe to orderbook and trade channels."""
        await self.send_subscribe(CHANNEL_DEPTH_BOOK, self.symbol)
        await self.send_subscribe(CHANNEL_PUBLIC_TRADE, self.symbol)

    async def handle_message(self, data: Dict) -> Optional[Dict]:
        """Process incoming WebSocket message."""
        channel = data.get("channel")

        if channel == CHANNEL_DEPTH_BOOK:
            processed = self._process_orderbook(data.get("data", {}))
            await self.buffer.add_orderbook(processed)
            self.orderbook_count += 1
            return processed

        elif channel == CHANNEL_PUBLIC_TRADE:
            processed = self._process_trade(data.get("data", {}))
            added = await self.buffer.add_trade(processed)
            if added:
                self.trade_count += 1
            return processed

        return None

    def _process_orderbook(self, data: Dict) -> Dict:
        """Process orderbook message into structured format."""
        received_at = datetime.now(timezone.utc)
        asks = data.get("asks", [])
        bids = data.get("bids", [])

        # Sort and limit levels
        bids_sorted, asks_sorted = sort_orderbook(bids, asks, self.config.orderbook_levels)

        # Validate
        is_valid, _ = validate_orderbook(bids_sorted, asks_sorted, self.symbol)

        # Extract prices and quantities
        bid_prices = [float(b[0]) for b in bids_sorted]
        bid_quantities = [float(b[1]) for b in bids_sorted]
        ask_prices = [float(a[0]) for a in asks_sorted]
        ask_quantities = [float(a[1]) for a in asks_sorted]

        # Pad to exact level count
        levels = self.config.orderbook_levels
        bid_prices.extend([None] * (levels - len(bid_prices)))
        bid_quantities.extend([None] * (levels - len(bid_quantities)))
        ask_prices.extend([None] * (levels - len(ask_prices)))
        ask_quantities.extend([None] * (levels - len(ask_quantities)))

        # Calculate derived values
        best_bid = bid_prices[0] if bid_prices[0] is not None else None
        best_ask = ask_prices[0] if ask_prices[0] is not None else None

        return {
            "timestamp": parse_timestamp(data.get("time")),
            "received_at": received_at,
            "symbol": self.symbol,
            "bid_prices": bid_prices,
            "bid_quantities": bid_quantities,
            "ask_prices": ask_prices,
            "ask_quantities": ask_quantities,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": calculate_spread(best_bid, best_ask),
            "mid_price": calculate_mid_price(best_bid, best_ask),
            "is_valid": is_valid
        }

    def _process_trade(self, data: Dict) -> Dict:
        """Process trade message into structured format."""
        received_at = datetime.now(timezone.utc)
        is_buyer_taker = data.get("is_buyer_taker", False)

        return {
            "timestamp": parse_timestamp(data.get("time")),
            "received_at": received_at,
            "symbol": self.symbol,
            "price": float(data.get("price", 0)),
            "quantity": float(data.get("qty", 0)),
            "quote_quantity": float(data.get("quote_qty", 0)) if data.get("quote_qty") else None,
            "side": "BUY" if is_buyer_taker else "SELL",
            "is_buyer_taker": is_buyer_taker
        }


class FlushTask:
    """Periodic task to flush buffer to Parquet files."""

    def __init__(self, buffer: DataBuffer, interval: int):
        self.buffer = buffer
        self.interval = interval
        self.running = False

    async def run(self):
        """Run periodic flush loop."""
        self.running = True
        while self.running:
            await asyncio.sleep(self.interval)
            if self.running:
                logger.info("Flushing buffers to Parquet...")
                await self.buffer.flush()

    def stop(self):
        """Stop the flush task."""
        self.running = False


async def main():
    """Main entry point."""
    config = Config()
    buffer = DataBuffer(config)

    # Create collectors for each symbol
    collectors = [DataCollector(symbol, config, buffer) for symbol in config.symbols]

    # Create flush task
    flush_task = FlushTask(buffer, config.flush_interval_seconds)

    # Setup graceful shutdown
    shutdown_event = asyncio.Event()

    def handle_shutdown(signum, frame):
        logger.info("Shutdown signal received...")
        shutdown_event.set()
        flush_task.stop()
        for collector in collectors:
            collector.stop()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    logger.info(f"Starting data collection for {config.symbols}...")
    logger.info("Press Ctrl+C to stop")

    # Start all tasks
    tasks = [
        asyncio.create_task(flush_task.run()),
        *[asyncio.create_task(collector.run()) for collector in collectors]
    ]

    # Wait for shutdown
    await shutdown_event.wait()

    # Cancel tasks
    for task in tasks:
        task.cancel()

    # Final flush
    logger.info("Final buffer flush before exit...")
    await buffer.flush()

    # Print summary
    for collector in collectors:
        stats = collector.stats
        logger.info(
            f"[{collector.symbol}] Total: {collector.orderbook_count} orderbooks, "
            f"{collector.trade_count} trades | "
            f"Reconnects: {stats['reconnect_count']}, "
            f"Uptime: {stats['total_uptime_seconds']:.0f}s"
        )

    logger.info("Data collection stopped")


if __name__ == "__main__":
    asyncio.run(main())

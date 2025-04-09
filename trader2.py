# trader.py
import json
from typing import Dict, List, Any

# Import the official classes from your datamodel file
# Make sure datamodel.py is in the same directory or your PYTHONPATH
from datamodel import TradingState, Order, OrderDepth # Added OrderDepth explicitly

# Default position limits for Round 1 products
DEFAULT_POSITION_LIMITS = {
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50,
}

# Fair value for stable product
RAINFOREST_RESIN_FAIR_VALUE = 10000

class Trader:

    def __init__(self, limits = DEFAULT_POSITION_LIMITS):
        self.position_limits = limits
        print("Initializing Trader...")

    def get_mid_price(self, symbol: str, state: TradingState) -> float | None:
        """Calculates the mid-price from the order book."""
        order_depth: OrderDepth | None = state.order_depths.get(symbol)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], Dict, str]: # Adjusted return type hint
        """
        Main trading logic method called each iteration by the competition environment.
        Returns orders, conversions (empty dict for now), and traderData.
        """
        print(f"Timestamp: {state.timestamp}, TraderData: {state.traderData}")
        result: Dict[str, List[Order]] = {}
        traderData = state.traderData
        conversions = {} # Initialize conversions as an empty dictionary

        for symbol in state.listings:
            if symbol == "RAINFOREST_RESIN":
                orders = self.run_strategy_rainforest_resin(state, symbol)
                result[symbol] = orders
            elif symbol == "KELP":
                 orders = self.run_strategy_kelp_squid(state, symbol)
                 result[symbol] = orders
            elif symbol == "SQUID_INK":
                 orders = self.run_strategy_kelp_squid(state, symbol)
                 result[symbol] = orders

        print(f"Orders to place: {result}")
        print("----------------------------------------------------")
        # *** THE FIX IS HERE: Return three items ***
        return result, conversions, traderData


    def run_strategy_rainforest_resin(self, state: TradingState, symbol: str) -> List[Order]:
        """Strategy for the stable Rainforest Resin."""
        orders: List[Order] = []
        product = symbol
        limit = self.position_limits.get(product, 50)
        current_position = state.position.get(product, 0)
        order_depth: OrderDepth | None = state.order_depths.get(product)

        if not order_depth or (not order_depth.buy_orders and not order_depth.sell_orders):
             print(f"{product}: No order depth data, skipping.")
             return orders

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        buy_price = RAINFOREST_RESIN_FAIR_VALUE - 1
        sell_price = RAINFOREST_RESIN_FAIR_VALUE + 1

        print(f"Running RESIN strategy: Position={current_position}, BestBid={best_bid}, BestAsk={best_ask}")

        sell_volume_available = limit - current_position
        if sell_volume_available > 0:
            if best_bid is not None and best_bid >= sell_price:
                sell_qty = min(sell_volume_available, order_depth.buy_orders[best_bid])
                print(f"RESIN: Placing aggressive sell order: {sell_qty}@{best_bid}")
                orders.append(Order(product, best_bid, -sell_qty))
                sell_volume_available -= sell_qty

            if sell_volume_available > 0:
                place_sell_price = sell_price
                print(f"RESIN: Placing passive sell order: {sell_volume_available}@{place_sell_price}")
                orders.append(Order(product, place_sell_price, -sell_volume_available))

        buy_volume_available = limit + current_position
        if buy_volume_available > 0:
            if best_ask is not None and best_ask <= buy_price:
                buy_qty = min(buy_volume_available, order_depth.sell_orders[best_ask])
                print(f"RESIN: Placing aggressive buy order: {buy_qty}@{best_ask}")
                orders.append(Order(product, best_ask, buy_qty))
                buy_volume_available -= buy_qty

            if buy_volume_available > 0:
                place_buy_volume = buy_volume_available
                print(f"RESIN: Placing passive buy order: {place_buy_volume}@{buy_price}")
                orders.append(Order(product, buy_price, place_buy_volume))

        return orders


    def run_strategy_kelp_squid(self, state: TradingState, symbol: str) -> List[Order]:
        """Generic strategy for KELP and SQUID_INK based on mid-price market making."""
        orders: List[Order] = []
        product = symbol
        limit = self.position_limits.get(product, 50)
        current_position = state.position.get(product, 0)
        order_depth: OrderDepth | None = state.order_depths.get(product)

        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            print(f"{product}: No order depth data, skipping.")
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2.0
        spread = 1

        buy_price = int(mid_price - spread)
        sell_price = int(mid_price + spread)

        print(f"Running {product} strategy: Position={current_position}, MidPrice={mid_price:.2f}, BestBid={best_bid}, BestAsk={best_ask}")

        sell_volume_available = limit - current_position
        if sell_volume_available > 0:
            if best_bid >= sell_price:
                sell_qty = min(sell_volume_available, order_depth.buy_orders[best_bid])
                print(f"{product}: Placing aggressive sell order: {sell_qty}@{best_bid}")
                orders.append(Order(product, best_bid, -sell_qty))
                sell_volume_available -= sell_qty

            if sell_volume_available > 0:
                place_sell_volume = sell_volume_available
                print(f"{product}: Placing passive sell order: {place_sell_volume}@{sell_price}")
                orders.append(Order(product, sell_price, -place_sell_volume))

        buy_volume_available = limit + current_position
        if buy_volume_available > 0:
            if best_ask <= buy_price:
                buy_qty = min(buy_volume_available, order_depth.sell_orders[best_ask])
                print(f"{product}: Placing aggressive buy order: {buy_qty}@{best_ask}")
                orders.append(Order(product, best_ask, buy_qty))
                buy_volume_available -= buy_qty

            if buy_volume_available > 0:
                place_buy_volume = buy_volume_available
                print(f"{product}: Placing passive buy order: {place_buy_volume}@{buy_price}")
                orders.append(Order(product, buy_price, place_buy_volume))

        return orders
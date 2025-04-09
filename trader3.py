import json
from typing import Any, List, Dict
import math
import numpy as np
import jsonpickle

from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        compressed = {}
        for symbol, depth in order_depths.items():
            compressed[symbol] = [depth.buy_orders, depth.sell_orders]
        return compressed

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> List[Any]:
        conversion_observations = {}
        for product, obs in observations.conversionObservations.items():
            conversion_observations[product] = [
                obs.bidPrice,
                obs.askPrice,
                obs.transportFees,
                obs.exportTariff,
                obs.importTariff,
                obs.sugarPrice,
                obs.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    """
    Example Trader that demonstrates how to:
      - Use logger.print() instead of print()
      - Implement a simple multi-product strategy (Rainforest Resin, Kelp, Squid Ink)
      - Keep state across runs using traderData
      - Return the required (orders, conversions, trader_data) from run()
    """

    def __init__(self):
        # Example parameter dictionary per product
        self.params = {
            "RAINFOREST_RESIN": {
                "fair_value": 10000,
                "take_width": 1,
                "clear_width": 0,
                "disregard_edge": 1,
                "join_edge": 2,
                "default_edge": 4,
                "soft_position_limit": 25,
            },
            "KELP": {
                "take_width": 2,
                "clear_width": 1,
                "prevent_adverse": True,
                "adverse_volume": 15,
                "reversion_beta": -0.10,
                "disregard_edge": 2,
                "join_edge": 2,
                "default_edge": 4,
                "soft_position_limit": 25,
            },
            "SQUID_INK": {
                "take_width": 3,
                "clear_width": 1,
                "prevent_adverse": True,
                "adverse_volume": 10,
                "reversion_beta": -0.20,
                "disregard_edge": 2,
                "join_edge": 2,
                "default_edge": 4,
                "soft_position_limit": 25,
            },
        }
        # Round 1 position limits
        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
        }

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        """
        The main entry point. Must return a tuple of:
         (dict[Symbol, list[Order]], conversions: int, trader_data: str)
        """
        # Decode stored data if available
        trader_object = {}
        if state.traderData:
            try:
                trader_object = jsonpickle.decode(state.traderData)
            except:
                trader_object = {}

        result: Dict[Symbol, List[Order]] = {}
        conversions = 0  # if you want to request conversions, set > 0
        # We'll store updated data (e.g. last prices) in trader_object and re-encode at the end
        # so it persists across runs
        orders_for_all_products: Dict[Symbol, List[Order]] = {}

        # For demonstration: handle up to these three products
        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            if product not in self.params:
                continue
            if product not in state.order_depths:
                continue

            symbol = product  # symbol == product name in default environment
            order_depth = state.order_depths[symbol]
            position = state.position.get(symbol, 0)
            p = self.params[symbol]

            # Step 1: Compute a fair value
            fair_value = self.estimate_fair_value(symbol, order_depth, trader_object)

            # Step 2: Attempt to 'take' best orders if they are obviously profitable
            take_orders = []
            buy_taken, sell_taken = 0, 0
            take_orders, buy_taken, sell_taken = self.take_orders(
                symbol,
                order_depth,
                fair_value,
                p["take_width"],
                position,
                p.get("prevent_adverse", False),
                p.get("adverse_volume", 0),
            )

            # Step 3: Attempt to 'clear' over-position near fair value
            clear_orders = []
            buy_cleared, sell_cleared = buy_taken, sell_taken
            clear_orders, buy_cleared, sell_cleared = self.clear_orders(
                symbol,
                order_depth,
                fair_value,
                p["clear_width"],
                position,
                buy_cleared,
                sell_cleared,
            )

            # Step 4: Post passive orders around fair_value
            make_orders = []
            make_buy, make_sell = buy_cleared, sell_cleared
            make_orders, make_buy, make_sell = self.make_orders(
                symbol,
                order_depth,
                fair_value,
                position,
                make_buy,
                make_sell,
                p["disregard_edge"],
                p["join_edge"],
                p["default_edge"],
                manage_position=True,
                soft_position_limit=p["soft_position_limit"],
            )

            # Collect final orders
            orders_for_all_products[symbol] = take_orders + clear_orders + make_orders

            logger.print(
                f"{symbol} | Pos={position} | FV={fair_value} | Orders={len(orders_for_all_products[symbol])}"
            )

        # Prepare to return
        # Put all final orders in 'result'
        for sym, odrs in orders_for_all_products.items():
            result[sym] = odrs

        # Re-encode any data for next iteration
        trader_data = jsonpickle.encode(trader_object)

        # Flush logs in required format
        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data

    ############################################################
    # ---------------  Sub-Methods for Strategy  --------------
    ############################################################

    def estimate_fair_value(self, product: str, order_depth: OrderDepth, trader_obj: dict) -> float:
        """
        Simple approach:
         - If Rainforest Resin, return a stable fair value (10_000).
         - If KELP or SQUID_INK, do a mid-price plus mean reversion to last mid.
        """
        p = self.params[product]
        limit = self.position_limits[product]

        # If it’s the stable product:
        if product == "RAINFOREST_RESIN":
            return p["fair_value"]

        # If we have no buy or sell orders in the book, fallback to a default
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return trader_obj.get(f"{product}_last_mid", 2000)

        # Basic best-bid / best-ask
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        # Filter out too-large volumes if we're being cautious about 'adverse selection'
        big_qty_threshold = p.get("adverse_volume", 999999) if p.get("prevent_adverse", False) else 999999
        valid_asks = [price for price, vol in order_depth.sell_orders.items() if abs(vol) <= big_qty_threshold]
        valid_bids = [price for price, vol in order_depth.buy_orders.items() if abs(vol) <= big_qty_threshold]

        if valid_asks and valid_bids:
            mid_price = (min(valid_asks) + max(valid_bids)) / 2
        else:
            mid_price = (best_ask + best_bid) / 2

        last_mid_key = f"{product}_last_mid"
        last_mid = trader_obj.get(last_mid_key, mid_price)

        # Mean reversion step
        reversion_beta = p.get("reversion_beta", 0.0)
        ret = (mid_price - last_mid) / (last_mid if abs(last_mid) > 1e-9 else 1)
        pred_ret = ret * reversion_beta
        fair = mid_price + mid_price * pred_ret

        # store the mid for next iteration
        trader_obj[last_mid_key] = mid_price
        return fair

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool,
        adverse_volume: int,
    ):
        """
        Aggress best quotes if they are obviously profitable.
        """
        orders: List[Order] = []
        buy_taken = 0
        sell_taken = 0

        # can buy if best ask <= fair_value - take_width
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_vol = -order_depth.sell_orders[best_ask]  # negative in the data

            if (not prevent_adverse or best_ask_vol <= adverse_volume) and best_ask <= fair_value - take_width:
                # how many can we buy, respecting limit
                limit = self.position_limits[product]
                buyable = limit - position
                qty = min(best_ask_vol, buyable)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    order_depth.sell_orders[best_ask] += qty
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]
                    buy_taken += qty

        # can sell if best bid >= fair_value + take_width
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_vol = order_depth.buy_orders[best_bid]

            if (not prevent_adverse or best_bid_vol <= adverse_volume) and best_bid >= fair_value + take_width:
                limit = self.position_limits[product]
                sellable = limit + position  # how many we can sell if position is negative
                qty = min(best_bid_vol, sellable)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    order_depth.buy_orders[best_bid] -= qty
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
                    sell_taken += qty

        return orders, buy_taken, sell_taken

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: float,
        position: int,
        buy_so_far: int,
        sell_so_far: int,
    ):
        """
        If we have leftover net position, try to flatten it near fair_value.
        """
        orders: List[Order] = []
        net_after = position + buy_so_far - sell_so_far

        # We'll define a quick approach: If net_after > 0, we want to sell some near (fair_value + clear_width).
        # If net_after < 0, we want to buy near (fair_value - clear_width).
        if net_after > 0:
            # check how many are bidding above or around fair_value + clear_width
            ask_px = math.floor(fair_value + clear_width)
            volume_at_above = 0
            for px, vol in order_depth.buy_orders.items():
                if px >= ask_px:
                    volume_at_above += vol
            limit_sell = self.position_limits[product] + position
            can_sell = min(volume_at_above, net_after, limit_sell)
            if can_sell > 0:
                orders.append(Order(product, ask_px, -can_sell))
                sell_so_far += can_sell

        elif net_after < 0:
            bid_px = math.ceil(fair_value - clear_width)
            volume_at_below = 0
            for px, vol in order_depth.sell_orders.items():
                if px <= bid_px:
                    volume_at_below += -vol
            limit_buy = self.position_limits[product] - position
            can_buy = min(volume_at_below, abs(net_after), limit_buy)
            if can_buy > 0:
                orders.append(Order(product, bid_px, can_buy))
                buy_so_far += can_buy

        return orders, buy_so_far, sell_so_far

    def make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_taken: int,
        sell_taken: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        """
        Post passive orders around fair_value,
        possibly pennying or joining existing quotes,
        and optionally adjusting based on net position.
        """
        orders: List[Order] = []

        # Identify best quotes above/below fair
        asks_above = [px for px in order_depth.sell_orders.keys() if px > fair_value + disregard_edge]
        bids_below = [px for px in order_depth.buy_orders.keys() if px < fair_value - disregard_edge]

        best_ask_above = min(asks_above) if asks_above else None
        best_bid_below = max(bids_below) if bids_below else None

        ask_px = round(fair_value + default_edge)
        if best_ask_above is not None:
            if abs(best_ask_above - fair_value) <= join_edge:
                ask_px = best_ask_above
            else:
                ask_px = best_ask_above - 1

        bid_px = round(fair_value - default_edge)
        if best_bid_below is not None:
            if abs(fair_value - best_bid_below) <= join_edge:
                bid_px = best_bid_below
            else:
                bid_px = best_bid_below + 1

        # If we want to skew for position control:
        if manage_position:
            if position > soft_position_limit:
                ask_px -= 1
            elif position < -soft_position_limit:
                bid_px += 1

        # create final orders, up to limit
        limit = self.position_limits[product]
        buy_cap = limit - (position + buy_taken)
        sell_cap = limit + (position - sell_taken)

        if buy_cap > 0:
            orders.append(Order(product, bid_px, buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, ask_px, -sell_cap))

        return orders, buy_taken, sell_taken

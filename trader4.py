import json
from typing import Any, List, Dict, Tuple
# Make sure all necessary types from datamodel are imported
from datamodel import (
    OrderDepth, UserId, TradingState, Order, Listing, Observation,
    ProsperityEncoder, Symbol, Trade # Removed Product from here if it's not in datamodel
)
import jsonpickle
import numpy as np
import math

# --- Define Product Constants ---
# This class defines the valid product symbols as constants.
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

# --- Logger Class (as provided before) ---
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_json = self.to_json(
            [
                self.compress_state(state, ""),
                self.compress_orders(orders),
                conversions,
                "",
                "",
            ]
        )
        base_length = len(base_json)
        max_item_length = (self.max_log_length - base_length) // 3
        if max_item_length < 0:
             max_item_length = 0

        output_list = [
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]
        print(self.to_json(output_list))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        # Ensure state.position is included in compression
        position_data = state.position if hasattr(state, 'position') else {}
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            position_data, # Use the position data
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        if listings: # Check if listings is not None or empty
             for listing in listings.values():
                   compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        if order_depths: # Check if order_depths is not None or empty
             for symbol, order_depth in order_depths.items():
                   compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        if trades: # Check if trades is not None or empty
             for arr in trades.values():
                   if arr: # Check if the list of trades is not None or empty
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

    def compress_observations(self, observations: Observation | None) -> list[Any]: # Allow None
        # Handle cases where observations might be None
        if observations is None:
            return [{}, {}] # Return empty dicts if no observations

        conversion_observations = {}
        # Check for existence of conversionObservations before iterating
        if hasattr(observations, 'conversionObservations') and observations.conversionObservations:
             for product, observation in observations.conversionObservations.items():
                  # Adjusted attributes based on common datamodels, ensure they exist
                  obs_data = []
                  for attr in ['bidPrice', 'askPrice', 'transportFees', 'exportTariff', 'importTariff', 'sunlight', 'humidity']:
                      obs_data.append(getattr(observation, attr, None)) # Use getattr with default None
                  conversion_observations[product] = obs_data

        plain_value_observations = {}
        if hasattr(observations, 'plainValueObservations') and observations.plainValueObservations:
            plain_value_observations = observations.plainValueObservations

        return [plain_value_observations, conversion_observations]


    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        if orders: # Check if orders is not None or empty
            for arr in orders.values():
                 if arr: # Check if the list of orders is not None or empty
                      for order in arr:
                           compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        encoder_cls = ProsperityEncoder if "ProsperityEncoder" in globals() else json.JSONEncoder
        try:
            return json.dumps(value, cls=encoder_cls, separators=(",", ":"))
        except TypeError as e:
            # Fallback for types ProsperityEncoder might not handle (like numpy types)
            # This basic fallback might lose precision for floats.
            def fallback_serializer(obj):
                 if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                  np.int16, np.int32, np.int64, np.uint8,
                                  np.uint16, np.uint32, np.uint64)):
                     return int(obj)
                 elif isinstance(obj, (np.float_, np.float16, np.float32,
                                     np.float64)):
                     return float(obj)
                 elif isinstance(obj, (np.ndarray,)):
                     return obj.tolist()
                 # Add other numpy types if needed
                 return f"Unserializable type: {type(obj)}" # Or raise error

            logger.print(f"Warning: Standard JSON encoding failed ({e}), attempting fallback serialization.")
            try:
                 # Use standard json with the fallback default
                 return json.dumps(value, default=fallback_serializer, separators=(",", ":"))
            except Exception as fallback_e:
                 logger.print(f"Error: Fallback JSON encoding also failed: {fallback_e}")
                 return '{"error": "JSON serialization failed"}'


    def truncate(self, value: str | Any, max_length: int) -> str: # Allow Any type for input
        if not isinstance(value, str):
             value = str(value) # Convert non-strings to string
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."

# Instantiate the logger globally
logger = Logger()

# --- Trader Class ---

# Parameters NEED TO BE DEFINED *AFTER* Product class
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 10,
        "manage_position": True,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.2, # Placeholder
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "manage_position": False,
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.2, # Placeholder
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "manage_position": False,
    },
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        # LIMITS also needs to be defined *AFTER* Product class
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
        }

    # --- Calculation and Order Logic Methods ---
    # (No changes needed in these methods themselves for this specific error)
    def calculate_fair_value(self, product: str, order_depth: OrderDepth, traderObject: Dict) -> float | None:
        params = self.params[product]
        # Use Product class constants correctly here
        if product == Product.RAINFOREST_RESIN:
            return params["fair_value"]

        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders: # Check order_depth itself too
            last_fair = traderObject.get(f"{product}_fair_value", None)
            # Don't log excessively if it's just missing transiently
            # if last_fair is None:
            #      logger.print(f"Warning: No order depth and no previous fair value for {product}.")
            return last_fair

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        adverse_vol = params.get("adverse_volume", 0)
        filtered_ask = [p for p, vol in order_depth.sell_orders.items() if abs(vol) < adverse_vol]
        filtered_bid = [p for p, vol in order_depth.buy_orders.items() if abs(vol) < adverse_vol]

        mm_ask = min(filtered_ask) if filtered_ask else best_ask
        mm_bid = max(filtered_bid) if filtered_bid else best_bid

        mid_price = (mm_ask + mm_bid) / 2.0 # Ensure float division
        last_mid_price_key = f"{product}_last_mid_price"
        fair_value_key = f"{product}_fair_value"
        fair_value = mid_price # Initialize with mid_price

        if last_mid_price_key in traderObject:
            last_mid = traderObject[last_mid_price_key]
            if isinstance(last_mid, (int, float)) and last_mid != 0: # Check type and non-zero
                last_returns = (mid_price - last_mid) / last_mid
                pred_returns = last_returns * params.get("reversion_beta", 0.0) # Default beta to 0.0
                fair_value = mid_price + (mid_price * pred_returns)
            # else: fair_value remains mid_price (if last_mid is zero or wrong type)
        # else: fair_value remains mid_price (if key doesn't exist)

        # Store calculated values (ensure they are json-serializable)
        traderObject[last_mid_price_key] = float(mid_price) # Store as float
        traderObject[fair_value_key] = float(fair_value) # Store as float
        return fair_value


    def take_best_orders(
        self, product: str, fair_value: float, take_width: float,
        orders: List[Order], order_depth: OrderDepth, position: int,
        buy_order_volume: int, sell_order_volume: int,
        prevent_adverse: bool = False, adverse_volume: int = 0,
    ) -> Tuple[int, int]:
        position_limit = self.LIMIT[product]

        # Take best ask (buy)
        if order_depth and order_depth.sell_orders: # Check depth exists
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = abs(order_depth.sell_orders[best_ask])

            if not prevent_adverse or best_ask_amount <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - (position + buy_order_volume))
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity

        # Take best bid (sell)
        if order_depth and order_depth.buy_orders: # Check depth exists
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = abs(order_depth.buy_orders[best_bid])

            if not prevent_adverse or best_bid_amount <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + (position - sell_order_volume))
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity

        return buy_order_volume, sell_order_volume

    def market_make(
        self, product: str, orders: List[Order], bid: float, ask: float,
        position: int, buy_order_volume: int, sell_order_volume: int,
    ) -> Tuple[int, int]:
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))

        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self, product: str, fair_value: float, width: float,
        orders: List[Order], order_depth: OrderDepth, position: int,
        buy_order_volume: int, sell_order_volume: int,
    ) -> Tuple[int, int]:

        if not order_depth: return buy_order_volume, sell_order_volume # Skip if no depth

        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = fair_value - width
        fair_for_ask = fair_value + width

        buy_capacity = self.LIMIT[product] - (position + buy_order_volume)
        sell_capacity = self.LIMIT[product] + (position - sell_order_volume)

        # If long position
        if position_after_take > 0 and order_depth.buy_orders: # Check buy_orders exist
            clear_quantity = 0
            sorted_bids = sorted(order_depth.buy_orders.items(), key=lambda item: item[0], reverse=True)
            target_price = fair_for_ask # Price we want to hit or beat
            actual_hit_price = -1 # Track the actual best price we hit

            for price, volume in sorted_bids:
                if price >= fair_for_ask:
                    sell_amount = min(abs(volume), position_after_take - clear_quantity)
                    if sell_amount <= 0: continue
                    clear_quantity += sell_amount
                    actual_hit_price = max(actual_hit_price, price) # Update best price hit
                    if clear_quantity >= position_after_take: break
                else:
                    break # Prices are too low

            sent_quantity = min(sell_capacity, clear_quantity)
            if sent_quantity > 0 and actual_hit_price != -1: # Ensure we actually found a price level
                 # logger.print(f"CLEAR SELL {product}: Qty {sent_quantity} @ Px {round(actual_hit_price)}")
                 orders.append(Order(product, round(actual_hit_price), -abs(sent_quantity)))
                 sell_order_volume += abs(sent_quantity)

        # If short position
        elif position_after_take < 0 and order_depth.sell_orders: # Check sell_orders exist
            clear_quantity = 0
            sorted_asks = sorted(order_depth.sell_orders.items(), key=lambda item: item[0])
            target_price = fair_for_bid # Price we want to lift or beat
            actual_lift_price = float('inf') # Track the actual best price we lift

            for price, volume in sorted_asks:
                if price <= fair_for_bid:
                    buy_amount = min(abs(volume), abs(position_after_take) - clear_quantity)
                    if buy_amount <= 0: continue
                    clear_quantity += buy_amount
                    actual_lift_price = min(actual_lift_price, price) # Update best price lifted
                    if clear_quantity >= abs(position_after_take): break
                else:
                    break # Prices are too high

            sent_quantity = min(buy_capacity, clear_quantity)
            if sent_quantity > 0 and actual_lift_price != float('inf'): # Ensure we found a price level
                 # logger.print(f"CLEAR BUY {product}: Qty {sent_quantity} @ Px {round(actual_lift_price)}")
                 orders.append(Order(product, round(actual_lift_price), abs(sent_quantity)))
                 buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def take_orders(
        self, product: str, order_depth: OrderDepth, fair_value: float, take_width: float, position: int,
        prevent_adverse: bool = False, adverse_volume: int = 0,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        # Pass initial volumes as 0, get cumulative volume after takes
        buy_vol, sell_vol = self.take_best_orders(
            product, fair_value, take_width, orders, order_depth, position,
            0, 0, prevent_adverse, adverse_volume, # Start with 0 volume for this step
        )
        return orders, buy_vol, sell_vol # Return orders and *cumulative* volume from this step

    def clear_orders(
        self, product: str, order_depth: OrderDepth, fair_value: float, clear_width: float, position: int,
        buy_order_volume: int, # Cumulative volume *before* this step
        sell_order_volume: int, # Cumulative volume *before* this step
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        # Pass cumulative volumes, get updated cumulative volumes
        buy_vol, sell_vol = self.clear_position_order(
            product, fair_value, clear_width, orders, order_depth, position,
            buy_order_volume, sell_order_volume,
        )
        return orders, buy_vol, sell_vol # Return orders and *updated* cumulative volume


    def make_orders(
        self, product: str, order_depth: OrderDepth, fair_value: float, position: int,
        buy_order_volume: int, # Cumulative volume *before* this step
        sell_order_volume: int, # Cumulative volume *before* this step
        disregard_edge: float, join_edge: float, default_edge: float,
        manage_position: bool = False, soft_position_limit: int = 0,
    ) -> Tuple[List[Order], int, int]: # Returns orders and *unmodified* cumulative volume
        orders: List[Order] = []

        if not order_depth: return orders, buy_order_volume, sell_order_volume # Skip if no depth

        # Filter based on disregard edge
        asks_above_fair = [p for p in order_depth.sell_orders.keys() if p > fair_value + disregard_edge] if order_depth.sell_orders else []
        bids_below_fair = [p for p in order_depth.buy_orders.keys() if p < fair_value - disregard_edge] if order_depth.buy_orders else []

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        # Determine ask price
        ask = fair_value + default_edge
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1

        # Determine bid price
        bid = fair_value - default_edge
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        # Adjust quotes based on position *after potential clears*
        if manage_position:
            # Position check uses cumulative volumes passed in (which reflect takes/clears)
            pos_after_clear = position + buy_order_volume - sell_order_volume
            if pos_after_clear > soft_position_limit:
                bid -= 1
                ask -= 1
            elif pos_after_clear < -soft_position_limit:
                bid += 1
                ask += 1

        # Ensure bid < ask before rounding
        bid = min(bid, ask - 1.0) # Use float comparison

        # market_make uses the cumulative volumes to calculate remaining capacity
        # It doesn't modify the volumes it returns here.
        _, _ = self.market_make(
            product, orders, bid, ask, position, buy_order_volume, sell_order_volume
        )
        # Return the orders created in this step, and the *same* cumulative volumes passed in
        return orders, buy_order_volume, sell_order_volume


    # --- Main run method ---
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        traderObject = {}
        if state.traderData:
            try:
                decoded_data = jsonpickle.decode(state.traderData)
                if isinstance(decoded_data, dict):
                    traderObject = decoded_data
                else:
                     # logger.print(f"Warning: Decoded traderData not dict: {type(decoded_data)}")
                     traderObject = {}
            except Exception as e:
                logger.print(f"Error decoding traderData: {e}. Resetting traderObject.")
                traderObject = {}

        result: Dict[Symbol, List[Order]] = {}

        # Ensure product iteration uses defined Product constants if needed elsewhere,
        # but here we iterate through keys from PARAMS which should align with product strings
        for product in self.params.keys():
            order_depth = state.order_depths.get(product) # Use .get for safety
            position = state.position.get(product, 0)
            params = self.params[product]

            # Skip product if no order depth data is available in the current state
            if order_depth is None:
                # logger.print(f"No order depth data for {product} in state.timestamp {state.timestamp}. Skipping.")
                result[product] = []
                continue # Skip to next product

            fair_value = self.calculate_fair_value(product, order_depth, traderObject)

            if fair_value is None:
                # logger.print(f"Warning: Fair value for {product} is None at ts {state.timestamp}. Skipping orders.")
                result[product] = []
                continue

            # --- Order Generation Pipeline ---
            # Start with zero cumulative volume for this tick
            current_buy_volume = 0
            current_sell_volume = 0

            # 1. Take Orders -> returns orders_step1, buy_vol_step1, sell_vol_step1
            take_orders, current_buy_volume, current_sell_volume = self.take_orders(
                product, order_depth, fair_value, params["take_width"], position,
                params.get("prevent_adverse", False), params.get("adverse_volume", 0),
            )

            # 2. Clear Orders -> uses buy/sell_vol_step1, returns orders_step2, buy/sell_vol_step2
            clear_orders, current_buy_volume, current_sell_volume = self.clear_orders(
                product, order_depth, fair_value, params["clear_width"], position,
                current_buy_volume, current_sell_volume, # Pass cumulative volume after takes
            )

            # 3. Make Orders -> uses buy/sell_vol_step2, returns orders_step3, buy/sell_vol_step2 (unmodified)
            make_orders, _, _ = self.make_orders(
                product, order_depth, fair_value, position,
                current_buy_volume, current_sell_volume, # Pass cumulative volume after clears
                params["disregard_edge"], params["join_edge"], params["default_edge"],
                params.get("manage_position", False), params.get("soft_position_limit", 0),
            )

            result[product] = take_orders + clear_orders + make_orders

        conversions = 0
        # Ensure traderObject is serializable before encoding
        serializable_trader_object = {}
        for k, v in traderObject.items():
             if isinstance(v, (dict, list, str, int, float, bool)) or v is None:
                  serializable_trader_object[k] = v
             else:
                  # Attempt conversion or skip/log non-serializable types
                  try:
                       serializable_trader_object[k] = float(v) # Example: try converting to float
                       # logger.print(f"Warning: Converted non-serializable value for key {k} to float.")
                  except (ValueError, TypeError):
                       # logger.print(f"Warning: Skipping non-serializable value for key {k} of type {type(v)}.")
                       pass # Skip if conversion fails

        trader_data = jsonpickle.encode(serializable_trader_object, unpicklable=False) # Make it plain JSON

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
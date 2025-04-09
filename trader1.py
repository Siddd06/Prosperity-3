import json
import math
from typing import List
from datamodel import OrderDepth, TradingState, Order, Trade

class Trader:
    """
    A more advanced Round 1 trading bot combining:
      - Rolling exponential moving averages (fast & slow)
      - Basic momentum detection (fast vs. slow EMA)
      - Market-making logic (posting both buy & sell orders around fair value)
      - Strict position limit checks
      - Data persistence across runs using traderData

    Tunable parameters:
      - FAST_ALPHA / SLOW_ALPHA: smoothing factors for EMAs
      - SPREAD: how aggressively or passively to post around fair value
      - MOMENTUM_BIAS: how much to tilt size toward the momentum side
    """

    def __init__(self):
        # Constants
        self.FAST_ALPHA = 0.2
        self.SLOW_ALPHA = 0.05
        self.SPREAD = 0.003  # e.g. 0.3% of fair price on each side
        self.MOMENTUM_BIAS = 0.3  # fraction of max size we might tilt in direction of momentum

        # Hard-coded position limits (from Round 1 docs)
        self.POSITION_LIMITS = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }

        # Initialize if needed, but keep in mind the simulation can re-instantiate the class.
        # We rely on traderData for real persistence.

    def run(self, state: TradingState):
        """
        The function called every iteration. We'll:
          1. Deserialize EMAs from traderData (if any).
          2. Update EMAs for each product based on new trades.
          3. Compute new fair values & momentum signals.
          4. Construct buy & sell orders with market-making logic.
          5. Return (orders, conversions=0, updatedTraderData).
        """
        # 1) Initialize or deserialize stored data
        # 'traderData' is a string we can use to store JSON for next run
        if state.traderData.strip():
            # Attempt to load previously stored data
            try:
                stored_data = json.loads(state.traderData)
            except:
                stored_data = {}
        else:
            stored_data = {}

        # We'll store two EMAs per product: 'fast' and 'slow'
        # If they aren't in stored_data, initialize them
        for product in self.POSITION_LIMITS:
            if product not in stored_data:
                stored_data[product] = {
                    "fast_ema": None,
                    "slow_ema": None
                }

        # This dictionary will map each product -> list of Orders
        result = {}

        # Conversions are not used in Round 1
        conversions = 0

        # 2) Update EMAs based on new trades
        # We'll use the *average execution price* of all trades since the last iteration
        # as the “new price” to feed into EMAs.
        for product in state.order_depths.keys():

            # Only track these products
            if product not in self.POSITION_LIMITS:
                continue

            new_price = None

            # Combine own_trades and market_trades from this iteration
            recent_trades: List[Trade] = []
            if product in state.own_trades:
                recent_trades += state.own_trades[product]
            if product in state.market_trades:
                recent_trades += state.market_trades[product]

            if len(recent_trades) > 0:
                # Weighted average price for all trades that occurred
                total_notional = 0
                total_volume = 0
                for tr in recent_trades:
                    price = tr.price
                    qty = abs(tr.quantity)  # quantity can be negative if we were the seller
                    total_notional += price * qty
                    total_volume += qty
                new_price = total_notional / total_volume if total_volume > 0 else None

            # Update the EMAs if we have a new trade price
            if new_price is not None:
                # If first time, initialize both EMAs to that price
                if stored_data[product]["fast_ema"] is None:
                    stored_data[product]["fast_ema"] = new_price
                    stored_data[product]["slow_ema"] = new_price
                else:
                    old_fast = stored_data[product]["fast_ema"]
                    old_slow = stored_data[product]["slow_ema"]

                    # EMA update
                    new_fast = (self.FAST_ALPHA * new_price) + ((1 - self.FAST_ALPHA) * old_fast)
                    new_slow = (self.SLOW_ALPHA * new_price) + ((1 - self.SLOW_ALPHA) * old_slow)

                    stored_data[product]["fast_ema"] = new_fast
                    stored_data[product]["slow_ema"] = new_slow

        # 3) Based on updated EMAs, compute new fair prices & momentum signals
        #    (If we have never seen a trade for a product, fallback to known "static" reference.)
        #    From the Tropical TV hints, we know approximate stable values:
        fallback_values = {
            "RAINFOREST_RESIN": 10_000,
            "KELP": 2_000,
            "SQUID_INK": 2_000
        }

        # 4) Construct buy & sell orders for each product
        for product, order_depth in state.order_depths.items():
            if product not in self.POSITION_LIMITS:
                continue

            # Acquire current position
            current_pos = state.position.get(product, 0)

            # Get the current EMAs
            fast_ema = stored_data[product]["fast_ema"]
            slow_ema = stored_data[product]["slow_ema"]

            # If we don't have valid EMAs yet, set them to fallback
            if fast_ema is None or slow_ema is None:
                fair_value = fallback_values[product]
            else:
                fair_value = fast_ema  # often you might average them, or just pick fast for reactivity

            # We'll incorporate slow_ema to gauge momentum
            # If fast > slow => uptrend; if fast < slow => downtrend
            momentum_factor = 1.0
            if fast_ema is not None and slow_ema is not None:
                # e.g., ratio or difference
                # If fast is 10% above slow => momentum = 1.1 => we skew bigger buy
                ratio = (fast_ema / slow_ema) if slow_ema != 0 else 1
                momentum_factor = ratio

            # Basic limit order logic
            # Let's see the best bid & ask
            asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])  # (price, qty)
            bids = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)

            # Prepare the list of new orders
            orders: List[Order] = []

            # Let's define a "tight" buy/sell price around fair_value with an adjustable spread
            # For example, if fair_value=2000, spread=0.003 => half-spread ~6 => we place
            #   buy around 1994 and sell around 2006 if we want symmetrical quotes.
            half_spread = math.ceil(fair_value * self.SPREAD)

            # Momentum tilt: if ratio > 1, we lean buy; if ratio < 1, we lean sell
            # We'll do that by adjusting the center of our quote up or down a little
            # e.g. if ratio = 1.01 => shift center up by 1% of half_spread => about +0.01*6 = +0.06 => ~ +0
            # This is a small effect, but you can tweak it.
            tilt = (momentum_factor - 1.0) * 0.5 * half_spread
            # So if momentum > 1 => tilt > 0 => shift center price upward
            # if momentum < 1 => tilt < 0 => shift center price downward
            # This tries to fill more on the side of the perceived short-term trend

            center_price = fair_value + tilt

            # Proposed quotes
            buy_quote_price = max(1, math.floor(center_price - half_spread))
            sell_quote_price = math.ceil(center_price + half_spread)

            # We'll decide how large each side should be.
            # Start with a baseline lot size, then tilt based on momentum
            # If ratio=1 => no tilt. If ratio=1.05 => 5% tilt => bigger buy size, smaller sell size, etc.
            # We ensure we never exceed position limits
            max_size = self.POSITION_LIMITS[product]
            # Currently available buy capacity
            buy_cap = max_size - current_pos
            # Currently available sell capacity
            sell_cap = max_size + current_pos  # if current_pos<0 => we can still sell more

            if buy_cap > 0 and sell_cap > 0:
                # If ratio>1 => shift size from the sell side to buy side
                # Example: ratio=1.10 => we want 10% more size on buy side
                # We'll keep total_lot = 2 * base_lot so we place symmetrical amounts in total
                # and then tilt them
                base_lot = 5  # you can tune this
                total_lot = 2 * base_lot

                # For momentum:
                #   buy_size = base_lot * ratio
                #   sell_size = base_lot * (2-ratio)
                # But keep them integer
                # Also clamp them to buy_cap / sell_cap
                ratio_clamped = max(0.5, min(1.5, momentum_factor))
                buy_size_f = base_lot * ratio_clamped
                sell_size_f = base_lot * (2.0 - ratio_clamped)

                buy_size = int(round(min(buy_size_f, buy_cap)))
                sell_size = int(round(min(sell_size_f, sell_cap)))

                # Place the buy order if we have capacity and reason
                if buy_size > 0 and buy_quote_price > 0:
                    orders.append(Order(product, buy_quote_price, +buy_size))

                # Place the sell order if we have capacity
                if sell_size > 0 and sell_quote_price > 0:
                    orders.append(Order(product, sell_quote_price, -sell_size))

            # Additionally, we can “cross” existing best quotes if they are better than our own fair logic
            # e.g. if best ask < buy_quote_price => we cross aggressively
            if len(asks) > 0:
                best_ask_price, best_ask_qty = asks[0]
                # best_ask_qty is negative in the dict, so actual quantity is -best_ask_qty
                ask_volume = -best_ask_qty
                if best_ask_price < buy_quote_price and buy_cap > 0:
                    # We'll try to buy up to the lesser of ask_volume or buy_cap
                    volume = min(ask_volume, buy_cap)
                    orders.append(Order(product, best_ask_price, volume))

            if len(bids) > 0:
                best_bid_price, best_bid_qty = bids[0]
                # best_bid_qty is positive in the dict
                if best_bid_price > sell_quote_price and sell_cap > 0:
                    # We'll sell up to the lesser of best_bid_qty or sell_cap
                    volume = min(best_bid_qty, sell_cap)
                    orders.append(Order(product, best_bid_price, -volume))

            # If we have any orders, record them
            if orders:
                result[product] = orders

        # 5) Serialize stored_data back into a string so it persists next iteration
        updated_traderData = json.dumps(stored_data)

        return result, conversions, updated_traderData

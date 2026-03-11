from dataclasses import dataclass

import numpy as np

from app.models.option import OptionContract, OptionType, Position


@dataclass
class StrategyMetrics:
    max_profit: float          # float or math.inf
    max_loss: float            # float or -math.inf
    breakeven_points: list[float]
    strategy_name: str


def compute_breakeven_points(price_range: np.ndarray, payoff: np.ndarray) -> list[float]:
    """
    Find price levels where the payoff curve crosses zero.

    Uses sign changes between consecutive array elements to locate crossings,
    then linearly interpolates to the exact zero.
    """
    breakevens: list[float] = []
    signs = np.sign(payoff)

    for i in range(len(signs) - 1):
        if signs[i] == 0.0:
            breakevens.append(float(price_range[i]))
        elif signs[i] != signs[i + 1] and signs[i] != 0.0 and signs[i + 1] != 0.0:
            # Linear interpolation to zero crossing
            p0, p1 = float(price_range[i]), float(price_range[i + 1]) 
            v0, v1 = float(payoff[i]), float(payoff[i + 1])
            crossing = p0 - v0 * (p1 - p0) / (v1 - v0)
            breakevens.append(round(crossing, 4))

    return breakevens
    

def detect_strategy_name(contracts: list[OptionContract]) -> str:
    """
    Identify common option strategies from the list of contracts.

    Covers: long/short call, long/short put, bull call spread, bear put spread,
    straddle, strangle, iron condor. Falls back to "Custom Strategy".
    """
    if len(contracts) == 1:
        c = contracts[0]
        if c.option_type == OptionType.CALL and c.position == Position.LONG:
            return "Long Call"
        if c.option_type == OptionType.CALL and c.position == Position.SHORT:
            return "Short Call"
        if c.option_type == OptionType.PUT and c.position == Position.LONG:
            return "Long Put"
        if c.option_type == OptionType.PUT and c.position == Position.SHORT:
            return "Short Put"

    if len(contracts) == 2:
        calls = [c for c in contracts if c.option_type == OptionType.CALL]
        puts = [c for c in contracts if c.option_type == OptionType.PUT]

        # Bull call spread: long call at lower strike + short call at higher strike
        if len(calls) == 2:
            long_calls = [c for c in calls if c.position == Position.LONG]
            short_calls = [c for c in calls if c.position == Position.SHORT]
            if len(long_calls) == 1 and len(short_calls) == 1:
                if long_calls[0].strike < short_calls[0].strike:
                    return "Bull Call Spread"
                if long_calls[0].strike > short_calls[0].strike:
                    return "Bear Call Spread"

        # Bear put spread: long put at higher strike + short put at lower strike
        if len(puts) == 2:
            long_puts = [c for c in puts if c.position == Position.LONG]
            short_puts = [c for c in puts if c.position == Position.SHORT]
            if len(long_puts) == 1 and len(short_puts) == 1:
                if long_puts[0].strike > short_puts[0].strike:
                    return "Bear Put Spread"
                if long_puts[0].strike < short_puts[0].strike:
                    return "Bull Put Spread"

        # Straddle: long call + long put at same strike
        if len(calls) == 1 and len(puts) == 1:
            c, p = calls[0], puts[0]
            if (
                c.position == Position.LONG
                and p.position == Position.LONG
                and c.strike == p.strike
            ):
                return "Long Straddle"
            if (
                c.position == Position.SHORT
                and p.position == Position.SHORT
                and c.strike == p.strike
            ):
                return "Short Straddle"

            # Strangle: long call + long put at different strikes
            if c.position == Position.LONG and p.position == Position.LONG:
                return "Long Strangle"
            if c.position == Position.SHORT and p.position == Position.SHORT:
                return "Short Strangle"

    # Long Call Butterfly: all calls, 3 distinct strikes, long wings, short body
    # Handles qty=2 on a single short call or two separate short call contracts
    all_calls = [c for c in contracts if c.option_type == OptionType.CALL]
    all_puts = [c for c in contracts if c.option_type == OptionType.PUT]
    if all_calls and not all_puts:
        # Net quantity per strike: positive=long, negative=short
        net: dict[float, int] = {}
        for c in all_calls:
            sign = 1 if c.position == Position.LONG else -1
            net[c.strike] = net.get(c.strike, 0) + sign * c.quantity
        strikes = sorted(net)
        if (
            len(strikes) == 3
            and net[strikes[0]] > 0   # long wing (low)
            and net[strikes[1]] < 0   # short body (middle)
            and net[strikes[2]] > 0   # long wing (high)
            and net[strikes[0]] + net[strikes[2]] == -net[strikes[1]]  # quantities balance
        ):
            return "Long Call Butterfly"

    # 4-leg strategies
    if len(contracts) == 4:
        calls = sorted(
            [c for c in contracts if c.option_type == OptionType.CALL],
            key=lambda x: x.strike,
        )
        puts = sorted(
            [c for c in contracts if c.option_type == OptionType.PUT],
            key=lambda x: x.strike,
        )
        if len(calls) == 2 and len(puts) == 2:
            long_put, short_put = (
                (puts[0], puts[1]) if puts[0].position == Position.LONG else (puts[1], puts[0])
            )
            short_call, long_call = (
                (calls[0], calls[1]) if calls[0].position == Position.SHORT else (calls[1], calls[0])
            )

            # Iron Butterfly: short call and short put share the same body strike
            if (
                long_put.position == Position.LONG
                and short_put.position == Position.SHORT
                and short_call.position == Position.SHORT
                and long_call.position == Position.LONG
                and short_put.strike == short_call.strike
                and long_put.strike < short_put.strike < long_call.strike
            ):
                return "Iron Butterfly"

            # Iron Condor: all four strikes are distinct
            if (
                long_put.position == Position.LONG
                and short_put.position == Position.SHORT
                and short_call.position == Position.SHORT
                and long_call.position == Position.LONG
                and long_put.strike < short_put.strike < short_call.strike < long_call.strike
            ):
                return "Iron Condor"

    return "Custom Strategy"


def compute_metrics(
    contracts: list[OptionContract],
    price_range: np.ndarray,
    payoff: np.ndarray,
) -> StrategyMetrics:
    """
    Derive strategy metrics from a payoff curve.

    Args:
        contracts: The list of OptionContract legs (used for strategy name detection).
        price_range: Array of underlying prices corresponding to the payoff values.
        payoff: Array of P&L values across the price range.

    Returns:
        StrategyMetrics with max_profit, max_loss, breakeven_points, strategy_name.
    """
    # If the curve is still rising at the right edge, profit is unbounded
    max_profit: float = float("inf") if payoff[-1] > payoff[-2] else float(np.max(payoff))
    # If the curve is still falling at the left edge, loss is unbounded
    max_loss: float = float("-inf") if payoff[0] < payoff[1] else float(np.min(payoff))

    breakevens = compute_breakeven_points(price_range, payoff)
    name = detect_strategy_name(contracts)

    return StrategyMetrics(
        max_profit=max_profit,
        max_loss=max_loss,
        breakeven_points=breakevens,
        strategy_name=name,
    )

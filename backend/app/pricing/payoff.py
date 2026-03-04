import numpy as np

from app.models.option import OptionContract, OptionType, Position
from app.models.strategy import Strategy


def call_payoff(S: np.ndarray, K: float, premium: float, position: Position) -> np.ndarray:
    """
    Compute call option payoff at expiration across a price range.

    Args:
        S: Array of underlying prices.
        K: Strike price.
        premium: Option premium paid (long) or received (short).
        position: LONG or SHORT.

    Returns:
        Array of P&L values corresponding to each price in S.
    """
    intrinsic = np.maximum(S - K, 0.0)
    if position == Position.LONG:
        return intrinsic - premium # what you recieve at expiration - what you paid for the option
    return premium - intrinsic # what you are paid for the option - what the option is now worth


def put_payoff(S: np.ndarray, K: float, premium: float, position: Position) -> np.ndarray:
    """
    Compute put option payoff at expiration across a price range.

    Args:
        S: Array of underlying prices.
        K: Strike price.
        premium: Option premium paid (long) or received (short).
        position: LONG or SHORT.

    Returns:
        Array of P&L values corresponding to each price in S.
    """
    intrinsic = np.maximum(K - S, 0.0) # How far the stock is below the strike
    if position == Position.LONG:
        return intrinsic - premium # what you recieve at expiration - what you paid for the option
    return premium - intrinsic # what you are paid for the option - what the option is now worth


def strategy_payoff(contracts: list[OptionContract]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the aggregate payoff curve for a multi-leg strategy.

    Price range is generated internally as np.linspace(0, 2 * max_strike, 10000).
    Each leg's P&L is scaled by its quantity.

    Args:
        contracts: List of OptionContract legs.

    Returns:
        Tuple of (price_range, payoff_curve) as numpy arrays.
    """
    if not contracts:
        raise ValueError("No contracts provided")

    max_strike = max(c.strike for c in contracts)
    price_range = np.linspace(0.0, 2.0 * max_strike, 10_000) # 0 is the lowest a stock can go and 2* max_strike is the highest

    total_payoff = np.zeros(len(price_range)) # Initialize payoff array

    for contract in contracts:
        if contract.option_type == OptionType.CALL:
            leg = call_payoff(price_range, contract.strike, contract.premium, contract.position) # payoff at expiration for each price in price_range
        else:
            leg = put_payoff(price_range, contract.strike, contract.premium, contract.position)

        total_payoff += leg * contract.quantity #Add payoff for each contract

    return price_range, total_payoff

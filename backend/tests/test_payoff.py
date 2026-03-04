"""
Unit tests for payoff computation functions.

All tests use analytical expected values — no approximations.
"""

import numpy as np
import pytest

from app.models.option import OptionType, Position
from app.pricing.payoff import call_payoff, put_payoff, strategy_payoff
from app.models.option import OptionContract
from datetime import date


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contract(
    option_type: OptionType,
    position: Position,
    strike: float,
    premium: float,
    quantity: int = 1,
) -> OptionContract:
    return OptionContract(
        ticker="TEST",
        option_type=option_type,
        position=position,
        strike=strike,
        premium=premium,
        quantity=quantity,
        expiration=date(2025, 12, 19),
    )


# ---------------------------------------------------------------------------
# Long Call
# ---------------------------------------------------------------------------

class TestLongCall:
    def test_above_strike_profit(self) -> None:
        """Long call above strike: payoff = S - K - premium."""
        S = np.array([110.0])
        result = call_payoff(S, K=100.0, premium=5.0, position=Position.LONG)
        assert result[0] == pytest.approx(5.0)

    def test_below_strike_loss(self) -> None:
        """Long call below strike: payoff = -premium (option expires worthless)."""
        S = np.array([90.0])
        result = call_payoff(S, K=100.0, premium=5.0, position=Position.LONG)
        assert result[0] == pytest.approx(-5.0)

    def test_at_strike_loss(self) -> None:
        """Long call at strike: payoff = -premium."""
        S = np.array([100.0])
        result = call_payoff(S, K=100.0, premium=5.0, position=Position.LONG)
        assert result[0] == pytest.approx(-5.0)

    def test_breakeven(self) -> None:
        """Long call breakeven at K + premium."""
        S = np.array([105.0])
        result = call_payoff(S, K=100.0, premium=5.0, position=Position.LONG)
        assert result[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Short Call
# ---------------------------------------------------------------------------

class TestShortCall:
    def test_max_profit_below_strike(self) -> None:
        """Short call below strike: max profit = premium received."""
        S = np.array([80.0])
        result = call_payoff(S, K=100.0, premium=5.0, position=Position.SHORT)
        assert result[0] == pytest.approx(5.0)

    def test_loss_above_strike(self) -> None:
        """Short call above strike: loss grows with S."""
        S = np.array([120.0])
        result = call_payoff(S, K=100.0, premium=5.0, position=Position.SHORT)
        assert result[0] == pytest.approx(-15.0)


# ---------------------------------------------------------------------------
# Long Put
# ---------------------------------------------------------------------------

class TestLongPut:
    def test_below_strike_profit(self) -> None:
        """Long put below strike: payoff = K - S - premium."""
        S = np.array([85.0])
        result = put_payoff(S, K=100.0, premium=4.0, position=Position.LONG)
        assert result[0] == pytest.approx(11.0)

    def test_above_strike_loss(self) -> None:
        """Long put above strike: payoff = -premium."""
        S = np.array([110.0])
        result = put_payoff(S, K=100.0, premium=4.0, position=Position.LONG)
        assert result[0] == pytest.approx(-4.0)

    def test_breakeven(self) -> None:
        """Long put breakeven at K - premium."""
        S = np.array([96.0])
        result = put_payoff(S, K=100.0, premium=4.0, position=Position.LONG)
        assert result[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Short Put
# ---------------------------------------------------------------------------

class TestShortPut:
    def test_max_profit_above_strike(self) -> None:
        """Short put max profit = premium received when S >= K."""
        S = np.array([110.0])
        result = put_payoff(S, K=100.0, premium=4.0, position=Position.SHORT)
        assert result[0] == pytest.approx(4.0)

    def test_max_loss_equals_premium_received(self) -> None:
        """Short put max loss when S = 0 equals K - premium."""
        S = np.array([0.0])
        result = put_payoff(S, K=100.0, premium=4.0, position=Position.SHORT)
        # At S=0: payoff = premium - K = 4 - 100 = -96
        assert result[0] == pytest.approx(-96.0)

    def test_loss_below_strike(self) -> None:
        """Short put loss = premium - (K - S) when S < K."""
        S = np.array([80.0])
        result = put_payoff(S, K=100.0, premium=4.0, position=Position.SHORT)
        assert result[0] == pytest.approx(-16.0)


# ---------------------------------------------------------------------------
# strategy_payoff — Bull Call Spread
# ---------------------------------------------------------------------------

class TestBullCallSpread:
    """
    Bull call spread: long call K1=100 @ 5, short call K2=110 @ 2.
    Net debit = 3.  Max profit = (K2-K1) - net_debit = 7.
    Breakeven = K1 + net_debit = 103.
    """

    @pytest.fixture
    def contracts(self) -> list[OptionContract]:
        return [
            _contract(OptionType.CALL, Position.LONG, strike=100.0, premium=5.0),
            _contract(OptionType.CALL, Position.SHORT, strike=110.0, premium=2.0),
        ]

    def test_max_loss_is_net_debit(self, contracts: list[OptionContract]) -> None:
        _, payoff = strategy_payoff(contracts)
        assert float(np.min(payoff)) == pytest.approx(-3.0, abs=1e-3)

    def test_max_profit_is_spread_minus_debit(self, contracts: list[OptionContract]) -> None:
        _, payoff = strategy_payoff(contracts)
        assert float(np.max(payoff)) == pytest.approx(7.0, abs=1e-3)

    def test_breakeven_equals_k1_plus_net_premium(self, contracts: list[OptionContract]) -> None:
        from app.analysis.metrics import compute_breakeven_points
        price_range, payoff = strategy_payoff(contracts)
        breakevens = compute_breakeven_points(price_range, payoff)
        assert len(breakevens) == 1
        assert breakevens[0] == pytest.approx(103.0, abs=0.05)


# ---------------------------------------------------------------------------
# strategy_payoff — Straddle
# ---------------------------------------------------------------------------

class TestStraddle:
    """
    Long straddle: long call K=100 @ 5, long put K=100 @ 4.
    Total debit = 9.  Breakevens at 91 and 109.
    Symmetric: payoff(100 + x) == payoff(100 - x).
    """

    @pytest.fixture
    def contracts(self) -> list[OptionContract]:
        return [
            _contract(OptionType.CALL, Position.LONG, strike=100.0, premium=5.0),
            _contract(OptionType.PUT, Position.LONG, strike=100.0, premium=4.0),
        ]

    def test_max_loss_at_strike(self, contracts: list[OptionContract]) -> None:
        price_range, payoff = strategy_payoff(contracts)
        idx = int(np.argmin(np.abs(price_range - 100.0)))
        assert payoff[idx] == pytest.approx(-9.0, abs=0.02)

    def test_symmetry_around_strike(self, contracts: list[OptionContract]) -> None:
        price_range, payoff = strategy_payoff(contracts)
        above_idx = int(np.argmin(np.abs(price_range - 115.0)))
        below_idx = int(np.argmin(np.abs(price_range - 85.0)))
        assert payoff[above_idx] == pytest.approx(payoff[below_idx], abs=0.02)

    def test_breakeven_points(self, contracts: list[OptionContract]) -> None:
        from app.analysis.metrics import compute_breakeven_points
        price_range, payoff = strategy_payoff(contracts)
        breakevens = sorted(compute_breakeven_points(price_range, payoff))
        assert len(breakevens) == 2
        assert breakevens[0] == pytest.approx(91.0, abs=0.05)
        assert breakevens[1] == pytest.approx(109.0, abs=0.05)


# ---------------------------------------------------------------------------
# strategy_payoff — quantity scaling
# ---------------------------------------------------------------------------

class TestQuantityScaling:
    def test_quantity_multiplies_payoff(self) -> None:
        """2 contracts should produce exactly 2x the payoff of 1 contract."""
        c1 = _contract(OptionType.CALL, Position.LONG, strike=100.0, premium=5.0, quantity=1)
        c2 = _contract(OptionType.CALL, Position.LONG, strike=100.0, premium=5.0, quantity=2)
        _, payoff1 = strategy_payoff([c1])
        _, payoff2 = strategy_payoff([c2])
        np.testing.assert_allclose(payoff2, 2 * payoff1)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_contracts_raises(self) -> None:
        with pytest.raises(ValueError, match="No contracts provided"):
            strategy_payoff([])

    def test_price_range_starts_at_zero(self) -> None:
        c = _contract(OptionType.CALL, Position.LONG, strike=100.0, premium=5.0)
        price_range, _ = strategy_payoff([c])
        assert price_range[0] == pytest.approx(0.0)

    def test_price_range_ends_at_2x_max_strike(self) -> None:
        c = _contract(OptionType.CALL, Position.LONG, strike=100.0, premium=5.0)
        price_range, _ = strategy_payoff([c])
        assert price_range[-1] == pytest.approx(200.0, abs=0.05)

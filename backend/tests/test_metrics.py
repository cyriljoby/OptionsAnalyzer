"""
Unit tests for metrics computation
(max_profit, max_loss, breakeven, strategy_name).
"""

from datetime import date

import pytest

from app.analysis.metrics import compute_metrics, detect_strategy_name
from app.models.option import OptionContract, OptionType, Position
from app.pricing.payoff import strategy_payoff


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


def _run(contracts: list[OptionContract]):
    price_range, payoff = strategy_payoff(contracts)
    return compute_metrics(contracts, price_range, payoff)


# ---------------------------------------------------------------------------
# Long Call metrics
# ---------------------------------------------------------------------------

class TestLongCallMetrics:
    @pytest.fixture
    def metrics(self):
        return _run([_contract(OptionType.CALL, Position.LONG, 100.0, 5.0)])

    def test_max_loss(self, metrics) -> None:
        assert metrics.max_loss == pytest.approx(-5.0, abs=0.01)

    def test_max_profit_is_unlimited(self, metrics) -> None:
        assert metrics.max_profit == float("inf")

    def test_breakeven(self, metrics) -> None:
        assert len(metrics.breakeven_points) == 1
        assert metrics.breakeven_points[0] == pytest.approx(105.0, abs=0.05)

    def test_strategy_name(self, metrics) -> None:
        assert metrics.strategy_name == "Long Call"


# ---------------------------------------------------------------------------
# Long Put metrics
# ---------------------------------------------------------------------------

class TestLongPutMetrics:
    @pytest.fixture
    def metrics(self):
        return _run([_contract(OptionType.PUT, Position.LONG, 100.0, 4.0)])

    def test_max_loss(self, metrics) -> None:
        assert metrics.max_loss == pytest.approx(-4.0, abs=0.01)

    def test_breakeven(self, metrics) -> None:
        assert len(metrics.breakeven_points) == 1
        assert metrics.breakeven_points[0] == pytest.approx(96.0, abs=0.05)

    def test_strategy_name(self, metrics) -> None:
        assert metrics.strategy_name == "Long Put"


# ---------------------------------------------------------------------------
# Short Put metrics
# ---------------------------------------------------------------------------

class TestShortPutMetrics:
    @pytest.fixture
    def metrics(self):
        return _run([_contract(OptionType.PUT, Position.SHORT, 100.0, 4.0)])

    def test_max_profit(self, metrics) -> None:
        assert metrics.max_profit == pytest.approx(4.0, abs=0.01)

    def test_max_loss_is_unlimited(self, metrics) -> None:
        assert metrics.max_loss == float("-inf")

    def test_strategy_name(self, metrics) -> None:
        assert metrics.strategy_name == "Short Put"


# ---------------------------------------------------------------------------
# Bull Call Spread metrics
# ---------------------------------------------------------------------------

class TestBullCallSpreadMetrics:
    @pytest.fixture
    def contracts(self):
        return [
            _contract(OptionType.CALL, Position.LONG, 100.0, 5.0),
            _contract(OptionType.CALL, Position.SHORT, 110.0, 2.0),
        ]

    def test_max_profit(self, contracts) -> None:
        m = _run(contracts)
        assert m.max_profit == pytest.approx(7.0, abs=0.01)

    def test_max_loss(self, contracts) -> None:
        m = _run(contracts)
        assert m.max_loss == pytest.approx(-3.0, abs=0.01)

    def test_breakeven(self, contracts) -> None:
        m = _run(contracts)
        assert len(m.breakeven_points) == 1
        assert m.breakeven_points[0] == pytest.approx(103.0, abs=0.05)

    def test_strategy_name(self, contracts) -> None:
        m = _run(contracts)
        assert m.strategy_name == "Bull Call Spread"


# ---------------------------------------------------------------------------
# Bear Put Spread metrics
# ---------------------------------------------------------------------------

class TestBearPutSpreadMetrics:
    @pytest.fixture
    def contracts(self):
        # Long put at 110, short put at 100 — classic bear put spread
        return [
            _contract(OptionType.PUT, Position.LONG, 110.0, 8.0),
            _contract(OptionType.PUT, Position.SHORT, 100.0, 3.0),
        ]

    def test_max_profit(self, contracts) -> None:
        # max profit = (K_long - K_short) - net_debit = 10 - 5 = 5
        m = _run(contracts)
        assert m.max_profit == pytest.approx(5.0, abs=0.01)

    def test_max_loss(self, contracts) -> None:
        # max loss = net debit = 5
        m = _run(contracts)
        assert m.max_loss == pytest.approx(-5.0, abs=0.01)

    def test_strategy_name(self, contracts) -> None:
        m = _run(contracts)
        assert m.strategy_name == "Bear Put Spread"


# ---------------------------------------------------------------------------
# Straddle metrics
# ---------------------------------------------------------------------------

class TestStraddleMetrics:
    @pytest.fixture
    def metrics(self):
        contracts = [
            _contract(OptionType.CALL, Position.LONG, 100.0, 5.0),
            _contract(OptionType.PUT, Position.LONG, 100.0, 4.0),
        ]
        return _run(contracts)

    def test_max_loss(self, metrics) -> None:
        assert metrics.max_loss == pytest.approx(-9.0, abs=0.02)

    def test_two_breakeven_points(self, metrics) -> None:
        assert len(metrics.breakeven_points) == 2

    def test_breakevens_symmetric(self, metrics) -> None:
        be = sorted(metrics.breakeven_points)
        assert be[0] == pytest.approx(91.0, abs=0.05)
        assert be[1] == pytest.approx(109.0, abs=0.05)

    def test_strategy_name(self, metrics) -> None:
        assert metrics.strategy_name == "Long Straddle"


# ---------------------------------------------------------------------------
# Iron Condor — strategy name detection
# ---------------------------------------------------------------------------

class TestIronCondorName:
    def test_detected(self) -> None:
        # long put 80, short put 90, short call 110, long call 120
        contracts = [
            _contract(OptionType.PUT, Position.LONG, 80.0, 1.0),
            _contract(OptionType.PUT, Position.SHORT, 90.0, 3.0),
            _contract(OptionType.CALL, Position.SHORT, 110.0, 3.0),
            _contract(OptionType.CALL, Position.LONG, 120.0, 1.0),
        ]
        assert detect_strategy_name(contracts) == "Iron Condor"


# ---------------------------------------------------------------------------
# Custom Strategy fallback
# ---------------------------------------------------------------------------

class TestCustomStrategy:
    def test_three_leg_is_custom(self) -> None:
        contracts = [
            _contract(OptionType.CALL, Position.LONG, 95.0, 4.0),
            _contract(OptionType.CALL, Position.SHORT, 100.0, 2.0),
            _contract(OptionType.PUT, Position.LONG, 90.0, 2.0),
        ]
        assert detect_strategy_name(contracts) == "Custom Strategy"


# ---------------------------------------------------------------------------
# Iron Butterfly
# ---------------------------------------------------------------------------

class TestIronButterflyName:
    @pytest.fixture
    def contracts(self) -> list[OptionContract]:
        # long put 90, short put 100, short call 100, long call 110
        return [
            _contract(OptionType.PUT, Position.LONG, 90.0, 1.0),
            _contract(OptionType.PUT, Position.SHORT, 100.0, 4.0),
            _contract(OptionType.CALL, Position.SHORT, 100.0, 4.0),
            _contract(OptionType.CALL, Position.LONG, 110.0, 1.0),
        ]

    def test_detected_as_iron_butterfly(
        self, contracts: list[OptionContract]
    ) -> None:
        assert detect_strategy_name(contracts) == "Iron Butterfly"

    def test_not_confused_with_iron_condor(
        self, contracts: list[OptionContract]
    ) -> None:
        assert detect_strategy_name(contracts) != "Iron Condor"


# ---------------------------------------------------------------------------
# Long Call Butterfly
# ---------------------------------------------------------------------------

class TestLongCallButterflyMetrics:
    @pytest.fixture
    def contracts(self) -> list[OptionContract]:
        # Buy K1=90 @3, sell 2x K2=100 @5, buy K3=110 @3
        # Net debit = 3 + 3 - 10 = -4 (credit of 4... use realistic premiums)
        # Use: buy K1=90 @6, sell 2x K2=100 @3, buy K3=110 @1
        # Net = -6 + 6 - 1 = -1 debit
        return [
            _contract(OptionType.CALL, Position.LONG, 90.0, 6.0, quantity=1),
            _contract(OptionType.CALL, Position.SHORT, 100.0, 3.0, quantity=2),
            _contract(OptionType.CALL, Position.LONG, 110.0, 1.0, quantity=1),
        ]

    def test_strategy_name(self, contracts: list[OptionContract]) -> None:
        assert detect_strategy_name(contracts) == "Long Call Butterfly"

    def test_two_breakeven_points(
        self, contracts: list[OptionContract]
    ) -> None:
        m = _run(contracts)
        assert len(m.breakeven_points) == 2

    def test_max_profit_positive(
        self, contracts: list[OptionContract]
    ) -> None:
        m = _run(contracts)
        assert m.max_profit > 0

    def test_max_loss_is_net_debit(
        self, contracts: list[OptionContract]
    ) -> None:
        # net debit = 6 + 1 - 2*3 = 1
        m = _run(contracts)
        assert m.max_loss == pytest.approx(-1.0, abs=0.02)

    def test_single_contract_qty2_also_detected(self) -> None:
        # qty=2 on the short leg instead of two separate contracts
        contracts = [
            _contract(OptionType.CALL, Position.LONG, 90.0, 6.0, quantity=1),
            _contract(OptionType.CALL, Position.SHORT, 100.0, 3.0, quantity=2),
            _contract(OptionType.CALL, Position.LONG, 110.0, 1.0, quantity=1),
        ]
        assert detect_strategy_name(contracts) == "Long Call Butterfly"

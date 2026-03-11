from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from app.analysis.metrics import StrategyMetrics, compute_metrics
from app.models.option import OptionContract, OptionType, Position
from app.pricing.payoff import strategy_payoff

router = APIRouter(prefix="/api/strategy", tags=["strategy"])


class OptionContractRequest(BaseModel):
    ticker: str
    option_type: OptionType
    position: Position
    strike: float
    premium: float
    quantity: int
    expiration: date

    @field_validator("strike", "premium")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("strike and premium must be positive")
        return v

    @field_validator("quantity")
    @classmethod
    def must_be_nonzero(cls, v: int) -> int:
        if v == 0:
            raise ValueError("quantity must be non-zero")
        return v


class PayoffPoint(BaseModel):
    price: float
    pnl: float


class StrategyRequest(BaseModel):
    contracts: list[OptionContractRequest]


class StrategyAnalysisResponse(BaseModel):
    strategy_name: str
    max_profit: float | None          # None represents +inf
    max_loss: float | None            # None represents -inf
    breakeven_points: list[float]
    payoff_curve: list[PayoffPoint]


CURVE_SAMPLE_POINTS = 200


@router.post("/analyze", response_model=StrategyAnalysisResponse)
async def analyze_strategy(request: StrategyRequest) -> StrategyAnalysisResponse:
    if not request.contracts:
        raise HTTPException(status_code=422, detail="At least one contract is required")

    domain_contracts = [
        OptionContract(
            ticker=c.ticker,
            option_type=c.option_type,
            position=c.position,
            strike=c.strike,
            premium=c.premium,
            quantity=c.quantity,
            expiration=c.expiration,
        )
        for c in request.contracts
    ]

    price_range, payoff = strategy_payoff(domain_contracts)
    metrics: StrategyMetrics = compute_metrics(domain_contracts, price_range, payoff)

    # Downsample the 10 000-point curve to CURVE_SAMPLE_POINTS for the HTTP response
    step = max(1, len(price_range) // CURVE_SAMPLE_POINTS)
    sampled_prices = price_range[::step]
    sampled_payoff = payoff[::step]

    curve = [
        PayoffPoint(price=round(float(p), 4), pnl=round(float(v), 4))
        for p, v in zip(sampled_prices, sampled_payoff)
    ]

    return StrategyAnalysisResponse(
        strategy_name=metrics.strategy_name,
        max_profit=None if metrics.max_profit == float("inf") else metrics.max_profit,
        max_loss=None if metrics.max_loss == float("-inf") else metrics.max_loss,
        breakeven_points=metrics.breakeven_points,
        payoff_curve=curve,
    )

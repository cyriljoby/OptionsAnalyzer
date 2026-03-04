from dataclasses import dataclass
from datetime import date
from enum import Enum


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class Position(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class OptionContract:
    ticker: str
    option_type: OptionType
    position: Position
    strike: float
    premium: float
    quantity: int
    expiration: date

from dataclasses import dataclass, field

from app.models.option import OptionContract


@dataclass
class Strategy:
    contracts: list[OptionContract] = field(default_factory=list)

    def add_leg(self, contract: OptionContract) -> None:
        self.contracts.append(contract)

    @property
    def legs(self) -> list[OptionContract]:
        return self.contracts

    @property
    def max_strike(self) -> float:
        if not self.contracts:
            raise ValueError("Strategy has no contracts")
        return max(c.strike for c in self.contracts)

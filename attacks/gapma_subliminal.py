from __future__ import annotations

from attacks.gapma_base import FixedGAPMACondition


class GAPMASubliminalCondition(FixedGAPMACondition):
    def __init__(self) -> None:
        super().__init__(
            strategy_name="subliminal",
            condition_name="gapma_subliminal",
        )
from __future__ import annotations

from attacks.gapma_base import FixedGAPMACondition


class GAPMAEmotionalCondition(FixedGAPMACondition):
    def __init__(self) -> None:
        super().__init__(
            strategy_name="emotional",
            condition_name="gapma_emotional",
        )
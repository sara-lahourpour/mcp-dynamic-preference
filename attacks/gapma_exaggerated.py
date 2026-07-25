from __future__ import annotations

from attacks.gapma_base import FixedGAPMACondition


class GAPMAExaggeratedCondition(FixedGAPMACondition):
    def __init__(self) -> None:
        super().__init__(
            strategy_name="exaggerated",
            condition_name="gapma_exaggerated",
        )
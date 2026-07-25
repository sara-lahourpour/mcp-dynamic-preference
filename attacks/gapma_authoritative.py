from __future__ import annotations

from attacks.gapma_base import FixedGAPMACondition


class GAPMAAuthoritativeCondition(FixedGAPMACondition):
    def __init__(self) -> None:
        super().__init__(
            strategy_name="authoritative",
            condition_name="gapma_authoritative",
        )
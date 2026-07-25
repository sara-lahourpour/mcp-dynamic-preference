from __future__ import annotations

import json
from pathlib import Path

from attacks.base import ToolMetadata


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OPTIMIZED_PATH = (
    PROJECT_ROOT
    / "data"
    / "gapma_ga_descriptions.json"
)


class GAPMAGeneticOptimizedCondition:
    condition_name = "gapma_ga_optimized"

    def __init__(
        self,
        strategy: str = "exaggerated",
    ) -> None:
        self.strategy = strategy

    def _load_description(self) -> str:
        if not OPTIMIZED_PATH.exists():
            raise FileNotFoundError(
                "GA-optimized descriptions were not found. "
                "Run scripts.optimize_gapma_ga first."
            )

        with OPTIMIZED_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        strategies = data.get(
            "optimized_strategies",
            {},
        )

        strategy_data = strategies.get(
            self.strategy
        )

        if not isinstance(strategy_data, dict):
            raise ValueError(
                f"No GA result exists for strategy "
                f"{self.strategy!r}."
            )

        description = strategy_data.get(
            "optimized_description"
        )

        if (
            not isinstance(description, str)
            or not description.strip()
        ):
            raise ValueError(
                "The optimized description is missing."
            )

        return description.strip()

    def transform(
        self,
        *,
        original_name: str,
        original_description: str,
    ) -> ToolMetadata:
        return ToolMetadata(
            name=original_name,
            description=self._load_description(),
        )
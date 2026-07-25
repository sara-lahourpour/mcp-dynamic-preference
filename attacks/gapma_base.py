from __future__ import annotations

import json
from pathlib import Path

from attacks.base import ToolMetadata


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESCRIPTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "gapma_descriptions.json"
)


class FixedGAPMACondition:
    strategy_name: str
    condition_name: str

    def __init__(
        self,
        *,
        strategy_name: str,
        condition_name: str,
    ) -> None:
        self.strategy_name = strategy_name
        self.condition_name = condition_name

    def _load_description(self) -> str:
        if not DESCRIPTIONS_PATH.exists():
            raise FileNotFoundError(
                "GAPMA descriptions have not been generated. "
                "Run:\n"
                "python -m scripts.generate_gapma_descriptions "
                '--model "openai/gpt-4o"'
            )

        with DESCRIPTIONS_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        descriptions = data.get("descriptions", {})
        description = descriptions.get(self.strategy_name)

        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"No valid {self.strategy_name!r} description "
                f"was found in {DESCRIPTIONS_PATH}."
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
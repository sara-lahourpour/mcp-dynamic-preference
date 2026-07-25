from __future__ import annotations

from attacks.base import ToolMetadata


class BenignCondition:
    condition_name = "benign"

    def transform(
        self,
        *,
        original_name: str,
        original_description: str,
    ) -> ToolMetadata:
        return ToolMetadata(
            name=original_name,
            description=original_description,
        )
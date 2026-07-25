from __future__ import annotations

from attacks.base import ToolMetadata


class DPMABestNameCondition:
    condition_name = "dpma_best_name"

    def transform(
        self,
        *,
        original_name: str,
        original_description: str,
    ) -> ToolMetadata:
        return ToolMetadata(
            name=f"best_{original_name}",
            description=original_description,
        )
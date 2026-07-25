from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    description: str


class AttackCondition(Protocol):
    """
    Every attack condition must implement this interface.
    """

    condition_name: str

    def transform(
        self,
        *,
        original_name: str,
        original_description: str,
    ) -> ToolMetadata:
        ...
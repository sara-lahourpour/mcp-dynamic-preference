from __future__ import annotations

from typing import TypeVar


ToolType = TypeVar("ToolType")


def balanced_tool_order(
    tools: list[ToolType],
    trial_index: int,
) -> list[ToolType]:
    """
    Cyclically rotate tools.

    With six tools, every tool appears in every position
    once during each six-trial block.
    """
    if not tools:
        raise ValueError("Tool list cannot be empty.")

    offset = trial_index % len(tools)

    return tools[offset:] + tools[:offset]
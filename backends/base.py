from __future__ import annotations

from typing import Protocol

from host.models import ConnectedTool, SelectionResponse


class ModelBackend(Protocol):
    async def select_tool(
        self,
        *,
        model: str,
        query: str,
        system_prompt: str,
        tools: list[ConnectedTool],
        temperature: float,
    ) -> SelectionResponse:
        ...
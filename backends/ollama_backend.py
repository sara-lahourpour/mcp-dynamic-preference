from __future__ import annotations

import json
from typing import Any

from ollama import AsyncClient

from host.mcp_manager import to_tool_schema
from host.models import ConnectedTool, SelectionResponse


def normalize_arguments(
    arguments: Any,
) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return dict(arguments)

    if isinstance(arguments, str):
        try:
            value = json.loads(arguments)

            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

    return {}


class OllamaBackend:
    def __init__(self) -> None:
        self.client = AsyncClient()

    async def select_tool(
        self,
        *,
        model: str,
        query: str,
        system_prompt: str,
        tools: list[ConnectedTool],
        temperature: float,
    ) -> SelectionResponse:
        response = await self.client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            tools=[
                to_tool_schema(tool)
                for tool in tools
            ],
            options={
                "temperature": temperature,
            },
            stream=False,
        )

        calls = response.message.tool_calls or []

        if not calls:
            return SelectionResponse(
                selected_name=None,
                selected_arguments={},
                model_content=response.message.content,
                error="Model did not call a tool.",
                provider_name="ollama",
            )

        call = calls[0]

        return SelectionResponse(
            selected_name=call.function.name,
            selected_arguments=normalize_arguments(
                call.function.arguments
            ),
            model_content=response.message.content,
            error=None,
            provider_name="ollama",
            prompt_tokens=getattr(
                response,
                "prompt_eval_count",
                None,
            ),
            completion_tokens=getattr(
                response,
                "eval_count",
                None,
            ),
        )
from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI

from dotenv import load_dotenv

from host.mcp_manager import to_tool_schema
from host.models import ConnectedTool, SelectionResponse

load_dotenv()

def normalize_arguments(
    arguments: Any,
) -> dict[str, Any]:
    """
    Convert model-generated function arguments into a dictionary.
    """
    if isinstance(arguments, dict):
        return dict(arguments)

    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)

            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {}


class OpenRouterBackend:
    """
    Send tool-selection requests to OpenRouter using its
    OpenAI-compatible API.
    """

    def __init__(self) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. "
                "Set it in Git Bash before running the experiment."
            )

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": (
                    "http://localhost/mcp-preference-research"
                ),
                "X-Title": "MCP Preference Research",
            },
        )

    async def select_tool(
        self,
        *,
        model: str,
        query: str,
        system_prompt: str,
        tools: list[ConnectedTool],
        temperature: float,
    ) -> SelectionResponse:
        """
        Ask one OpenRouter model to select exactly one tool.
        """
        try:
            response = await self.client.chat.completions.create(
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
                tool_choice="required",
                temperature=temperature,
                max_tokens=100,
                extra_body={
                    "provider": {
                        "allow_fallbacks": False,
                        "require_parameters": True,
                    }
                },
            )

        except Exception as error:
            return SelectionResponse(
                selected_name=None,
                selected_arguments={},
                model_content=None,
                error=(
                    f"OpenRouter request failed: "
                    f"{type(error).__name__}: {error}"
                ),
                provider_name=None,
            )

        if not response.choices:
            return SelectionResponse(
                selected_name=None,
                selected_arguments={},
                model_content=None,
                error="OpenRouter returned no choices.",
                provider_name=None,
            )

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        usage = response.usage

        prompt_tokens = (
            usage.prompt_tokens
            if usage is not None
            else None
        )

        completion_tokens = (
            usage.completion_tokens
            if usage is not None
            else None
        )

        total_tokens = (
            usage.total_tokens
            if usage is not None
            else None
        )

        provider_name = None

        model_extra = getattr(
            response,
            "model_extra",
            None,
        )

        if isinstance(model_extra, dict):
            provider_name = model_extra.get("provider")

        if not tool_calls:
            return SelectionResponse(
                selected_name=None,
                selected_arguments={},
                model_content=(
                    str(message.content)
                    if message.content is not None
                    else None
                ),
                error="Model did not call a tool.",
                provider_name=provider_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        selected_call = tool_calls[0]

        selected_name = (
            selected_call.function.name
        )

        selected_arguments = normalize_arguments(
            selected_call.function.arguments
        )

        valid_tool_names = {
            tool.exposed_name
            for tool in tools
        }

        error: str | None = None

        if selected_name not in valid_tool_names:
            error = (
                "Model selected an unknown tool: "
                f"{selected_name}"
            )

        return SelectionResponse(
            selected_name=selected_name,
            selected_arguments=selected_arguments,
            model_content=(
                str(message.content)
                if message.content is not None
                else None
            ),
            error=error,
            provider_name=provider_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
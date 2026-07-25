from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from host.models import ConnectedTool


async def connect_server(
    *,
    stack: AsyncExitStack,
    server_script: Path,
    server_id: str,
    original_name: str,
    exposed_name: str,
    raw_description: str,
    exposed_description: str,
) -> ConnectedTool:
    process_environment = os.environ.copy()

    process_environment.update(
        {
            "SERVER_ID": server_id,
            "TOOL_NAME": exposed_name,
            "TOOL_DESCRIPTION": exposed_description,
        }
    )

    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        env=process_environment,
    )

    read_stream, write_stream = await stack.enter_async_context(
        stdio_client(parameters)
    )

    session = await stack.enter_async_context(
        ClientSession(read_stream, write_stream)
    )

    await session.initialize()

    listed = await session.list_tools()

    if len(listed.tools) != 1:
        raise RuntimeError(
            f"{server_id} exposed {len(listed.tools)} tools; "
            "exactly one was expected."
        )

    tool = listed.tools[0]

    return ConnectedTool(
        server_id=server_id,
        original_name=original_name,
        exposed_name=tool.name,
        raw_description=raw_description,
        exposed_description=tool.description or "",
        input_schema=tool.inputSchema,
        session=session,
    )


def to_tool_schema(
    tool: ConnectedTool,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.exposed_name,
            "description": tool.exposed_description,
            "parameters": tool.input_schema,
        },
    }


async def invoke_tool(
    *,
    selected_name: str,
    arguments: dict[str, Any],
    tools: list[ConnectedTool],
) -> tuple[str | None, str | None]:
    selected_tool = next(
        (
            tool
            for tool in tools
            if tool.exposed_name == selected_name
        ),
        None,
    )

    if selected_tool is None:
        return None, f"Unknown selected tool: {selected_name}"

    try:
        result = await selected_tool.session.call_tool(
            selected_name,
            arguments=arguments,
        )
    except Exception as error:
        return (
            None,
            f"{type(error).__name__}: {error}",
        )

    structured = getattr(
        result,
        "structuredContent",
        None,
    )

    if structured is not None:
        return (
            json.dumps(
                structured,
                ensure_ascii=False,
            ),
            None,
        )

    text_parts: list[str] = []

    for item in getattr(result, "content", []):
        text = getattr(item, "text", None)

        if text:
            text_parts.append(text)

    return (
        "\n".join(text_parts)
        if text_parts
        else str(result),
        None,
    )
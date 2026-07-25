from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import replace
from typing import Any

from attacks.base import ToolMetadata
from host.experiment_config import SERVER_SCRIPT
from host.mcp_manager import connect_server
from host.models import ConnectedTool


async def connect_experiment_tools(
    *,
    stack: AsyncExitStack,
    competitors: list[dict[str, Any]],
    attacker_server_id: str,
    static_condition: Any | None,
    is_adversarial: bool,
) -> list[ConnectedTool]:
    connected_tools: list[ConnectedTool] = []

    for competitor in competitors:
        server_id = competitor["server_id"]
        original_name = competitor["tool_name"]
        original_description = competitor[
            "raw_description"
        ]

        if (
            server_id == attacker_server_id
            and not is_adversarial
        ):
            if static_condition is None:
                raise RuntimeError(
                    "Static attack condition is missing."
                )

            metadata = static_condition.transform(
                original_name=original_name,
                original_description=original_description,
            )
        else:
            metadata = ToolMetadata(
                name=original_name,
                description=original_description,
            )

        connected_tool = await connect_server(
            stack=stack,
            server_script=SERVER_SCRIPT,
            server_id=server_id,
            original_name=original_name,
            exposed_name=metadata.name,
            raw_description=original_description,
            exposed_description=metadata.description,
        )

        connected_tools.append(connected_tool)

    return connected_tools


def apply_attacker_description(
    *,
    tools: list[ConnectedTool],
    attacker_server_id: str,
    description: str,
) -> list[ConnectedTool]:
    result: list[ConnectedTool] = []

    for tool in tools:
        if tool.server_id == attacker_server_id:
            result.append(
                replace(
                    tool,
                    exposed_description=description,
                )
            )
        else:
            result.append(tool)

    return result


def find_attacker_tool(
    *,
    tools: list[ConnectedTool],
    attacker_server_id: str,
) -> ConnectedTool:
    try:
        return next(
            tool
            for tool in tools
            if tool.server_id == attacker_server_id
        )
    except StopIteration as error:
        raise RuntimeError(
            "No connected tool matched attacker server ID "
            f"{attacker_server_id!r}."
        ) from error
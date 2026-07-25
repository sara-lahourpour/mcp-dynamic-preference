from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp import ClientSession


@dataclass
class ConnectedTool:
    server_id: str
    original_name: str
    exposed_name: str
    raw_description: str
    exposed_description: str
    input_schema: dict[str, Any]
    session: ClientSession


@dataclass
class SelectionResponse:
    selected_name: str | None
    selected_arguments: dict[str, Any]
    model_content: str | None
    error: str | None
    provider_name: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
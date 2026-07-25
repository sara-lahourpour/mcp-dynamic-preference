from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP


SERVER_ID = os.environ["SERVER_ID"]
TOOL_NAME = os.environ["TOOL_NAME"]
TOOL_DESCRIPTION = os.environ["TOOL_DESCRIPTION"]

mcp = FastMCP(f"Equivalent Tool Server: {SERVER_ID}")


def synthetic_search(query: str) -> dict[str, object]:
    """
    All six servers use exactly the same implementation and result.

    This prevents genuine quality differences from influencing
    the client's tool-selection decision.
    """
    return {
        "query": query,
        "server_id": SERVER_ID,
        "results": [
            {
                "title": "Synthetic research result",
                "content": (
                    "The Model Context Protocol allows language-model "
                    "applications to connect with external tools and data."
                ),
            }
        ],
    }


# FastMCP normally obtains the tool name from the decorated function.
# Registering dynamically lets every subprocess expose a different name
# while using the same implementation.
mcp.tool(
    name=TOOL_NAME,
    description=TOOL_DESCRIPTION,
)(synthetic_search)


if __name__ == "__main__":
    mcp.run(transport="stdio")
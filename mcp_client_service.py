from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import settings

_TOOLS_CACHE: list[Any] | None = None


def load_mcp_servers_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or settings.mcp_servers_config
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return payload.get("mcpServers", payload)


async def load_mcp_tools() -> list[Any]:
    global _TOOLS_CACHE
    if _TOOLS_CACHE is not None:
        return _TOOLS_CACHE

    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(load_mcp_servers_config())
    _TOOLS_CACHE = await client.get_tools()
    return _TOOLS_CACHE


async def list_mcp_tools() -> list[dict[str, str]]:
    tools = await load_mcp_tools()
    results: list[dict[str, str]] = []
    for tool in tools:
        results.append(
            {
                "name": getattr(tool, "name", ""),
                "description": getattr(tool, "description", "") or "",
            }
        )
    return results

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

from tools.local_tools import route_plan


mcp = FastMCP("travel-route")


@mcp.tool()
async def plan_route(origin: str, destination: str, transport: str = "高铁/地铁") -> dict:
    """Plan a route between two places."""
    return await route_plan(origin, destination, transport)


if __name__ == "__main__":
    asyncio.run(mcp.run_stdio_async())

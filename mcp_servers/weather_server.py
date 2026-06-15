from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

from tools.local_tools import weather_query


mcp = FastMCP("travel-weather")


@mcp.tool()
async def query_weather(city: str, date: str | None = None) -> dict:
    """Query destination weather and travel advice."""
    return await weather_query(city, date)


if __name__ == "__main__":
    asyncio.run(mcp.run_stdio_async())

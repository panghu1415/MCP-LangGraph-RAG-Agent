from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

from tools.local_tools import hotel_price_search


mcp = FastMCP("travel-hotel")


@mcp.tool()
async def search_hotel_price(destination: str, days: int, style: str = "standard") -> dict:
    """Search hotel candidates and estimated/real prices."""
    return await hotel_price_search(destination, days, style)


if __name__ == "__main__":
    asyncio.run(mcp.run_stdio_async())

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

from tools.local_tools import search_flight, search_train_12306, smart_transport_dispatch


mcp = FastMCP("travel-transport")


@mcp.tool()
async def search_train(origin: str, destination: str, travel_date: str | None = None) -> dict:
    """Search train schedules from 12306 or configured train API."""
    return await search_train_12306(origin, destination, travel_date)


@mcp.tool()
async def search_flight_ticket(origin: str, destination: str, travel_date: str | None = None) -> dict:
    """Search flight schedules from configured flight API."""
    return await search_flight(origin, destination, travel_date)


@mcp.tool()
async def dispatch_transport(origin: str, destination: str, message: str, preference: dict | None = None) -> dict:
    """Choose train, flight or driving based on distance and user preference."""
    return await smart_transport_dispatch(origin, destination, message, preference)


if __name__ == "__main__":
    asyncio.run(mcp.run_stdio_async())

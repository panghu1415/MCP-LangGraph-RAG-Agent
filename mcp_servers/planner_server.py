from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

from tools.local_tools import itinerary_plan


mcp = FastMCP("travel-planner")


@mcp.tool()
def create_itinerary(destination: str, days: int, interests: str, knowledge_context: str = "") -> dict:
    """Create a day-by-day travel itinerary."""
    return itinerary_plan(destination, days, interests, knowledge_context)


if __name__ == "__main__":
    mcp.run()

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

from tools.local_tools import save_itinerary


mcp = FastMCP("travel-writer")


@mcp.tool()
def write_itinerary(content: str, filename: str | None = None) -> dict:
    """Save an itinerary markdown file."""
    return save_itinerary(content, filename)


if __name__ == "__main__":
    mcp.run()

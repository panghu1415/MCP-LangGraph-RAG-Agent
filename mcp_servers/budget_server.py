from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

from tools.local_tools import budget_estimate


mcp = FastMCP("travel-budget")


@mcp.tool()
def estimate_budget(destination: str, days: int, people: int = 1, style: str = "standard") -> dict:
    """Estimate a travel budget."""
    return budget_estimate(destination, days, people, style)


if __name__ == "__main__":
    mcp.run()

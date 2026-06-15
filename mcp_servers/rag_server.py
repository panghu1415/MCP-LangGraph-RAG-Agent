from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

from tools.local_tools import travel_rag_search


mcp = FastMCP("travel-rag")


@mcp.tool()
def search_travel_knowledge(query: str, top_k: int = 3) -> dict:
    """Search travel guides, policies and attraction notes."""
    return travel_rag_search(query, top_k)


if __name__ == "__main__":
    mcp.run()

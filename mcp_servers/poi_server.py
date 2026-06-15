from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

from tools.local_tools import amap_search_nearby_poi, amap_search_poi, travel_poi_search


mcp = FastMCP("travel-poi")


@mcp.tool()
async def search_poi(city: str, keywords: str, types: str = "", limit: int = 5) -> dict:
    """Search POI data from AMap, including attractions, hotels and restaurants."""
    return await amap_search_poi(city, keywords, types, limit)


@mcp.tool()
async def search_travel_poi(destination: str, interests: str = "") -> dict:
    """Search travel POI bundles for a destination."""
    return await travel_poi_search(destination, interests)


@mcp.tool()
async def search_nearby_poi(location: str, keywords: str, types: str = "", radius: int = 5000, limit: int = 5) -> dict:
    """Search nearby POI around a longitude,latitude location with AMap."""
    return await amap_search_nearby_poi(location, keywords, types, radius, limit)


if __name__ == "__main__":
    asyncio.run(mcp.run_stdio_async())

"""
Retrieval MCP server.

Wraps hybrid_search() as an MCP TOOL rather than a plain function call.
This is the piece that lets the LangGraph agent discover and call
retrieval dynamically through the MCP protocol (list_tools/call_tool).
"""
import sys
from mcp.server.fastmcp import FastMCP

from app.retrieval.hybrid import hybrid_search

mcp = FastMCP("retrieval")


@mcp.tool()
def search_class_documents(class_name: str, query: str, n_results: int = 5) -> list[dict]:
    """
    Search one class's ingested documents for chunks relevant to a query.
    Uses hybrid retrieval (BM25 + semantic search, fused via RRF).
    """
    return hybrid_search(class_name, query, n_results=n_results)


if __name__ == "__main__":
    if "--list-tools" in sys.argv:
        print("Tools exposed by this MCP server:")
        print("  - search_class_documents(class_name, query, n_results=5)")
    else:
        mcp.run(transport="stdio")
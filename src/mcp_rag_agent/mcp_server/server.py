"""MCP server for RAG agent with semantic search capabilities."""
import logging

from mcp.server.fastmcp import FastMCP

from mcp_rag_agent.core.config import config
from mcp_rag_agent.core.log_setup import setup_logging
from mcp_rag_agent.mcp_server.tools import search_documents

setup_logging()
logger = logging.getLogger("MCP Server")

# Initialize MCP
mcp = FastMCP(
    name= config.mcp_name,
    host=config.mcp_host,
    port=config.mcp_port
)

@mcp.tool()(search_documents)

@mcp.prompt()
def grounded_qa_prompt() -> str:
    """
    Provide a prompt template for grounded question-answering using retrieved documents.

    This prompt instructs language models to answer questions strictly based on
    provided document context, preventing hallucination or speculation beyond
    the given information. Designed for RAG workflows where factual accuracy
    is critical.
    """
    return (
        "You answer ONLY using the provided documents. "
        "If information is missing, say you don't know."
    )


if __name__ == "__main__":
    logger.info("Initializing MCP server...")
    # Run the following command to start a UI to test the server:
    # >>> mcp dev src/mcp_rag_agent/mcp_server/server.py
    mcp.run(transport="stdio")

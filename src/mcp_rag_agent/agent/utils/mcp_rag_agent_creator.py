import logging

from typing import List

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.prompts import load_mcp_prompt

from mcp_rag_agent.core.config import Config, config
from mcp_rag_agent.core.log_setup import setup_logging


setup_logging()
logger = logging.getLogger("Agent Creator")

# -------------------------------------------------------------------
# MCP client configuration
# -------------------------------------------------------------------
logger.info("Setting up MCP client configuration...")

# Name you give this connection in the MCP client (can be anything)
MCP_SERVER_NAME = config.mcp_name

# How to start your MCP server (stdio transport)
# ⚠️ Make sure the args match how you normally start `server.py`
MCP_CONNECTIONS = {
    MCP_SERVER_NAME: {
        "command": "python",
        # Option A: run as a module (recommended if it's on PYTHONPATH)
        "args": ["-m", "mcp_rag_agent.mcp_server.server"],
        # Option B (alternative): absolute path to server.py
        # "args": ["/abs/path/to/src/mcp_rag_agent/mcp_server/server.py"],
        "transport": "stdio",
    }
}


# -------------------------------------------------------------------
# Load MCP tools + grounded QA prompt
# -------------------------------------------------------------------
logger.info("Loading MCP tools and grounded QA prompt...")

async def _load_mcp_tools_and_prompt() -> tuple[List[BaseTool], str]:
    """
    - Connects to your MCP server via MultiServerMCPClient
    - Loads all MCP tools as LangChain tools (including `search_documents`)
    - Loads the `grounded_qa_prompt` MCP prompt and turns it into a system prompt
    """
    client = MultiServerMCPClient(MCP_CONNECTIONS)

    # 1) Load all tools from this MCP server
    # Each tool call will internally open a short-lived MCP session.
    tools: List[BaseTool] = await client.get_tools(server_name=MCP_SERVER_NAME)

    # 2) Load the grounded QA prompt once (it's static text from your server)
    async with client.session(MCP_SERVER_NAME) as session:
        prompt_messages = await load_mcp_prompt(
            session,
            name="grounded_qa_prompt",  # <- name from your @mcp.prompt()
        )

    # `prompt_messages` is a list of HumanMessage/AIMessage – flatten to a single string
    grounded_prompt = " ".join(msg.content for msg in prompt_messages)

    return tools, grounded_prompt


# -------------------------------------------------------------------
# Build the LangGraph ReAct agent wired to MCP
# -------------------------------------------------------------------
logger.info("Creating MCP RAG agent...")

async def create_mcp_rag_agent(system_prompt: str, config: Config):
    """
    Returns a LangGraph compiled graph that:
    - Uses ChatOpenAI as the LLM
    - Can call MCP tools (e.g. `search_documents`)
    - Is guided by the `grounded_qa_prompt` from the MCP server
    """
    # Tools
    tools, grounded_prompt = await _load_mcp_tools_and_prompt()

    # System message
    full_system_prompt = system_prompt + "\n" + grounded_prompt
    system_prompt_template = SystemMessage(content=full_system_prompt)

    # LLM
    model = ChatOpenAI(
        api_key=config.model_api_key,
        model=config.text_model,
        **config.text_generation_kwargs
    )

    # Prebuilt agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt_template,
    )

    return agent

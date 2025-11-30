import logging

from typing import List

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from mcp_rag_agent.core.config import Config
from mcp_rag_agent.core.log_setup import setup_logging


setup_logging()
logger = logging.getLogger("Agent Creator")


# -------------------------------------------------------------------
# Build the LangGraph ReAct agent
# -------------------------------------------------------------------
logger.info("Creating RAG agent...")

async def create_rag_agent(system_prompt: str, tools: List[BaseTool], config: Config):
    """
    Returns a LangGraph compiled graph that:
    - Uses ChatOpenAI as the LLM
    - Can call LangChain tools (e.g. `search_documents`)
    """
    # System message
    system_prompt_template = SystemMessage(content=system_prompt)

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
        debug=config.debug
    )

    return agent

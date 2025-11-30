import asyncio
import logging

from mcp_rag_agent.agent.prompts import system_prompt
from mcp_rag_agent.core.config import config
from mcp_rag_agent.core.log_setup import setup_logging



setup_logging()
logger = logging.getLogger("Agent Creator")


if config.ff_mcp_server:
    from mcp_rag_agent.agent.utils.mcp_rag_agent_creator import create_mcp_rag_agent

    agent = asyncio.run(create_mcp_rag_agent(system_prompt=system_prompt, config=config))
else:
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field
    from mcp_rag_agent.mcp_server.tools import search_documents

    logger.info("Creating LangChain structured tools...")

    class SearchDocumentsInput(BaseModel):
        """Input schema for the search_documents tool."""
        query: str = Field(
            description="The search query text to find relevant documents."
        )
        top_k: int = Field(
            default=3,
            description="The maximum number of results to return.",
            ge=1,
            le=10
        )

    search_documents_langchain = StructuredTool(
        name="search_policy_documents",
        description=(
            "This tool understands the meaning of the query, "
            "searches through the stored policy documents, "
            "and returns the most relevant documents "
            "so they can be used to support the final answer."
        ),
        func=lambda query, top_k=3: asyncio.run(search_documents(query, top_k)),
        coroutine=search_documents,
        args_schema=SearchDocumentsInput
    )
    
    tools = [search_documents_langchain]

    from mcp_rag_agent.agent.utils.rag_agent_creator import create_rag_agent
    agent = asyncio.run(create_rag_agent(system_prompt=system_prompt, tools=tools, config=config))


# -------------------------------------------------------------------
# Simple CLI test
# -------------------------------------------------------------------

if __name__ == "__main__":
    async def demo():
        # LangGraph ReAct agent expects a state with "messages"
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "In the UK, how many days of annual leave do employees receive?",
                    }
                ]
            }
        )

        # Final answer is the last message in the state
        final_message = result["messages"][-1]
        print(final_message.content)

    asyncio.run(demo())

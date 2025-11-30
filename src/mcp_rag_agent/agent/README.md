# Agent Module

The agent module implements a LangGraph ReAct agent that integrates with the Model Context Protocol (MCP) server to provide RAG-based question answering capabilities.

## Overview

This module creates an intelligent agent that:
- Connects to the MCP server to access RAG tools (document search)
- Uses OpenAI's language models for natural language understanding and generation
- Implements the ReAct (Reasoning + Acting) pattern for tool-based interactions
- Follows the COSTAR prompting framework for structured, high-quality responses

## Architecture

```
agent/
├── create_agent.py          # Main agent creation and configuration
└── prompts/
    ├── __init__.py          # Prompts module exports
    └── system_prompt.py     # COSTAR-based system prompt
```

## Components

### `create_agent.py`

The main module responsible for:

1. **MCP Client Configuration**: Sets up connection to the MCP server using stdio transport
2. **Tool Loading**: Dynamically loads all MCP tools (including `search_documents`)
3. **Prompt Integration**: Combines the system prompt with the grounded QA prompt from MCP
4. **Agent Creation**: Builds a LangGraph ReAct agent with OpenAI LLM and MCP tools

#### Key Functions

##### `_load_mcp_tools_and_prompt()`

```python
async def _load_mcp_tools_and_prompt() -> tuple[List[BaseTool], str]
```

- Connects to the MCP server via `MultiServerMCPClient`
- Loads all available MCP tools as LangChain-compatible tools
- Retrieves the `grounded_qa_prompt` from the MCP server
- Returns both tools and the grounded prompt text

##### `create_mcp_rag_agent()`

```python
async def create_mcp_rag_agent(system_prompt: str, config: Config)
```

- **Parameters**:
  - `system_prompt`: The base system prompt defining agent behavior
  - `config`: Configuration object containing API keys and model settings
- **Returns**: A compiled LangGraph agent ready to process queries
- **Process**:
  1. Loads MCP tools and grounded prompt
  2. Combines system prompt with grounded prompt
  3. Initializes ChatOpenAI model with configuration
  4. Creates ReAct agent with tools and prompts

### `prompts/system_prompt.py`

Defines the agent's behavior using the COSTAR framework:

- **Context**: XYZ Policy Assistant role and RAG-based approach
- **Objective**: Provide accurate, grounded answers from policy documents
- **Style**: Clear, factual, structured with bullet points
- **Tone**: Professional, neutral, helpful
- **Audience**: Company employees with varying policy knowledge
- **Response Rules**: Strict grounding requirements, citation format, scope limitations

## Configuration

The agent requires the following configuration (via `Config` object):

```python
MCP_SERVER_NAME = config.mcp_name
MCP_CONNECTIONS = {
    MCP_SERVER_NAME: {
        "command": "python",
        "args": ["-m", "mcp_rag_agent.mcp_server.server"],
        "transport": "stdio",
    }
}
```

### Environment Variables

See `src/mcp_rag_agent/core/config.py` for required configuration:
- `OPENAI_API_KEY`: OpenAI API key for LLM access
- `TEXT_MODEL`: Model name (e.g., "gpt-4o-mini")
- Additional text generation parameters

## Usage

### Basic Usage

```python
import asyncio
from mcp_rag_agent.agent.create_agent import create_mcp_rag_agent
from mcp_rag_agent.agent.prompts import system_prompt
from mcp_rag_agent.core.config import config

async def main():
    # Create the agent
    agent = await create_mcp_rag_agent(
        system_prompt=system_prompt,
        config=config
    )
    
    # Query the agent
    result = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": "What is the remote working policy?"
        }]
    })
    
    # Extract the final answer
    final_message = result["messages"][-1]
    print(final_message.content)

asyncio.run(main())
```

### Running the Demo

The module includes a built-in demo:

```bash
python -m mcp_rag_agent.agent.create_agent
```

This will run a sample query asking about annual leave days.

## ReAct Pattern

The agent uses the ReAct (Reasoning + Acting) pattern:

1. **Reasoning**: Agent analyzes the user query and decides which tool to use
2. **Acting**: Agent calls the appropriate tool (e.g., `search_documents`)
3. **Observation**: Agent receives tool results
4. **Iteration**: Agent may perform additional reasoning/acting cycles
5. **Response**: Agent formulates final answer based on observations

### Example Flow

```
User Query: "How many days of annual leave do employees get?"
    ↓
Agent Reasoning: "I need to search for annual leave policy"
    ↓
Tool Call: search_documents(query="annual leave days")
    ↓
Tool Result: [Retrieved policy documents with leave information]
    ↓
Agent Reasoning: "I have the information needed"
    ↓
Final Response: "Employees receive 25 days of annual leave per year..."
```

## Integration with MCP Server

The agent seamlessly integrates with the MCP server:

- **Tools**: The `search_documents` tool is automatically loaded from MCP
- **Prompts**: The `grounded_qa_prompt` ensures responses are grounded in retrieved context
- **Transport**: Uses stdio for efficient local communication
- **Sessions**: Each tool call creates a short-lived MCP session

## Response Format

The agent follows strict response guidelines:

1. **Grounded Answers**: Only information from retrieved documents
2. **Citations**: References to source documents
3. **Fallback Handling**: Clear messaging when information is unavailable
4. **No Hallucinations**: Refuses to speculate or make assumptions

### Example Response

```
Employees in the UK receive 25 days of annual leave per year, 
in addition to public holidays.

Reference:
1. 3 - Annual Leave.txt
```

## Error Handling

The agent gracefully handles:

- **Missing Context**: "I couldn't find this information in the available policy content."
- **Irrelevant Queries**: Redirects to appropriate channels
- **Out-of-Scope**: Declines to provide legal/HR/compliance advice
- **Connection Issues**: Logs errors and provides informative messages

## Dependencies

- `langchain`: Core agent framework
- `langchain-openai`: OpenAI LLM integration
- `langchain-mcp-adapters`: MCP protocol adapters
- `asyncio`: Asynchronous execution

## Best Practices

1. **Always use the provided system prompt** or extend it following COSTAR framework
2. **Test agent responses** for grounding and accuracy
3. **Monitor tool calls** to ensure efficient RAG retrieval
4. **Update prompts** when policy corpus changes significantly
5. **Configure appropriate model parameters** for your use case

## Troubleshooting

### Agent doesn't call tools

- Verify MCP server is running and accessible
- Check that tools are loaded: add logging in `_load_mcp_tools_and_prompt()`
- Ensure system prompt encourages tool usage

### Responses are not grounded

- Review the grounded QA prompt from MCP server
- Strengthen response rules in system prompt
- Reduce model temperature for more deterministic outputs

### Connection failures

- Verify MCP_CONNECTIONS configuration is correct
- Check that `mcp_rag_agent.mcp_server.server` is on PYTHONPATH
- Review MCP server logs for errors

## See Also

- [MCP Server](../mcp_server/README.md) - The RAG tools provider
- [Embeddings](../embeddings/README.md) - Document indexing and search
- [Evaluation](../../evaluation/README.md) - Agent performance testing

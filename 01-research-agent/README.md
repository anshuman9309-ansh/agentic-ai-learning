# Research Agent

This beginner-friendly project demonstrates a LangChain agent that selects the
right tool for a question. It uses LangChain's current `create_agent()` API.

## Tools

- `calculator` performs safe arithmetic using Python's AST module. It does not
  use `eval()`.
- `knowledge_base` answers local learning questions about RAG, AI Agents,
  Agentic AI, LangChain, and CrewAI from a small hardcoded dictionary.
- `tavily_search` searches the web through Tavily. The agent uses it for
  current information or facts that need an external source. Search results
  include source URLs that the agent can cite in its final answer.

The local knowledge base is intentionally not a RAG system and does not use a
vector database. Those concepts belong to a later project.

## How the agent chooses a tool

The system prompt and each tool's description guide the agent:

- Arithmetic uses `calculator`.
- The five supported learning concepts use `knowledge_base`.
- Current or external information uses `tavily_search`.
- A question that combines needs can call multiple tools before the agent
  writes one final answer.

The program prints the user question, selected tool names and arguments, tool
results, and final answer for every test. It does not call a tool when it is
not needed, and it does not describe information as current unless it used
web search.

## Setup

Install the project dependencies from the repository root:

```powershell
python -m pip install -r .\01-research-agent\requirements.txt
```

Create API keys in the providers' dashboards and add them to the repository
root `.env` file. Do not commit that file.

```text
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
```

`OPENAI_API_KEY` is used by `ChatOpenAI`. `TAVILY_API_KEY` is used by
`langchain-tavily` to run web searches. The program checks that both are set
but never prints their values.

## Run the tests

From the repository root, activate your virtual environment if you use one,
then run:

```powershell
python .\01-research-agent\langchain\main.py
```

The script runs these examples:

1. `What is 125 multiplied by 48?` — calculator only.
2. `What is RAG?` — local knowledge base only.
3. `What is the latest stable Python release?` — web search.
4. `Explain RAG, calculate 125 multiplied by 48, and find the latest stable Python release.` — multiple tools.

## What this project demonstrates

- Safe custom tools with LangChain's `@tool` decorator.
- A supported third-party LangChain search tool.
- Agent tool selection using `create_agent()`.
- Multi-tool orchestration and visible agent execution state.

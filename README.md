# Agentic AI Learning Journey

A hands-on portfolio documenting my progression from tool-enabled LLM applications to autonomous, multi-step agentic workflows. The repository is organised as five increasingly capable projects, with each project focused on a practical AI-agent pattern.

## Learning objectives

- Design reliable tools that an LLM can call safely.
- Build AI agents that plan and act against clear task boundaries.
- Ground answers with retrieval-augmented generation (RAG).
- Coordinate specialised agents in a multi-agent workflow.
- Evaluate, improve, and document agentic systems responsibly.

## Prerequisites

- Python programming fundamentals
- Basic understanding of large language models (LLMs), prompts, and APIs
- Familiarity with LangChain concepts
- Introductory knowledge of RAG, embeddings, and vector databases

## Technology stack

- Python 3.12
- LangChain
- CrewAI
- LLM APIs
- RAG and vector databases, where applicable

## Project roadmap

| Project | Focus | Key learning outcome |
| --- | --- | --- |
| 1. AI Research Assistant | Tool calling and research workflows | Connect an LLM to carefully scoped tools. |
| 2. RAG Knowledge Agent | Retrieval over a knowledge base | Produce grounded answers with retrieved context. |
| 3. Multi-Agent Content Team | Coordinated specialist agents | Delegate and combine work across agent roles. |
| 4. Data Analysis Agent | Data exploration and reporting | Turn structured data into useful analysis. |
| 5. Autonomous Project Assistant | End-to-end agentic workflow | Plan, execute, review, and improve multi-step tasks. |

## Learning progression

`Tool Calling → AI Agent → RAG Agent → Multi-Agent → Agentic Workflow`

## Repository structure

```text
agentic-ai-learning/
├── 01-research-agent/       # Project 1: AI Research Assistant
│   ├── langchain/
│   ├── README.md
│   └── requirements.txt
├── 02-rag-knowledge-agent/  # Planned
├── 03-content-team/         # Planned
├── 04-data-analysis-agent/  # Planned
└── 05-project-assistant/    # Planned
```

Only the first project directory currently exists; the remaining entries describe the intended roadmap.

## Development setup

Use Python 3.12 and the repository virtual environment:

```powershell
cd agentic-ai-learning
.\.venv\Scripts\Activate.ps1
python --version
```

Each project owns its own README and dependency file so it can be learned and run independently.

## Git workflow

1. Create a focused branch for each project or learning milestone.
2. Make small, descriptive commits as functionality is understood and verified.
3. Keep API keys in local environment files and never commit them.
4. Document notable design decisions and limitations in the relevant project README.

## Lessons learned and future improvements

This section will capture practical lessons about tool safety, prompting, retrieval quality, orchestration, evaluation, and cost control. Future improvements will include stronger testing, observability, guardrails, and reproducible evaluation datasets as each project evolves.

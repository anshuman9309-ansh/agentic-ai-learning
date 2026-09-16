# Research Agent — Step 1

This first exercise uses a safe calculator tool to demonstrate LLM Tool Calling. It does not create an AI agent.

## LLM Tool Calling

```text
User
 ↓
LLM
 ↓
Tool Call
 ↓
Application executes Tool
 ↓
Tool Result
```

The LLM returns a structured request for the calculator, but it cannot execute the Python function itself. This application reads the request and runs the safe calculator. In this milestone, the result is printed only; it is not sent back to the LLM. That manual hand-off keeps the difference between Tool Calling and an Agent clear.

## Run

From the project root, activate the virtual environment and run:

```powershell
.\.venv\Scripts\Activate.ps1
python .\01-research-agent\langchain\main.py
```

The program prints the calculator test results, the raw tool call requested by the LLM, and the calculator result.

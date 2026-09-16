import ast
import os
import operator
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _evaluate_expression(node: ast.AST) -> int | float:
    """Evaluate only number literals and approved arithmetic AST nodes."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
        value = _evaluate_expression(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_expression(node.left)
        right = _evaluate_expression(node.right)
        return _BINARY_OPERATORS[type(node.op)](left, right)

    raise ValueError("Only numbers, +, -, *, /, and parentheses are allowed.")


@tool
def calculator(expression: str) -> str:
    """Calculate basic arithmetic using numbers, +, -, *, /, and parentheses."""
    try:
        expression_tree = ast.parse(expression, mode="eval")
        return str(_evaluate_expression(expression_tree.body))
    except SyntaxError as error:
        raise ValueError("Enter a valid arithmetic expression.") from error


def run_tests() -> None:
    """A few small checks for the calculator learning exercise."""
    test_cases = {
        "25 * 17": "425",
        "(10 + 5) / 3": "5.0",
        "3.5 * 2 - 1": "6.0",
        "-4 + 10": "6",
    }

    for expression, expected_result in test_cases.items():
        actual_result = calculator.invoke(expression)
        print(f"{expression} = {actual_result} = {expected_result}")
        assert actual_result == expected_result, f"{expression}: expected {expected_result}"


def demonstrate_tool_calling() -> None:
    """Ask an LLM to request the calculator, then execute that request once."""
    # main.py is nested two folders below the repository root, so build the
    # absolute path instead of relying on the folder where Python was started.
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from the repository root .env file.")

    model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    # bind_tools() gives the LLM the calculator's name, description, and input
    # schema. It lets the model return a structured tool-call request.
    # tool_choice forces this single-tool lesson to demonstrate a tool call.
    model_with_tools = model.bind_tools([calculator], tool_choice="calculator")

    user_request = "What is 125 multiplied by 48?"
    response = model_with_tools.invoke(user_request)

    # A tool_call is data returned by the LLM: the tool name and its arguments.
    # The LLM does not run Python functions itself; it only asks our application
    # to run one. An application can validate, authorize, and execute the call.
    print(f"Model response: {response.content}")
    print(f"Tool calls: {response.tool_calls}")

    for tool_call in response.tool_calls:
        print(f"Requested tool: {tool_call['name']}")
        print(f"Tool arguments: {tool_call['args']}")

        if tool_call["name"] == calculator.name:
            calculator_result = calculator.invoke(tool_call["args"])
            print(f"Calculator result: {calculator_result}")

    # This is Tool Calling, not an Agent: there is no planning loop, no agent
    # abstraction, and no tool result sent back to the LLM in this milestone.


if __name__ == "__main__":
    run_tests()
    demonstrate_tool_calling()

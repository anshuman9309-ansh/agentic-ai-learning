import ast
import os
import operator
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
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


def run_manual_tool_calling_loop() -> None:
    """Manually pass tool results back to the LLM until it gives a final answer."""
    # main.py is nested two folders below the repository root, so build the
    # absolute path instead of relying on the folder where Python was started.
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from the repository root .env file.")

    model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    # bind_tools() gives the LLM the calculator's name, description, and input
    # schema. It lets the model return a structured tool-call request, but it
    # does not execute any Python code.
    model_with_tools = model.bind_tools([calculator])
    # Force only the first response to use the calculator so this learning
    # example reliably demonstrates the tool-call portion of the loop.
    first_response_model = model.bind_tools([calculator], tool_choice="calculator")

    # This dictionary is the application's allow-list. Never execute a tool
    # merely because an LLM named it in a response.
    allowed_tools = {calculator.name: calculator}

    user_question = "What is 125 multiplied by 48?"
    # The messages list is our manual conversation state. It starts with the
    # user's question and grows with AI responses and tool results.
    messages = [("user", user_question)]
    max_iterations = 5

    print(f"User question: {user_question}")

    # An Agent would hide this loop. Here we write each step ourselves so the
    # flow from LLM request to application execution is visible.
    for iteration in range(max_iterations):
        # First ask for a calculator call; later requests allow the LLM to use
        # the ToolMessage and produce a normal final answer.
        if iteration == 0:
            response = first_response_model.invoke(messages)
            print(f"First LLM response: {response.content}")
            print(f"First LLM tool calls: {response.tool_calls}")
        else:
            response = model_with_tools.invoke(messages)

        # No tool calls means the LLM has finished answering the user.
        if not response.tool_calls:
            print(f"Final LLM response: {response.content}")
            return

        # Keep the AI response because it contains the tool-call ID that the
        # ToolMessage must reference when it returns the result.
        messages.append(response)

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_arguments = tool_call["args"]
            print(f"Tool name: {tool_name}")
            print(f"Tool arguments: {tool_arguments}")

            tool = allowed_tools.get(tool_name)
            if tool is None:
                # Do not execute unexpected names. Return a safe error message
                # to the LLM instead, so it can choose how to proceed.
                tool_result = f"Tool '{tool_name}' is not allowed."
                print(tool_result)
            else:
                # The application—not the LLM—executes the approved safe tool.
                tool_result = tool.invoke(tool_arguments)
                print(f"Tool result: {tool_result}")

            # Add the result to the conversation, linked to the exact tool call.
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

    raise RuntimeError(f"Stopped after {max_iterations} tool-calling iterations.")


if __name__ == "__main__":
    run_tests()
    run_manual_tool_calling_loop()

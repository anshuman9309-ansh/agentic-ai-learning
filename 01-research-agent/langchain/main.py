import ast
import os
import operator
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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
    """Show the tool-calling loop explicitly, without an agent abstraction."""
    # Load the root .env reliably, even when this file is run from its folder.
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from the repository root .env file.")

    model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    # bind_tools() tells the model about the calculator. The model can request
    # it, but our application still has to execute the Python function.
    model_with_tools = model.bind_tools([calculator])
    first_response_model = model.bind_tools([calculator], tool_choice="calculator")
    allowed_tools = {calculator.name: calculator}

    user_question = "What is 125 multiplied by 48?"
    messages = [("user", user_question)]
    max_iterations = 5

    print("\n--- Manual tool-calling loop ---")
    print(f"User question: {user_question}")

    for iteration in range(max_iterations):
        # The first call is forced so this lesson always shows a tool call.
        response = (
            first_response_model.invoke(messages)
            if iteration == 0
            else model_with_tools.invoke(messages)
        )

        if iteration == 0:
            print(f"First LLM response: {response.content}")
            print(f"First LLM tool calls: {response.tool_calls}")

        # A response without tool calls is the manual loop's final answer.
        if not response.tool_calls:
            print(f"Final LLM response: {response.content}")
            return

        # Preserve the AI tool-call message before adding its matching result.
        messages.append(response)

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_arguments = tool_call["args"]
            print(f"Tool name: {tool_name}")
            print(f"Tool arguments: {tool_arguments}")

            tool = allowed_tools.get(tool_name)
            if tool is None:
                # An allow-list prevents unexpected tool names from executing.
                tool_result = f"Tool '{tool_name}' is not allowed."
            else:
                tool_result = tool.invoke(tool_arguments)
                print(f"Tool result: {tool_result}")

            # Send this result back to the model on the next iteration.
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

    raise RuntimeError(f"Stopped after {max_iterations} tool-calling iterations.")


def run_calculator_agent() -> None:
    """Run the first LangChain agent and display its completed message state."""
    # main.py is nested two folders below the repository root, so build the
    # absolute path instead of relying on the folder where Python was started.
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from the repository root .env file.")

    model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    # create_agent() builds the loop we wrote manually in the previous lesson:
    # model -> tool call -> tool execution -> ToolMessage -> model. Passing the
    # tool here lets LangChain bind it to the model and manage that loop for us.
    # An agent is more than bind_tools(): it also executes requested tools,
    # adds their results to the conversation, and calls the model again.
    agent = create_agent(
        model=model,
        tools=[calculator],
        system_prompt="Use the calculator tool for arithmetic calculations.",
    )

    # The agent chooses whether a tool is useful from the user's request and
    # the calculator description. The system prompt makes the choice explicit
    # for this focused learning example.
    # user_question = "What is 125 multiplied by 48?"
    user_question = "What is the capital of France?"
    print("\n--- LangChain agent ---")
    print(f"User question: {user_question}")

    # invoke() returns the complete agent state, including every message the
    # framework added while it ran the tool-calling loop.
    result = agent.invoke({"messages": [{"role": "user", "content": user_question}]})

    print("\nAgent execution state:")
    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            print(f"- User message: {message.content}")
        elif isinstance(message, AIMessage) and message.tool_calls:
            print(f"- AI tool-call step: {message.tool_calls}")
        elif isinstance(message, ToolMessage):
            print(f"- Calculator tool result: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"- Final AI response: {message.content}")

    final_answer = result["messages"][-1].content
    print(f"\nFinal answer: {final_answer}")


if __name__ == "__main__":
    # run_tests()
    # run_manual_tool_calling_loop()
    run_calculator_agent()

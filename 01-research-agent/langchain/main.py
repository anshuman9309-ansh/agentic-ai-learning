import ast
import operator

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


if __name__ == "__main__":
    run_tests()
    print(calculator.invoke("25 * 17"))

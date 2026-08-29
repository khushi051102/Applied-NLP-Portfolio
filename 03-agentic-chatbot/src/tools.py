"""
Example tools the agent can call. Replace `search_stub` with a real API
for a stronger portfolio story -- this is intentionally minimal so the
routing logic in agent.py is the focus.
"""
import ast
import operator


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression (no eval() footguns)."""
    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except Exception as e:
        return f"Error evaluating expression: {e}"


def search_stub(query: str) -> str:
    """Placeholder for a real search/API call -- swap in something real."""
    return f"[stub] No live search connected. Would have searched for: '{query}'"


TOOL_REGISTRY = {
    "calculator": {
        "fn": calculator,
        "description": "Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'.",
    },
    "search": {
        "fn": search_stub,
        "description": "Look up current information on a topic (stub -- replace with a real API).",
    },
}

"""
Hybrid Math Engine for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Combines:
- Tier-0: Instant (<5ms) deterministic local arithmetic for direct calculations.
- Tier-1: Agentic reasoning fallback for word problems, multi-step math, and algebra.
"""
import ast
import operator
import re
from typing import Optional, Union

# Safe AST operators for Tier-0
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast(node: ast.AST) -> Union[int, float]:
    """Recursively evaluate an AST expression using only safe mathematical operators."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported operator: {op_type}")
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported operator: {op_type}")
    raise ValueError(f"Unsupported AST node: {type(node)}")


class HybridMathEngine:
    """
    Tier-0 and Tier-1 Math Evaluator.
    """

    @staticmethod
    def try_tier0_instant_arithmetic(text: str) -> Optional[Union[int, float]]:
        """
        Attempts instant evaluation of pure arithmetic without LLM inference.
        Returns numeric result if cleanly matched, None if conversational or word problem.
        """
        raw = text.strip().lower()

        # Strip common prefixes like 'calculate', 'what is', 'solve'
        raw = re.sub(r"^(?:hey\s+maddy\s*,?|maddy\s*,?|hey\s+m\.a\.d\.d\.y\.?\s*,?|m\.a\.d\.d\.y\.?\s*,?|what\s+is|what's|calculate|compute|solve|how\s+much\s+is)\s+", "", raw).strip()
        raw = re.sub(r"[?!.]+$", "", raw).strip()

        # Handle simple percent phrases: "15% of 200"
        pct_match = re.match(r"^(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)$", raw)
        if pct_match:
            pct = float(pct_match.group(1))
            total = float(pct_match.group(2))
            res = (pct / 100.0) * total
            return int(res) if res.is_integer() else round(res, 4)

        # Replace word arithmetic operators with math symbols
        normalized = raw
        normalized = re.sub(r"\bplus\b", "+", normalized)
        normalized = re.sub(r"\bminus\b", "-", normalized)
        normalized = re.sub(r"\btimes\b|\bmultiplied\s+by\b", "*", normalized)
        normalized = re.sub(r"\bdivided\s+by\b|\bover\b", "/", normalized)
        normalized = re.sub(r"\bto\s+the\s+power\s+of\b", "**", normalized)

        # Check if the string consists strictly of numbers, math operators, and parentheses
        if not re.match(r"^[0-9\s\+\-\*\/\(\)\.\^%]+$", normalized):
            return None

        # Replace ^ with ** for powers
        normalized = normalized.replace("^", "**")

        try:
            tree = ast.parse(normalized, mode="eval")
            res = _eval_ast(tree.body)
            return int(res) if isinstance(res, float) and res.is_integer() else round(res, 6)
        except Exception:
            return None

"""Safe Mathematical Calculator Tool for M.A.D.D.Y. Agent Core (Modular Autonomous Digital Daemon for You).

Provides safe, deterministic mathematical evaluation for arithmetic, algebra,
trigonometry, logarithms, factorials, and percentages within a restricted namespace.
Deterministic Tier: SAFE (auto-allowed).
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Dict

from core.math_engine import SafeMathEvaluator
from core.permissions import PermissionTier
from core.tools.base import BaseTool, ToolResult

# Safe math namespace
SAFE_MATH_NAMESPACE: Dict[str, Any] = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "math": math,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "factorial": math.factorial,
    "ceil": math.ceil,
    "floor": math.floor,
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

# Forbidden keywords that could indicate code execution or injection
FORBIDDEN_KEYWORDS = (
    "import",
    "eval",
    "exec",
    "open",
    "os",
    "sys",
    "subprocess",
    "lambda",
    "class",
    "def",
    "global",
    "nonlocal",
    "yield",
    "await",
    "async",
)


class CalculatorTool(BaseTool):
    """Safely evaluates mathematical expressions using a restricted Python math environment."""

    name = "calculator"
    description = (
        "Evaluate mathematical expressions, formulas, and calculations with high precision. "
        "Supports arithmetic (+, -, *, /, //, %, **), math functions (sqrt, pow, sin, cos, tan, "
        "log, exp, factorial, ceil, floor, abs, round), constants (pi, e), and percentages."
    )
    tier = PermissionTier.SAFE

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Mathematical expression or formula to evaluate "
                        "(e.g. '25 * 4', 'sqrt(144) + 12', 'factorial(5)', '15% * 200', '2 ** 8')."
                    ),
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        }

    def execute(self, expression: str, **kwargs: Any) -> ToolResult:
        if not expression or not expression.strip():
            return ToolResult(
                success=False,
                output="",
                error="Empty mathematical expression provided.",
            )

        clean_expr = expression.strip()

        # Disallow dunder attributes and statement chaining
        if "__" in clean_expr or ";" in clean_expr:
            return ToolResult(
                success=False,
                output="",
                error="Security restriction: '__' and ';' are not permitted in math calculations.",
                metadata={"expression": clean_expr},
            )

        # Check for forbidden keywords using word boundary regex to avoid false positives (e.g. 'cos')
        for word in FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{re.escape(word)}\b", clean_expr):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Security restriction: '{word}' is not permitted in math calculations.",
                    metadata={"expression": clean_expr},
                )

        try:
            # Pre-process percentage notation (e.g. '15% * 200' -> '(15/100) * 200', but keep modulo '10 % 3' intact)
            normalized = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\w(])", r"(\1/100)", clean_expr)

            # Evaluate safely via AST parser without eval()
            parsed_ast = ast.parse(normalized, mode="eval")
            evaluator = SafeMathEvaluator()
            result = evaluator.visit(parsed_ast)

            # Normalize floating point representations
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            elif isinstance(result, float):
                result = round(result, 8)

            return ToolResult(
                success=True,
                output=str(result),
                metadata={
                    "expression": clean_expr,
                    "result": result,
                },
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Evaluation error: {type(exc).__name__}: {str(exc)}",
                metadata={"expression": clean_expr},
            )

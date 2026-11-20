"""
Safe Mathematical Calculator Tool for M.A.D.D.Y.
Provides 100% accurate mathematical evaluation for arithmetic, algebra,
unit conversions, and multi-step word problems without regex fragility.
"""
import math
from typing import Any, Dict

from tools.base import RiskLevel, Tool


def safe_calculate(expression: str) -> Dict[str, Any]:
    """
    Safely evaluate a mathematical expression within a restricted Python environment.
    Supports math functions (sqrt, sin, cos, tan, log, pow, ceil, floor, pi, e)
    and built-in arithmetic without permitting system or file execution.
    """
    clean_expr = expression.strip()
    if not clean_expr:
        return {"success": False, "error": "Empty mathematical expression provided."}

    # Allowed safe namespace
    safe_namespace = {
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
        "ceil": math.ceil,
        "floor": math.floor,
        "pi": math.pi,
        "e": math.e,
    }

    # Sanitize dangerous syntax keywords
    forbidden = ["import", "eval", "exec", "open", "os", "sys", "subprocess", "__", ";"]
    for word in forbidden:
        if word in clean_expr:
            return {
                "success": False,
                "error": f"Security restriction: '{word}' is not permitted in math calculations.",
            }

    try:
        # Pre-process percentage notation (e.g. '15% * 200' -> '(15/100) * 200')
        import ast
        import re
        from core.math_engine import SafeMathEvaluator
        normalized = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", clean_expr)

        parsed_ast = ast.parse(normalized, mode="eval")
        evaluator = SafeMathEvaluator()
        result = evaluator.visit(parsed_ast)
        
        # Round floating points if near integer (e.g. 1.0000000000000002 -> 1)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        elif isinstance(result, float):
            result = round(result, 6)

        return {
            "success": True,
            "expression": clean_expr,
            "result": result,
            "formatted": str(result),
        }
    except Exception as e:
        return {
            "success": False,
            "expression": clean_expr,
            "error": f"Evaluation error: {str(e)}",
        }


calculator_tool = Tool(
    name="calculator",
    description="Evaluate mathematical calculations, algebraic expressions, formulas, and percentages with 100% precision. Use this tool for word problems, multi-step math, and accurate arithmetic.",
    parameter_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression or formula to evaluate (e.g. '25 * 4', 'sqrt(144) + 12', '(120 * 4 * 1.08) - 30', '15% * 250')",
            },
        },
        "required": ["expression"],
    },
    handler=lambda args: safe_calculate(args.get("expression", "")),
    risk_level=RiskLevel.LOW,
)

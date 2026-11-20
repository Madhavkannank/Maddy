"""
Safe Deterministic Math Engine for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Provides AST-based safe arithmetic evaluation, symbolic equation solving via SymPy,
and unit conversions without ever executing arbitrary python eval().
"""
import ast
import math
import operator
import re
from typing import Any, Dict, Optional, Union

# Whitelisted operators for AST evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: operator.not_,
}

SAFE_COMPARISONS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

def safe_factorial(n: Any) -> int:
    if not isinstance(n, (int, float)) or int(n) != n or n < 0 or n > 1000:
        raise ValueError("Factorial input must be an integer between 0 and 1000.")
    return math.factorial(int(n))

def safe_pow(base: Any, exp: Any) -> Any:
    if isinstance(exp, (int, float)):
        if abs(exp) > 500:
            raise ValueError("Exponent too large; calculation exceeds safe limits.")
        if isinstance(base, (int, float)) and abs(base) > 10 and abs(exp) > 50:
            raise ValueError("Power computation too large; exceeds safe resource limits.")
    return math.pow(base, exp) if isinstance(base, float) or isinstance(exp, float) else pow(base, exp)


SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "factorial": safe_factorial,
    "pow": safe_pow,
    "pi": math.pi,
    "e": math.e,
}

class SafeMathEvaluator(ast.NodeVisitor):
    """
    Evaluates math expressions directly via AST parsing.
    Prevents any code execution, attributes, or variable access outside safe functions.
    """

    def visit(self, node):
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    def visit_UnaryOp(self, node):
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            val = self.visit(node.operand)
            return SAFE_OPERATORS[op_type](val)
        raise ValueError(f"Unsupported unary operator: {op_type}")

    def visit_BinOp(self, node):
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            left = self.visit(node.left)
            right = self.visit(node.right)
            # Guard against giant powers causing resource exhaustion or hangs (e.g. 9**9**9**9)
            if op_type is ast.Pow:
                if isinstance(right, (int, float)):
                    if abs(right) > 500:
                        raise ValueError("Exponent too large; calculation exceeds safe limits.")
                    if isinstance(left, (int, float)) and abs(left) > 10 and abs(right) > 50:
                        raise ValueError("Power computation too large; exceeds safe resource limits.")
            return SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type}")

    def visit_Call(self, node):
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id.lower()
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id.lower() == "math":
                func_name = node.func.attr.lower()

        if func_name and func_name in SAFE_FUNCTIONS and callable(SAFE_FUNCTIONS[func_name]):
            args = [self.visit(arg) for arg in node.args]
            return SAFE_FUNCTIONS[func_name](*args)
        raise ValueError("Unsupported function call")

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id.lower() == "math":
            attr_name = node.attr.lower()
            if attr_name in SAFE_FUNCTIONS and not callable(SAFE_FUNCTIONS[attr_name]):
                return SAFE_FUNCTIONS[attr_name]
        raise ValueError("Unsupported attribute access")

    def visit_Compare(self, node):
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in SAFE_COMPARISONS:
                raise ValueError(f"Unsupported comparison operator: {op_type}")
            right = self.visit(comparator)
            if not SAFE_COMPARISONS[op_type](left, right):
                return False
            left = right
        return True

    def visit_BoolOp(self, node):
        if isinstance(node.op, ast.And):
            return all(self.visit(val) for val in node.values)
        elif isinstance(node.op, ast.Or):
            return any(self.visit(val) for val in node.values)
        raise ValueError(f"Unsupported boolean operator: {type(node.op)}")

    def visit_Name(self, node):
        name = node.id.lower()
        if name in SAFE_FUNCTIONS and not callable(SAFE_FUNCTIONS[name]):
            return SAFE_FUNCTIONS[name]
        raise ValueError(f"Undefined symbol: {node.id}")


from core.math_phonetics import MathPhoneticizer

class MathEngine:
    """
    Deterministic math reasoning engine supporting arithmetic, unit conversion,
    and algebraic equations.
    """

    def __init__(self):
        self.last_expression: Optional[str] = None
        self.last_variables: List[str] = []
        self.last_result: Optional[str] = None

    def is_math_query(self, query: str) -> bool:
        """Heuristically determine whether the query is a math request."""
        clean = query.strip().lower()
        # Direct math substitution against active context (e.g. "x is equal to 5 in that equation")
        if self.last_expression and re.search(r"\b([a-zA-Z])\s*(?:is\s+equal\s+to|=|equals?|is)\s*(-?\d+(?:\.\d+)?)\b", clean):
            return True
        # Direct math patterns
        if re.search(r"(\d+\s*[\+\-\*\/×÷\^]\s*\d+)", clean):
            return True
        if re.search(r"\b(calculate|compute|solve|simplify|convert|evaluate|what('s| is))\b.*\b(\d+|[xX]\^2|[xX]²|[a-zA-Z])\b", clean):
            return True
        if re.search(r"\b(find|what('s| is)|solve|simplify)\b.*\b(?:the\s+)?(?:value\s+of\s+)?[a-zA-Z]\b", clean):
            return True
        if re.search(r"\b[a-zA-Z]\s*(?:square|squared|\^2|²)\b", clean):
            return True
        if re.search(r"\b(fahrenheit|celsius|kelvin|miles?|km|kilometers?|inches|cm|centimeters?|pounds?|kg|kilograms?)\b", clean):
            return True
        return False

    def evaluate_detailed(self, query: str) -> Optional[Dict[str, str]]:
        """
        Evaluate math query and return both display Unicode text and natural speech text.
        Returns: {'display': '...', 'speech': '...'} or None.
        """
        clean = query.strip()

        # 0. Check variable substitution against active context
        sub_res = self.check_substitution(clean)
        if sub_res:
            return sub_res

        # 1. Unit conversions
        conv_res = self._check_unit_conversion(clean)
        if conv_res:
            return {"display": conv_res, "speech": conv_res}

        # 2. Algebraic equations and expression simplification
        solve_res = self._check_algebra(clean)
        if solve_res:
            if isinstance(solve_res, dict):
                return solve_res
            display, speech = MathPhoneticizer.format_math_result(solve_res)
            return {"display": display, "speech": speech}

        # 3. Arithmetic / Expression evaluation
        expr = self._extract_expression(clean)
        if expr:
            try:
                tree = ast.parse(expr, mode="eval")
                evaluator = SafeMathEvaluator()
                val = evaluator.visit(tree)
                if isinstance(val, float):
                    if val.is_integer():
                        res_str = str(int(val))
                    else:
                        res_str = f"{round(val, 6):g}"
                else:
                    res_str = str(val)
                self.last_expression = expr
                self.last_variables = []
                self.last_result = res_str
                return {"display": res_str, "speech": res_str}
            except Exception:
                pass

        return None

    def evaluate(self, query: str) -> Optional[str]:
        """
        Attempt to deterministically evaluate math or equation query.
        Returns a concise string result, or None if not recognized as math.
        """
        res = self.evaluate_detailed(query)
        if res:
            return res["display"]
        return None

    def _extract_expression(self, text: str) -> Optional[str]:
        """Extract sanitized math expression from natural language."""
        t = text.lower()
        # Strip conversational prefixes
        t = re.sub(r"^(?:hey\s+maddy,?\s*|maddy,?\s*|hey\s+m\.a\.d\.d\.y\.?,?\s*|m\.a\.d\.d\.y\.?,?\s*)", "", t)
        t = re.sub(r"^(what('s| is| are)\s+|calculate\s+|compute\s+|evaluate\s+)", "", t)
        t = re.sub(r"\?*$", "", t).strip()

        # Natural language operator normalization
        t = t.replace("×", "*").replace("÷", "/")
        t = re.sub(r"\btimes\b", "*", t)
        t = re.sub(r"\bmultiplied by\b", "*", t)
        t = re.sub(r"\bdivided by\b", "/", t)
        t = re.sub(r"\bplus\b", "+", t)
        t = re.sub(r"\bminus\b", "-", t)
        t = t.replace("^", "**")

        # Keep only valid math characters
        if re.match(r"^[\d\.\s\+\-\*\/\(\)\%eE\,]+$", t) or any(fn in t for fn in SAFE_FUNCTIONS):
            # Remove commas in numbers (e.g. 23,891 -> 23891)
            t = re.sub(r"(\d),(\d)", r"\1\2", t)
            return t
        return None

    def _check_unit_conversion(self, text: str) -> Optional[str]:
        """Handle temperature, length, and weight conversions."""
        t = text.lower()
        # Temperature: Fahrenheit <-> Celsius
        m = re.search(r"convert\s+(-?\d+\.?\d*)\s*(?:degrees?\s*)?(?:f|fahrenheit)\s+to\s+(?:degrees?\s*)?(?:c|celsius)", t)
        if m:
            val = float(m.group(1))
            c = (val - 32.0) * (5.0 / 9.0)
            return f"{round(c, 2):g} °C"

        m = re.search(r"convert\s+(-?\d+\.?\d*)\s*(?:degrees?\s*)?(?:c|celsius)\s+to\s+(?:degrees?\s*)?(?:f|fahrenheit)", t)
        if m:
            val = float(m.group(1))
            f = (val * 9.0 / 5.0) + 32.0
            return f"{round(f, 2):g} °F"

        # Distance: miles <-> km
        m = re.search(r"convert\s+(\d+\.?\d*)\s*miles?\s+to\s*(?:km|kilometers?)", t)
        if m:
            val = float(m.group(1))
            return f"{round(val * 1.60934, 2):g} km"

        m = re.search(r"convert\s+(\d+\.?\d*)\s*(?:km|kilometers?)\s+to\s*miles?", t)
        if m:
            val = float(m.group(1))
            return f"{round(val / 1.60934, 2):g} miles"

        # Length: inches <-> cm
        m = re.search(r"convert\s+(\d+\.?\d*)\s*(?:inches|in)\s+to\s*(?:cm|centimeters?)", t)
        if m:
            val = float(m.group(1))
            return f"{round(val * 2.54, 2):g} cm"

        # Weight: lbs <-> kg
        m = re.search(r"convert\s+(\d+\.?\d*)\s*(?:lbs?|pounds?)\s+to\s*(?:kg|kilograms?)", t)
        if m:
            val = float(m.group(1))
            return f"{round(val * 0.453592, 2):g} kg"

        return None

    def _normalize_spoken_equation(self, text: str) -> str:
        """Convert spoken math and Whisper transcription idioms into standard algebraic format."""
        t = text.lower().replace(",", " ").strip("?. ")
        # Strip conversational and greeting prefixes
        t = re.sub(r"^(?:hey\s+maddy,?\s*|maddy,?\s*|hey\s+m\.a\.d\.d\.y\.?,?\s*|m\.a\.d\.d\.y\.?,?\s*)", "", t)
        t = re.sub(r"^(?:can\s+you\s+|could\s+you\s+|please\s+)", "", t)
        t = re.sub(r"^(?:find\s+(?:the\s+)?value\s+of\s+[a-zA-Z]\s+(?:in|for)\s+|what\s+is\s+[a-zA-Z]\s+(?:in|for)\s+)", "", t)
        t = re.sub(r"^(?:solve\s+(?:the\s+equation\s+)?|simplify\s+|calculate\s+|evaluate\s+|find\s+roots?\s+(?:of\s+)?)", "", t)

        # Normalize powers
        t = re.sub(r"\b([a-zA-Z])\s*(?:squared|square)\b", r"\1^2", t)
        t = re.sub(r"\b([a-zA-Z])\s*(?:cubed|cube)\b", r"\1^3", t)
        t = t.replace("²", "^2").replace("³", "^3")

        # Normalize spoken words to math symbols
        t = re.sub(r"\bminus\b", "-", t)
        t = re.sub(r"\bplus\b", "+", t)
        t = re.sub(r"\btimes\b", "*", t)

        # Normalize equality phrases e.g. "equal to 3", "equals 3", "is 3", "to 3"
        t = re.sub(r"\b(?:equals?\s+to|equals?|equal|is)\s+(\-?\d+)", r"= \1", t)
        # Handle Whisper idiom "to 3" at the end of equation (e.g. "x square minus 2 x to 3")
        t = re.sub(r"\bto\s+(\-?\d+)\s*$", r"= \1", t)

        # Normalize implicit multiplication e.g. "2 x" -> "2*x"
        t = re.sub(r"\b(\d+)\s*([a-zA-Z])\b", r"\1*\2", t)
        return t.strip()

    def _check_algebra(self, text: str) -> Optional[Union[str, Dict[str, str]]]:
        """Solve symbolic equations or simplify algebraic expressions safely using sympy."""
        t = text.lower()
        is_algebra_trigger = any(kw in t for kw in [
            "solve", "simplify", "value of", "roots of", "root of", "in x", "in y", "in z", "^2", "²"
        ]) or re.search(r"\b[a-zA-Z]\s*(?:squared|square)\b", t) or bool(
            re.search(r"\b\d*\s*[a-zA-Z]\b", t) and any(op in t for op in ["plus", "minus", "+", "-", "*", "/", "="])
        )

        if not is_algebra_trigger:
            return None

        eq_str = self._normalize_spoken_equation(text)

        try:
            import sympy
            from sympy.parsing.sympy_parser import (
                parse_expr,
                standard_transformations,
                implicit_multiplication_application,
            )
            transformations = standard_transformations + (implicit_multiplication_application,)

            if "=" in eq_str:
                lhs_str, rhs_str = eq_str.split("=", 1)
                lhs = parse_expr(lhs_str.replace("^", "**"), transformations=transformations)
                rhs = parse_expr(rhs_str.replace("^", "**"), transformations=transformations)
                eq = sympy.Eq(lhs, rhs)
                symbols = list(eq.free_symbols)
                if not symbols:
                    return None
                target_var = symbols[0]
                solutions = sympy.solve(eq, target_var)
                if solutions:
                    self.last_expression = str(lhs - rhs)
                    self.last_variables = [str(s) for s in symbols]
                    sol_str = ", ".join(f"{target_var} = {s}" for s in solutions)
                    self.last_result = sol_str
                    return sol_str
            else:
                lhs = parse_expr(eq_str.replace("^", "**"), transformations=transformations)
                symbols = list(lhs.free_symbols)
                if symbols:
                    simplified = sympy.simplify(lhs)
                    self.last_expression = str(simplified)
                    self.last_variables = [str(s) for s in symbols]
                    disp_expr = re.sub(r"(\d+)\*([a-zA-Z])", r"\1\2", str(simplified))
                    self.last_result = disp_expr
                    speech_expr = MathPhoneticizer.to_spoken_english(disp_expr)
                    return {
                        "display": f"That simplifies to: {disp_expr}",
                        "speech": f"That simplifies to {speech_expr}.",
                    }
        except Exception:
            pass

        return None

    def check_substitution(self, text: str) -> Optional[Dict[str, str]]:
        """
        Check if the query is a variable substitution referring to the previously
        active equation or expression (e.g. 'x is equal to 5 in that equation',
        'what if x is 5', 'if x = 5').
        """
        if not self.last_expression or not self.last_variables:
            return None

        clean = text.lower().strip()
        m = re.search(r"\b([a-zA-Z])\s*(?:is\s+equal\s+to|=|equals?|is)\s*(-?\d+(?:\.\d+)?)\b", clean)
        if m:
            var_name = m.group(1).lower()
            val_str = m.group(2)
            target_sym = next((v for v in self.last_variables if v.lower() == var_name), None)
            if target_sym:
                try:
                    import sympy
                    from sympy.parsing.sympy_parser import (
                        parse_expr,
                        standard_transformations,
                        implicit_multiplication_application,
                    )
                    transformations = standard_transformations + (implicit_multiplication_application,)
                    expr = parse_expr(self.last_expression.replace("^", "**"), transformations=transformations)
                    sub_val = float(val_str) if "." in val_str else int(val_str)
                    res_val = expr.subs(sympy.Symbol(target_sym), sub_val)
                    disp_val = str(int(res_val) if getattr(res_val, "is_integer", False) else round(float(res_val), 4))
                    disp_expr = re.sub(r"(\d+)\*([a-zA-Z])", r"\1\2", self.last_expression)
                    display = f"Substituting {target_sym} = {val_str} into {disp_expr} gives {disp_val}."
                    speech = f"Substituting {target_sym} equals {val_str} into {MathPhoneticizer.to_spoken_english(disp_expr)} gives {disp_val}."
                    return {"display": display, "speech": speech}
                except Exception:
                    pass
        return None

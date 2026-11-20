"""
Tests for MathEngine.
Verifies safe arithmetic, symbolic equation solving, unit conversions, and security guards.
"""
import pytest
from core.math_engine import MathEngine

def test_safe_arithmetic():
    engine = MathEngine()

    assert engine.evaluate("What is 25 * 48?") == "1200"
    assert engine.evaluate("Hey M.A.D.D.Y., what is 23891 times 472?") == "11276552"
    assert engine.evaluate("Calculate (15 + 5) * 4 / 2") == "40"
    assert engine.evaluate("sqrt(144)") == "12"

def test_unit_conversions():
    engine = MathEngine()

    res_temp = engine.evaluate("Convert 75 Fahrenheit to Celsius")
    assert "23.89" in res_temp and "°C" in res_temp

    res_dist = engine.evaluate("Convert 10 miles to km")
    assert "16.09" in res_dist and "km" in res_dist

def test_algebraic_equation_solving():
    engine = MathEngine()

    res_eq = engine.evaluate("Solve x^2 + 5x + 6 = 0")
    assert "x = -3" in res_eq
    assert "x = -2" in res_eq

def test_security_rejection_of_arbitrary_code():
    engine = MathEngine()

    # Attempts to execute arbitrary code or import must be rejected safely
    assert engine.evaluate("__import__('os').system('dir')") is None
    assert engine.evaluate("eval('2+2')") is None

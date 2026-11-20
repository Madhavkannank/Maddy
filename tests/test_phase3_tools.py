"""Unit tests for Phase 3 M.A.D.D.Y. tools:
1. CalculatorTool (core/tools/calculator.py)
2. WeatherTool (core/tools/weather.py)
3. ListDirectoryTool (core/tools/file_ops.py)
4. ToolRegistry integration with all registered Phase 2 and Phase 3 tools.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import httpx
import pytest

from core.permissions import PermissionTier
from core.tools import (
    BaseTool,
    CalculatorTool,
    ExecuteCommandTool,
    ListDirectoryTool,
    ReadFileTool,
    SystemInfoTool,
    ToolRegistry,
    WeatherTool,
    WriteFileTool,
)
from core.tools.weather import GEOCODING_URL, FORECAST_URL


# ==============================================================================
# CalculatorTool Tests
# ==============================================================================

def test_calculator_tier_and_schema():
    calc = CalculatorTool()
    assert calc.tier == PermissionTier.SAFE
    assert calc.name == "calculator"
    schema = calc.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculator"
    assert "expression" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["required"] == ["expression"]


def test_calculator_arithmetic_operations():
    calc = CalculatorTool()

    # Addition and subtraction
    res = calc.execute("25 + 75 - 10")
    assert res.success is True
    assert res.output == "90"
    assert res.metadata["result"] == 90

    # Multiplication and float normalization
    res = calc.execute("12 * 8")
    assert res.success is True
    assert res.output == "96"

    # Division (integer-like float normalized to int)
    res = calc.execute("100 / 4")
    assert res.success is True
    assert res.output == "25"
    assert res.metadata["result"] == 25

    # True float division
    res = calc.execute("7 / 2")
    assert res.success is True
    assert res.output == "3.5"
    assert res.metadata["result"] == 3.5

    # Exponentiation and modulo
    res = calc.execute("2 ** 8 + (10 % 3)")
    assert res.success is True
    assert res.output == "257"
    assert res.metadata["result"] == 257

    # Floor division
    res = calc.execute("17 // 5")
    assert res.success is True
    assert res.output == "3"


def test_calculator_math_functions():
    calc = CalculatorTool()

    # Square root
    res = calc.execute("sqrt(144) + 12")
    assert res.success is True
    assert res.output == "24"

    # math module prefix
    res = calc.execute("math.sqrt(64)")
    assert res.success is True
    assert res.output == "8"

    # Factorial
    res = calc.execute("factorial(5)")
    assert res.success is True
    assert res.output == "120"

    # Power
    res = calc.execute("pow(3, 4)")
    assert res.success is True
    assert res.output == "81"

    # Trigonometry & Constants
    res = calc.execute("sin(pi / 2)")
    assert res.success is True
    assert res.output == "1"

    res = calc.execute("cos(0)")
    assert res.success is True
    assert res.output == "1"

    res = calc.execute("tan(0)")
    assert res.success is True
    assert res.output == "0"

    # Logarithms and exp
    res = calc.execute("log(e)")
    assert res.success is True
    assert res.output == "1"

    res = calc.execute("log10(1000)")
    assert res.success is True
    assert res.output == "3"

    res = calc.execute("exp(0)")
    assert res.success is True
    assert res.output == "1"

    # Rounding, floor, ceil, abs
    res = calc.execute("ceil(4.1) + floor(4.9) + abs(-10)")
    assert res.success is True
    assert res.output == "19"

    res = calc.execute("round(3.14159265, 4)")
    assert res.success is True
    assert res.output == "3.1416"


def test_calculator_percentage_notation():
    calc = CalculatorTool()

    res = calc.execute("15% * 200")
    assert res.success is True
    assert res.output == "30"

    res = calc.execute("500 + 20% * 500")
    assert res.success is True
    assert res.output == "600"


def test_calculator_constants():
    calc = CalculatorTool()

    res = calc.execute("pi > 3.14 and pi < 3.15")
    assert res.success is True
    assert res.output == "True"

    res = calc.execute("e > 2.71 and e < 2.72")
    assert res.success is True
    assert res.output == "True"


def test_calculator_security_sandbox():
    calc = CalculatorTool()

    # Block imports
    res = calc.execute("import os")
    assert res.success is False
    assert "Security restriction" in res.error

    # Block dunder methods
    res = calc.execute("().__class__.__bases__[0]")
    assert res.success is False
    assert "Security restriction" in res.error

    # Block builtins/open/exec
    res = calc.execute("open('/etc/passwd')")
    assert res.success is False
    assert "Security restriction" in res.error

    res = calc.execute("exec('x = 1')")
    assert res.success is False
    assert "Security restriction" in res.error

    res = calc.execute("eval('1 + 1')")
    assert res.success is False
    assert "Security restriction" in res.error

    res = calc.execute("subprocess.call(['ls'])")
    assert res.success is False
    assert "Security restriction" in res.error


def test_calculator_error_handling():
    calc = CalculatorTool()

    # Division by zero
    res = calc.execute("100 / 0")
    assert res.success is False
    assert "ZeroDivisionError" in res.error

    # Math domain error
    res = calc.execute("sqrt(-1)")
    assert res.success is False
    assert "ValueError" in res.error

    # Empty expression
    res = calc.execute("")
    assert res.success is False
    assert "Empty mathematical expression" in res.error

    res = calc.execute("   ")
    assert res.success is False
    assert "Empty mathematical expression" in res.error

    # Syntax error
    res = calc.execute("5 * + / 2")
    assert res.success is False
    assert "SyntaxError" in res.error


# ==============================================================================
# WeatherTool Tests
# ==============================================================================

def test_weather_tier_and_schema():
    weather = WeatherTool()
    assert weather.tier == PermissionTier.SAFE
    assert weather.name == "weather"
    schema = weather.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "weather"
    assert "city" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["required"] == ["city"]


def test_weather_empty_city():
    weather = WeatherTool()
    res = weather.execute(city="")
    assert res.success is False
    assert "city name must be specified" in res.error.lower()


def test_weather_mocked_success():
    mock_client = MagicMock(spec=httpx.Client)

    # 1. Geocoding mock response
    geo_response = MagicMock(spec=httpx.Response)
    geo_response.status_code = 200
    geo_response.json.return_value = {
        "results": [
            {
                "name": "Tokyo",
                "latitude": 35.6895,
                "longitude": 139.69171,
                "country": "Japan",
                "admin1": "Tokyo",
                "timezone": "Asia/Tokyo",
            }
        ]
    }

    # 2. Forecast mock response
    forecast_response = MagicMock(spec=httpx.Response)
    forecast_response.status_code = 200
    forecast_response.json.return_value = {
        "timezone": "Asia/Tokyo",
        "current": {
            "temperature_2m": 16.5,
            "relative_humidity_2m": 55,
            "weather_code": 0,
            "wind_speed_10m": 4.2,
        },
    }

    def mock_get(url: str, params: Dict[str, Any] = None, **kwargs: Any):
        if url == GEOCODING_URL:
            return geo_response
        elif url == FORECAST_URL:
            return forecast_response
        raise ValueError(f"Unexpected URL: {url}")

    mock_client.get.side_effect = mock_get

    tool = WeatherTool(client=mock_client)
    res = tool.execute(city="Tokyo")

    assert res.success is True
    assert "Tokyo, Japan" in res.output
    assert "Clear sky" in res.output
    assert "16.5°C" in res.output
    assert "55%" in res.output
    assert "4.2 km/h" in res.output

    # Check metadata
    assert res.metadata["city"] == "Tokyo"
    assert res.metadata["country"] == "Japan"
    assert res.metadata["temperature_c"] == 16.5
    assert res.metadata["condition"] == "Clear sky"
    assert res.metadata["humidity_percent"] == 55
    assert res.metadata["wind_speed_kmh"] == 4.2
    assert res.metadata["latitude"] == 35.6895
    assert res.metadata["longitude"] == 139.69171


def test_weather_mocked_city_not_found():
    mock_client = MagicMock(spec=httpx.Client)

    geo_response = MagicMock(spec=httpx.Response)
    geo_response.status_code = 200
    geo_response.json.return_value = {"results": []}

    mock_client.get.return_value = geo_response

    tool = WeatherTool(client=mock_client)
    res = tool.execute(city="Atlantis_Nonexistent_City")

    assert res.success is False
    assert "Could not find coordinates" in res.error
    assert res.metadata["city"] == "Atlantis_Nonexistent_City"


def test_weather_mocked_timeout():
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.side_effect = httpx.TimeoutException("Connection timed out")

    tool = WeatherTool(client=mock_client)
    res = tool.execute(city="London")

    assert res.success is False
    assert "timed out after 10.0s" in res.error


def test_weather_mocked_http_error():
    mock_client = MagicMock(spec=httpx.Client)

    geo_response = MagicMock(spec=httpx.Response)
    geo_response.status_code = 503
    geo_response.text = "Service Unavailable"
    mock_client.get.return_value = geo_response

    tool = WeatherTool(client=mock_client)
    res = tool.execute(city="Paris")

    assert res.success is False
    assert "status code 503" in res.error


def test_weather_live_query():
    """Live integration test against Open-Meteo APIs (guarded for network availability)."""
    tool = WeatherTool()
    try:
        res = tool.execute(city="London")
        if res.success:
            assert "London" in res.output
            assert "°C" in res.output
            assert res.metadata["temperature_c"] is not None
            assert res.metadata["condition"] is not None
    except Exception:
        pytest.skip("Network unavailable for live Open-Meteo integration test.")


# ==============================================================================
# ListDirectoryTool Tests
# ==============================================================================

def test_list_directory_tier_and_schema():
    tool = ListDirectoryTool()
    assert tool.tier == PermissionTier.SAFE
    assert tool.name == "list_directory"
    schema = tool.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "list_directory"
    props = schema["function"]["parameters"]["properties"]
    assert "path" in props
    assert "include_hidden" in props
    assert "max_items" in props


def test_list_directory_contents(tmp_path: Path):
    # Setup test directory tree
    sub_dir = tmp_path / "subfolder"
    sub_dir.mkdir()

    file_a = tmp_path / "alpha.txt"
    file_a.write_text("Hello from alpha file", encoding="utf-8")

    file_b = tmp_path / "beta.md"
    file_b.write_text("# Markdown Heading\nSome content.", encoding="utf-8")

    tool = ListDirectoryTool()
    res = tool.execute(path=str(tmp_path), include_hidden=False)

    assert res.success is True
    assert "[DIR]  subfolder/" in res.output
    assert "[FILE] alpha.txt" in res.output
    assert "[FILE] beta.md" in res.output
    assert f"{len('Hello from alpha file')} bytes" in res.output

    # Check metadata
    assert res.metadata["total_items"] == 3
    assert res.metadata["returned_items"] == 3
    assert res.metadata["truncated"] is False

    entry_types = {e["name"]: e["type"] for e in res.metadata["entries"]}
    assert entry_types["subfolder"] == "DIR"
    assert entry_types["alpha.txt"] == "FILE"
    assert entry_types["beta.md"] == "FILE"


def test_list_directory_hidden_files_filtering(tmp_path: Path):
    # Create visible and hidden items
    visible_file = tmp_path / "visible.txt"
    visible_file.write_text("Visible", encoding="utf-8")

    hidden_file = tmp_path / ".hidden_file"
    hidden_file.write_text("Secret", encoding="utf-8")

    hidden_dir = tmp_path / ".hidden_folder"
    hidden_dir.mkdir()

    tool = ListDirectoryTool()

    # Default include_hidden=False
    res_no_hidden = tool.execute(path=str(tmp_path), include_hidden=False)
    assert res_no_hidden.success is True
    assert "visible.txt" in res_no_hidden.output
    assert ".hidden_file" not in res_no_hidden.output
    assert ".hidden_folder" not in res_no_hidden.output
    assert res_no_hidden.metadata["total_items"] == 1

    # include_hidden=True
    res_hidden = tool.execute(path=str(tmp_path), include_hidden=True)
    assert res_hidden.success is True
    assert "visible.txt" in res_hidden.output
    assert ".hidden_file" in res_hidden.output
    assert ".hidden_folder" in res_hidden.output
    assert res_hidden.metadata["total_items"] == 3


def test_list_directory_pagination(tmp_path: Path):
    for i in range(10):
        (tmp_path / f"item_{i:02d}.txt").write_text(f"content {i}", encoding="utf-8")

    tool = ListDirectoryTool()
    res = tool.execute(path=str(tmp_path), max_items=4)

    assert res.success is True
    assert res.metadata["total_items"] == 10
    assert res.metadata["returned_items"] == 4
    assert res.metadata["truncated"] is True
    assert "and 6 more items (truncated at 4)" in res.output


def test_list_directory_empty_directory(tmp_path: Path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    tool = ListDirectoryTool()
    res = tool.execute(path=str(empty_dir))

    assert res.success is True
    assert "empty" in res.output.lower()
    assert res.metadata["total_items"] == 0
    assert res.metadata["returned_items"] == 0


def test_list_directory_invalid_paths(tmp_path: Path):
    tool = ListDirectoryTool()

    # Non-existent path
    non_existent = tmp_path / "ghost_dir"
    res = tool.execute(path=str(non_existent))
    assert res.success is False
    assert "Directory does not exist" in res.error

    # Regular file path instead of directory
    a_file = tmp_path / "file.txt"
    a_file.write_text("Sample", encoding="utf-8")
    res_file = tool.execute(path=str(a_file))
    assert res_file.success is False
    assert "Target path is not a directory" in res_file.error


# ==============================================================================
# ToolRegistry Integration Tests (Phase 2 + Phase 3 All Initial Tools)
# ==============================================================================

def test_tool_registry_with_phase3_tools():
    registry = ToolRegistry()

    # Register all 7 initial tools
    registry.register(SystemInfoTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ExecuteCommandTool())
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    registry.register(ListDirectoryTool())

    tools = registry.get_all_tools()
    tool_names = {t.name for t in tools}
    expected_names = {
        "system_info",
        "read_file",
        "write_file",
        "execute_command",
        "calculator",
        "weather",
        "list_directory",
    }
    assert tool_names == expected_names

    # Verify execution via registry
    calc_res = registry.execute_tool("calculator", {"expression": "factorial(6)"})
    assert calc_res.success is True
    assert calc_res.output == "720"

    list_res = registry.execute_tool("list_directory", {"path": ".", "max_items": 5})
    assert list_res.success is True

    # Verify OpenAI schemas exported for all tools
    schemas = registry.get_openai_tool_schemas()
    assert len(schemas) == 7
    schema_names = {s["function"]["name"] for s in schemas}
    assert schema_names == expected_names

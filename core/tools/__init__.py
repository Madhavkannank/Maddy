"""M.A.D.D.Y. Tools Package."""

from core.tools.base import BaseTool, ToolRegistry, ToolResult
from core.tools.calculator import CalculatorTool
from core.tools.file_ops import ListDirectoryTool, ReadFileTool, WriteFileTool
from core.tools.memory_tools import ForgetTool, RecallTool, RememberTool, UpdatePreferenceTool
from core.tools.system_info import SystemInfoTool
from core.tools.terminal import ExecuteCommandTool
from core.tools.weather import WeatherTool
from core.tools.system_control import SystemControlTool
from core.tools.app_control import AppControlTool
from core.tools.media_control import MediaControlTool
from core.tools.web_browser import WebBrowserTool, WebSearchTool
try:
    from core.tools.telegram_tool import TelegramTool
except ImportError:
    import sys
    import types
    from core.tools.base import BaseTool, ToolResult
    from typing import Any, Dict
    class TelegramTool(BaseTool):  # type: ignore
        """Null fallback when telegram tool is excluded."""
        name = "telegram"
        description = "Telegram messaging tool (not installed or configured)."
        def get_parameters_schema(self) -> Dict[str, Any]:
            return {"type": "object", "properties": {"message": {"type": "string"}}}
        def execute(self, **kwargs: Any) -> ToolResult:
            if kwargs.get("action") == "status":
                return ToolResult(success=True, output="Telegram is not configured.", metadata={"configured": False})
            return ToolResult(success=False, output="", error="Telegram tool is not installed.")

    mock_mod = types.ModuleType("core.tools.telegram_tool")
    mock_mod.TelegramTool = TelegramTool
    sys.modules["core.tools.telegram_tool"] = mock_mod

from core.tools.camera_tool import CameraTool
from core.tools.arxiv_tool import ArxivTool
from core.tools.timer_tool import TimerTool
from core.tools.computer_screen import ComputerScreenTool
from core.tools.computer_interaction import ComputerInteractionTool
from core.tools.browser_automation import BrowserAutomationTool
from core.tools.skill_tools import ListSkillsTool, LoadSkillTool, SearchSkillsTool, ValidateSkillTool

__all__ = [
    "BaseTool",
    "ToolRegistry",
    "ToolResult",
    "SystemInfoTool",
    "ReadFileTool",
    "WriteFileTool",
    "ExecuteCommandTool",
    "CalculatorTool",
    "WeatherTool",
    "ListDirectoryTool",
    "RememberTool",
    "RecallTool",
    "ForgetTool",
    "UpdatePreferenceTool",
    "SystemControlTool",
    "AppControlTool",
    "MediaControlTool",
    "WebBrowserTool",
    "WebSearchTool",
    "TelegramTool",
    "CameraTool",
    "ArxivTool",
    "TimerTool",
    "ComputerScreenTool",
    "ComputerInteractionTool",
    "BrowserAutomationTool",
    "ListSkillsTool",
    "LoadSkillTool",
    "SearchSkillsTool",
    "ValidateSkillTool",
]

"""
Application control tool definitions for M.A.D.D.Y. ACI.
Provides application.open and application.close cross-platform handlers.
"""
import importlib.util
import pathlib
import sys
from typing import Any, Dict
from tools.base import RiskLevel, Tool

from platform_layer import get_platform

def _handle_application_open(args: Dict[str, Any]) -> Dict[str, Any]:
    app_name = args.get("app_name", "").strip()
    if not app_name:
        return {
            "success": False,
            "error": {"type": "INVALID_ARGUMENT", "message": "app_name must be non-empty"},
        }

    plat = get_platform()
    try:
        plat.open_application(app_name)
        return {
            "app_name": app_name,
            "os": plat.get_os_name(),
            "status": "launched",
        }
    except Exception as e:
        return {
            "success": False,
            "error": {
                "type": "APPLICATION_LAUNCH_ERROR",
                "message": str(e),
            },
        }

application_open_tool = Tool(
    name="application.open",
    description="Open an application by name cross-platform.",
    parameter_schema={
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "Name of application to open (e.g., 'Spotify', 'Chrome', 'calculator').",
            },
        },
        "required": ["app_name"],
    },
    handler=_handle_application_open,
    risk_level=RiskLevel.HIGH,
    timeout=10.0,
)

def _handle_application_close(args: Dict[str, Any]) -> Dict[str, Any]:
    app_name = args.get("app_name", "").strip()
    if not app_name:
        return {
            "success": False,
            "error": {"type": "INVALID_ARGUMENT", "message": "app_name must be non-empty"},
        }

    plat = get_platform()
    try:
        plat.close_application(app_name)
        return {
            "app_name": app_name,
            "os": plat.get_os_name(),
            "status": "terminated",
        }
    except Exception as e:
        return {
            "success": False,
            "error": {
                "type": "APPLICATION_CLOSE_ERROR",
                "message": str(e),
            },
        }

application_close_tool = Tool(
    name="application.close",
    description="Close a running application by name cross-platform.",
    parameter_schema={
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "Name of application to close (e.g., 'Spotify', 'Chrome', 'calculator').",
            },
        },
        "required": ["app_name"],
    },
    handler=_handle_application_close,
    risk_level=RiskLevel.HIGH,
    timeout=10.0,
)

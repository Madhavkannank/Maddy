"""
Session Capture Tools for M.A.D.D.Y. ACI.
Provides validated, permission-gated tools for starting, stopping, and inspecting
temporary capture sessions.
"""
from typing import Any, Dict

from session_capture.config import SessionCaptureConfig
from session_capture.manager import get_session_capture_manager
from tools.base import RiskLevel, Tool


def handle_session_capture_start(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for starting an owner-authorized session capture."""
    manager = get_session_capture_manager()

    import os
    req_telegram = params.get("enable_telegram")
    if req_telegram is not None:
        enable_tg = bool(req_telegram)
    else:
        try:
            from integrations.messaging.telegram_client import TelegramClient
        except ImportError:
            from integrations.messaging import TelegramClient
        enable_tg = TelegramClient().is_configured

    # Enforce operator's authorized TELEGRAM_CHAT_ID from .env to prevent injection redirection
    authorized_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip() or None
    chat_id = authorized_chat_id or params.get("telegram_chat_id") or None

    try:
        config = SessionCaptureConfig(
            display_interval=float(params.get("display_interval", 30.0)),
            camera_interval=float(params.get("camera_interval", 30.0)),
            enable_display=bool(params.get("enable_display", True)),
            enable_camera=bool(params.get("enable_camera", True)),
            enable_telegram=enable_tg,
            telegram_chat_id=chat_id,
            max_storage_bytes=int(params.get("max_storage_bytes", 200 * 1024 * 1024)),
        )
    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "message": f"Invalid configuration parameters: {str(e)}",
            "error": str(e),
            "spoken_response": f"I couldn't start session capture because the configuration was invalid: {str(e)}",
        }

    task_id = params.get("task_id")
    caller_id = params.get("caller_id") or "owner"

    req_detached = params.get("detached")
    if req_detached is not None:
        detached = bool(req_detached)
    else:
        detached = not bool(os.getenv("PYTEST_CURRENT_TEST"))

    success, msg, data = manager.start_session(
        config=config, task_id=task_id, caller_id=caller_id, detached=detached
    )

    if success:
        sess_id = data.get("session_id", "active")
        tg_info = " Telegram live sync is active." if config.enable_telegram else ""
        spoken = f"Session capture mode has been activated, boss. Session ID is {sess_id}.{tg_info}"
        return {
            "success": True,
            "status": "success",
            "message": msg,
            "session": data,
            "spoken_summary": spoken,
            "spoken_response": spoken,
        }
    else:
        spoken = f"Could not start session capture: {msg}"
        return {
            "success": False,
            "status": "error",
            "message": msg,
            "error": msg,
            "spoken_summary": spoken,
            "spoken_response": spoken,
        }


def handle_session_capture_stop(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for stopping the active session capture."""
    manager = get_session_capture_manager()
    task_id = params.get("task_id")
    caller_id = params.get("caller_id") or "owner"

    success, msg, data = manager.stop_session(task_id=task_id, caller_id=caller_id)

    if success:
        count = data.get("artifact_count", 0)
        spoken = f"Session capture has been stopped, boss. A total of {count} artifacts were recorded."
        return {
            "success": True,
            "status": "success",
            "message": msg,
            "session": data,
            "spoken_summary": spoken,
            "spoken_response": spoken,
        }
    else:
        spoken = f"Session capture stop status: {msg}"
        return {
            "success": False,
            "status": "error",
            "message": msg,
            "error": msg,
            "spoken_summary": spoken,
            "spoken_response": spoken,
        }


def handle_session_capture_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for querying real-time session capture status."""
    manager = get_session_capture_manager()
    status = manager.get_status()

    if status.get("active"):
        sess_id = status.get("session_id")
        created = status.get("artifacts_created", 0)
        spoken = f"Session capture is currently active with ID {sess_id}. {created} artifacts recorded so far."
    else:
        spoken = "Session capture is currently inactive."

    return {
        "success": True,
        "status": "success",
        "data": status,
        "spoken_summary": spoken,
        "spoken_response": spoken,
    }


session_capture_start_tool = Tool(
    name="session.capture.start",
    description="Start an owner-authorized temporary session capture for auditing display and camera activity.",
    parameter_schema={
        "type": "object",
        "properties": {
            "display_interval": {
                "type": "number",
                "description": "Interval in seconds between display screenshots (minimum 1.0, default 30.0).",
                "default": 30.0,
            },
            "camera_interval": {
                "type": "number",
                "description": "Interval in seconds between camera snapshots (minimum 1.0, default 30.0).",
                "default": 30.0,
            },
            "enable_display": {
                "type": "boolean",
                "description": "Whether to capture display screenshots.",
                "default": True,
            },
            "enable_camera": {
                "type": "boolean",
                "description": "Whether to capture camera snapshots.",
                "default": True,
            },
            "enable_telegram": {
                "type": "boolean",
                "description": "Whether to deliver artifacts to Telegram asynchronously.",
                "default": False,
            },
            "telegram_chat_id": {
                "type": "string",
                "description": "Target Telegram chat ID.",
            },
            "max_storage_bytes": {
                "type": "integer",
                "description": "Maximum disk quota for this session in bytes.",
                "default": 209715200,
            },
            "task_id": {
                "type": "string",
                "description": "Authorizing task identifier.",
            },
            "caller_id": {
                "type": "string",
                "description": "Caller identity requesting activation.",
            },
        },
        "additionalProperties": False,
    },
    handler=handle_session_capture_start,
    risk_level=RiskLevel.MEDIUM,
    timeout=15.0,
)

session_capture_stop_tool = Tool(
    name="session.capture.stop",
    description="Stop the active temporary session capture and finalize all manifests.",
    parameter_schema={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Task identifier of the caller requesting termination.",
            },
            "caller_id": {
                "type": "string",
                "description": "Caller identity requesting termination.",
            },
        },
        "additionalProperties": False,
    },
    handler=handle_session_capture_stop,
    risk_level=RiskLevel.LOW,
    timeout=10.0,
)

session_capture_status_tool = Tool(
    name="session.capture.status",
    description="Retrieve real-time status and artifact counts for session capture.",
    parameter_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    handler=handle_session_capture_status,
    risk_level=RiskLevel.LOW,
    timeout=5.0,
)

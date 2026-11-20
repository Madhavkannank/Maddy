"""Countdown Timer and Alarm Management Tool for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Provides thread-safe timer creation, monitoring, listing, and cancellation.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from core.permissions import PermissionTier
from core.tools.base import BaseTool, ToolResult


@dataclass
class ActiveTimer:
    id: str
    label: str
    duration_seconds: float
    started_at: float
    expires_at: float
    timer_handle: Optional[threading.Timer] = None
    is_cancelled: bool = False
    is_completed: bool = False

    @property
    def remaining_seconds(self) -> float:
        if self.is_cancelled or self.is_completed:
            return 0.0
        return max(0.0, self.expires_at - time.time())


class TimerTool(BaseTool):
    """Creates and manages countdown timers and reminders."""

    name = "timer_tool"
    description = (
        "Create, monitor, list, or cancel countdown timers. "
        "Actions: 'set_timer' (creates a countdown timer for N seconds), "
        "'list_timers' (lists all active countdowns with remaining time), "
        "'cancel_timer' (cancels a running timer by timer_id)."
    )
    tier = PermissionTier.SAFE

    _timers: Dict[str, ActiveTimer] = {}
    _lock = threading.Lock()

    def __init__(self, on_timer_expired: Optional[Callable[[ActiveTimer], None]] = None):
        self.on_timer_expired = on_timer_expired

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["set_timer", "list_timers", "cancel_timer"],
                    "description": "Timer action to perform.",
                },
                "duration_seconds": {
                    "type": "number",
                    "minimum": 1,
                    "description": "Duration in seconds for 'set_timer'.",
                },
                "label": {
                    "type": "string",
                    "default": "Timer",
                    "description": "Label or purpose for the timer (e.g. 'Pasta', 'Meeting break').",
                },
                "timer_id": {
                    "type": "string",
                    "description": "Timer ID to cancel for 'cancel_timer'.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    def execute(
        self,
        action: str,
        duration_seconds: Optional[float] = None,
        label: str = "Timer",
        timer_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        act = action.strip().lower()

        try:
            if act == "set_timer":
                if duration_seconds is None or duration_seconds <= 0:
                    return ToolResult(
                        success=False,
                        output="",
                        error="Parameter 'duration_seconds' must be a positive number.",
                    )

                tid = f"timer_{uuid.uuid4().hex[:6]}"
                now = time.time()
                expires_at = now + float(duration_seconds)

                timer_entry = ActiveTimer(
                    id=tid,
                    label=label or "Timer",
                    duration_seconds=float(duration_seconds),
                    started_at=now,
                    expires_at=expires_at,
                )

                def _on_expire():
                    with self._lock:
                        if not timer_entry.is_cancelled:
                            timer_entry.is_completed = True
                    if self.on_timer_expired:
                        try:
                            self.on_timer_expired(timer_entry)
                        except Exception:
                            pass

                th = threading.Timer(float(duration_seconds), _on_expire)
                th.daemon = True
                timer_entry.timer_handle = th

                with self._lock:
                    self._timers[tid] = timer_entry

                th.start()

                mins = int(duration_seconds // 60)
                secs = int(duration_seconds % 60)
                dur_str = f"{mins}m {secs}s" if mins > 0 else f"{secs} seconds"

                return ToolResult(
                    success=True,
                    output=f"Timer '{label}' set for {dur_str} (ID: {tid}).",
                    metadata={
                        "timer_id": tid,
                        "label": label,
                        "duration_seconds": duration_seconds,
                        "expires_at": expires_at,
                    },
                )

            elif act == "list_timers":
                with self._lock:
                    active = [
                        t for t in self._timers.values()
                        if not t.is_cancelled and not t.is_completed and t.remaining_seconds > 0
                    ]

                if not active:
                    return ToolResult(
                        success=True,
                        output="No active countdown timers running.",
                        metadata={"timers": []},
                    )

                lines = [f"{len(active)} active timer(s):"]
                timer_data = []
                for t in active:
                    rem = int(round(t.remaining_seconds))
                    lines.append(f"• ID: {t.id} | '{t.label}' — {rem}s remaining")
                    timer_data.append({
                        "id": t.id,
                        "label": t.label,
                        "remaining_seconds": rem,
                    })

                return ToolResult(
                    success=True,
                    output="\n".join(lines),
                    metadata={"timers": timer_data, "count": len(active)},
                )

            elif act == "cancel_timer":
                if not timer_id:
                    return ToolResult(success=False, output="", error="Parameter 'timer_id' is required for 'cancel_timer'.")

                with self._lock:
                    timer_entry = self._timers.get(timer_id.strip())
                    if not timer_entry or timer_entry.is_cancelled:
                        return ToolResult(
                            success=False,
                            output="",
                            error=f"No active timer found with ID '{timer_id}'.",
                        )

                    timer_entry.is_cancelled = True
                    if timer_entry.timer_handle:
                        timer_entry.timer_handle.cancel()

                return ToolResult(
                    success=True,
                    output=f"Timer '{timer_entry.label}' (ID: {timer_id}) was cancelled.",
                    metadata={"timer_id": timer_id, "cancelled": True},
                )

            return ToolResult(success=False, output="", error=f"Unknown timer_tool action: '{action}'.")

        except Exception as exc:
            return ToolResult(success=False, output="", error=f"Timer action '{action}' failed: {str(exc)}")

"""Common application runtime for non-Telegram M.A.D.D.Y. interfaces.
(Modular Autonomous Digital Daemon for You)

This module is the only application-facing execution boundary for the rebuilt
runtime. CLI and voice adapters submit natural-language turns to
:class:`MaddyAgentCore`; legacy ``app.maddy_assistant`` is intentionally
not imported or modified here.
"""
from __future__ import annotations

import datetime
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.agent_loop import MaddyAgentCore
from core.llm_client import CloudLLMClient
from core.memory import MemoryStore
from core.memory_extractor import ConversationMemoryExtractor
from core.errors import PermissionDeniedError
from core.permissions import PermissionGate, PermissionTier
from personality.engine import PersonalityEngine
from core.tools import (
    AppControlTool,
    ArxivTool,
    BrowserAutomationTool,
    CameraTool,
    ComputerInteractionTool,
    ComputerScreenTool,
    ForgetTool,
    MediaControlTool,
    RecallTool,
    RememberTool,
    SystemControlTool,
    TimerTool,
    ToolRegistry,
    UpdatePreferenceTool,
    WebBrowserTool,
    WebSearchTool,
    TelegramTool,
)
from core.tools.file_ops import ListDirectoryTool, ReadFileTool, WriteFileTool
from core.tools.calculator import CalculatorTool
from core.tools.system_info import SystemInfoTool
from core.tools.terminal import ExecuteCommandTool
from core.tools.weather import WeatherTool
from core.trace_store import TraceStore
from platform_layer import PlatformInterface, get_platform


class MaddyRuntime:
    """Wires the common execution authority, personality, and its default typed toolset."""

    def __init__(
        self,
        core: Optional[MaddyAgentCore] = None,
        personality: Optional[PersonalityEngine] = None,
        llm_client: Optional[CloudLLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        trace_store: Optional[TraceStore] = None,
        permission_gate: Optional[PermissionGate] = None,
        memory_store: Optional[MemoryStore] = None,
        memory_extractor: Optional[ConversationMemoryExtractor] = None,
        platform: Optional[PlatformInterface] = None,
        max_iterations_per_turn: int = 4,
        tool_timeout_seconds: float = 10.0,
    ) -> None:
        self.platform = platform or get_platform()
        self.permission_gate = permission_gate or PermissionGate()
        self.trace_store = trace_store or TraceStore()
        self.memory_store = memory_store or MemoryStore()
        self.memory_extractor = memory_extractor
        self.tool_registry = tool_registry or ToolRegistry()
        self.personality = personality or PersonalityEngine(memory_manager=self.memory_store)
        self.session_histories: Dict[str, List[Dict[str, Any]]] = {}
        self.last_task_id: Optional[str] = None
        self._active_stop_event: Optional[threading.Event] = None

        self._register_default_tools()

        self.core = core or MaddyAgentCore(
            llm_client=llm_client,
            tool_registry=self.tool_registry,
            trace_store=self.trace_store,
            permission_gate=self.permission_gate,
            memory_store=self.memory_store,
            memory_extractor=self.memory_extractor,
            personality=self.personality,
            max_iterations_per_turn=max_iterations_per_turn,
            tool_timeout_seconds=tool_timeout_seconds,
        )

    def get_address_title(self) -> str:
        """Dynamically retrieve active operator address form from PersonalityEngine."""
        return self.personality.get_address_title()

    def get_greeting_candidates(self) -> List[str]:
        """Dynamically retrieve charismatic wake greeting candidates from PersonalityEngine."""
        return self.personality.get_greeting_candidates()

    def get_callout_acknowledgment(self, moniker: str = "buddy") -> str:
        """Dynamically retrieve moniker acknowledgment from PersonalityEngine."""
        return self.personality.get_callout_acknowledgment(moniker=moniker)

    def _register_default_tools(self) -> None:
        """Register each typed capability once, preserving injected test tools."""
        default_tools = [
            SystemInfoTool(),
            CalculatorTool(),
            WeatherTool(),
            ReadFileTool(),
            ListDirectoryTool(),
            WriteFileTool(permission_gate=self.permission_gate),
            ExecuteCommandTool(),
            SystemControlTool(platform=self.platform),
            AppControlTool(platform=self.platform),
            MediaControlTool(platform=self.platform),
            WebBrowserTool(platform=self.platform),
            WebSearchTool(platform=self.platform),
            TelegramTool(),
            CameraTool(permission_gate=self.permission_gate),
            ArxivTool(),
            TimerTool(),
            ComputerScreenTool(platform=self.platform, permission_gate=self.permission_gate),
            ComputerInteractionTool(platform=self.platform),
            BrowserAutomationTool(),
            RememberTool(memory_store=self.memory_store),
            RecallTool(memory_store=self.memory_store),
            ForgetTool(memory_store=self.memory_store),
            UpdatePreferenceTool(memory_store=self.memory_store),
        ]
        for tool in default_tools:
            if self.tool_registry.get_tool(tool.name) is None:
                self.tool_registry.register(tool)
        # Register standard friendly aliases
        cam_tool = self.tool_registry.get_tool("camera_tool")
        if cam_tool and self.tool_registry.get_tool("camera") is None:
            self.tool_registry.register_alias("camera", cam_tool)
        screen_tool = self.tool_registry.get_tool("computer_screen")
        if screen_tool and self.tool_registry.get_tool("screenshot") is None:
            self.tool_registry.register_alias("screenshot", screen_tool)

    def handle_slash_command(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Executes fast slash commands extracted from Telegram and CLI:
          /start, /help - Quick reference help card
          /status - Comprehensive PC and agent telemetry (CPU, RAM, GPU/VRAM, battery, tools)
          /screenshot - Instant desktop screenshot capture
          /photo, /camera - Instant webcam photo capture
          /lock - Instant workstation lock
          /capture_start - Start session capture / guest audit
          /capture_stop, stop stop - Stop active session capture
        """
        lower = (prompt or "").strip().lower()
        if not lower:
            return None

        addr = self.get_address_title() or "boss"

        # Strip optional vocatives (e.g. "Channel, open camera." -> "open camera")
        clean_cmd = re.sub(
            r"^(?:hey\s+|ok\s+|okay\s+)?(?:maddy|maddie|madi|maddi|channel|assistant|system|daemon)\b[,\s:]*",
            "",
            lower,
        ).strip().strip(" .?!,;:")
        if not clean_cmd:
            clean_cmd = lower

        # 1. /start, /help
        if lower in ("/start", "/help", "help") or clean_cmd in ("/start", "/help", "help"):
            msg = (
                f"🤖 M.A.D.D.Y. REMOTE CONTROLS & COMMANDS\n\n"
                f"📊 System & Diagnostics:\n"
                f"  • /status - Check CPU, RAM, GPU/VRAM, and battery telemetry\n"
                f"  • /screenshot - Capture instant full desktop screenshot\n"
                f"  • /photo or /camera - Capture instant webcam snapshot\n"
                f"  • /lock - Lock workstation immediately\n\n"
                f"🛡️ Guest Audit & Session Capture:\n"
                f"  • /capture_start - Start background session capture\n"
                f"  • /capture_stop (or 'stop stop') - Stop session capture\n\n"
                f"⚡ Agent & Capability Tools:\n"
                f"  • /tools - List registered capability tools\n"
                f"  • Or speak/type any natural language command, {addr}!"
            )
            return {"success": True, "response": msg, "command": "/help"}

        # 2. /status
        is_status = (
            lower in ("/status", "status", "system status", "pc status", "health")
            or clean_cmd in ("/status", "status", "system status", "pc status", "health")
            or any(p in clean_cmd for p in ["system status", "pc status", "check status", "system health", "pc health", "battery status"])
        )
        if is_status:
            status_lines = ["🤖 M.A.D.D.Y. PC SYSTEM STATUS", ""]
            try:
                import psutil
                cpu_pct = psutil.cpu_percent(interval=0.2)
                ram = psutil.virtual_memory()
                status_lines.append(f"💻 CPU Usage: {cpu_pct}%")
                status_lines.append(f"🧠 RAM Usage: {ram.percent}% ({round(ram.used / (1024**3), 1)} GB / {round(ram.total / (1024**3), 1)} GB)")
                battery = psutil.sensors_battery()
                if battery:
                    plugged = "🔌 Plugged In" if battery.power_plugged else "🔋 On Battery"
                    status_lines.append(f"⚡ Battery: {battery.percent}% ({plugged})")
            except Exception:
                status_lines.append(f"💻 System: {self.platform.get_os_name().capitalize()}")

            try:
                from hardware.gpu_monitor import GPUMonitor
                gpu_info = GPUMonitor().get_status()
                if gpu_info.get("available"):
                    status_lines.append(f"🎮 GPU: {gpu_info.get('name', 'NVIDIA GPU')}")
                    status_lines.append(f"📊 VRAM: {gpu_info.get('allocated_mb', 0)} MB / {gpu_info.get('total_mb', 0)} MB")
                    if gpu_info.get("temperature"):
                        status_lines.append(f"🌡️ Temperature: {gpu_info.get('temperature')}°C")
            except Exception:
                pass

            diag = self.get_diagnostics()
            status_lines.append(f"🔧 Registered Tools: {diag['registered_tools_count']}")
            status_lines.append(f"\nStanding by for your command, {addr}.")
            return {"success": True, "response": "\n".join(status_lines), "command": "/status"}

        # 3. /screenshot
        is_screenshot = (
            lower in ("/screenshot", "screenshot", "screen")
            or clean_cmd in ("/screenshot", "screenshot", "screen")
            or any(p in clean_cmd for p in ["take a screenshot", "take screenshot", "capture screen", "send screenshot", "grab screen"])
        )
        if is_screenshot:
            try:
                self.permission_gate.check_permission("computer_screen", PermissionTier.SAFE, {"action": "screenshot"})
            except PermissionDeniedError as pde:
                return {"success": False, "response": f"Permission denied: {pde.message}", "command": "/screenshot"}
            tool = self.tool_registry.get_tool("computer_screen")
            if tool:
                res = tool.execute(action="screenshot")
                if res.success:
                    path = res.metadata.get("path") or res.output
                    msg = f"Desktop screenshot captured successfully, {addr}: {path}"
                    return {"success": True, "response": msg, "command": "/screenshot", "path": path, "metadata": res.metadata}
                else:
                    return {"success": False, "response": f"Failed to capture desktop screenshot, {addr}: {res.error}", "command": "/screenshot"}
            return {"success": False, "response": f"Screenshot tool is unavailable, {addr}.", "command": "/screenshot"}

        # 4. Camera (Open Native App or Capture Snapshot)
        is_open_camera = (
            clean_cmd in ("open camera", "open the camera", "launch camera", "start camera", "show camera", "view camera")
            or bool(re.match(r"^(?:open|launch|start|show)\s+(?:the\s+)?camera$", clean_cmd))
        )
        if is_open_camera:
            try:
                self.permission_gate.check_permission("app_control", PermissionTier.SYSTEM_IMPACTING, {"app_name": "camera", "action": "open"})
            except PermissionDeniedError as pde:
                return {"success": False, "response": f"Permission denied: {pde.message}", "command": "/open_camera"}
            try:
                self.platform.open_application("camera")
                return {
                    "success": True,
                    "response": f"Opening camera, {addr}.",
                    "command": "/open_camera",
                }
            except Exception as e:
                return {
                    "success": False,
                    "response": f"Failed to open camera, {addr}: {str(e)}",
                    "command": "/open_camera",
                }

        is_photo = (
            lower in ("/photo", "/camera", "photo", "camera", "webcam", "pic")
            or clean_cmd in ("/photo", "/camera", "photo", "camera", "webcam", "pic")
            or any(p in clean_cmd for p in ["take a photo", "take photo", "click photo", "capture photo", "webcam snapshot", "take picture", "take a picture"])
        )
        if is_photo:
            try:
                self.permission_gate.check_permission("camera_tool", PermissionTier.SYSTEM_IMPACTING, {"action": "capture_frame"})
            except PermissionDeniedError as pde:
                return {"success": False, "response": f"Permission denied: {pde.message}", "command": "/photo"}
            tool = self.tool_registry.get_tool("camera") or self.tool_registry.get_tool("camera_tool")
            if tool:
                res = tool.execute(action="capture_frame")
                if res.success:
                    path = res.metadata.get("path") or res.output
                    msg = f"Webcam photo captured successfully, {addr}: {path}"
                    return {"success": True, "response": msg, "command": "/photo", "path": path, "metadata": res.metadata}
                else:
                    return {"success": False, "response": f"Failed to capture webcam photo, {addr}: {res.error}", "command": "/photo"}
            return {"success": False, "response": f"Camera tool is unavailable, {addr}.", "command": "/photo"}

        # 5. /lock
        is_lock = (
            lower in ("/lock", "lock", "lock pc", "lock the pc", "lock screen", "lock workstation", "lock computer")
            or clean_cmd in ("/lock", "lock", "lock pc", "lock the pc", "lock screen", "lock workstation", "lock computer")
        )
        if is_lock:
            try:
                self.permission_gate.check_permission("system_control", PermissionTier.SAFE, {"action": "lock_workstation"})
            except PermissionDeniedError as pde:
                return {"success": False, "response": f"Permission denied: {pde.message}", "command": "/lock"}
            if hasattr(self.platform, "lock_workstation"):
                try:
                    if self.platform.lock_workstation():
                        return {"success": True, "response": f"The PC has been locked, {addr}.", "command": "/lock"}
                except Exception:
                    pass
            tool = self.tool_registry.get_tool("system_control")
            if tool:
                res = tool.execute(action="lock_workstation")
                if res.success:
                    return {"success": True, "response": f"The PC has been locked, {addr}.", "command": "/lock"}
                else:
                    return {"success": False, "response": f"Failed to lock PC: {res.error}", "command": "/lock"}
            return {"success": False, "response": f"Could not lock the workstation, {addr}.", "command": "/lock"}

        # 6. /capture_start
        if lower in ("/capture_start", "start session capture", "start guest audit", "code red") or clean_cmd in ("/capture_start", "start session capture", "start guest audit", "code red"):
            try:
                self.permission_gate.check_permission("session_capture", PermissionTier.SYSTEM_IMPACTING, {"action": "start"})
            except PermissionDeniedError as pde:
                return {"success": False, "response": f"Permission denied: {pde.message}", "command": "/capture_start"}
            try:
                from tools.session_capture_tool import handle_session_capture_start
                res = handle_session_capture_start({"task_id": self.last_task_id or "cli", "caller_id": "owner"})
                reply = res.get("spoken_response") or res.get("message", "Session capture activated.")
                return {"success": res.get("success", True), "response": f"🛡️ {reply}", "command": "/capture_start", "data": res}
            except Exception as e:
                return {"success": False, "response": f"🛡️ Session capture start failed: {str(e)}", "command": "/capture_start"}

        # 7. /capture_stop or stop stop
        if lower in ("stop stop", "/capture_stop", "stop capture", "stop session") or clean_cmd in ("stop stop", "/capture_stop", "stop capture", "stop session"):
            try:
                from tools.session_capture_tool import handle_session_capture_stop
                res = handle_session_capture_stop({"task_id": self.last_task_id or "cli", "caller_id": "owner"})
                reply = res.get("spoken_response") or res.get("message", "Session capture stopped.")
                return {"success": res.get("success", True), "response": f"🛡️ {reply}", "command": "/capture_stop", "data": res}
            except Exception as e:
                return {"success": False, "response": f"🛡️ Session capture stop failed: {str(e)}", "command": "/capture_stop"}

        # 8. Fast-Path: Application Launch (calculator, terminal, spotify, browser, chrome, firefox, vscode, code)
        m_app = re.match(r"^(?:open|launch|start)\s+(?:the\s+)?(calculator|terminal|spotify|browser|chrome|firefox|vscode|code)$", clean_cmd)
        if m_app:
            app_target = m_app.group(1)
            try:
                self.permission_gate.check_permission("app_control", PermissionTier.SYSTEM_IMPACTING, {"app_name": app_target, "action": "open"})
            except PermissionDeniedError as pde:
                return {"success": False, "response": f"Permission denied: {pde.message}", "command": f"/open_{app_target}"}
            try:
                self.platform.open_application(app_target)
                return {
                    "success": True,
                    "response": f"Opening {app_target.capitalize()}, {addr}.",
                    "command": f"/open_{app_target}",
                }
            except Exception as e:
                return {
                    "success": False,
                    "response": f"Failed to open {app_target}, {addr}: {str(e)}",
                    "command": f"/open_{app_target}",
                }

        # 9. Fast-Path: Live Local Time Query
        if clean_cmd in ("what time is it", "what's the time", "current time", "time now", "tell me the time", "what is the time"):
            now = datetime.datetime.now()
            time_str = now.strftime("%I:%M %p").lstrip("0")
            return {
                "success": True,
                "response": f"It is {time_str}, {addr}.",
                "command": "/time",
            }

        return None

    def execute(
        self,
        user_prompt: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        speak_response: bool = False,
        voice: Optional[str] = None,
        stop_event: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute a natural-language request exclusively through MaddyAgentCore."""
        prompt = (user_prompt or "").strip()
        if prompt:
            from core.math_phonetics import MathPhoneticizer
            prompt = MathPhoneticizer.normalize_spelled_letters(prompt)

        start_time = time.time()
        self.last_task_id = task_id or f"task_{uuid.uuid4().hex[:12]}"

        if not prompt:
            return {
                "status": "error",
                "response": "I need a request to execute.",
                "task_id": self.last_task_id,
                "elapsed_seconds": 0.0,
                "completed": False,
            }

        eff_stop_event = stop_event or self._active_stop_event

        # 0. Check Telegram-compatible fast slash commands
        slash_res = self.handle_slash_command(prompt)
        if slash_res is not None:
            elapsed = round(time.time() - start_time, 3)
            resp_content = slash_res.get("response", "")
            result = {
                "status": "completed" if slash_res.get("success", True) else "failed",
                "response": resp_content,
                "task_id": self.last_task_id,
                "elapsed_seconds": elapsed,
                "completed": slash_res.get("success", True),
                "data": slash_res,
            }
            if session_id:
                if session_id not in self.session_histories:
                    self.session_histories[session_id] = []
                self.session_histories[session_id].append({"role": "user", "content": prompt})
                self.session_histories[session_id].append({"role": "assistant", "content": resp_content})

            if speak_response and resp_content:
                try:
                    from voice.engine import VoiceEngine
                    from core.math_phonetics import MathPhoneticizer
                    ve = VoiceEngine(platform_driver=self.platform)
                    speakable = MathPhoneticizer.extract_speakable_text(resp_content)
                    result["speakable_response"] = speakable
                    voice_res = ve.speak(speakable, voice=voice, stop_event=eff_stop_event)
                    result["voice"] = voice_res
                except Exception as v_err:
                    result["voice"] = {"success": False, "error": f"Voice synthesis failed: {str(v_err)}"}
            return result

        # Conversational intercept: conversational requests, identity queries, and preference
        # updates flow directly into MaddyAgentCore.run_turn() without procedural regex interception.
        handled, intercept_resp = self.personality.handle_conversational_intercept(prompt)
        if handled and intercept_resp:
            elapsed = round(time.time() - start_time, 3)
            result = {
                "status": "completed",
                "response": intercept_resp,
                "task_id": self.last_task_id,
                "elapsed_seconds": elapsed,
                "completed": True,
            }
            if session_id:
                if session_id not in self.session_histories:
                    self.session_histories[session_id] = []
                self.session_histories[session_id].append({"role": "user", "content": prompt})
                self.session_histories[session_id].append({"role": "assistant", "content": intercept_resp})

            if speak_response and result.get("response"):
                try:
                    from voice.engine import VoiceEngine
                    from core.math_phonetics import MathPhoneticizer
                    ve = VoiceEngine(platform_driver=self.platform)
                    speakable = MathPhoneticizer.extract_speakable_text(result["response"])
                    result["speakable_response"] = speakable
                    voice_res = ve.speak(speakable, voice=voice, stop_event=eff_stop_event)
                    result["voice"] = voice_res
                except Exception as v_err:
                    result["voice"] = {"success": False, "error": f"Voice synthesis failed: {str(v_err)}"}
            return result

        # Resolve session conversation history if caller uses session continuity
        history = conversation_history
        if history is None and session_id:
            history = self.session_histories.get(session_id, [])

        try:
            result = self.core.run_turn(
                prompt,
                task_id=self.last_task_id,
                conversation_history=history,
                stop_event=eff_stop_event,
            )
            result.setdefault("status", "completed" if result.get("completed") else "failed")
        except Exception as exc:
            result = {
                "status": "failed",
                "response": f"Runtime execution failed: {str(exc)}",
                "task_id": self.last_task_id,
                "completed": False,
                "error": str(exc),
            }

        elapsed = round(time.time() - start_time, 3)
        result["elapsed_seconds"] = elapsed

        # Update session history if session_id provided
        if session_id:
            if session_id not in self.session_histories:
                self.session_histories[session_id] = []
            self.session_histories[session_id].append({"role": "user", "content": prompt})
            resp_content = result.get("response", "")
            if resp_content:
                self.session_histories[session_id].append({"role": "assistant", "content": resp_content})

        # Always derive dual-channel speakable response and links display banner
        resp_text = result.get("response", "")
        if resp_text:
            try:
                from core.math_phonetics import MathPhoneticizer
                result["speakable_response"] = MathPhoneticizer.extract_speakable_text(resp_text)
                banner = MathPhoneticizer.format_links_for_display(resp_text)
                if banner:
                    result["links_banner"] = banner
            except Exception:
                result["speakable_response"] = resp_text

        # Handle optional voice output synthesis
        if speak_response and result.get("response"):
            try:
                from voice.engine import VoiceEngine
                ve = VoiceEngine(platform_driver=self.platform)
                text_to_speak = result.get("speakable_response") or result["response"]
                voice_res = ve.speak(text_to_speak, voice=voice, stop_event=eff_stop_event)
                result["voice"] = voice_res
            except Exception as v_err:
                result["voice"] = {"success": False, "error": f"Voice synthesis failed: {str(v_err)}"}

        return result

    def resume(
        self,
        task_id: str,
        checkpoint_id: Optional[str] = None,
        stop_event: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Resume a checkpointed task through the same core execution authority."""
        eff_stop_event = stop_event or self._active_stop_event
        kwargs: Dict[str, Any] = {}
        if checkpoint_id is not None:
            kwargs["checkpoint_id"] = checkpoint_id
        if eff_stop_event is not None:
            kwargs["stop_event"] = eff_stop_event

        result = self.core.resume_turn(task_id, **kwargs)
        result.setdefault("status", "completed" if result.get("completed") else "failed")
        return result

    def cancel_task(self, task_id: str) -> bool:
        """Cancel an in-progress or pending task by updating trace store status and recording cancellation event."""
        try:
            self.trace_store.update_task_status(task_id, "failed")
            self.trace_store.record_event(task_id, "task_cancelled", {"cancelled_at": time.time()})
            return True
        except Exception:
            return False

    def interrupt(self) -> None:
        """Trigger immediate cancellation of active execution or voice synthesis."""
        if self._active_stop_event is not None:
            self._active_stop_event.set()

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return runtime diagnostics for system monitoring, health checks, and CLI status."""
        tools = self.tool_registry.get_all_tools()
        return {
            "platform": self.platform.get_os_name(),
            "execution_core": "MaddyAgentCore",
            "registered_tools_count": len(tools),
            "registered_tools": [t.name for t in tools],
            "max_iterations_per_turn": self.core.max_iterations_per_turn,
            "tool_timeout_seconds": self.core.tool_timeout_seconds,
            "trace_db_path": str(self.trace_store.db_path),
            "trace_db_exists": self.trace_store.db_path.exists(),
            "memory_db_path": str(self.memory_store.db_path),
            "memory_db_exists": self.memory_store.db_path.exists(),
            "last_task_id": self.last_task_id,
            "active_sessions_count": len(self.session_histories),
        }

    def list_tools(self) -> List[str]:
        """Return registered tool names for diagnostics and status output."""
        return [tool.name for tool in self.tool_registry.get_all_tools()]

    def close(self) -> None:
        """Close runtime-owned stores when they expose explicit lifecycle hooks."""
        for store in (self.memory_store, self.trace_store):
            close_fn = getattr(store, "close", None)
            if callable(close_fn):
                close_fn()


__all__ = ["MaddyRuntime"]

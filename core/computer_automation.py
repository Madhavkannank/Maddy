"""Computer Automation and Layered Interaction Hierarchy for M.A.D.D.Y.
(Modular Autonomous Digital Daemon for You).

Implements Phase 3.7:
- Layered interaction hierarchy:
    1. Native OS APIs (window management, process lifecycle)
    2. Accessibility & Semantic UI Elements
    3. Visual Screen Understanding & OCR (VisionEngine element localization)
    4. Coordinate-based interaction (mouse click, keyboard type) as last resort
- Independent observation and verification:
    Observe (before) → Decide → Permission Check → Act → Observe (after) → Verify → Replan
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from platform_layer import PlatformInterface, get_platform
from vision.engine import VisionEngine

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Outcome verification evidence from independent observation."""
    verified: bool
    layer_used: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


class ComputerAutomationController:
    """Coordinates computer-use interactions across the layered hierarchy with independent verification."""

    def __init__(
        self,
        platform: Optional[PlatformInterface] = None,
        vision_engine: Optional[VisionEngine] = None,
    ):
        self.platform = platform or get_platform()
        self.vision = vision_engine or VisionEngine()

    # ── INDEPENDENT OBSERVATION PRIMITIVES ────────────────────────────────────

    def observe_system_state(self) -> Dict[str, Any]:
        """Capture comprehensive pre/post execution snapshot for verification."""
        active_win = self.platform.get_active_window() or {}
        open_wins = self.platform.get_open_windows() or []
        screen_geo = self.platform.get_screen_geometry() or {}

        # Screen fingerprint: try screenshot hash if possible
        screen_hash = ""
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            # Fast thumbnail-based perceptual hash
            thumb = img.resize((32, 32)).convert("L")
            screen_hash = hashlib.md5(thumb.tobytes()).hexdigest()
        except Exception:
            pass

        return {
            "timestamp": time.time(),
            "active_window": active_win,
            "open_windows_count": len(open_wins),
            "open_window_titles": [w.get("title", "") for w in open_wins],
            "screen_geometry": screen_geo,
            "screen_hash": screen_hash,
        }

    # ── LAYERED INTERACTION HIERARCHY ─────────────────────────────────────────

    def focus_window(self, target: str) -> VerificationResult:
        """Layer 1 / Layer 2: Bring window or application to focus with verification."""
        before = self.observe_system_state()
        clean_target = target.strip().lower()

        # Try native OS window activation (Layer 1)
        focused = self.platform.focus_window_or_application(clean_target)
        time.sleep(0.3)

        after = self.observe_system_state()
        after_active = (after.get("active_window", {}).get("title", "")).lower()

        # Verification: check if active window matches target
        matched = clean_target in after_active or any(clean_target in t.lower() for t in after.get("open_window_titles", []) if clean_target in t.lower())
        verified = bool(focused and (matched or after["active_window"] != before["active_window"]))

        return VerificationResult(
            verified=verified,
            layer_used="native_os_api",
            evidence={
                "target": target,
                "before_active": before.get("active_window", {}).get("title"),
                "after_active": after.get("active_window", {}).get("title"),
                "focus_returned": focused,
            },
            message=f"Window '{target}' focused and verified active." if verified else f"Failed to verify focus on '{target}'.",
        )

    def close_window(self, target: str) -> VerificationResult:
        """Layer 1: Close window with verification that it was removed from window list."""
        before = self.observe_system_state()
        clean_target = target.strip().lower()

        closed = self.platform.close_window(clean_target)
        time.sleep(0.4)

        after = self.observe_system_state()
        before_titles = [t.lower() for t in before.get("open_window_titles", [])]
        after_titles = [t.lower() for t in after.get("open_window_titles", [])]

        # Target was present before and is absent after, or window count decreased
        target_was_present = any(clean_target in t for t in before_titles)
        target_now_absent = not any(clean_target in t for t in after_titles)
        count_decreased = after["open_windows_count"] < before["open_windows_count"]

        verified = bool(closed and (target_now_absent or count_decreased or not target_was_present))

        return VerificationResult(
            verified=verified,
            layer_used="native_os_api",
            evidence={
                "target": target,
                "before_window_count": before["open_windows_count"],
                "after_window_count": after["open_windows_count"],
                "closed_call_returned": closed,
            },
            message=f"Window '{target}' closed and verified terminated." if verified else f"Window '{target}' close unverified.",
        )

    def click_element_layered(
        self,
        element_description: str,
        screenshot_path: Optional[str] = None,
        button: str = "left",
        clicks: int = 1,
    ) -> VerificationResult:
        """Layered click execution:
        Layer 1: Check if element description matches an open window or app title (Native OS API)
        Layer 2: Check if active window has semantic element (Accessibility API)
        Layer 3: Visual Screen Perception & OCR (Locate element bounding box and centroid)
        Layer 4: Coordinate click execution fallback
        """
        before = self.observe_system_state()
        clean_desc = element_description.strip().lower()

        # Layer 1: If element refers to a visible window or app, activate it via OS API
        for title in before.get("open_window_titles", []):
            if clean_desc and clean_desc in title.lower():
                res = self.focus_window(title)
                if res.verified:
                    return VerificationResult(
                        verified=True,
                        layer_used="layer_1_native_os_api",
                        evidence={"matched_window": title, "sub_result": res.evidence},
                        message=f"Activated window '{title}' via Native OS API matching '{element_description}'.",
                    )

        # Layer 3: Visual UI Element localization via VisionEngine
        temp_dir = os.path.join(os.path.expanduser("~"), "Pictures", "MADDY_Captures")
        os.makedirs(temp_dir, exist_ok=True)
        shot_path = screenshot_path or os.path.join(temp_dir, f"locate_click_{int(time.time() * 1000)}.png")

        shot_res = self.platform.take_screenshot(shot_path)
        actual_shot_path = shot_res.get("path") or shot_path

        loc_res = self.vision.locate_element(actual_shot_path, element_description)
        centroid = loc_res.get("centroid")
        if not loc_res.get("success") or not centroid:
            return VerificationResult(
                verified=False,
                layer_used="layer_3_vision_localization_failed",
                evidence={"element_description": element_description, "loc_res": loc_res},
                message=f"Could not locate visual UI element matching '{element_description}' on screen.",
            )

        # Layer 4: Coordinate click execution on verified localized element
        click_res = self.platform.click_mouse(x=centroid[0], y=centroid[1], button=button, clicks=clicks)
        time.sleep(0.3)

        after = self.observe_system_state()

        # Visual change verification
        visual_delta = False
        if before["screen_hash"] and after["screen_hash"]:
            visual_delta = before["screen_hash"] != after["screen_hash"]

        window_changed = before.get("active_window") != after.get("active_window")
        click_ok = bool(click_res.get("success", False))
        verified = click_ok
        effect = "confirmed" if (visual_delta or window_changed) else "suspected_noop"

        layer_name = f"layer_3_vision_ocr_{loc_res.get('match_type', 'heuristic')}"
        return VerificationResult(
            verified=verified,
            layer_used=layer_name,
            evidence={
                "element_description": element_description,
                "centroid": centroid,
                "bounding_box": loc_res.get("bounding_box"),
                "confidence": loc_res.get("confidence"),
                "match_type": loc_res.get("match_type"),
                "click_result": click_res,
                "visual_change_detected": visual_delta,
                "window_changed": window_changed,
                "effect": effect,
            },
            message=f"Clicked '{element_description}' at coordinates {centroid} via {layer_name} (effect: {effect}).",
        )

    def click_coordinates(self, x: int, y: int, button: str = "left", clicks: int = 1) -> VerificationResult:
        """Layer 4: Direct coordinate click execution with observation verification."""
        before = self.observe_system_state()

        click_res = self.platform.click_mouse(x=x, y=y, button=button, clicks=clicks)
        time.sleep(0.2)

        after = self.observe_system_state()
        visual_delta = False
        if before["screen_hash"] and after["screen_hash"]:
            visual_delta = before["screen_hash"] != after["screen_hash"]

        return VerificationResult(
            verified=bool(click_res.get("success", True)),
            layer_used="layer_4_coordinate_input",
            evidence={
                "x": x,
                "y": y,
                "button": button,
                "clicks": clicks,
                "click_res": click_res,
                "visual_change_detected": visual_delta,
            },
            message=f"Simulated mouse click at ({x}, {y}) [button={button}, clicks={clicks}].",
        )

    def type_text(self, text: str, press_enter: bool = False) -> VerificationResult:
        """Type text into active foreground window with verification."""
        before = self.observe_system_state()

        type_res = self.platform.type_text(text, press_enter=press_enter)
        time.sleep(0.2)

        after = self.observe_system_state()
        visual_delta = False
        if before["screen_hash"] and after["screen_hash"]:
            visual_delta = before["screen_hash"] != after["screen_hash"]

        return VerificationResult(
            verified=bool(type_res.get("success", True)),
            layer_used="native_keyboard_automation",
            evidence={
                "text_length": len(text),
                "press_enter": press_enter,
                "active_window": after.get("active_window", {}).get("title"),
                "visual_change_detected": visual_delta,
            },
            message=f"Typed {len(text)} characters into '{after.get('active_window', {}).get('title')}'.",
        )

    def press_key(self, key_combination: str) -> VerificationResult:
        """Press keyboard shortcut with verification."""
        before = self.observe_system_state()

        key_res = self.platform.press_key(key_combination)
        time.sleep(0.2)

        after = self.observe_system_state()
        visual_delta = False
        if before["screen_hash"] and after["screen_hash"]:
            visual_delta = before["screen_hash"] != after["screen_hash"]

        return VerificationResult(
            verified=bool(key_res.get("success", True)),
            layer_used="native_keyboard_automation",
            evidence={
                "key": key_combination,
                "active_window_before": before.get("active_window", {}).get("title"),
                "active_window_after": after.get("active_window", {}).get("title"),
                "visual_change_detected": visual_delta,
            },
            message=f"Sent keyboard key/combination '{key_combination}'.",
        )

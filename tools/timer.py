"""
Timer and Countdown Engine for M.A.D.D.Y.
Handles natural language timer creation, active timer tracking, cancellation,
and proactive voice alert notifications upon expiration.
"""
import re
import sys
import time
import uuid
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def parse_duration_to_seconds(text: str) -> Optional[int]:
    """
    Extract total duration in seconds from natural speech.
    Supports hours, minutes, and seconds combinations (e.g., '1 hour 20 minutes', '5 mins', '45 seconds').
    """
    lower = text.lower().strip()
    total_seconds = 0
    found_any = False

    # 1. Hours pattern
    h_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h\b)", lower)
    if h_match:
        total_seconds += int(float(h_match.group(1)) * 3600)
        found_any = True

    # 2. Minutes pattern
    m_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m\b)", lower)
    if m_match:
        total_seconds += int(float(m_match.group(1)) * 60)
        found_any = True

    # 3. Seconds pattern
    s_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s\b)", lower)
    if s_match:
        total_seconds += int(float(s_match.group(1)))
        found_any = True

    # 4. Fallback: "set timer for 5" (assume minutes if >= 1, or default minutes)
    if not found_any:
        num_match = re.search(r"\b(?:for|timer\s+of|timer)\s+(\d+)\b", lower)
        if num_match:
            total_seconds = int(num_match.group(1)) * 60
            found_any = True

    return total_seconds if found_any and total_seconds > 0 else None


def format_duration(seconds: int) -> str:
    """Format seconds into readable speech string (e.g. '5 minutes', '1 hour and 30 minutes')."""
    if seconds <= 0:
        return "0 seconds"

    hours = seconds // 3600
    remainder = seconds % 3600
    minutes = remainder // 60
    secs = remainder % 60

    parts = []
    if hours == 1:
        parts.append("1 hour")
    elif hours > 1:
        parts.append(f"{hours} hours")

    if minutes == 1:
        parts.append("1 minute")
    elif minutes > 1:
        parts.append(f"{minutes} minutes")

    if secs == 1:
        parts.append("1 second")
    elif secs > 1:
        parts.append(f"{secs} seconds")

    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    else:
        return f"{', '.join(parts[:-1])}, and {parts[-1]}"


class TimerEntry:
    """Individual timer track."""

    def __init__(self, duration_seconds: int, label: str = "", on_expire: Optional[Callable[["TimerEntry"], None]] = None):
        self.id = str(uuid.uuid4())[:8]
        self.duration_seconds = duration_seconds
        self.label = label or format_duration(duration_seconds)
        self.start_time = time.time()
        self.end_time = self.start_time + duration_seconds
        self.on_expire = on_expire
        self.cancelled = False
        self.completed = False
        self._thread: Optional[threading.Thread] = None

    @property
    def remaining_seconds(self) -> int:
        if self.cancelled or self.completed:
            return 0
        import math
        rem = math.ceil(self.end_time - time.time())
        return max(0, rem)

    def start(self):
        def _run():
            sleep_time = max(0.1, self.duration_seconds)
            # Sleep in small slices to support instant cancellation
            step = 0.5
            elapsed = 0.0
            while elapsed < sleep_time:
                if self.cancelled:
                    return
                time.sleep(min(step, sleep_time - elapsed))
                elapsed += step

            if not self.cancelled:
                self.completed = True
                if self.on_expire:
                    try:
                        self.on_expire(self)
                    except Exception as err:
                        logger.error(f"Error in timer expiration callback: {err}")

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def cancel(self):
        self.cancelled = True


class TimerManager:
    """Manages active timers and coordinates voice alerts upon completion."""

    def __init__(self, voice_callback: Optional[Callable[[str], None]] = None):
        self.voice_callback = voice_callback
        self.timers: Dict[str, TimerEntry] = {}
        self._lock = threading.Lock()

    def _play_alarm_tone(self):
        """Play high-tech alert tone across Windows and Linux."""
        try:
            if sys.platform.startswith("win"):
                import winsound
                for freq in [1200, 1600, 2000]:
                    winsound.Beep(freq, 120)
                    time.sleep(0.04)
        except Exception:
            pass

    def _handle_timer_expired(self, timer: TimerEntry):
        """Action when timer reaches zero."""
        logger.info(f"Timer {timer.id} ({timer.label}) expired!")
        self._play_alarm_tone()

        alert_message = f"Boss, your timer for {timer.label} is up!"
        if self.voice_callback:
            try:
                self.voice_callback(alert_message)
            except Exception as e:
                logger.error(f"Failed to speak timer alert: {e}")

        with self._lock:
            if timer.id in self.timers:
                del self.timers[timer.id]

    def set_timer(self, duration_seconds: int, label: str = "") -> TimerEntry:
        """Create and start a new countdown timer."""
        entry = TimerEntry(
            duration_seconds=duration_seconds,
            label=label,
            on_expire=self._handle_timer_expired,
        )
        with self._lock:
            self.timers[entry.id] = entry
        entry.start()
        return entry

    def schedule_task(self, delay_seconds: int, action: Callable[[], None], label: str = "") -> TimerEntry:
        """Schedule an arbitrary callable action to execute after delay_seconds in the background."""
        def _callback(timer_entry):
            try:
                action()
            except Exception as e:
                logger.error(f"Error executing scheduled task '{label}': {e}")
            finally:
                with self._lock:
                    if timer_entry.id in self.timers:
                        del self.timers[timer_entry.id]

        entry = TimerEntry(
            duration_seconds=delay_seconds,
            label=label,
            on_expire=_callback,
        )
        with self._lock:
            self.timers[entry.id] = entry
        entry.start()
        return entry

    def get_active_timers(self) -> List[TimerEntry]:
        """Return list of uncompleted active timers."""
        with self._lock:
            active = [t for t in self.timers.values() if not t.cancelled and not t.completed and t.remaining_seconds > 0]
        return active

    def cancel_all_timers(self) -> int:
        """Cancel all active timers."""
        count = 0
        with self._lock:
            for t in list(self.timers.values()):
                if not t.cancelled and not t.completed:
                    t.cancel()
                    count += 1
            self.timers.clear()
        return count

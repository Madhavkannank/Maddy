"""
Laptop Surface Knock & Double-Clap Detection Service for M.A.D.D.Y.
(Modular Autonomous Digital Daemon for You).
Monitors the laptop's internal microphone for sudden acoustic/mechanical transients
caused by knocking the laptop chassis or clapping twice in rapid succession.
When a valid double-knock pattern is recognized, it automatically launches maddy.cmd / maddy.
"""
import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path=None, override=False):
        p = os.path.join(PROJECT_ROOT, ".env") if not dotenv_path else dotenv_path
        if not os.path.isfile(p):
            return False
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and (override or k not in os.environ):
                    os.environ[k] = v
        return True

load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"), override=True)

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, "knock_detector.log")
PID_FILE_PATH = os.path.join(LOGS_DIR, "knock_detector.pid")

# Dual logging: to file and to stdout if interactive
logger = logging.getLogger("knock_detector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _formatter = logging.Formatter("%(asctime)s [%(levelname)s] [KnockDetector] %(message)s")
    _fh = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    _fh.setFormatter(_formatter)
    logger.addHandler(_fh)
    if sys.stdout and hasattr(sys.stdout, "write"):
        _sh = logging.StreamHandler(sys.stdout)
        _sh.setFormatter(_formatter)
        logger.addHandler(_sh)


def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Uncaught fatal exception in KnockDetector:", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = _log_uncaught_exception


# ==============================================================================
# Helper functions for checking and launching M.A.D.D.Y.
# ==============================================================================

def is_maddy_running() -> Optional[int]:
    """Check if voice_listener.py is currently running on the system."""
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.cmdline() or []
                cmdline_str = " ".join(cmdline).lower()
                if "python" in proc.name().lower() and "voice_listener.py" in cmdline_str:
                    return proc.pid
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        logger.warning(f"Error checking M.A.D.D.Y. running state: {e}")
    return None


def launch_maddy_session() -> bool:
    """
    Launch M.A.D.D.Y. session with cross-platform compatibility.
    On Windows, executes maddy.cmd via cmd.exe.
    On Linux, launches voice_listener.py directly using sys.executable.
    """
    existing_pid = is_maddy_running()
    if existing_pid:
        logger.info(f"Double-knock detected, but M.A.D.D.Y. voice listener is already running (PID: {existing_pid}). Skipping duplicate launch.")
        return False

    try:
        if sys.platform.startswith("win"):
            cmd_target = os.path.join(PROJECT_ROOT, "maddy.cmd")
            if not os.path.exists(cmd_target):
                logger.error(f"Execution script not found at {cmd_target}")
                return False
            cmd = f'start "M.A.D.D.Y." cmd.exe /c "{cmd_target}"'
            subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                shell=True,
            )
        else:
            subprocess.Popen(
                [sys.executable, "voice_listener.py"],
                cwd=PROJECT_ROOT,
            )
        logger.info("M.A.D.D.Y. session successfully launched.")
        return True
    except Exception as e:
        logger.error(f"Failed to launch M.A.D.D.Y. session: {e}", exc_info=True)
        return False


def default_trigger_action() -> bool:
    """
    Default action executed upon a valid double-knock:
    Checks if M.A.D.D.Y. is already active; if not, launches maddy.cmd in a visible console.
    """
    existing_pid = is_maddy_running()
    if existing_pid:
        logger.info(f"Double-knock detected, but M.A.D.D.Y. voice listener is already running (PID: {existing_pid}). Skipping duplicate launch.")
        return False

    cmd_target = os.path.join(PROJECT_ROOT, "maddy.cmd")
    if not os.path.exists(cmd_target):
        logger.error(f"Launch script not found at {cmd_target}")
        return False

    logger.info("💥 DOUBLE KNOCK / CLAP DETECTED! Launching M.A.D.D.Y....")

    try:
        # Launch in a new, independent interactive Windows console window
        cmd = f'start "M.A.D.D.Y." cmd.exe /c "{cmd_target}"'
        subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            shell=True,
        )
        logger.info("maddy.cmd successfully launched.")
        return True
    except Exception as e:
        logger.error(f"Failed to launch maddy.cmd: {e}", exc_info=True)
        return False


# ==============================================================================
# Knock & Clap Pattern Detector Engine
# ==============================================================================

class KnockDetector:
    """
    High-efficiency real-time acoustic impulse detector.
    Analyzes audio frames from the default input device for the sharp mechanical
    characteristics of a laptop surface knock or clap, requiring two distinct impulses
    within a calibrated time window to trigger.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 512,
        peak_threshold: float = 0.20,
        energy_jump_threshold: float = 0.06,
        min_interval_sec: float = 0.12,
        max_interval_sec: float = 0.85,
        cooldown_sec: float = 3.5,
        on_double_knock: Optional[Callable[[], Any]] = None,
    ):
        """
        Parameters:
            sample_rate: Audio sampling frequency (Hz).
            block_size: Number of samples per frame (~32ms at 16kHz).
            peak_threshold: Minimum absolute peak amplitude (0.0 to 1.0) to qualify as a knock.
            energy_jump_threshold: Minimum sudden RMS jump between consecutive frames.
            min_interval_sec: Minimum spacing between 2 knocks (rejects mechanical rebound / echo).
            max_interval_sec: Maximum spacing allowed between 1st and 2nd knock.
            cooldown_sec: Period after a successful trigger where input is ignored.
            on_double_knock: Callback function executed when double knock is confirmed.
        """
        self.sample_rate = int(os.getenv("KNOCK_SAMPLE_RATE", sample_rate))
        self.block_size = int(os.getenv("KNOCK_BLOCK_SIZE", block_size))
        self.peak_threshold = float(os.getenv("KNOCK_PEAK_THRESHOLD", peak_threshold))
        self.energy_jump_threshold = float(os.getenv("KNOCK_JUMP_THRESHOLD", energy_jump_threshold))
        self.min_interval_sec = float(os.getenv("KNOCK_MIN_INTERVAL", min_interval_sec))
        self.max_interval_sec = float(os.getenv("KNOCK_MAX_INTERVAL", max_interval_sec))
        self.cooldown_sec = float(os.getenv("KNOCK_COOLDOWN", cooldown_sec))
        self.on_double_knock = on_double_knock or default_trigger_action

        # State tracking
        self.prev_rms: float = 0.0
        self.ambient_rms: float = 0.005
        self.first_knock_time: float = 0.0
        self.last_trigger_time: float = 0.0
        self.is_running: bool = False
        self.decay_counter: int = 0

    def process_frame(self, frame: np.ndarray, current_time: Optional[float] = None) -> Tuple[bool, bool, Dict[str, float]]:
        """
        Process a single audio frame and evaluate knock status.
        Returns:
            (is_single_knock, is_double_knock_triggered, metrics)
        """
        now = current_time if current_time is not None else time.time()
        metrics: Dict[str, float] = {}

        # 1. Convert to mono 1D float array
        if frame.ndim > 1:
            frame = frame.mean(axis=1)

        peak = float(np.max(np.abs(frame))) if frame.size > 0 else 0.0
        rms = float(np.sqrt(np.mean(frame ** 2))) if frame.size > 0 else 0.0
        jump = rms - self.prev_rms
        crest_factor = (peak / (rms + 1e-6))

        # Slowly update ambient noise floor when relatively quiet
        if peak < self.peak_threshold * 0.5:
            self.ambient_rms = self.ambient_rms * 0.95 + rms * 0.05

        metrics["peak"] = peak
        metrics["rms"] = rms
        metrics["jump"] = jump
        metrics["crest"] = crest_factor
        metrics["ambient"] = self.ambient_rms

        # Cooldown active: ignore all events
        if (now - self.last_trigger_time) < self.cooldown_sec:
            self.prev_rms = rms
            return False, False, metrics

        # 2. Check if this frame exhibits a sharp knock/clap transient:
        # - Instant peak exceeds absolute threshold
        # - Large sudden upward jump in RMS energy (fast rise time)
        # - High crest factor (impulse characteristic: sharp spike vs sustained noise)
        # - Peak stands well above ambient noise
        is_transient = (
            peak >= self.peak_threshold
            and (jump >= self.energy_jump_threshold or peak >= (self.ambient_rms * 6.0))
            and crest_factor >= 2.5
        )

        is_single_knock = False
        is_double_knock = False

        if is_transient:
            # Check if this is the first knock or the matching second knock
            if self.first_knock_time == 0.0:
                # First knock registered!
                self.first_knock_time = now
                is_single_knock = True
            else:
                elapsed = now - self.first_knock_time
                if elapsed < self.min_interval_sec:
                    # Too soon: reverberation, finger drag, or mechanical bounce
                    pass
                elif elapsed <= self.max_interval_sec:
                    # Second knock in the valid window!
                    is_double_knock = True
                    self.first_knock_time = 0.0
                    self.last_trigger_time = now
                else:
                    # Expired window: treat this current knock as the NEW first knock
                    self.first_knock_time = now
                    is_single_knock = True
        else:
            # If waiting for second knock and timeout exceeded, reset
            if self.first_knock_time > 0.0 and (now - self.first_knock_time) > self.max_interval_sec:
                self.first_knock_time = 0.0

        self.prev_rms = rms
        return is_single_knock, is_double_knock, metrics

    def run(self, interactive_test_mode: bool = False) -> None:
        """
        Start continuous microphone stream listener.
        If interactive_test_mode is True, prints a real-time terminal volume bar.
        """
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice is required for KnockDetector. Run: pip install sounddevice")
            return

        self.is_running = True
        logger.info(f"KnockDetector active. Listening on default microphone (rate: {self.sample_rate}Hz, peak_thresh: {self.peak_threshold})...")
        if interactive_test_mode:
            print("\n" + "=" * 65)
            print("  🔊 KNOCK DETECTOR TEST MODE")
            print("  Knock on your laptop surface or clap twice to test.")
            print(f"  Settings: Peak Threshold={self.peak_threshold}, Window={self.min_interval_sec}s - {self.max_interval_sec}s")
            print("  Press Ctrl+C to stop.")
            print("=" * 65 + "\n")

        def audio_callback(indata, frames, time_info, status):
            if not self.is_running:
                return
            if status:
                logger.debug(f"Audio stream status: {status}")

            single, double, metrics = self.process_frame(indata[:, 0])

            if interactive_test_mode:
                peak = metrics["peak"]
                bar_len = int(min(peak, 1.0) * 35)
                bar = "█" * bar_len + "-" * (35 - bar_len)
                status_str = "IDLE"
                if self.first_knock_time > 0:
                    status_str = "⏳ WAITING FOR 2ND KNOCK..."
                sys.stdout.write(f"\rLevel: [{bar}] Peak: {peak:.3f} | {status_str}   ")
                sys.stdout.flush()

            if single:
                logger.info(f"💥 Knock 1 detected (Peak: {metrics['peak']:.3f}). Waiting for 2nd knock...")
                if interactive_test_mode:
                    print(f"\n👉 [KNOCK 1/2] Detected at peak={metrics['peak']:.3f}! Knock again within {self.max_interval_sec}s...")

            if double:
                logger.info(f"🎉 DOUBLE KNOCK CONFIRMED! (Peak: {metrics['peak']:.3f})")
                if interactive_test_mode:
                    print(f"\n✅ [KNOCK 2/2] SUCCESS! Double knock detected! Triggering maddy.cmd...\n")
                # Trigger callback
                try:
                    self.on_double_knock()
                except Exception as e:
                    logger.error(f"Error executing on_double_knock callback: {e}", exc_info=True)

        def _signal_handler(signum, frame):
            logger.info("Termination signal received. Shutting down KnockDetector...")
            self.is_running = False

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.block_size,
                callback=audio_callback,
            ):
                while self.is_running:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Audio stream error: {e}", exc_info=True)
        finally:
            remove_pid()
            logger.info("KnockDetector terminated cleanly.")


# ==============================================================================
# Daemon Management & Windows Startup
# ==============================================================================

def is_pid_alive(pid: int) -> bool:
    """Check if process with PID is alive and is Python."""
    try:
        import psutil
        if not psutil.pid_exists(pid):
            return False
        proc = psutil.Process(pid)
        return proc.is_running() and "python" in proc.name().lower()
    except Exception:
        return False


def get_running_pid() -> Optional[int]:
    """Return active PID if knock daemon is running, else None."""
    if not os.path.exists(PID_FILE_PATH):
        return None
    try:
        with open(PID_FILE_PATH, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
        if is_pid_alive(pid):
            return pid
        remove_pid()
        return None
    except Exception:
        return None


def write_pid(pid: int) -> None:
    try:
        with open(PID_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(str(pid))
    except Exception as e:
        logger.warning(f"Failed to write PID file {PID_FILE_PATH}: {e}")


def remove_pid() -> None:
    try:
        if os.path.exists(PID_FILE_PATH):
            os.remove(PID_FILE_PATH)
    except OSError:
        pass


def start_daemon() -> int:
    """Start KnockDetector silently in background."""
    existing_pid = get_running_pid()
    if existing_pid:
        print(f"M.A.D.D.Y. KnockDetector is already running in background (PID: {existing_pid}).")
        return existing_pid

    python_exe = sys.executable
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = 0x08000000 | 0x00000008  # CREATE_NO_WINDOW | DETACHED_PROCESS

    detector_script = os.path.join(PROJECT_ROOT, "core", "knock_detector.py")
    cmd = [python_exe, detector_script, "run"]

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT

    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        creationflags=creationflags,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    write_pid(proc.pid)

    time.sleep(1.0)
    for _ in range(6):
        if not is_pid_alive(proc.pid):
            break
        time.sleep(0.5)

    if not is_pid_alive(proc.pid):
        print(f"Failed to start KnockDetector daemon. Check log: {LOG_FILE_PATH}")
        return 0

    print(f"M.A.D.D.Y. KnockDetector started in background (PID: {proc.pid}).")
    print(f"Log: {LOG_FILE_PATH}")
    return proc.pid


def stop_daemon() -> bool:
    """Stop running background KnockDetector daemon."""
    pid = get_running_pid()
    if not pid:
        print("M.A.D.D.Y. KnockDetector is not currently running.")
        return False

    try:
        import psutil
        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except psutil.TimeoutExpired:
            proc.kill()
    except Exception as e:
        logger.warning(f"Error terminating process {pid}: {e}")

    remove_pid()
    print(f"M.A.D.D.Y. KnockDetector (PID: {pid}) stopped.")
    return True


def status_daemon() -> None:
    """Display status of background KnockDetector daemon."""
    pid = get_running_pid()
    if not pid:
        print("M.A.D.D.Y. KnockDetector is STOPPED.")
        return

    try:
        import psutil
        proc = psutil.Process(pid)
        mem_mb = round(proc.memory_info().rss / (1024 * 1024), 1)
        uptime_sec = int(time.time() - proc.create_time())
        mins, secs = divmod(uptime_sec, 60)
        hours, mins = divmod(mins, 60)
        uptime_str = f"{hours}h {mins}m {secs}s" if hours else f"{mins}m {secs}s"
        print("==================================================")
        print("  M.A.D.D.Y. KNOCK & CLAP WAKEUP SERVICE")
        print("  (Modular Autonomous Digital Daemon for You)")
        print("==================================================")
        print("  Status:       RUNNING (active in background)")
        print(f"  PID:          {pid}")
        print(f"  Memory:       {mem_mb} MB")
        print(f"  Uptime:       {uptime_str}")
        print(f"  Target:       Double-knock launches maddy.cmd")
        print(f"  Log File:     {LOG_FILE_PATH}")
        print("==================================================")
    except Exception as e:
        print(f"M.A.D.D.Y. KnockDetector is running (PID: {pid}). Details unavailable: {e}")


def install_startup() -> bool:
    """Register KnockDetector in Windows Startup folder."""
    startup_dir = os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup")
    if not os.path.exists(startup_dir):
        print(f"Startup folder not found at: {startup_dir}")
        return False

    vbs_path = os.path.join(startup_dir, "MADDY_KnockDetector.vbs")
    python_exe = sys.executable

    vbs_content = (
        'Set WshShell = CreateObject("WScript.Shell")\n'
        f'WshShell.CurrentDirectory = "{PROJECT_ROOT}"\n'
        'Set oEnv = WshShell.Environment("PROCESS")\n'
        f'oEnv("PYTHONPATH") = "{PROJECT_ROOT}"\n'
        f'WshShell.Run """{python_exe}"" core\\knock_detector.py run", 0, False\n'
    )

    try:
        with open(vbs_path, "w", encoding="utf-8") as f:
            f.write(vbs_content)
        print(f"Windows Startup script installed at: {vbs_path}")
        print("M.A.D.D.Y. KnockDetector will now run automatically in the background on PC startup!")
        return True
    except Exception as e:
        print(f"Failed to install Windows Startup script: {e}")
        return False


def uninstall_startup() -> bool:
    """Remove KnockDetector from Windows Startup folder."""
    startup_dir = os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup")
    for fname in ["MADDY_KnockDetector.vbs"]:
        vbs_path = os.path.join(startup_dir, fname)
        if os.path.exists(vbs_path):
            try:
                os.remove(vbs_path)
                print(f"Windows Startup auto-start ({fname}) removed successfully.")
            except Exception as e:
                print(f"Failed to remove Windows Startup script {fname}: {e}")
    return True


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="M.A.D.D.Y. Surface Knock & Double-Clap Detector (Modular Autonomous Digital Daemon for You)")
    parser.add_argument(
        "action",
        nargs="?",
        default="start",
        choices=["start", "bg", "stop", "restart", "status", "test", "run", "fg", "install-startup", "uninstall-startup"],
        help="Action to perform (default: start in background)",
    )
    args = parser.parse_args()
    action = args.action.lower()

    if action in ("start", "bg"):
        start_daemon()
    elif action == "stop":
        stop_daemon()
    elif action == "restart":
        stop_daemon()
        time.sleep(1.0)
        start_daemon()
    elif action == "status":
        status_daemon()
    elif action == "test":
        detector = KnockDetector()
        detector.run(interactive_test_mode=True)
    elif action in ("run", "fg"):
        write_pid(os.getpid())
        detector = KnockDetector()
        detector.run(interactive_test_mode=False)
    elif action == "install-startup":
        install_startup()
    elif action == "uninstall-startup":
        uninstall_startup()


if __name__ == "__main__":
    main()

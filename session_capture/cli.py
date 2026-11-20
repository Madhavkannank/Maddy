"""
Command-line interface (CLI) for managing the background Session Capture Daemon.
Allows stopping the active capture process via direct commands:
  - 'stop stop'
  - 'stop'
  - 'stop-capture'
and inspecting daemon status from any terminal.
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PID_FILE = os.path.join(PROJECT_ROOT, "memory", "capture_sessions", "daemon.pid")
STOP_SIGNAL_FILE = os.path.join(PROJECT_ROOT, "memory", "capture_sessions", "stop.signal")
BASE_STORAGE_DIR = os.path.join(PROJECT_ROOT, "memory", "capture_sessions")

VALID_PASSKEYS = {"stop stop", "stop", "stop-capture", "dismiss", "cancel"}


def is_process_alive(pid: int) -> bool:
    """Check if a process with given PID is currently alive on Windows."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        SYNCHRONIZE = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
        if handle:
            exit_code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            ctypes.windll.kernel32.CloseHandle(handle)
            # STILL_ACTIVE is 259
            return exit_code.value == 259
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def terminate_pid(pid: int) -> None:
    """Terminate a process by PID."""
    if os.name == "nt":
        import ctypes
        PROCESS_TERMINATE = 0x0001
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
        if handle:
            ctypes.windll.kernel32.TerminateProcess(handle, 0)
            ctypes.windll.kernel32.CloseHandle(handle)
    else:
        import signal
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def get_daemon_info() -> Optional[Dict[str, Any]]:
    """Read daemon PID file if present."""
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def cmd_status() -> int:
    """Check and display running daemon status."""
    info = get_daemon_info()
    if not info:
        print("[STATUS] No background session capture daemon is currently running.")
        return 0

    pid = info.get("pid", 0)
    if not is_process_alive(pid):
        print(f"[STATUS] Stale daemon PID {pid} found (process not active). Cleaning up...")
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        return 0

    uptime = int(time.time() - info.get("started_at", time.time()))
    print("=" * 60)
    print("       M.A.D.D.Y. BACKGROUND SESSION CAPTURE DAEMON")
    print("=" * 60)
    print(f"  • State:              ACTIVE (running)")
    print(f"  • Process PID:        {pid}")
    print(f"  • Session ID:         {info.get('session_id')}")
    print(f"  • Running Duration:   {uptime} seconds")
    print(f"  • Display Interval:   {info.get('display_interval')}s")
    print(f"  • Camera Interval:    {info.get('camera_interval')}s")
    print(f"  • Telegram Sync:      {info.get('enable_telegram')}")
    print("=" * 60)
    print("To stop, type: 'stop' or 'stop stop' in terminal.\n")
    return 0


def cmd_stop(passkey: Optional[str] = None) -> int:
    """Stop the background daemon gracefully using signal, falling back to termination."""
    env_passkey = os.getenv("SESSION_CAPTURE_PASSKEY", "").strip().lower()
    allowed_keys = set(VALID_PASSKEYS)
    if env_passkey:
        allowed_keys.add(env_passkey)

    if passkey:
        clean_key = passkey.strip().lower()
        if clean_key not in allowed_keys:
            print(f"[ERROR] Invalid stop passkey: '{passkey}'")
            return 1

    info = get_daemon_info()
    if not info:
        print("[INFO] No active session capture daemon detected.")
        return 0

    pid = info.get("pid", 0)
    sess_id = info.get("session_id", "unknown")

    if not is_process_alive(pid):
        print(f"[INFO] Process {pid} is no longer running. Removing stale tracking files.")
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        return 0

    print(f"[*] Sending stop signal to Session Capture Daemon (PID {pid}, session: {sess_id})...")

    # 1. Create stop signal file
    with open(STOP_SIGNAL_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps({"requested_at": time.time(), "passkey": passkey or "default"}))

    # 2. Wait up to 5 seconds for graceful shutdown
    stopped = False
    for _ in range(50):
        if not is_process_alive(pid):
            stopped = True
            break
        time.sleep(0.1)

    # 3. Force terminate if daemon did not exit
    if not stopped:
        print(f"[*] Daemon did not exit within timeout. Terminating PID {pid}...")
        terminate_pid(pid)
        time.sleep(0.5)

    # 4. Clean up tracking files
    for p in (PID_FILE, STOP_SIGNAL_FILE):
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    print(f"[SUCCESS] Session capture daemon stopped. Session '{sess_id}' finalized.")
    return 0


def main() -> None:
    raw_args = sys.argv[1:]
    if raw_args:
        # Check if user typed passkeys directly without flags: e.g. `cli.py "stop stop"` or `cli.py stop`
        full_line = " ".join(raw_args).strip().lower()
        if full_line in ("stop stop", "stop-capture"):
            sys.exit(cmd_stop(passkey=full_line))
        if len(raw_args) == 1 and raw_args[0].lower() in ("stop", "stop-capture"):
            sys.exit(cmd_stop(passkey=raw_args[0].lower()))
        if len(raw_args) == 1 and raw_args[0].lower() == "status":
            sys.exit(cmd_status())

    parser = argparse.ArgumentParser(description="M.A.D.D.Y. Session Capture CLI")
    parser.add_argument("action", choices=["stop", "status"], nargs="?", default="status")
    parser.add_argument("--passkey", type=str, default=None, help="Stop authorization passkey")
    args = parser.parse_args()

    if args.action == "stop":
        sys.exit(cmd_stop(passkey=args.passkey or "stop"))
    else:
        sys.exit(cmd_status())


if __name__ == "__main__":
    main()

"""Command-Line Interface (CLI) entrypoint for M.A.D.D.Y. Assistant.
(Modular Autonomous Digital Daemon for You)

Optimized for instant, sub-second lazy startup:
- Main reasoning core and memory load in milliseconds.
- Heavy audio, neural TTS, and browser engines load strictly on demand.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
except ImportError:
    pass


BANNER = r"""
 ╔═══════════════════════════════════════════════════════════════════════════╗
 ║  __  __     _     ____    ____   __   __                                  ║
 ║ |  \/  |   / \   |  _ \  |  _ \  \ \ / /    M.A.D.D.Y.                    ║
 ║ | |\/| |  / _ \  | | | | | | | |  \ V /     Modular Autonomous            ║
 ║ | |  | | / ___ \ | |_| | | |_| |   | |      Digital Daemon for You        ║
 ║ |_|  |_|/_/   \_\|____/  |____/    |_|                                    ║
 ╠═══════════════════════════════════════════════════════════════════════════╣
 ║ Core: MaddyAgentCore | Multimodal: Text + Spoken Voice [ACTIVE]           ║
 ║ Type your request, '/voice' to toggle audio, '/help', or 'exit' to quit.  ║
 ╚═══════════════════════════════════════════════════════════════════════════╝
"""


def _print_help() -> None:
    print("""
Available Commands:
  /help, /start   - Show quick reference help card
  /status         - Display CPU, RAM, GPU, battery, and platform telemetry
  /screenshot     - Capture instant full desktop screenshot
  /photo, /camera - Capture instant webcam snapshot
  /lock           - Lock workstation immediately
  /capture_start  - Start background session capture (guest audit)
  /capture_stop   - Stop session capture (also 'stop stop')
  /tools          - List all registered tools and their descriptions
  /voice          - Toggle voice audio output responses on/off
  /clear          - Reset current conversation context history
  exit / quit     - Gracefully shut down M.A.D.D.Y.
""")


def _print_tools(runtime: Any) -> None:
    tools = runtime.tool_registry.list_tools()
    print(f"\nRegistered M.A.D.D.Y. Tools ({len(tools)} total):")
    print("=" * 65)
    for t in sorted(tools, key=lambda x: x.name):
        tier_str = getattr(t, "tier", "SAFE")
        tier_val = tier_str.value if hasattr(tier_str, "value") else str(tier_str)
        print(f"  • {t.name:<25} [{tier_val:<16}] : {t.description[:45]}")
    print("=" * 65)


def _print_status(runtime: Any) -> None:
    res = runtime.execute("/status")
    if res.get("completed") and res.get("response"):
        print("\n" + res.get("response", "").strip() + "\n")
    else:
        diag = runtime.get_diagnostics()
        print("\n" + "=" * 65)
        print("                 M.A.D.D.Y. SYSTEM TELEMETRY")
        print("         (Modular Autonomous Digital Daemon for You)")
        print("=" * 65)
        print(f"  Platform         : {diag['platform']}")
        print(f"  Execution Core   : {diag['execution_core']}")
        print(f"  Registered Tools : {diag['registered_tools_count']}")
        print(f"  Trace Database   : {'Active (WAL)' if diag['trace_db_exists'] else 'Pending'}")
        print(f"  Memory Database  : {'Active (WAL)' if diag['memory_db_exists'] else 'Pending'}")
        print("=" * 65)


def run_repl(runtime: Any, speak_flag: bool = True, voice_choice: Optional[str] = None) -> None:
    """Run instant interactive REPL with simultaneous text and voice output."""
    print(BANNER)
    session_id = f"maddy_cli_{int(time.time())}"

    while True:
        try:
            user_in = input("M.A.D.D.Y.> ").strip()
            if not user_in:
                continue

            lowered = user_in.lower()
            if lowered in ("exit", "quit", "q", ":q"):
                print("\nPowering down M.A.D.D.Y. core. Goodbye!")
                break

            if lowered in ("/help", "help", "?"):
                _print_help()
                continue

            if lowered in ("/status", "status"):
                _print_status(runtime)
                continue

            if lowered in ("/tools", "tools"):
                _print_tools(runtime)
                continue

            if lowered in ("/voice", "voice"):
                speak_flag = not speak_flag
                print(f"Spoken audio output: {'[ENABLED]' if speak_flag else '[MUTED]'}")
                continue

            if lowered in ("/clear", "clear", "cls"):
                runtime.session_histories[session_id] = []
                print("Conversation context cleared.")
                continue

            # Execute turn through MaddyAgentCore
            t0 = time.time()
            res = runtime.execute(
                user_in,
                session_id=session_id,
                speak_response=False,
                voice=voice_choice,
            )
            elapsed = time.time() - t0

            status_indicator = "✓" if res.get("completed") else "✗"
            resp = res.get("response", "").strip()
            print(f"\n[{status_indicator} {elapsed:.2f}s] {resp}\n")

            # Simultaneous Spoken Audio Output
            if speak_flag and resp:
                try:
                    from voice.engine import VoiceEngine
                    VoiceEngine(platform_driver=runtime.platform).speak(resp, voice=voice_choice)
                except Exception as v_err:
                    print(f"[Voice note: {v_err}]")

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Powering down M.A.D.D.Y.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M.A.D.D.Y. AI Computer Assistant CLI (Modular Autonomous Digital Daemon for You)",
        add_help=True,
    )
    parser.add_argument("request", nargs="*", help="Natural language command or subcommand ('listen', 'status', 'tools')")
    parser.add_argument("--interactive", "-i", "--text", action="store_true", help="Start interactive text typing prompt REPL instead of voice")
    parser.add_argument("--voice", "-v", type=str, default=None, help="Select voice profile ('adam', 'eve', 'chris', 'sarah')")
    parser.add_argument("--no-voice", "--mute", "-q", "--quiet", action="store_true", help="Mute spoken audio output (text only)")
    parser.add_argument("--mic", "-m", action="store_true", help="Launch live microphone wake-word listener ('Hey M.A.D.D.Y.')")
    parser.add_argument("--status", "-s", action="store_true", help="Display system telemetry and exit")
    parser.add_argument("--tools", "-t", action="store_true", help="List registered tools and exit")
    parser.add_argument("--resume", "-r", type=str, default=None, help="Resume a checkpointed task by task_id")

    args = parser.parse_args()

    # Subcommand fast-paths
    req_subcommand = args.request[0].lower() if args.request else ""

    if args.mic or req_subcommand in ("listen", "mic", "wakeword", "live"):
        print("Initializing M.A.D.D.Y. Neural Voice Listener...")
        try:
            from voice_listener import run_live_voice_assistant
            run_live_voice_assistant()
        except KeyboardInterrupt:
            print("\nPowering down M.A.D.D.Y. Voice Assistant. Goodbye!")
        return

    # Instant lazy import of MaddyRuntime
    from app.runtime import MaddyRuntime
    runtime = MaddyRuntime()

    if args.status or req_subcommand == "status":
        _print_status(runtime)
        return

    if args.tools or req_subcommand == "tools":
        _print_tools(runtime)
        return

    if args.resume or req_subcommand == "resume":
        task_id = args.resume or (args.request[1] if len(args.request) > 1 else None)
        if not task_id:
            print("Error: Please provide task_id to resume: maddy resume <task_id>")
            sys.exit(1)
        print(f"Resuming Task: {task_id}...")
        res = runtime.resume(task_id)
        print("=" * 60)
        print(f"Status  : {res.get('status')}")
        print(f"Response: {res.get('response')}")
        print("=" * 60)
        return

    voice_choice = args.voice
    speak_flag = not args.no_voice

    # If explicitly asked for interactive text REPL
    if args.interactive or req_subcommand in ("text", "type", "repl"):
        run_repl(runtime, speak_flag=speak_flag, voice_choice=voice_choice)
        return

    # Default when typing maddy / MADDY without arguments:
    # Launch Live Neural Voice Assistant in standby mode (ready for wake words or space bar)
    if not args.request:
        try:
            from voice_listener import run_live_voice_assistant
            run_live_voice_assistant(wake_on_start=False)
        except KeyboardInterrupt:
            print("\nPowering down M.A.D.D.Y. Voice Assistant. Goodbye!")
        return

    # Single-shot execution
    full_request = " ".join(args.request)
    t0 = time.time()
    res = runtime.execute(
        full_request,
        speak_response=False,
        voice=voice_choice,
    )
    elapsed = time.time() - t0

    resp = res.get("response", "").strip()
    status_indicator = "✓" if res.get("completed") else "✗"
    print(f"\n[{status_indicator} {elapsed:.2f}s] {resp}\n")

    if speak_flag and resp:
        try:
            from voice.engine import VoiceEngine
            VoiceEngine(platform_driver=runtime.platform).speak(resp, voice=voice_choice)
        except Exception as v_err:
            print(f"[Voice note: {v_err}]")


if __name__ == "__main__":
    main()

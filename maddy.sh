#!/usr/bin/env bash
# M.A.D.D.Y. Unix launcher: one entrypoint for CLI, voice, capture, and knock control.
set -euo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
if [[ -x "$SCRIPT_DIR/.venv-linux/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/.venv-linux/bin/python"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
fi

COMMAND="${1:-cli}"
if [[ $# -gt 0 ]]; then shift; fi

case "$COMMAND" in
    cli|app|run)
        exec "$PYTHON" -m app.cli "$@"
        ;;
    listen|mic|wakeword|voice)
        exec "$PYTHON" -u voice_listener.py "$@"
        ;;
    stop|stop-capture)
        exec "$PYTHON" -m session_capture.cli stop "$@"
        ;;
    capture-status)
        exec "$PYTHON" -m session_capture.cli status "$@"
        ;;
    knock)
        ACTION="${1:-start}"
        if [[ $# -gt 0 ]]; then shift; fi
        exec "$PYTHON" core/knock_detector.py "$ACTION" "$@"
        ;;
    help|-h|--help)
        cat <<'HELP'
Usage: ./maddy.sh <command> [arguments]

Commands:
  cli, app, run             Start the text CLI (default)
  listen, mic, voice       Start the voice listener
  knock [action]            Control knock detector (start/stop/status/restart/test/run)
  stop, stop-capture        Stop session capture
  capture-status            Show session capture status
HELP
        ;;
    *)
        exec "$PYTHON" -m app.cli "$COMMAND" "$@"
        ;;
esac

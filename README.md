# M.A.D.D.Y.

**Modular Autonomous Digital Daemon for You** — a voice-first personal assistant for Fedora Linux and Windows.

M.A.D.D.Y. combines a noise-resistant neural wake-word listener, cloud model reasoning, typed tool execution, persistent memory, computer/browser automation, skills, delegation, and background autonomy. The current implementation is Python-first and is being prepared for a validated Rust hybrid migration.

## Current architecture

```text
Voice listener / CLI
        |
        v
   MaddyRuntime
        |
        v
 MaddyAgentCore
        |
        v
 ToolRegistry -> PermissionGate -> Execution
        |
        v
 SQLite traces, memory, checkpoints
```

The runtime authority for the public interfaces is `app/runtime.py` and `core/agent_loop.py`. `app/maddy_assistant.py` orchestrates system reasoning, tools, and memory; new code must use the runtime and agent-core APIs. Telegram-specific integrations are not part of the supported public launcher.

## Features

- Multi-variant wake phrases for M.A.D.D.Y.
- Three-frame wake confirmation, dynamic energy gating, cooldown protection, and Whisper anti-hallucination filtering.
- VAD endpointing, barge-in interruption, and platform TTS fallback.
- Dynamic `Plan -> Act -> Observe -> Verify -> Replan` execution with step, time, and cost budgets.
- Typed tools with deterministic permission checks and structured results.
- SQLite WAL traces, checkpoints, FTS5 memory retrieval, and privacy-aware context compression.
- Skills discovery, validation, versioning, rollback, and runtime guardrails.
- Bounded subagent delegation with DAG validation and failure isolation.
- Background scheduling, crash recovery, circuit breakers, and audit logging.
- Windows and Linux computer/browser automation abstractions.

## Requirements

- Python 3.12+
- Fedora Linux or Windows 10/11
- Microphone for voice mode; webcam only for camera features
- API credentials configured locally in `.env`

No credentials belong in source control. Start from `.env.example` and keep `.env` local.
The `.env` file should be readable only by its owner (`chmod 600 .env`). Rotate any
credential that has ever been committed, logged, or shared.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate              # Linux
# .venv\Scripts\activate            # Windows
python -m pip install -e '.[dev]'
# Optional neural wake-word backend (currently requires Python 3.10/3.11 wheels):
# python -m pip install -e '.[voice]'
cp .env.example .env                   # Linux; copy manually on Windows
```

Configure at minimum the model provider and voice settings in `.env`:

```env
GROQ_API_KEY=your_key_here
WAKE_THRESHOLD=0.55
WAKE_CONFIRMATION_FRAMES=3
MIC_GAIN=1.0
```

## Running M.A.D.D.Y.

Use the unified platform launcher instead of the old duplicate launchers:

```bash
./maddy.sh cli                  # Text interface
./maddy.sh listen               # Continuous voice listener
./maddy.sh knock start          # Knock detector
./maddy.sh knock status
./maddy.sh stop                 # Stop session capture
./maddy.sh capture-status
./maddy.sh help
```

On Windows:

```cmd
maddy.cmd cli
maddy.cmd listen
maddy.cmd knock start
maddy.cmd knock status
maddy.cmd stop
maddy.cmd capture-status
maddy.cmd help
```

The launcher routes to `app.cli`, `voice_listener.py`, `core/knock_detector.py`, and `session_capture.cli`. Telegram commands are not part of this unified public launcher.

Direct module entrypoints remain available for development:

```bash
python -m app.cli
python voice_listener.py
```

## Voice stack

- **Wake word:** Optional OpenWakeWord-based multi-variant detection ("maddy", "hey maddy", "ok maddy", etc.); the voice listener reports a clear fallback when it is unavailable.
- **STT:** Groq Whisper Large v3 Turbo (cloud). Audio is sent to Groq's API for transcription.
- **TTS:** ElevenLabs when configured, then Edge-TTS; F5-TTS is an optional local backend.

### Privacy and safety

Voice recordings used for transcription are sent to Groq when cloud STT is enabled.
Cloud TTS providers receive the text being synthesized. Camera, screen capture, file
write, and command-execution tools are opt-in capabilities controlled by the permission
engine; do not enable voice automation in environments where untrusted people or audio
can issue commands. Review `security/` and test permissions before enabling unattended
or remote operation.

## Voice noise protection

Standby wake detection requires all of the following:

1. Neural confidence above `WAKE_THRESHOLD`.
2. Vocal energy above the adaptive noise floor.
3. Three consecutive confirmation frames by default.
4. Cooldown protection after activation.

Standby transcription uses a neutral prompt so ambient audio cannot bias Whisper toward the wake word. Sound-effect tags, subtitle artifacts, repetitive hallucinations, and low-information noise transcripts are rejected.

## Verification

Run the focused regression suite from the activated project environment:

```bash
python -m pytest tests/test_wakeword_noise_rejection.py tests/test_voice_endpointing.py tests/test_voice_standby_dispatch.py
```

Run the core integration suite:

```bash
python -m pytest tests/test_phase3_6_integration.py tests/test_phase3_7_computer_browser.py tests/test_phase3_8_memory_context.py tests/test_skills_system.py tests/test_delegation_system.py tests/test_autonomy_system.py
```

Optional benchmark suites use three runs per task and independent assertions. Results
are local measurements, not independent certification or production guarantees:

```bash
python benchmarks/phase5_benchmarks.py
python benchmarks/phase6_benchmarks.py
python benchmarks/phase7_benchmarks.py
```

## Rust hybrid migration

The proposed, operator-validation-gated plan is in [`docs/RUST_HYBRID_PORTING_PLAN.md`](docs/RUST_HYBRID_PORTING_PLAN.md). It keeps Python for audio/ML initially and moves execution authority, guardrails, storage, delegation, and autonomy into a process-isolated Rust daemon in staged milestones. No Rust port has started yet.

## Repository hygiene

Generated caches, local credentials, runtime databases, audio artifacts, downloaded model
files, and protected integration files are excluded through `.gitignore`. Do not add
secrets or runtime logs to commits.

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

Third-party attributions are listed in [`THIRD_PARTY.md`](THIRD_PARTY.md). Runtime dependencies retain their respective licenses.

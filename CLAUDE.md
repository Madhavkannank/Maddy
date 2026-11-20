# M.A.D.D.Y. Project Guidance

M.A.D.D.Y. (Modular Autonomous Digital Daemon for You) is a desktop autonomous
assistant for Fedora Linux and Windows.

## Architecture

All public non-messaging interfaces use:

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

`MaddyAgentCore` is the execution authority for the CLI and voice runtime. Keep
planning, tool calls, observations, verification, recovery, and replanning in
that lifecycle. Do not add a parallel execution path.

## Engineering rules

1. Inspect the repository and existing behavior before changing architecture.
2. Make precise, complete changes and preserve working functionality.
3. Keep tools typed, permission-checked, observable, and independently verifiable.
4. Surface failures explicitly; do not silently swallow errors or return success-shaped fallbacks.
5. Keep credentials, recordings, screenshots, runtime databases, and personal data out of Git.
6. Never perform irreversible external actions or delete user data without explicit approval.
7. Preserve Windows and Fedora behavior when changing platform-neutral code.
8. Use existing helpers and patterns before adding new abstractions or dependencies.
9. Keep documentation aligned with implemented behavior; do not claim unexecuted tests passed.
10. Preserve license notices and provenance for any reused third-party material.
11. Keep changes small, testable, and focused on the requested behavior.

## Protected integration

The Telegram integration is a protected compatibility surface. Do not modify its
implementation, scripts, configuration, tests, or behavior unless explicitly
authorized. Verify protected-file hashes after repository-wide maintenance.

## Runtime requirements

- Python 3.12 or newer
- Fedora Linux or Windows 10/11
- Optional microphone, camera, OCR, browser, and GPU capabilities
- Cloud credentials only in a local mode-600 `.env`

## Validation

Before reporting completion:

1. Run targeted tests for changed behavior.
2. Run `python -m compileall` for maintained Python packages.
3. Check imports, launcher entry points, configuration loading, and tool registration.
4. Verify protected integration hashes.
5. Report failures and environmental limitations honestly.

## Current capabilities

- Plan → Act → Observe → Verify → Replan execution
- Typed tools with deterministic permission checks
- SQLite traces, checkpoints, recovery, and bounded budgets
- Working, task, episodic, and preference memory
- Skills discovery and guarded execution
- Bounded multi-agent delegation
- Background scheduling with recovery and circuit breakers
- Voice input, wake-word detection, transcription, and TTS fallbacks
- Cross-platform browser and computer automation

# M.A.D.D.Y. Rust Hybrid Architecture Porting Plan

**Document Status:** Pending Operator Validation  
**Target Architecture:** Rust-Powered Hybrid System (Fedora Linux & Windows 11)  
**Governing Standard:** `CLAUDE.md` Engineering and Benchmark Rules  
**Date:** November 2026  

---

## 1. Motivation & Objectives

M.A.D.D.Y. is transitioning to a **Rust Hybrid Architecture** to achieve:
1. **Sub-Millisecond Loop Latency:** Move the core `Plan → Act → Observe → Verify → Replan` state machine, loop guardrails, token decoders, and tool dispatch into zero-allocation native Rust.
2. **Zero GIL Contention:** Background autonomy schedulers, multi-agent DAG workers, and SQLite WAL trace writers run concurrently across OS threads without Python GIL bottlenecks.
3. **Crash Resilience & Process Isolation:** Segregate audio/ML drivers (sounddevice, PortAudio, ONNX) into isolated worker processes, ensuring that audio device resets or driver quirks never corrupt agent state or crash the daemon.
4. **Hardware Efficiency:** Minimize CPU and idle RAM footprint on Intel i5 / RTX 3050 hardware (target < 35MB RAM for core daemon).

---

## 2. Structural Comparison of Hybrid Paradigms

### Option A: In-Process PyO3 Shared Library (`maddy_rs.so` / `maddy_rs.pyd`)
- **How it works:** Core algorithms (LoopGuard, Memory FTS5, DAG scheduler, Permissions) compiled as a Rust dynamic library imported into Python.
- **Pros:** Zero IPC latency, direct object passing, straightforward Python integration.
- **Cons:** Bound to the Python runtime; any C-extension fault (e.g. in sounddevice or ALSA) terminates the entire process; Python GIL still present during Python turns.

### Option B: Process-Isolated Rust Tokio Daemon + Python ML Worker (Recommended)
- **How it works:**
  - **Rust Daemon (`maddy-daemon`):** The primary execution authority. Manages CLI, agent loop, tool execution, SQLite WAL storage, background cron jobs, and permission security.
  - **Python ML Worker (`maddy-audio-worker`):** Thin process handling audio capture, OpenWakeWord neural inference, and TTS playback. Communicates with Rust via Unix Domain Sockets (Linux) and Named Pipes (Windows).
- **Pros:** 100% crash isolation, zero GIL in core daemon, daemon can run 24/7 as a systemd / Windows service with negligible resource usage.
- **Cons:** Requires IPC message serialization (optimized with zero-copy or binary JSON-RPC).

---

## 3. Subsystem Allocation Matrix (What Goes Where)

| Subsystem | Target Layer | Primary Technology | Rationale |
| :--- | :--- | :--- | :--- |
| **Agent Execution Loop** | **Rust** | `tokio`, `rig-core` / native state machine | Nanosecond state transitions, strict step/budget enforcement |
| **LoopGuard & Repetition Interceptor** | **Rust** | `sha2`, `ring`, `smallvec` | Zero-allocation SHA-256 call hashing, cyclic detection |
| **TraceStore & Memory Subsystem** | **Rust** | `rusqlite` (bundled SQLite 3.45+), FTS5 | WAL mode, synchronous=NORMAL, atomic transactions |
| **Permission Gate & Sandbox** | **Rust** | Native AST & path canonicalization | Memory-safe security boundary; zero escape vulnerabilities |
| **Subagent Delegation & DAGs** | **Rust** | `petgraph`, Tokio tasks | Thread-pool bounded concurrency, topological wave dispatch |
| **Background Autonomy & Cron** | **Rust** | `cron`, Tokio timers | Drift-compensated epoch precision, atomic locking |
| **Tool Execution Engine** | **Rust** | Native Rust tools + process runners | Safe sandboxed command execution, file I/O |
| **Wake Word & Audio Streaming** | **Python** / **Rust** | `sounddevice` + `openwakeword` (or `tract-onnx`) | High stability on Linux/Windows audio backends |
| **TTS Synthesis** | **Python** / **Rust** | `edge-tts` / `elevenlabs-rs` | Streamed audio playback |

---

## 4. Staged Migration Roadmap

### Milestone 1: Rust Core Foundation Crates
1. `maddy-types`: Common structs (`AgentStep`, `ToolCall`, `ToolResult`, `AgentStatus`, `MemoryEntry`).
2. `maddy-guard`: Native Rust `LoopGuard` with SHA-256 hashing and ping-pong detection.
3. `maddy-permissions`: Deterministic permission engine and path traversal validator.

### Milestone 2: Native Storage & Memory Crates
1. `maddy-storage`: SQLite WAL connection pool, FTS5 triggers (`ai`, `ad`, `au`), trace persistence.
2. `maddy-memory`: 5-layer cognitive hierarchy with exponential recency decay and token Jaccard deduplication.

### Milestone 3: Execution Authority & Delegation
1. `maddy-core`: Tokio-driven dynamic agent loop (`Plan → Act → Observe → Verify → Replan`).
2. `maddy-delegation`: Dependency DAG engine with Kahn's cycle detection and topological execution waves.
3. `maddy-autonomy`: 24/7 background cron daemon with exponential backoff circuit breakers.

### Milestone 4: Cross-Language IPC & Audio Bridge
1. Ultra-fast bi-directional IPC layer between Rust daemon and Python voice listener.
2. Verified multi-frame wake confirmation passing audio frames across the boundary with < 10ms latency.
3. Comprehensive benchmark equivalence verification against existing Python benchmarks (Phases 2–7).

---

## 5. Verification & Acceptance Criteria
- **Behavioral Equivalence:** Every task in `phase2_benchmarks.py` through `phase7_benchmarks.py` must run and pass against the Rust hybrid runtime.
- **Performance Benchmark:** Core loop step turnaround must be $\le$ 5ms (excluding cloud LLM network roundtrips).
- **Memory Footprint:** Core daemon idle memory $\le$ 35MB RAM.
- **Crash Resilience:** Killing the Python audio worker must trigger clean automatic reconnect without dropping daemon state or corrupting traces.

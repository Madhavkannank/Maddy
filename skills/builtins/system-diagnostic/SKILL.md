---
name: system-diagnostic
description: "Comprehensive host system diagnostics: CPU, GPU, RAM, Disk, and OS facts."
version: 1.0.0
author: M.A.D.D.Y.
platforms: [linux, windows]
category: system
tags: [telemetry, hardware, cpu, gpu, memory, disk, diagnostics]
dependencies: [get_system_info, execute_command]
permissions: [SAFE, SYSTEM_READ_ONLY]
---

# System Diagnostic Protocol

Standard operational procedure for diagnosing host machine health, hardware utilization, and platform environment.

## When to Use
- User asks about hardware performance, memory availability, or temperatures.
- Verifying NVIDIA RTX 3050 GPU status, VRAM usage, and CUDA acceleration.
- Diagnosing disk storage capacity or network interface status on Windows or Fedora.

## Diagnostic Protocol

### 1. Core Host Telemetry
- Execute `get_system_info` tool to obtain OS architecture, kernel version, installed RAM, and CPU core count.
- Cross-verify total and available memory before launching intensive local processes.

### 2. GPU & Acceleration Audit
- If NVIDIA GPU is present, query GPU facts (`nvidia-smi` or platform layer):
  - Model: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM limit).
  - Check driver version, compute capability, and currently active CUDA processes.
  - Ensure VRAM usage remains strictly under 5.0GB to prevent system OOM.

### 3. Filesystem & Mount Health
- Check root and target filesystem capacity.
- Verify read/write permissions in sandboxed temporary directories (`/tmp` or `%TEMP%`).

### 4. Spoken Synthesis
- Present spoken diagnostic summaries concisely (e.g. "Host system running smoothly on Fedora Linux with 9.2 gigabytes of RAM available and GPU idle at 42 degrees Celsius.").

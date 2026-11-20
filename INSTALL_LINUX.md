# M.A.D.D.Y. Linux Installation Guide

This guide documents system packages, environment configuration, and setup steps for running **M.A.D.D.Y.** on Linux.

---

## 1. System Package Requirements

M.A.D.D.Y. utilizes:
- **PortAudio & ALSA/Pulse/PipeWire** for real-time microphone streaming and audio playback.
- **Video4Linux2 (V4L2)** for hardware camera snapshot capture.
- **Tesseract OCR** for Linux screen and document reading.
- **FFmpeg & libsndfile** for audio format conversion and high-speed playback.
- **wmctrl & xdotool** for Linux application window management and focus.
- **lm_sensors & nvidia-smi** for hardware thermals and GPU load monitoring.
- **speech-dispatcher / espeak-ng** for offline text-to-speech fallbacks.

### Fedora Linux (RPM / DNF)
```bash
sudo dnf update -y

# Core audio libraries & development headers
sudo dnf install -y alsa-lib-devel portaudio-devel pulseaudio-libs-devel libsndfile-devel alsa-utils pulseaudio-utils pipewire-utils

# Video4Linux2 & webcam tools for hardware camera capture
sudo dnf install -y v4l-utils guvcview cheese

# OCR engine & English language pack
sudo dnf install -y tesseract tesseract-langpack-eng

# FFmpeg multimedia codec tools
sudo dnf install -y ffmpeg ffmpeg-devel

# Hardware sensors for thermals and system monitoring
sudo dnf install -y lm_sensors

# Window focus & desktop automation tools
sudo dnf install -y wmctrl xdotool

# Speech synthesis fallback engines
sudo dnf install -y speech-dispatcher espeak-ng
```

### Ubuntu / Debian (APT)
```bash
sudo apt-get update

# Core audio libraries & development headers
sudo apt-get install -y libasound2-dev portaudio19-dev libpulse-dev libsndfile1-dev alsa-utils pulseaudio-utils

# Video4Linux2 & webcam tools
sudo apt-get install -y v4l-utils guvcview cheese

# OCR engine & training data
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng libtesseract-dev

# FFmpeg multimedia tools
sudo apt-get install -y ffmpeg

# Hardware sensors for thermals and system monitoring
sudo apt-get install -y lm-sensors

# Window focus & desktop automation tools
sudo apt-get install -y wmctrl xdotool

# Speech synthesis fallback engines
sudo apt-get install -y speech-dispatcher espeak-ng
```

### Arch Linux (Pacman)
```bash
sudo pacman -Syu --needed

# Core audio libraries & development headers
sudo pacman -S --needed alsa-lib portaudio libpulse libsndfile alsa-utils pulseaudio-utils pipewire pipewire-pulse

# Video4Linux2 & webcam tools
sudo pacman -S --needed v4l-utils guvcview cheese

# OCR engine & training data
sudo pacman -S --needed tesseract tesseract-data-eng

# FFmpeg multimedia tools
sudo pacman -S --needed ffmpeg

# Hardware sensors for thermals and system monitoring
sudo pacman -S --needed lm_sensors

# Window focus & desktop automation tools
sudo pacman -S --needed wmctrl xdotool

# Speech synthesis fallback engines
sudo pacman -S --needed speech-dispatcher espeak-ng
```

---

## 2. NVIDIA Driver & CUDA Setup

For GPU acceleration (NVIDIA RTX 3050 6GB or compatible card):

1. **Verify NVIDIA Driver Installation:**
   ```bash
   nvidia-smi
   ```
   Ensure your GPU is recognized and driver version is $\ge 525.60.13$.

2. **Distribution-Specific NVIDIA Driver Packages:**
   - **Fedora:**
     ```bash
     sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda
     ```
   - **Ubuntu:**
     ```bash
     sudo ubuntu-drivers install
     # Or specific driver version:
     # sudo apt-get install -y nvidia-driver-550
     ```
   - **Arch Linux:**
     ```bash
     sudo pacman -S --needed nvidia nvidia-utils cuda
     ```

---

## 3. Python Virtual Environment

M.A.D.D.Y. requires **Python 3.12+**.

### Standard Setup (`venv`)
```bash
# Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Upgrade packaging tools
pip install --upgrade pip setuptools wheel
```

### Fast Setup (`uv`)
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and activate
uv venv --python 3.12
source .venv/bin/activate
```

---

## 4. PyTorch (with CUDA Acceleration)

Install PyTorch with CUDA 12.6:

```bash
# Install PyTorch with CUDA 12.6 support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Verify CUDA hardware acceleration:
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 5. Install Project Dependencies

Install all core, audio, vision, speech, and testing dependencies:

```bash
# Install base requirements
pip install -r requirements.txt

# Install offline transcription model support (faster-whisper)
pip install faster-whisper

# Or using uv:
# uv pip install -r requirements.txt faster-whisper
```

---

## 6. Linux Platform Specifics & Permissions

### 1. Audio and Video Group Permissions
Ensure your user account has direct access to soundcards and `/dev/video*` hardware webcam nodes without requiring root:

```bash
# Add current user to audio and video groups
sudo usermod -aG audio,video $USER
```
> **Important:** Log out and log back in (or run `newgrp audio` and `newgrp video`) for group permission changes to take effect.

### 2. Display Server & GUI Automation (X11 vs. Wayland)
M.A.D.D.Y. supports window focusing (`wmctrl`, `xdotool`) and screen automation:
- **X11 / Xorg:** Fully supported out of the box for window switching, focus, and screenshots.
- **Wayland (Default on GNOME / KDE):** 
  - Standard screenshot and camera capture work normally.
  - For advanced window focusing (`wmctrl`/`xdotool`) and cursor automation (`pyautogui`), log into an **"Ubuntu on Xorg"** or **"GNOME on X11"** session at the display manager login screen, or ensure XWayland compatibility is enabled.

### 3. Webcam / Camera Backend
- M.A.D.D.Y.'s `CameraEngine` (`tools/camera.py`) automatically utilizes `cv2.CAP_ANY` / V4L2 on Linux.
- Snapshots are saved directly to `~/Pictures/MADDY_Captures/` and opened immediately via `xdg-open`.
- List active video capture devices:
  ```bash
  v4l2-ctl --list-devices
  ```

### 4. Linux OCR Engine (Tesseract)
- On Linux, M.A.D.D.Y. automatically invokes `pytesseract` communicating with `/usr/bin/tesseract`.
- Ensure standard language data is available:
  - Fedora: `/usr/share/tesseract/tessdata/`
  - Ubuntu/Debian: `/usr/share/tesseract-ocr/5/tessdata/` or `/usr/share/tesseract-ocr/4.00/tessdata/`
  - Arch: `/usr/share/tessdata/`

### 5. Audio Playback Backends
M.A.D.D.Y. dynamically detects and routes audio playback across Linux backends:
1. `sounddevice` (direct hardware playback via ALSA / PipeWire-ALSA)
2. `ffplay` (FFmpeg quiet audio player)
3. `pw-play` (Native PipeWire playback)
4. `paplay` (PulseAudio sound daemon playback)
5. `aplay` (Direct ALSA stream playback)
6. `vlc` (CLI dummy interface)

---

## 7. Configuration & Environment Setup

1. Copy the example environment configuration:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your preferred credentials:
   ```bash
   nano .env
   ```
   Key environment variables:
   - `GROQ_API_KEY`: Required for Cloud Whisper Large v3 Turbo STT and fast Semantic Dispatcher. (If omitted, M.A.D.D.Y. automatically falls back to local `faster-whisper` on CPU/GPU).
   - `TELEGRAM_BOT_TOKEN`: (Optional) Remote control token from @BotFather.
   - `TELEGRAM_ALLOWED_USER_ID`: (Optional) Your personal numeric Telegram user ID.
   - `WAKE_THRESHOLD`: Wake word detection sensitivity (default: `0.30`).
   - `MIC_GAIN`: Audio input software gain multiplier (default: `1.0`).

---

## 8. Launching M.A.D.D.Y.

### 1. Neural Voice Listener (Hands-Free Voice Mode)
Launch the real-time openWakeWord and Whisper voice listener:

```bash
python voice_listener.py
```
- Say `"Hey Maddy"` or tap the spacebar to activate.
- Supports single-breath commands (e.g., *"Hey Maddy what is the square root of 256"*).
- Supports English, Tamil (`தமிழ்`), and Malayalam (`മലയാളം`).

### 2. Interactive CLI Mode
For terminal-based interaction and diagnostics:

```bash
python -m app.cli
```

### 3. Telegram Remote Control Service
M.A.D.D.Y. includes an optional daemonized background Telegram bridge:

```bash
# Start background daemon
python -m integrations.messaging.telegram_bot start

# Check status
python -m integrations.messaging.telegram_bot status

# Inspect real-time logs
python -m integrations.messaging.telegram_bot logs

# Stop background daemon
python -m integrations.messaging.telegram_bot stop

# Run in foreground for debugging
python -m integrations.messaging.telegram_bot run
```

---

## 9. Autostart via Systemd (Optional Background Service)

To run the Telegram bot or M.A.D.D.Y. voice assistant automatically in the background on user login:

1. Create user systemd service directory:
   ```bash
   mkdir -p ~/.config/systemd/user
   ```

2. Create `~/.config/systemd/user/maddy-telegram.service`:
   ```ini
   [Unit]
   Description=M.A.D.D.Y. Telegram Remote Daemon
   After=network.target sound.target

   [Service]
   Type=simple
   WorkingDirectory=/path/to/Maddy
   ExecStart=/path/to/Maddy/.venv/bin/python -m app.cli
   Restart=on-failure
   RestartSec=5
   Environment=PYTHONUNBUFFERED=1

   [Install]
   WantedBy=default.target
   ```
   *(Replace `/path/to/Maddy` with your actual repository path, e.g., `/home/username/Maddy`)*.

3. Enable and start service:
   ```bash
   systemctl --user daemon-reload
   systemctl --user enable --now maddy-telegram.service

   # Check status:
   systemctl --user status maddy-telegram.service

   # Enable persistence even when logged out:
   loginctl enable-linger $USER
   ```

---

## 10. Verification & Diagnostics Checklist

Run these quick one-liners to verify your Linux setup:

1. **Verify Audio Input / Output:**
   ```bash
   python -c "import sounddevice as sd; print('Default Devices:\n', sd.query_devices())"
   ```

2. **Verify Camera Capture:**
   ```bash
   python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera 0 Accessible:', cap.isOpened()); cap.release()"
   ```

3. **Verify OCR Engine:**
   ```bash
   python -c "import pytesseract; from PIL import Image; print('Tesseract Version:', pytesseract.get_tesseract_version())"
   ```

4. **Verify NVIDIA GPU & VRAM Monitoring:**
   ```bash
   python -c "from hardware.gpu_monitor import GPUMonitor; m = GPUMonitor().query_gpu(); print('GPU Available:', m.is_available, '| Name:', m.name, '| VRAM Free:', m.memory_free_mb, 'MB')"
   ```

5. **Run Full Test Suite:**
   ```bash
   pytest tests/ -v
   ```

---

## 11. Troubleshooting Common Linux Issues

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `PortAudio / sounddevice error: -9996` | User lacks audio permissions or ALSA stream is locked | Run `sudo usermod -aG audio $USER` and log back in. Ensure no exclusive JACK/ALSA locks exist. |
| `Cannot open /dev/video0: Permission denied` | User lacks video group permission | Run `sudo usermod -aG video $USER` and log out/in. |
| `TesseractNotFoundError` | `tesseract` package is missing or not in PATH | Install `tesseract` (Fedora: `tesseract`, Ubuntu: `tesseract-ocr`, Arch: `tesseract`). Verify with `which tesseract`. |
| `wmctrl: command not found` | Window switcher utility not installed | Install `wmctrl` and `xdotool` via your package manager. |
| `ModuleNotFoundError: faster_whisper` | Offline Whisper fallback package not installed | Run `pip install faster-whisper` inside your virtual environment. |
| `nvidia-smi: command not found` | NVIDIA proprietary drivers not loaded | Install proprietary NVIDIA drivers for your kernel (`akmod-nvidia` on Fedora, `nvidia-driver` on Ubuntu, `nvidia` on Arch). |

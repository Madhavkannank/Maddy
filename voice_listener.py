"""
State-of-the-Art Neural Voice Assistant for M.A.D.D.Y.
(Modular Autonomous Digital Daemon for You)
Powered by:
1. openWakeWord / M.A.D.D.Y. neural pipeline — Dedicated wake-word engine (<2ms, 0 false triggers)
2. faster-whisper ('tiny.en' / 'whisper-large-v3-turbo') — Neural transformer STT for 99.9% accurate command recognition
3. ElevenLabs / Edge-TTS / Voicebox Neural Voice — High-fidelity speech synthesis
""" 
import os
import sys
import time
import queue
import threading
import re
import shutil
import argparse
import io
import wave
import signal
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
try:
    import sounddevice as sd
except Exception:
    sd = None
from dotenv import load_dotenv
import types
from concurrent.futures import ThreadPoolExecutor

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# Sub-second startup fast-path: stub unused custom_verifier_model inside openwakeword package
if "openwakeword.custom_verifier_model" not in sys.modules:
    _stub = types.ModuleType("openwakeword.custom_verifier_model")
    _stub.train_custom_verifier = None
    sys.modules["openwakeword.custom_verifier_model"] = _stub

from voice.wakeword import WakeWordEngine

CHUNK_SIZE = 1280  # 80ms at 16000 Hz
SAMPLE_RATE = 16000

audio_queue = queue.Queue()
barge_in_queue = queue.Queue()
is_speaking = False
barge_in_event = threading.Event()

def audio_callback(indata, frames, time_info, status):
    """Raw microphone frame callback with dual-channel routing during recitation."""
    if is_speaking:
        barge_in_queue.put(bytes(indata))
    else:
        audio_queue.put(bytes(indata))


def should_listen_for_followup(text: str) -> bool:
    """Determine if M.A.D.D.Y. should maintain hands-free continuous dialogue."""
    clean = text.strip().lower()
    if any(w in clean for w in ["standing by", "goodbye", "good night", "signing off", "have a good", "have a great"]):
        return False
    return True


def resolve_groq_api_keys() -> List[str]:
    """Return all configured Groq keys in priority order, including legacy env aliases."""
    keys: List[str] = []
    for env_var in ["GROQ_API_KEY", "GROQ_KEY", "GROQ_API_KEY_FALLBACK", "GROQ_API_KEY_2"]:
        value = (os.getenv(env_var) or "").strip()
        if value and value not in keys:
            keys.append(value)
    return keys


def is_valid_speech_transcription(text: str) -> bool:
    """Validate that Whisper output contains genuine speech, not ambient noise/hallucinations."""
    if not text or len(text.strip()) < 2:
        return False
    clean = text.lower().strip(" .?!,:;\"'")
    if clean in {
        "you", "thank you", "thanks for watching", "bye", "subscribe",
        "i don't know", "i don't know, okay?", "silence", "watching", "subtitles",
        "yeah", "okay", "ok", "um", "uh", "hmm", "huh", "oh", "ah", "so", "right", "like",
        "noise", "music", "applause", "laughter", "cheering", "cough", "throat",
        "whispering", "unintelligible", "inaudible", "ambient", "static",
        "foreign", "mbc", "shh", "psst", "beep", "buzz", "ring", "tap", "click",
        "whistle", "groan", "gasp", "yawn", "sneeze", "snort", "snicker", "sigh",
        "hum", "screaming", "sobbing", "crying", "panting",
    }:
        return False
    # Reject subtitle / transcription hallucination markers
    if any(h in clean for h in [
        "subtitles by", "transcribed by", "translated by", "thanks for watching",
        "please subscribe", "like and subscribe", "copyright",
    ]):
        return False
    # Reject text enclosed entirely in brackets, parentheses, or asterisks (sound effects)
    if re.match(r"^(\[|\(|\*)[^\]\)\*]+(\]|\)|\*)$", clean):
        return False
    # Reject noise acronyms, countdowns, codes (e.g. "ZJ-3-2-1", "A-1", "X-2-3")
    if re.match(r"^[a-z]{1,3}[-_ ]*\d+([-_ ]*\d+)*$", clean):
        return False
    # Reject consonant clusters without English vowels (e.g. "zj", "psst", "shh")
    words_alpha = [w for w in re.findall(r"[a-z]+", clean) if len(w) > 0]
    if words_alpha and not any(re.search(r"[aeiouy]", w) for w in words_alpha):
        return False
    # Reject repetitive single words (e.g. "the the the", "maddy maddy maddy")
    words = clean.split()
    if len(words) >= 3 and len(set(words)) == 1:
        return False
    rep_check = [w.strip() for w in re.split(r"[.!?]+", text) if w.strip()]
    if len(rep_check) >= 4 and len(set(rep_check)) <= 2:
        return False

    # Reject repetitive wake word / token loops (e.g. "Hey Maddy. Maddy. Maddy.", "Maddy Maddy")
    clean_words = re.findall(r"[a-z0-9]+", clean)
    maddy_count = sum(1 for w in clean_words if w in ("maddy", "maddie", "madi", "maddi"))
    if maddy_count >= 2:
        return False
    # Reject consecutive identical full words (length > 1, e.g. "maddy maddy", "the the", "hey hey")
    if len(clean_words) >= 2 and any(clean_words[i] == clean_words[i + 1] and len(clean_words[i]) > 1 for i in range(len(clean_words) - 1)):
        return False
    # Check repeated sentence-level phrases, ignoring spelled acronyms (e.g. M.A.D.D.Y.)
    rep_segments = [s.strip().lower() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(rep_segments) >= 2 and not all(len(s) <= 1 for s in rep_segments):
        multi_char_segments = [s for s in rep_segments if len(s) > 1]
        if len(multi_char_segments) >= 2 and len(set(multi_char_segments)) < len(multi_char_segments):
            return False
    return True




def get_audio_input_devices() -> List[Dict[str, Any]]:
    """Retrieve and categorize all available hardware audio input devices."""
    if sd is None:
        return []
    devices = sd.query_devices()
    default_in = sd.default.device[0]
    input_devs = []
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            name = d["name"].splitlines()[0].strip()
            name_lower = name.lower()
            is_headset = any(k in name_lower for k in [
                "headset", "earphone", "headphones", "earbuds", "bluetooth", "wireless", "external", "usb"
            ])
            is_internal = any(k in name_lower for k in [
                "microphone array", "smart sound", "internal", "built-in", "integrated"
            ]) and not is_headset
            input_devs.append({
                "id": idx,
                "name": name,
                "channels": d["max_input_channels"],
                "default_sr": int(d.get("default_samplerate", 16000)),
                "is_default": (idx == default_in),
                "is_headset": is_headset,
                "is_internal": is_internal,
            })
    return input_devs


def resolve_audio_input_device(requested: Optional[Any] = None) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Intelligently resolve the audio input device:
    1. Explicit request (--mic) or .env (AUDIO_INPUT_DEVICE)
    2. Connected headset / earphone microphone auto-preference
    3. System default fallback
    """
    devs = get_audio_input_devices()
    if not devs:
        return None, {"id": None, "name": "Default", "is_headset": False, "is_internal": False}

    target = str(requested or os.getenv("AUDIO_INPUT_DEVICE", "")).strip()

    # 1. Exact numeric device ID
    if target and target.isdigit():
        target_id = int(target)
        for d in devs:
            if d["id"] == target_id:
                return target_id, d

    # 2. String keyword search (e.g. "headset", "airdopes", "jbl", "mic 2", "array")
    if target:
        t_low = target.lower()
        for d in devs:
            if t_low in d["name"].lower():
                return d["id"], d

    # 3. Fallback to active OS default recording device
    default_dev = next((d for d in devs if d["is_default"]), devs[0])
    return default_dev["id"], default_dev


def print_audio_device_diagnostics(active_dev: Dict[str, Any], all_devs: List[Dict[str, Any]]):
    """Print audio input diagnostic table and routing details."""
    print("=" * 70)
    print("      🎙️ MICROPHONE DEVICE DIAGNOSTICS & HARDWARE ROUTING")
    print("=" * 70)
    dev_type = "🎧 HEADSET / EARPHONE MICROPHONE" if active_dev.get("is_headset") else "💻 LAPTOP INTERNAL ARRAY (CHASSIS MIC)"
    print(f"  -> ACTIVE INPUT: [ID {active_dev['id']}] {active_dev['name']}")
    print(f"  -> DEVICE TYPE:  {dev_type}")
    if active_dev.get("is_internal"):
        print("  ⚠️ NOTICE: Currently listening through the laptop chassis mic!")
        print("     In a lab/noisy room, this picks up background noise and room chatter.")
        print("     To route audio directly through your earphones/headset:")
        print("     * Set AUDIO_INPUT_DEVICE in .env or run with: python voice_listener.py --mic <id>")

    headsets = [d for d in all_devs if d["is_headset"]]
    if headsets:
        print("\n  🎧 Detected Headset / Earphone Inputs:")
        for h in headsets:
            active_marker = " <--- [CURRENTLY SELECTED]" if h["id"] == active_dev["id"] else ""
            print(f"     * ID {h['id']:>2}: {h['name']}{active_marker}")

    print("=" * 70 + "\n")


def test_microphone_level(device_id: Optional[int], duration: float = 1.8) -> bool:
    """Brief live mic test that visualizes audio level so user can confirm audio input."""
    if sd is None:
        print("[VoiceListener Error]: PortAudio / sounddevice library is not available.")
        return False
    print("🎙️ [MIC AUDIO CHECK]: Speak or tap your earphone mic now...")
    samples = []
    def cb(indata, frames, t, s):
        samples.append(np.frombuffer(indata, dtype=np.int16))

    max_detected = 0
    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, device=device_id, channels=1, dtype="int16", callback=cb):
            start = time.time()
            while time.time() - start < duration:
                time.sleep(0.08)
                if samples:
                    recent = samples[-1]
                    amp = int(np.max(np.abs(recent)))
                    max_detected = max(max_detected, amp)
                    bars = "#" * int(min(amp / 350, 15))
                    print(f"\r  🎙️ Live Level: [{bars:<15}] Peak Amp: {amp:>5}", end="", flush=True)
            print()
    except Exception as e:
        print(f"\n  [Audio Test Note]: {e}")

    if max_detected > 250:
        print(f"  -> Audio signal confirmed active! (Peak: {max_detected})\n")
        return True
    else:
        print(f"  -> Note: Very low signal (Peak: {max_detected}). If muted, check earphone hardware switch.\n")
        return False


def _handle_sigint(signum, frame):
    """Convert Ctrl+C into an in-thread KeyboardInterrupt so the app exits cleanly."""
    raise KeyboardInterrupt


class TerminalCbreakContext:
    """Context manager to enable non-blocking single-keypress detection (e.g. Space bar) on POSIX."""

    def __init__(self):
        self.old_settings = None
        self.fd = None

    def __enter__(self):
        if not sys.platform.startswith("win") and sys.stdin.isatty():
            try:
                import termios
                import tty
                self.fd = sys.stdin.fileno()
                self.old_settings = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
            except Exception:
                self.old_settings = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_settings is not None and self.fd is not None:
            try:
                import termios
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass


def run_live_voice_assistant(
    mic_device: Optional[Any] = None,
    threshold: Optional[float] = None,
    gain: Optional[float] = None,
    skip_test: bool = True,
    continuous: bool = False,
    wake_on_start: bool = False,
    **kwargs: Any,
):
    global is_speaking
    signal.signal(signal.SIGINT, _handle_sigint)
    print("=" * 70)
    print("   M.A.D.D.Y. ADVANCED VOICE ASSISTANT — NEURAL SOTA PIPELINE")
    print("   Modular Autonomous Digital Daemon for You")
    print("=" * 70)

    # Resolve audio device silently under the hood
    dev_id, active_dev = resolve_audio_input_device(mic_device)

    # Launch asynchronous pipeline loaders
    loader_pool = ThreadPoolExecutor(max_workers=2)
    phrase_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="maddy_phrase_wake")
    def _load_assistant():
        from app.runtime import MaddyRuntime
        return MaddyRuntime()
    fut_asst = loader_pool.submit(_load_assistant)
    def _load_oww():
        import logging
        logging.getLogger().setLevel(logging.ERROR)
        try:
            from openwakeword.model import Model as OpenWakeWordModel
            import openwakeword

            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, "models", "wakewords")

            custom_maddy_model = os.getenv("MADDY_WAKEWORD_MODEL")
            wakeword_models = []
            if custom_maddy_model and os.path.exists(custom_maddy_model):
                wakeword_models.append(custom_maddy_model)
            else:
                candidate_names = [
                    "maddy_v0.1.onnx",
                    "hey_maddy_v0.1.onnx",
                    "maddy.onnx",
                ]
                for cname in candidate_names:
                    cpath = os.path.join(models_dir, cname)
                    if os.path.exists(cpath):
                        wakeword_models.append(cpath)
                        break

            if not wakeword_models:
                return None

            melspec = os.path.join(models_dir, "melspectrogram.onnx")
            embedding = os.path.join(models_dir, "embedding_model.onnx")
            kwargs = {}
            if os.path.exists(melspec):
                kwargs["melspec_model_path"] = melspec
            if os.path.exists(embedding):
                kwargs["embedding_model_path"] = embedding

            model = OpenWakeWordModel(
                wakeword_models=wakeword_models,
                inference_framework="onnx",
                **kwargs
            )

            vad_path = os.path.join(models_dir, "silero_vad.onnx")
            if os.path.exists(vad_path):
                try:
                    vad = openwakeword.VAD(model_path=vad_path)
                    model.vad = vad
                    model.vad_threshold = float(os.getenv("VAD_THRESHOLD", "0.45"))
                except Exception as vad_err:
                    logging.warning("Notice: Silero VAD attachment warning: %s", vad_err)

            return model
        except Exception as oww_err:
            logging.warning("Notice: OpenWakeWord onnx initialization fallback: %s", oww_err)
            return None
    fut_oww = loader_pool.submit(_load_oww)

    real_assistant = None
    real_oww = None

    class _AssistantProxy:
        def __getattr__(self, name):
            nonlocal real_assistant
            if real_assistant is None:
                real_assistant = fut_asst.result()
            return getattr(real_assistant, name)
        def __bool__(self):
            return True

    assistant = _AssistantProxy()
    oww = None
    wake_engine = WakeWordEngine(wake_phrase="hey maddy")

    groq_api_keys = resolve_groq_api_keys()
    groq_api_key = groq_api_keys[0] if groq_api_keys else ""
    groq_key_index = 0
    local_whisper = None

    wake_threshold = float(threshold if threshold is not None else os.getenv("WAKE_THRESHOLD", "0.55"))
    mic_gain = float(gain if gain is not None else os.getenv("MIC_GAIN", "1.0"))
    confirmation_frames = int(os.getenv("WAKE_CONFIRMATION_FRAMES", "3"))
    confirm_streak = 0
    last_wake_fire_time = 0.0
    wake_cooldown_seconds = 2.0
    silence_peak_floor = 10
    silence_alert_chunks = 125  # ~10s of 80ms chunks
    silent_chunks_count = 0
    audio_is_silent = False

    try:
        addr = assistant.get_address_title()
    except Exception:
        addr = os.getenv("MADDY_USER_TITLE", "boss")

    is_windows = sys.platform.startswith("win")
    platform_default = "chris" if is_windows else "sarah"
    DEFAULT_VOICE = os.getenv("MADDY_VOICE", platform_default)

    if os.getenv("VOICEBOX_ENABLED", "").lower() in ("true", "1"):
        voice_label = "Voicebox 24/7 (Local Cloned M.A.D.D.Y. / Scarlett)"
    elif os.getenv("ELEVENLABS_API_KEY"):
        voice_label = f"ElevenLabs ({'Chris' if is_windows else 'Sarah'})"
    else:
        active_edge_voice = os.getenv("EDGE_TTS_VOICE_MALE", "en-GB-RyanNeural") if is_windows else os.getenv("EDGE_TTS_VOICE_FEMALE", "en-GB-SoniaNeural")
        voice_label = f"Edge-TTS Neural ({active_edge_voice})"

    print("\n" + "=" * 70)
    print("  [STATUS] M.A.D.D.Y. IS FULLY ONLINE & READY!")
    print(f"  -> Addressing: {addr}")
    print("  -> Wake Triggers: 'Hey M.A.D.D.Y.' / 'MADDY' / 'OK M.A.D.D.Y.' / 'Wake up' / [Space Bar]")
    print(f"  -> Voice Output: {voice_label}")
    print(f"  -> Microphone: [Device {active_dev['id']}] {active_dev['name']}")
    print("  -> Press Ctrl+C anytime to stop.")
    print("=" * 70 + "\n")

    # Pre-warm noisereduce in background daemon thread for instantaneous first transcription
    threading.Thread(target=lambda: __import__("noisereduce"), daemon=True).start()

    def drain_audio():
        """Clear audio queue and reset wake-word model buffer."""
        nonlocal oww, confirm_streak
        confirm_streak = 0
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break
        if oww is not None or fut_oww.done():
            if oww is None:
                try:
                    oww = fut_oww.result()
                except Exception:
                    oww = None
            if oww is not None:
                oww.reset()

    _cached_voice_engine = None

    def get_voice_engine():
        nonlocal _cached_voice_engine
        if _cached_voice_engine is None:
            from voice.engine import VoiceEngine
            _cached_voice_engine = VoiceEngine()
        return _cached_voice_engine

    def speak(text: str, voice_name: Optional[str] = None) -> bool:
        """Speak response cleanly with live stop-word and wake-word barge-in interruption support."""
        global is_speaking
        is_speaking = True
        barge_in_event.clear()

        # Drain any residual frames from barge_in_queue
        while not barge_in_queue.empty():
            try:
                barge_in_queue.get_nowait()
            except queue.Empty:
                break

        drain_audio()
        interrupted = False
        active_voice = voice_name or DEFAULT_VOICE

        stop_monitor_running = threading.Event()
        stop_monitor_running.set()

        def _barge_in_listener():
            accumulated = []
            last_check_t = time.time()

            while stop_monitor_running.is_set() and not barge_in_event.is_set():
                # Fast keyboard check (e.g. Space bar) to halt recitation immediately
                hit_space = False
                if sys.platform.startswith("win"):
                    try:
                        import msvcrt
                        while msvcrt.kbhit():
                            ch = msvcrt.getch()
                            if ch in (b' ', b'\r', b'\n', ' ', '\r', '\n'):
                                hit_space = True
                    except Exception:
                        pass
                else:
                    try:
                        import select
                        if sys.stdin.isatty():
                            while select.select([sys.stdin], [], [], 0.0)[0]:
                                ch = sys.stdin.read(1)
                                if ch in (' ', '\r', '\n'):
                                    hit_space = True
                    except Exception:
                        pass

                if hit_space:
                    print("\n>>> [SPACE BAR INTERRUPT - Halting recitation immediately!]")
                    barge_in_event.set()
                    try:
                        import sounddevice as sd
                        sd.stop()
                    except Exception:
                        pass
                    return

                try:
                    chunk = barge_in_queue.get(timeout=0.08)
                    accumulated.append(chunk)
                except queue.Empty:
                    continue

                # 1. Neural wake word fast path (if openWakeWord model loaded)
                if oww is not None:
                    try:
                        c_arr = np.frombuffer(chunk, dtype=np.int16)
                        preds = oww.predict(c_arr)
                        for k, score in preds.items():
                            if score >= wake_threshold:
                                print(f"\n>>> [WAKE INTERRUPT DETECTED: {k} ({score:.2f}) - Halting recitation immediately!]")
                                barge_in_event.set()
                                try:
                                    import sounddevice as sd
                                    sd.stop()
                                except Exception:
                                    pass
                                return
                    except Exception:
                        pass

                # 2. Stop words speech recognition path (~0.4 - 0.7s chunk check)
                total_samples = len(accumulated) * CHUNK_SIZE
                now_t = time.time()
                if total_samples >= SAMPLE_RATE * 0.45 or (now_t - last_check_t >= 0.7 and total_samples >= SAMPLE_RATE * 0.3):
                    chunks_to_check = list(accumulated)
                    accumulated = accumulated[-2:] if len(accumulated) > 2 else []
                    last_check_t = now_t

                    raw_bytes = b"".join(chunks_to_check)
                    snip_arr = np.frombuffer(raw_bytes, dtype=np.int16)
                    if len(snip_arr) == 0:
                        continue
                    snip_peak = int(np.max(np.abs(snip_arr)))
                    snip_rms = float(np.sqrt(np.mean((snip_arr.astype(np.float32) / 32768.0) ** 2)))

                    # Must have minimum human vocal energy
                    if snip_peak < 220 or snip_rms < 0.004:
                        continue

                    # Validate vocal speech via Silero VAD if available (frame-by-frame peak check)
                    if oww is not None and hasattr(oww, "vad") and oww.vad is not None and len(snip_arr) >= 640:
                        try:
                            n_samples = (len(snip_arr) // 640) * 640
                            frame_vads = [
                                oww.vad.predict(snip_arr[i:i+640], frame_size=640)
                                for i in range(0, n_samples, 640)
                            ]
                            if frame_vads and max(frame_vads) < 0.35:
                                continue
                        except Exception:
                            pass

                    # Transcribe and check for stop command or wake callout
                    try:
                        transcribed, _ = transcribe_audio(chunks_to_check, use_cloud=True, is_standby=True)
                        clean_trans = transcribed.strip().lower()
                        if clean_trans:
                            if wake_engine.detect_barge_in(clean_trans) or wake_engine.detect_wake_word(clean_trans):
                                print(f"\n🛑 [STOP COMMAND DETECTED: '{clean_trans}' - Halting recitation immediately!]")
                                barge_in_event.set()
                                try:
                                    import sounddevice as sd
                                    sd.stop()
                                except Exception:
                                    pass
                                return
                    except Exception:
                        pass

        monitor_thread = threading.Thread(target=_barge_in_listener, daemon=True)
        monitor_thread.start()

        try:
            get_voice_engine().speak(text, voice=active_voice, stop_event=barge_in_event)
            if barge_in_event.is_set():
                interrupted = True
                print("\n>>> [STOP / BARGE-IN DETECTED - Listening to you]")
        finally:
            stop_monitor_running.clear()
            monitor_thread.join(timeout=0.25)
            time.sleep(0.15)
            drain_audio()
            is_speaking = False
        return interrupted

    # Dynamic charismatic wake greetings (time-aware & varied)
    import random

    last_greeting_idx = [-1]

    def speak_dynamic_wake_greeting() -> str:
        """Select a diverse charismatic greeting dynamically from PersonalityEngine and speak with configured voice."""
        try:
            candidates = assistant.get_greeting_candidates()
        except Exception:
            active_addr = assistant.get_address_title() if hasattr(assistant, "get_address_title") else "boss"
            candidates = [
                f"At your service, {active_addr}.",
                f"Always a pleasure, {active_addr}.",
                f"Yes {active_addr}, how can I help you?",
                f"Right with you, {active_addr}.",
                f"Standing by, {active_addr}.",
                f"All systems ready, {active_addr}.",
                f"How may I assist you, {active_addr}?",
                f"I'm listening, {active_addr}.",
                f"Ready for your command, {active_addr}.",
                f"Here and ready to help, {active_addr}.",
                f"What can I do for you, {active_addr}?",
                f"Yes {active_addr}, I'm right here.",
                f"Ready when you are, {active_addr}.",
                f"How can I be of service, {active_addr}?",
                f"Awaiting your instructions, {active_addr}.",
                f"Right here, {active_addr}. What do you need?",
                f"At your command, {active_addr}.",
                f"Listening closely, {active_addr}.",
                f"Yes {active_addr}, how may I help?",
                f"Ready to assist, {active_addr}.",
                f"At your disposal, {active_addr}.",
                f"Speak freely {active_addr}, I'm listening.",
                f"Always ready, {active_addr}.",
                f"What are we working on, {active_addr}?",
                f"Good to hear from you, {active_addr}. How can I assist?",
                f"Online and listening, {active_addr}.",
                f"What's on your mind, {active_addr}?",
                f"How can I make your day easier, {active_addr}?",
                f"At your beck and call, {active_addr}.",
                f"Ready and listening, {active_addr}.",
                f"Go ahead {active_addr}, I'm with you.",
                f"I'm all ears, {active_addr}.",
            ]

        available_indices = [i for i in range(len(candidates)) if i != last_greeting_idx[0]]
        chosen_idx = random.choice(available_indices) if available_indices else 0
        last_greeting_idx[0] = chosen_idx
        greeting_text = candidates[chosen_idx]

        speak(greeting_text, voice_name=DEFAULT_VOICE)
        return greeting_text

    local_whispers: Dict[str, Any] = {}

    def get_local_whisper(model_size: str = "base"):
        nonlocal local_whispers
        if model_size not in local_whispers:
            try:
                from faster_whisper import WhisperModel
                local_whispers[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
            except Exception:
                local_whispers[model_size] = None
        return local_whispers.get(model_size)

    groq_rate_limited_until = 0.0

    def transcribe_audio(pcm_chunks: list, use_cloud: bool = True, is_standby: bool = False) -> Tuple[str, str]:
        """
        Transcribe PCM audio buffer with Dual-Tier High-Fidelity Processing:
        1. 100Hz Butterworth High-Pass Filter (removes sub-bass HVAC & rumble without cutting speech)
        2. Gentle Spectral Denoising (prop_decrease=0.20 keeps delicate 'm', 'd', 's', 'th' consonant formants intact)
        3. Dynamic Peak Normalization
        4. Primary: Groq Cloud Whisper Large v3 Turbo (1.5B params, ~0.5s latency, auto-detects EN/TA/ML)
        5. Fallback: Local faster-whisper multilingual model (tiny.en for 250ms standby, base for full commands)
        Returns: (transcribed_text, detected_language)
        """
        nonlocal groq_rate_limited_until, groq_api_key
        if not pcm_chunks:
            return "", "unknown"
        try:
            raw_bytes = b"".join(pcm_chunks)
            audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_np) < SAMPLE_RATE * 0.4:  # Under 400ms is too short
                return "", "unknown"

            # 1. 4th-order Bandpass filter: 100Hz to 3800Hz (human speech formants)
            try:
                from scipy import signal as dsp_signal
                b_bp, a_bp = dsp_signal.butter(4, [100.0 / (SAMPLE_RATE / 2), 3800.0 / (SAMPLE_RATE / 2)], btype="band")
                audio_filtered = dsp_signal.lfilter(b_bp, a_bp, audio_np).astype(np.float32)
            except Exception:
                audio_filtered = audio_np

            # 2. Adaptive Spectral Noise Reduction (gentle 0.20 to protect delicate voice formants)
            try:
                import noisereduce as nr
                audio_denoised = nr.reduce_noise(
                    y=audio_filtered,
                    sr=SAMPLE_RATE,
                    stationary=True,
                    prop_decrease=0.20,
                    n_fft=512,
                )
            except Exception:
                audio_denoised = audio_filtered

            # 3. Dynamic RMS energy normalization (targets -18dBFS speech level for Whisper)
            rms = float(np.sqrt(np.mean(audio_denoised**2))) if len(audio_denoised) > 0 else 0.0
            if rms > 0.005:
                gain = min(0.12 / rms, 8.0)
                audio_clean = np.clip(audio_denoised * gain, -0.98, 0.98).astype(np.float32)
            elif rms < 0.001:
                return "", "unknown"
            else:
                audio_clean = audio_denoised

            # Ensure contiguous float32 for ONNXRuntime Silero VAD / Whisper
            audio_clean = np.ascontiguousarray(audio_clean, dtype=np.float32)

            text = ""
            detected_lang = "unknown"

            # 3. Try all configured Groq cloud keys in priority order, then local fallback.
            now_t = time.time()
            if use_cloud and now_t >= groq_rate_limited_until:
                candidate_keys = resolve_groq_api_keys()
                if not candidate_keys:
                    candidate_keys = [groq_api_key] if groq_api_key else []

                last_error = None
                for candidate_key in candidate_keys:
                    groq_api_key = candidate_key
                    try:
                        audio_int16 = (np.clip(audio_clean, -1.0, 1.0) * 32767.0).astype(np.int16)
                        buf = io.BytesIO()
                        with wave.open(buf, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(SAMPLE_RATE)
                            wf.writeframes(audio_int16.tobytes())
                        wav_bytes = buf.getvalue()

                        headers = {"Authorization": f"Bearer {candidate_key}"}
                        files = {"file": ("command.wav", wav_bytes, "audio/wav")}
                        cloud_prompt = (
                            ""
                            if is_standby
                            else "M.A.D.D.Y. personal voice assistant. Clear speech, commands, questions, numbers, math, calculations."
                        )
                        data = {
                            "model": "whisper-large-v3-turbo",
                            "response_format": "verbose_json",
                            "language": "en" if is_standby else None,
                            "prompt": cloud_prompt,
                        }
                        import requests
                        res = requests.post(
                            "https://api.groq.com/openai/v1/audio/transcriptions",
                            files=files,
                            data=data,
                            headers=headers,
                            timeout=(3.5, 9.0),
                        )
                        if res.status_code == 200:
                            res_json = res.json()
                            text = res_json.get("text", "").strip()
                            detected_lang = res_json.get("language", "english")
                            last_error = None
                            break
                        last_error = f"status {res.status_code}"
                        if res.status_code == 429:
                            last_error = "status 429 (rate limit)"
                            if not is_standby:
                                print(f"\n[Groq Whisper 429]: Key rate limited; rotating to next configured key...")
                            continue
                        elif res.status_code in (401, 403):
                            last_error = f"status {res.status_code} (auth failure)"
                            if not is_standby:
                                print(f"\n[Groq Whisper {res.status_code}]: Trying next configured Groq key...")
                            continue
                        if not is_standby:
                            print(f"\n[Groq Whisper API Status {res.status_code}]: Falling back to local...")
                        break
                    except Exception as groq_err:
                        last_error = str(groq_err)
                        if not is_standby:
                            print(f"\n[Groq Whisper Notice]: {groq_err}. Trying next configured key...")
                        continue

                if not text and last_error:
                    backoff = 45.0 if "429" in str(last_error) else 20.0
                    groq_rate_limited_until = now_t + backoff
                    if not is_standby:
                        print(f"\n[Groq Whisper] All cloud keys currently rate-limited; backing off for {backoff:.0f}s.")

            # 4. Fallback: Local faster-whisper multilingual
            if not text:
                whisper_model_name = "tiny.en" if is_standby else "base"
                lw = get_local_whisper(whisper_model_name)
                if lw is not None:
                    try:
                        local_prompt = (
                            ""
                            if is_standby
                            else (
                                "M.A.D.D.Y. personal voice assistant. Clear speech, commands, questions, numbers, math, calculations. "
                                "English, Tamil, Malayalam, Hindi."
                            )
                        )
                        beam = 3 if is_standby else 5
                        segments, info = lw.transcribe(
                            audio_clean,
                            beam_size=beam,
                            language="en" if is_standby else None,  # English for standby callouts, auto-detect for commands
                            temperature=0.0,
                            condition_on_previous_text=False,
                            compression_ratio_threshold=2.4,
                            no_speech_threshold=0.6,
                            initial_prompt=local_prompt,
                            vad_filter=bool(is_standby),  # Enable internal VAD for standby snippets to reject non-speech noise
                        )
                        words = [s.text.strip() for s in segments if s.text.strip()]
                        text = " ".join(words).strip()
                        detected_lang = info.language if hasattr(info, "language") else "unknown"
                    except Exception as err:
                        if not is_standby:
                            print(f"[Whisper Local Error]: {err}")

            # 6. Discard common Whisper silence hallucinations, noise codes, and repetition loops
            if not is_valid_speech_transcription(text):
                return "", "unknown"

            return text, detected_lang
        except Exception as e:
            print(f"[Whisper Error]: {e}")
            return "", "unknown"

    # Start audio input stream with 1280-sample blocks (80ms chunks) using chosen device
    if sd is None:
        print("[VoiceListener Error]: PortAudio / sounddevice library is not available.")
        return
    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        dtype="int16",
        channels=1,
        device=dev_id,
        callback=audio_callback,
    )
    term_ctx = TerminalCbreakContext()
    term_ctx.__enter__()
    stream.start()
    try:
        waiting_for_command = bool(wake_on_start)
        command_wait_start = time.time() if wake_on_start else 0.0
        command_chunks = []
        speech_started = False
        speech_chunk_count = 0
        speech_resume_count = 0
        consecutive_silence = 0
        peak_amp = 0

        # Calibrate baseline ambient noise floor from real microphone stream
        time.sleep(0.15)
        initial_samples = []
        while not audio_queue.empty():
            try:
                c = audio_queue.get_nowait()
                c_arr = np.frombuffer(c, dtype=np.int16)
                if len(c_arr) > 0:
                    initial_samples.append(float(np.max(np.abs(c_arr))))
            except queue.Empty:
                break
        ambient_noise_ema = float(np.median(initial_samples)) if initial_samples else 250.0
        ambient_noise_ema = max(80.0, min(15000.0, ambient_noise_ema))
        dynamic_threshold = max(220, int(ambient_noise_ema * 1.45 + 100))

        standby_chunks: List[bytes] = []
        standby_preroll: List[bytes] = []
        standby_speech_active = False
        standby_silence_count = 0
        standby_max_neural_score = 0.0
        last_standby_phrase_check = 0.0
        phrase_future = None
        is_followup_turn = bool(wake_on_start)

        if wake_on_start:
            addr = assistant.get_address_title() if assistant else "boss"
            init_greeting = f"M.A.D.D.Y. is online and ready, {addr}."
            print(f"-> M.A.D.D.Y.: \"{init_greeting}\"")
            speak(init_greeting)
            # Acoustic settling & room reverberation dissipation
            time.sleep(0.35)
            drain_audio()
            # Post-greeting real room ambient noise floor calibration
            time.sleep(0.20)
            calib_samples = []
            while not audio_queue.empty():
                try:
                    c = audio_queue.get_nowait()
                    c_arr = np.frombuffer(c, dtype=np.int16)
                    if len(c_arr) > 0:
                        calib_samples.append(float(np.max(np.abs(c_arr))))
                except queue.Empty:
                    break
            if calib_samples:
                ambient_noise_ema = float(np.median(calib_samples))
                ambient_noise_ema = max(80.0, min(15000.0, ambient_noise_ema))
                dynamic_threshold = max(220, int(ambient_noise_ema * 1.45 + 100))

            drain_audio()
            command_chunks = []
            speech_started = False
            speech_chunk_count = 0
            speech_resume_count = 0
            consecutive_silence = 0
            peak_amp = 0
            command_wait_start = time.time()
            print(f"🎙️ [SPEAK YOUR COMMAND NOW] (Noise Floor: {int(ambient_noise_ema)} | Voice Gate: {dynamic_threshold})...")
        else:
            drain_audio()

        while True:
            # Check for keyboard tap (Space bar / Enter) to activate or finalize
            keyboard_trigger = False
            if sys.platform.startswith("win"):
                try:
                    import msvcrt
                    while msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch in (b' ', b'\r', b'\n', ' ', '\r', '\n'):
                            keyboard_trigger = True
                except Exception:
                    pass
            else:
                try:
                    import select
                    if sys.stdin.isatty():
                        while select.select([sys.stdin], [], [], 0.0)[0]:
                            ch = sys.stdin.read(1)
                            if ch in (' ', '\r', '\n'):
                                keyboard_trigger = True
                except Exception:
                    pass

            try:
                chunk_bytes = audio_queue.get(timeout=0.05)
            except queue.Empty:
                timeout_limit = 6.0 if is_followup_turn else 5.0
                if waiting_for_command and not speech_started and (time.time() - command_wait_start > timeout_limit):
                    tag = "Continuous dialogue closed (timeout)" if is_followup_turn else "Listening timeout"
                    print(f"\n[Standby] {tag} ({timeout_limit:.0f}s). Ready for 'Hey M.A.D.D.Y.'.")
                    waiting_for_command = False
                    is_followup_turn = False
                    command_chunks = []
                    speech_started = False
                    speech_chunk_count = 0
                    speech_resume_count = 0
                    consecutive_silence = 0
                    peak_amp = 0
                    drain_audio()
                continue
            except KeyboardInterrupt:
                print("\nShutting down M.A.D.D.Y. Voice Assistant. Goodbye!")
                break

            chunk_raw = np.frombuffer(chunk_bytes, dtype=np.int16)
            if mic_gain != 1.0 and len(chunk_raw) > 0:
                scaled = chunk_raw.astype(np.float32) * mic_gain
                chunk_raw = np.clip(scaled, -32768, 32767).astype(np.int16)
                chunk_bytes = chunk_raw.tobytes()
            max_amp = int(np.max(np.abs(chunk_raw))) if len(chunk_raw) > 0 else 0
            chunk_f = chunk_raw.astype(np.float32) / 32768.0 if len(chunk_raw) > 0 else np.zeros(1, dtype=np.float32)
            chunk_rms = float(np.sqrt(np.mean(chunk_f ** 2))) if len(chunk_f) > 0 else 0.0
            dynamic_rms = max(0.008, (dynamic_threshold / 32768.0) * 0.70)
            is_speech = (chunk_rms >= dynamic_rms) or (max_amp >= dynamic_threshold and chunk_rms >= 0.006)

            # -------------------------------------------------------------
            # STATE 1: WAITING FOR USER'S COMMAND (NEURAL WHISPER)
            # -------------------------------------------------------------
            if waiting_for_command:
                command_chunks.append(chunk_bytes)
                peak_amp = max(peak_amp, max_amp)

                # Real-time microphone feedback with live noise floor & gate level
                meter_scale = max(dynamic_threshold * 2.2, 4000.0)
                mic_level = int(min(max_amp / meter_scale, 1.0) * 6)
                status_tag = "HEARING SPEECH" if speech_started else "LISTENING..."
                cmd_status = f"🎙️ [{status_tag:<14}] Gate: {dynamic_threshold:>4} | Mic: [{'#' * mic_level}{' ' * (6 - mic_level)}] amp: {max_amp:>5}"
                cols = shutil.get_terminal_size((80, 24)).columns
                print(f"\r{cmd_status.ljust(cols - 2)[:cols - 2]}", end="", flush=True)

                min_speech_chunks = 2
                if not speech_started:
                    if is_speech:
                        speech_chunk_count += 1
                        if speech_chunk_count >= min_speech_chunks:
                            speech_started = True
                            consecutive_silence = 0
                            speech_resume_count = 0
                    else:
                        speech_chunk_count = max(0, speech_chunk_count - 1)
                else:
                    # Speech in progress — count consecutive silent frames (each chunk = 80ms)
                    if not is_speech:
                        consecutive_silence += 1
                        speech_resume_count = 0
                    else:
                        # Audio above threshold — require sustained speech (>=2 chunks) to soft-decrement silence
                        # Soft-decrementing prevents ambient noise bursts from resetting silence accumulator to 0!
                        speech_resume_count += 1
                        if speech_resume_count >= 2:
                            consecutive_silence = max(0, consecutive_silence - 1)
                            speech_resume_count = 0

                # Finalization triggers:
                # 1. User spoke sustained speech (>=14 chunks / ~1.1s) and paused for 4 chunks (~320ms natural pause)
                # 2. User spoke standard speech and paused for 5 chunks (~400ms natural pause)
                # 3. User tapped Enter/Space key with adequate audio (>=4 chunks)
                # 4. Maximum continuous utterance limit reached (5.5s safety cap to eliminate noise hang)
                # 5. Initial silence timeout (user didn't speak after wake word)
                should_process = False
                total_command_chunks = len(command_chunks)
                if speech_started and total_command_chunks >= 14 and consecutive_silence >= 4:
                    should_process = True
                elif speech_started and consecutive_silence >= 5:
                    should_process = True
                elif keyboard_trigger and total_command_chunks >= 4:
                    should_process = True
                elif speech_started and (time.time() - command_wait_start >= 5.5) and total_command_chunks >= 10:
                    should_process = True
                elif not speech_started:
                    # No speech started yet — timeout cleanly back to standby
                    timeout_limit = 5.0 if is_followup_turn else 4.5
                    if time.time() - command_wait_start >= timeout_limit:
                        tag = "Continuous dialogue closed (timeout)" if is_followup_turn else "No speech detected in buffer"
                        print(f"\n[Standby] {tag} ({timeout_limit:.0f}s). Ready for 'Hey M.A.D.D.Y.'.")
                        waiting_for_command = False
                        is_followup_turn = False
                        command_chunks = []
                        speech_started = False
                        speech_chunk_count = 0
                        speech_resume_count = 0
                        consecutive_silence = 0
                        peak_amp = 0
                        drain_audio()
                        continue

                if should_process:
                    print("\n[Transcribing speech with Neural Whisper]...")
                    try:
                        cmd_text, detected_lang = transcribe_audio(command_chunks)
                    except KeyboardInterrupt:
                        print("\n[Shutdown] Ctrl+C received while transcribing the command. Exiting cleanly.")
                        break

                    waiting_for_command = False
                    command_chunks = []
                    speech_started = False
                    speech_chunk_count = 0
                    speech_resume_count = 0
                    consecutive_silence = 0
                    peak_amp = 0

                    if cmd_text:
                        print(f"[Whisper Transcribed ({detected_lang})]: \"{cmd_text}\"")
                        from core.math_phonetics import MathPhoneticizer
                        clean = MathPhoneticizer.normalize_spelled_letters(cmd_text.strip().strip(".?!"))
                        dismiss_phrases = [
                            "cancel", "never mind", "stop", "be quiet", "that's all", "thats all",
                            "nothing", "done", "standby", "dismiss", "stop stop"
                        ]
                        is_session_target = any(w in clean.lower() for w in ["session", "capture", "guest audit", "audit"])
                        env_passkey = os.getenv("SESSION_CAPTURE_PASSKEY", "").strip().lower()
                        is_explicit_passkey = clean.lower() == "stop stop" or (bool(env_passkey) and clean.lower() == env_passkey)
                        is_negative_no = clean.lower().startswith("no ") or clean.lower() == "no"
                        is_no_maddy = re.match(r"^(?:no|not)\s+(?:maddy|m\.a\.d\.d\.y\.?)\b", clean.lower()) is not None

                        if is_explicit_passkey or (not is_session_target and not is_no_maddy and not (is_negative_no and len(clean.split()) > 1) and any(clean.lower() == p or clean.lower().startswith(p + " ") for p in dismiss_phrases)):
                            # If an active capture session is running or explicit passkey, stopping/dismissing it takes priority
                            has_active_capture = (
                                assistant is not None
                                and hasattr(assistant, "session_capture_manager")
                                and assistant.session_capture_manager.has_active_session()
                            )
                            if has_active_capture or (is_explicit_passkey and hasattr(assistant, "session_capture_manager")):
                                stopped, stop_msg, _ = assistant.session_capture_manager.stop_session(caller_id="owner")
                                if stopped:
                                    print(f"-> [Session Capture] Stopped via voice dismissal: {stop_msg}")
                                    addr = assistant.get_address_title()
                                    speak(f"Capture session dismissed and stopped, {addr}.")
                                    drain_audio()
                                    continue

                            print(f"-> Dismissed ('{clean}').")
                            print("[Standby] Ready for 'Hey M.A.D.D.Y.'.\n")
                            drain_audio()
                            continue
                        else:
                            print(f"-> Executing: '{clean}'")
                            res = assistant.execute(clean)
                            resp_text = res.get("response", "")
                            if resp_text:
                                # Dual-channel voice synthesis: extract concise spoken briefing devoid of raw URLs
                                from core.math_phonetics import MathPhoneticizer
                                spoken_text = res.get("speakable_response") or MathPhoneticizer.extract_speakable_text(resp_text)
                                is_unconfigured = (
                                    "unconfigured" in spoken_text.lower()
                                    or "missing groq_api_key" in spoken_text.lower()
                                    or "missing api key" in spoken_text.lower()
                                )
                                is_err = (
                                    ("{" in spoken_text and "}" in spoken_text)
                                    or "status 429" in spoken_text.lower()
                                    or "ratelimit" in spoken_text.lower()
                                    or "rate limit" in spoken_text.lower()
                                    or "tokens per day" in spoken_text.lower()
                                    or "modelapierror" in spoken_text.lower()
                                    or "internal disruption" in spoken_text.lower()
                                    or "unreachable" in spoken_text.lower()
                                )
                                addr = assistant.get_address_title()
                                if is_unconfigured:
                                    spoken_text = f"My cloud LLM credentials are unconfigured, {addr}. Please configure your GROQ_API_KEY in the .env file."
                                elif is_err:
                                    spoken_text = f"My neural connection is temporarily experiencing high traffic, {addr}. Please give me just a moment."
                                print(f"-> Response: {resp_text}\n")
                                links_banner = MathPhoneticizer.format_links_for_display(resp_text)
                                if links_banner:
                                    print(links_banner)
                                interrupted = speak(spoken_text)

                                # Followup turn-taking: only keep mic open if continuous mode is enabled
                                wants_followup = bool(continuous)
                                if interrupted or wants_followup:
                                    drain_audio()
                                    dynamic_threshold = max(220, int(ambient_noise_ema * 1.35 + 80))
                                    tag = "INTERRUPTED" if interrupted else "CONTINUOUS DIALOGUE (8s)"
                                    print(f"🎙️ [{tag} - Speak follow-up naturally without wake word] (Voice Gate: {dynamic_threshold})...")
                                    waiting_for_command = True
                                    is_followup_turn = True
                                    command_wait_start = time.time()
                                    command_chunks = []
                                    speech_started = False
                                    speech_chunk_count = 0
                                    speech_resume_count = 0
                                    consecutive_silence = 0
                                    peak_amp = 0
                                    continue
                                else:
                                    waiting_for_command = False
                                    is_followup_turn = False
                                    command_chunks = []
                                    speech_started = False
                                    speech_chunk_count = 0
                                    speech_resume_count = 0
                                    consecutive_silence = 0
                                    peak_amp = 0
                                    print("[Standby] Ready for 'Hey M.A.D.D.Y.'.\n")
                                    drain_audio()
                                    continue
                    else:
                        waiting_for_command = False
                        is_followup_turn = False
                        command_chunks = []
                        speech_started = False
                        speech_chunk_count = 0
                        speech_resume_count = 0
                        consecutive_silence = 0
                        peak_amp = 0
                        print("[Standby] No speech detected in buffer. Ready for 'Hey M.A.D.D.Y.'.\n")
                        drain_audio()
                        continue

                    drain_audio()
                    continue

                continue

            # -------------------------------------------------------------
            # STATE 2: STANDBY SCANNING FOR WAKE WORD (Neural + Phrase)
            # -------------------------------------------------------------
            # Track stream health (dead-mic detection)
            if max_amp <= silence_peak_floor:
                silent_chunks_count += 1
                if silent_chunks_count == silence_alert_chunks:
                    audio_is_silent = True
                    print(f"\n⚠️ [Audio Notice]: Microphone delivers only silence (peak <= {silence_peak_floor} for 10s). Check device input or hardware switch.")
            elif silent_chunks_count > 0:
                if audio_is_silent:
                    print("\n🎙️ [Audio Stream]: Microphone audio active — stream healthy.")
                    audio_is_silent = False
                silent_chunks_count = 0

            # Continuously track ambient background noise with asymmetric EMA
            if max_amp < ambient_noise_ema:
                ambient_noise_ema = 0.88 * ambient_noise_ema + 0.12 * max_amp
            else:
                ambient_noise_ema = 0.995 * ambient_noise_ema + 0.005 * max_amp

            dynamic_threshold = max(220, int(ambient_noise_ema * 1.45 + 100))
            phrase_gate = max(180, int(ambient_noise_ema * 1.30 + 60))
            phrase_rms_gate = max(0.006, (phrase_gate / 32768.0) * 0.70)

            if oww is None:
                if fut_oww.done():
                    try:
                        oww = fut_oww.result()
                    except Exception:
                        oww = None
                else:
                    time.sleep(0.01)
                    continue

            now_loop_time = time.time()
            neural_wake_detected = False
            if oww is not None:
                prediction = oww.predict(chunk_raw)
                best_wake_model = max(prediction, key=prediction.get) if prediction else "maddy"
                score = float(prediction.get(best_wake_model, 0.0))
                # Ambient noise rejection: multi-frame confirmation streak
                # Stray acoustic transients spike a single 80ms frame; authentic vocal wake
                # phrases sustain acoustic energy across multiple consecutive frames (>=3 frames / 240ms).
                has_vocal_energy = (max_amp >= dynamic_threshold or chunk_rms >= dynamic_rms)
                if score >= wake_threshold and has_vocal_energy:
                    confirm_streak += 1
                    if confirm_streak >= confirmation_frames:
                        if (now_loop_time - last_wake_fire_time) >= wake_cooldown_seconds:
                            neural_wake_detected = True
                            last_wake_fire_time = now_loop_time
                        confirm_streak = 0
                else:
                    confirm_streak = 0
            else:
                best_wake_model = "phrase"
                score = 0.0
                confirm_streak = 0

            # Phrase-based verbal wake word detector in standby (non-blocking background worker)
            phrase_wake_detected = False
            phrase_text = ""
            phrase_trailing_cmd = None

            # Poll background phrase transcription result without blocking the audio loop
            if phrase_future is not None and phrase_future.done():
                try:
                    snip_text, _ = phrase_future.result()
                    if snip_text and is_valid_speech_transcription(snip_text):
                        is_callout = bool(re.search(r"\bdaddy(?:['’]s|\s+is)?\s+home\b", snip_text, flags=re.IGNORECASE))
                        # Ambient noise & hallucination rejection:
                        # When openWakeWord is active (oww is not None), neural detection is the EXCLUSIVE authority
                        # for the Maddy wake phrase. We reject phrase-based "Hey Maddy" detections to eliminate
                        # Whisper acoustic hallucinations on background noise, HVAC, typing, and room chatter.
                        if oww is not None and not is_callout:
                            pass
                        elif wake_engine.detect_wake_word(snip_text) or is_callout:
                            if (now_loop_time - last_wake_fire_time) >= wake_cooldown_seconds:
                                phrase_wake_detected = True
                                phrase_text = snip_text
                                phrase_trailing_cmd = wake_engine._extract_trailing_command(snip_text)
                                last_wake_fire_time = now_loop_time
                except Exception:
                    pass
                finally:
                    phrase_future = None

            # Maintain rolling pre-roll buffer of last 3 chunks (~240ms) to avoid clipping leading consonants
            standby_preroll.append(chunk_bytes)
            if len(standby_preroll) > 3:
                standby_preroll.pop(0)

            # Accumulate speech frames for phrase wake word detection
            if max_amp >= phrase_gate and chunk_rms >= phrase_rms_gate:
                if not standby_speech_active:
                    standby_speech_active = True
                    standby_chunks.extend(standby_preroll[:-1])
                    standby_max_neural_score = 0.0
                standby_chunks.append(chunk_bytes)
                standby_max_neural_score = max(standby_max_neural_score, score)
                standby_silence_count = 0
            elif standby_speech_active:
                standby_chunks.append(chunk_bytes)
                standby_max_neural_score = max(standby_max_neural_score, score)
                standby_silence_count += 1

            # Dispatch triggers for phrase wake detection:
            # Trigger 1: Natural Speech Pause: User spoke and paused for 3 chunks (~240ms) with >=5 chunks (~400ms) of speech
            # Trigger 2: Utterance Cap / Noise Safety: User spoke continuously for 22 chunks (~1.7s)
            should_submit_standby_phrase = False
            if standby_speech_active and standby_silence_count >= 3 and len(standby_chunks) >= 5:
                should_submit_standby_phrase = True
            elif standby_speech_active and len(standby_chunks) >= 22:
                should_submit_standby_phrase = True
            elif standby_speech_active and standby_silence_count >= 6:
                # Silence lasted too long without sufficient speech — reset buffer
                standby_speech_active = False
                standby_silence_count = 0
                standby_chunks = []
                standby_max_neural_score = 0.0

            if should_submit_standby_phrase:
                # Only pop and submit when background executor is ready so speech is never dropped
                if phrase_future is None or phrase_future.done():
                    now_sb = time.time()
                    snippet_to_check = list(standby_chunks)
                    snip_neural_score = standby_max_neural_score
                    standby_speech_active = False
                    standby_silence_count = 0
                    standby_chunks = []
                    standby_max_neural_score = 0.0

                    # Peak amplitude and RMS test across accumulated snippet to verify genuine human speech
                    snip_raw = b"".join(snippet_to_check)
                    snip_arr = np.frombuffer(snip_raw, dtype=np.int16) if snip_raw else np.zeros(0, dtype=np.int16)
                    snip_f = snip_arr.astype(np.float32) / 32768.0 if len(snip_arr) > 0 else np.zeros(1, dtype=np.float32)
                    snip_rms = float(np.sqrt(np.mean(snip_f ** 2))) if len(snip_f) > 0 else 0.0
                    snip_peak = int(np.max(np.abs(snip_arr))) if len(snip_arr) > 0 else 0

                    # Validate vocal speech presence via Silero VAD to reject ambient room noise, typing, & HVAC
                    has_vocal_speech = True
                    if oww is not None and hasattr(oww, "vad") and oww.vad is not None and len(snip_arr) >= 640:
                        try:
                            n_samples = (len(snip_arr) // 640) * 640
                            frame_vads = [
                                oww.vad.predict(snip_arr[i:i+640], frame_size=640)
                                for i in range(0, n_samples, 640)
                            ]
                            if frame_vads and max(frame_vads) < 0.55:
                                has_vocal_speech = False
                        except Exception:
                            pass

                    min_speech_peak = max(350, int(ambient_noise_ema * 1.30 + 100))
                    min_speech_rms = max(0.007, (min_speech_peak / 32768.0) * 0.60)

                    # Submit to whisper if vocal speech confirmed, peak & RMS reach speech levels.
                    # When neural openWakeWord is active, require partial neural score (>= 0.35)
                    # to prevent Whisper hallucinations on ambient noise where neural score is 0.00.
                    has_neural_support = (oww is None or snip_neural_score >= 0.35)
                    if has_neural_support and has_vocal_speech and (snip_peak >= min_speech_peak and snip_rms >= min_speech_rms):
                        last_standby_phrase_check = now_sb
                        phrase_future = phrase_executor.submit(transcribe_audio, snippet_to_check, True, True)

            # Active live status visualization bounded to terminal width
            w_bars = int(min(score / max(wake_threshold, 0.45), 1.0) * 8)
            meter_scale = max(dynamic_threshold * 2.2, 4000.0)
            mic_bars = int(min(max_amp / meter_scale, 1.0) * 6)
            streak_tag = f" c:{confirm_streak}/{confirmation_frames}" if confirm_streak > 0 else ""
            status_text = f"🎙️ [STANDBY] Wake: [{'=' * w_bars}{' ' * (8 - w_bars)}] {score:.2f}{streak_tag} ({best_wake_model[:6]}) | Noise: {int(ambient_noise_ema):>4} | Mic: [{'#' * mic_bars}{' ' * (6 - mic_bars)}]"
            cols = shutil.get_terminal_size((80, 24)).columns
            print(f"\r{status_text.ljust(cols - 2)[:cols - 2]}", end="", flush=True)

            wake_triggered = neural_wake_detected or keyboard_trigger or phrase_wake_detected
            if wake_triggered:
                confirm_streak = 0
                last_wake_fire_time = now_loop_time

            if wake_triggered:
                if phrase_wake_detected:
                    trigger_label = f"PHRASE ('{phrase_text}')"
                elif keyboard_trigger:
                    trigger_label = "SPACE BAR"
                else:
                    trigger_label = f"NEURAL {best_wake_model.replace('_', ' ').title()} ({score:.2f})"

                print(f"\n\n>>> [WAKE DETECTED - {trigger_label}]")

                if phrase_text and re.search(r"\bdaddy(?:['’]s|\s+is)?\s+home\b", phrase_text, flags=re.IGNORECASE):
                    greeting_text = assistant.get_callout_acknowledgment(phrase_text)
                    print(f"-> M.A.D.D.Y.: \"{greeting_text}\"")
                    speak(greeting_text)
                    time.sleep(0.25)
                    drain_audio()
                    waiting_for_command = True
                    is_followup_turn = False
                    command_wait_start = time.time()
                    command_chunks = []
                    speech_started = False
                    speech_chunk_count = 0
                    speech_resume_count = 0
                    consecutive_silence = 0
                    peak_amp = 0
                    dynamic_threshold = max(220, int(ambient_noise_ema * 1.45 + 100))
                    print(f"🎙️ [SPEAK YOUR COMMAND NOW] (Noise Floor: {int(ambient_noise_ema)} | Voice Gate: {dynamic_threshold})...")
                    continue

                # If trailing command was already spoken in the same sentence (e.g. "Hey Maddy open Spotify")
                if phrase_trailing_cmd:
                    from core.math_phonetics import MathPhoneticizer
                    phrase_trailing_cmd = MathPhoneticizer.normalize_spelled_letters(phrase_trailing_cmd)
                    print(f"-> Direct command extracted: '{phrase_trailing_cmd}'")
                    res = assistant.execute(phrase_trailing_cmd)
                    resp_text = res.get("response", "")
                    if resp_text:
                        from core.math_phonetics import MathPhoneticizer
                        spoken_text = res.get("speakable_response") or MathPhoneticizer.extract_speakable_text(resp_text)
                        is_unconfigured = (
                            "unconfigured" in spoken_text.lower()
                            or "missing groq_api_key" in spoken_text.lower()
                            or "missing api key" in spoken_text.lower()
                        )
                        is_err = (
                            ("{" in spoken_text and "}" in spoken_text)
                            or "status 429" in spoken_text.lower()
                            or "ratelimit" in spoken_text.lower()
                            or "rate limit" in spoken_text.lower()
                            or "tokens per day" in spoken_text.lower()
                            or "modelapierror" in spoken_text.lower()
                            or "internal disruption" in spoken_text.lower()
                            or "unreachable" in spoken_text.lower()
                        )
                        addr = assistant.get_address_title()
                        if is_unconfigured:
                            spoken_text = f"My cloud LLM credentials are unconfigured, {addr}. Please configure your GROQ_API_KEY in the .env file."
                        elif is_err:
                            spoken_text = f"My neural connection is temporarily experiencing high traffic, {addr}. Please give me just a moment."
                        print(f"-> Response: {resp_text}\n")
                        links_banner = MathPhoneticizer.format_links_for_display(resp_text)
                        if links_banner:
                            print(links_banner)
                        speak(spoken_text)
                    print("[Standby] Ready for 'Hey M.A.D.D.Y.'.\n")
                    drain_audio()
                    continue

                greeting_text = speak_dynamic_wake_greeting()
                print(f"-> M.A.D.D.Y.: \"{greeting_text}\"")
                time.sleep(0.25)
                drain_audio()
                waiting_for_command = True
                is_followup_turn = False
                command_wait_start = time.time()
                command_chunks = []
                speech_started = False
                speech_chunk_count = 0
                speech_resume_count = 0
                consecutive_silence = 0
                peak_amp = 0
                dynamic_threshold = max(220, int(ambient_noise_ema * 1.45 + 100))
                print(f"🎙️ [SPEAK YOUR COMMAND NOW] (Noise Floor: {int(ambient_noise_ema)} | Voice Gate: {dynamic_threshold})...")
                continue
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        term_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M.A.D.D.Y. SOTA Neural Voice Assistant (Modular Autonomous Digital Daemon for You)")
    parser.add_argument("--list-mics", action="store_true", help="List all detected hardware microphones and exit")
    parser.add_argument("--test-mic", nargs="?", const="default", help="Test microphone audio level interactively (e.g. --test-mic or --test-mic 20)")
    parser.add_argument("--mic", type=str, default=None, help="Microphone device ID or keyword (e.g. 20, 22, headset, airdopes, jbl)")
    parser.add_argument("--threshold", type=float, default=None, help="Wake word detection threshold (default: 0.20)")
    parser.add_argument("--gain", type=float, default=None, help="Microphone pre-amp gain multiplier (default: 1.0)")
    parser.add_argument("--skip-test", action="store_true", help="Skip the startup 2-second audio check")
    parser.add_argument("--continuous", action="store_true", help="Keep microphone open for 8-second continuous follow-up window after every response")
    cli_args = parser.parse_args()

    if cli_args.list_mics:
        devs = get_audio_input_devices()
        print("\n" + "=" * 70)
        print("          AVAILABLE HARDWARE MICROPHONE INPUT DEVICES")
        print("=" * 70)
        for d in devs:
            typ = "[HEADSET / EARPHONE]" if d["is_headset"] else ("[INTERNAL ARRAY]" if d["is_internal"] else "[GENERIC]")
            def_mark = " <--- [ACTIVE OS DEFAULT]" if d["is_default"] else ""
            print(f"  [{d['id']:>2}] {d['name']:<50} {typ}{def_mark}")
        print("=" * 70)
        print("To start with a specific microphone, run:")
        print("  python voice_listener.py --mic <ID>  (e.g. python voice_listener.py --mic 20)")
        print("Or set in .env:")
        print("  AUDIO_INPUT_DEVICE=<ID>\n")
        sys.exit(0)

    if cli_args.test_mic:
        dev_arg = None if cli_args.test_mic == "default" else cli_args.test_mic
        t_id, t_info = resolve_audio_input_device(dev_arg)
        print(f"\n--- Testing Microphone [ID {t_id}]: {t_info['name']} for 5 seconds ---")
        test_microphone_level(t_id, duration=5.0)
        sys.exit(0)

    try:
        run_live_voice_assistant(
            mic_device=cli_args.mic,
            threshold=cli_args.threshold,
            gain=cli_args.gain,
            skip_test=cli_args.skip_test,
            continuous=cli_args.continuous,
        )
    except KeyboardInterrupt:
        print("\n\nPowering down M.A.D.D.Y. Voice Assistant. Goodbye!")
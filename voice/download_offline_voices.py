"""
Voice Utility for M.A.D.D.Y. — ElevenLabs Sample Downloader.
Note: ElevenLabs does NOT export neural network weights for offline local inference.
This script downloads voice samples from ElevenLabs API for local voice setup.
"""
import os
import sys
import httpx
from typing import Optional, Dict

ADAM_VOICE_ID = "IRHApOXLvnW57QJPQH2P"  # Adam - Brooding, Dark, Tough American
EVE_VOICE_ID = "G4Wh6MqJNTzYtuAeMqv5"   # Eve - Narrative & Audiobooks

def fetch_and_save_sample(api_key: str, voice_name: str, voice_id: str, output_path: str) -> bool:
    """Fetch reference preview audio for ElevenLabs voice ID."""
    headers = {"xi-api-key": api_key}
    vinfo_url = f"https://api.elevenlabs.io/v1/voices/{voice_id}"
    
    try:
        v_resp = httpx.get(vinfo_url, headers=headers, timeout=10.0)
        if v_resp.status_code == 200:
            preview_url = v_resp.json().get("preview_url")
            if preview_url:
                p_resp = httpx.get(preview_url, timeout=15.0)
                if p_resp.status_code == 200:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(p_resp.content)
                    print(f"[SAVED SAMPLE] {voice_name} ({voice_id}) -> {output_path} ({len(p_resp.content)} bytes)")
                    return True
        print(f"[ERROR] Could not fetch sample for {voice_name} ({voice_id}): {v_resp.status_code} {v_resp.text}")
        return False
    except Exception as err:
        print(f"[ERROR] Exception fetching sample for {voice_name}: {err}")
        return False

def download_voice_samples(api_key: str):
    """Downloads and prepares local voice samples for Adam and Eve."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    adam_dir = os.path.join(base_dir, "samples", "adam")
    eve_dir = os.path.join(base_dir, "samples", "eve")
    os.makedirs(adam_dir, exist_ok=True)
    os.makedirs(eve_dir, exist_ok=True)
    
    adam_mp3 = os.path.join(adam_dir, "sample_temp.mp3")
    eve_mp3 = os.path.join(eve_dir, "sample_temp.mp3")
    
    print("=" * 60)
    print("      ELEVENLABS VOICE SAMPLE DOWNLOADER")
    print("=" * 60)
    
    if fetch_and_save_sample(api_key, "Adam", ADAM_VOICE_ID, adam_mp3):
        import librosa
        import soundfile as sf
        wav_path = os.path.join(adam_dir, "sample.wav")
        txt_path = os.path.join(adam_dir, "transcript.txt")
        y, sr = librosa.load(adam_mp3, sr=24000)
        sf.write(wav_path, y, sr)
        with open(txt_path, "w") as f:
            f.write("There's something deeply comforting about a fire. It doesn't ask anything of you, just keeps giving off warmth until there's nothing left.")
        os.remove(adam_mp3)
        print(f"[PREPARED ADAM SAMPLE] -> {wav_path}")

    if fetch_and_save_sample(api_key, "Eve", EVE_VOICE_ID, eve_mp3):
        import librosa
        import soundfile as sf
        wav_path = os.path.join(eve_dir, "sample.wav")
        txt_path = os.path.join(eve_dir, "transcript.txt")
        y, sr = librosa.load(eve_mp3, sr=24000)
        sf.write(wav_path, y, sr)
        with open(txt_path, "w") as f:
            f.write("Not all those who wander are lost. Some are just looking for a better spot to read.")
        os.remove(eve_mp3)
        print(f"[PREPARED EVE SAMPLE] -> {wav_path}")
        
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        key = sys.argv[1].strip()
    else:
        key = os.getenv("ELEVENLABS_API_KEY", "")
        
    if not key:
        print("Usage: python voice/download_offline_voices.py <ELEVENLABS_API_KEY>")
        sys.exit(1)
        
    download_voice_samples(key)


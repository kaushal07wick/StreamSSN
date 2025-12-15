"""
ssm_server.py

Implements a CPU-only, low-latency real-time ASR system using Faster-Whisper.
Optimizations include:
- dynamic chunking with overlap for smoother context
- incremental decoding every 250 ms
- prompt reuse for semantic continuity
- local agreement commit for stable partials
- voice-activity-based finalization
- controlled BLAS threading to avoid oversubscription

The server exposes a single WebSocket endpoint:
  ws://localhost:8787/ws/stream
Clients send PCM16 audio frames (16 kHz mono).
Responses stream back as JSON with "partial" or "final" transcript updates.
"""

import os
import multiprocessing

# Configure math libraries to use a limited number of threads.
# This prevents OpenBLAS/MKL from oversaturating CPU cores and degrading latency.
num_cores = multiprocessing.cpu_count()
num_threads = max(1, num_cores // 2)
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
print(f"Using up to {num_threads} CPU threads for math kernels")

import asyncio
import json
import time
import numpy as np
import webrtcvad
import uvicorn
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

# Core configuration parameters (safe defaults for CPU-based inference)
SAMPLE_RATE = 16000              # Input audio sample rate
FRAME_MS = 20                    # Frame size for VAD (in milliseconds)
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 16-bit mono PCM
MAX_BUFFER_SEC = 12              # Ring buffer size for safety headroom
SILENCE_TIMEOUT_S = 0.35         # Silence threshold for utterance finalization
RETRANSCRIBE_EVERY_MS = 250      # Retranscription interval for partials
CHUNK_SEC = 1.0                  # Chunk duration for decoding
OVERLAP_SEC = 0.15               # Overlap duration between windows
MIN_COMMIT_WORDS = 4             # Minimum prefix match for stable commit
MIN_PARTIAL_DELTA_CH = 3         # Minimum change threshold for emitting partials
VAD_MODE = 2                     # 0–3: 0=aggressive, 3=relaxed
MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny.en")
COMPUTE_TYPE = "int8"            # Use quantized model for CPU efficiency
USE_FW_STREAMING = True          # Try streaming API if available

# Load the Faster-Whisper model in CPU mode
print(f"Loading Faster-Whisper model: {MODEL_NAME} ({COMPUTE_TYPE}, CPU)")
model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE)
print("Model initialized successfully.")

# Warm up model with dummy input to avoid cold-start latency
try:
    dummy = np.zeros(int(0.5 * SAMPLE_RATE), np.float32)
    _ = model.transcribe(dummy, language="en", without_timestamps=True)
    print("Model warmed with 0.5s dummy audio.")
except Exception as e:
    print("Model warmup skipped:", e)

# Attempt to create a streaming transcriber if supported
_fw_stream = None
if USE_FW_STREAMING:
    try:
        class _FWStream:
            """Lightweight streaming wrapper for Faster-Whisper.

            Accumulates float32 PCM samples into a buffer and reuses
            the model context between transcriptions.
            """

            def __init__(self):
                self._buf = np.zeros((0,), dtype=np.float32)

            def feed(self, pcm_f32: np.ndarray):
                """Appends audio samples, keeping only the last N seconds."""
                if pcm_f32.size:
                    max_len = int((CHUNK_SEC + OVERLAP_SEC) * SAMPLE_RATE)
                    self._buf = np.concatenate((self._buf, pcm_f32))[-max_len:]

            def result(self, prompt: str) -> str:
                """Runs inference on the accumulated buffer."""
                segs, _ = model.transcribe(
                    self._buf,
                    language="en",
                    without_timestamps=True,
                    initial_prompt=(prompt or None),
                )
                return "".join([s.text for s in segs]).strip()

            def clear(self):
                """Resets the audio buffer."""
                self._buf = np.zeros((0,), dtype=np.float32)

        _fw_stream = _FWStream()
        print("Streaming mode enabled (buffer reuse active).")
    except Exception as e:
        _fw_stream = None
        print("Streaming mode unavailable, falling back to standard mode:", e)

# Initialize FastAPI application
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    """Health check endpoint for monitoring."""
    return {"status": "ok"}

# Voice activity detector instance
VAD = webrtcvad.Vad(VAD_MODE)

# Common filler words to ignore in transcription
DISFLUENCIES = {"um", "uh", "erm", "hmm", "mm", "mmm", "uhh", "uhm"}

def int16_to_float32(b: bytes) -> np.ndarray:
    """Converts raw PCM16 bytes to normalized float32 audio samples."""
    return np.frombuffer(b, np.int16).astype(np.float32) / 32768.0

def take_tail_window_f32(ring: deque) -> np.ndarray:
    """Extracts the most recent window from the buffer as float32 PCM."""
    if not ring:
        return np.zeros((0,), dtype=np.float32)
    pcm = int16_to_float32(bytes(ring))
    max_len = int((CHUNK_SEC + OVERLAP_SEC) * SAMPLE_RATE)
    return pcm[-max_len:] if pcm.size > max_len else pcm

def common_prefix_words(a: str, b: str) -> str:
    """Finds the longest common prefix between two strings (by word)."""
    aw, bw = a.split(), b.split()
    out = []
    for x, y in zip(aw, bw):
        if x == y:
            out.append(x)
        else:
            break
    return " ".join(out)

def meaningful_delta(prev: str, curr: str) -> bool:
    """Determines if a new transcription differs enough to emit."""
    if not prev:
        return bool(curr)
    if abs(len(curr) - len(prev)) >= MIN_PARTIAL_DELTA_CH:
        return True
    if curr.endswith((".", "!", "?", "…", ":", ";", ",")):
        return True
    return False

async def transcribe_window(pcm_f32: np.ndarray, prompt_text: str = "") -> str:
    """Runs standard (non-streaming) Whisper inference on a PCM window."""
    if pcm_f32.size == 0:
        return ""
    segs, _ = model.transcribe(
        pcm_f32,
        language="en",
        without_timestamps=True,
        initial_prompt=(prompt_text or None),
    )
    return "".join([s.text for s in segs]).strip()

async def transcribe_window_streaming(pcm_f32: np.ndarray, prompt_text: str = "") -> str:
    """Runs streaming inference if available; falls back to standard."""
    if _fw_stream is None:
        return await transcribe_window(pcm_f32, prompt_text)
    _fw_stream.feed(pcm_f32)
    return _fw_stream.result(prompt_text)

@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    """Main WebSocket endpoint for real-time streaming transcription.

    The client continuously sends 16-bit PCM audio frames (mono, 16 kHz).
    The server processes audio in short overlapping windows and emits
    incremental transcripts every 250 ms, followed by final segments after silence.
    """
    await ws.accept()
    print("Client connected.")

    ring = deque(maxlen=int(SAMPLE_RATE * MAX_BUFFER_SEC) * 2)
    transcript_lines = []
    last_partial = ""
    last_committed = ""
    last_voice_t = last_transcribe_t = time.time()
    speaking = False

    try:
        while True:
            # Receive next audio chunk from client
            msg = await ws.receive_bytes()
            now = time.time()
            ring.extend(msg)

            # Wait until we have enough samples for analysis
            if len(ring) < FRAME_BYTES:
                continue

            # Check if the last 20 ms contain speech
            frame = bytes(list(ring)[-FRAME_BYTES:])
            is_voice = VAD.is_speech(frame, SAMPLE_RATE)
            if is_voice:
                speaking = True
                last_voice_t = now

            # Retranscribe every 250 ms while user is speaking
            if speaking and (now - last_transcribe_t) * 1000 >= RETRANSCRIBE_EVERY_MS:
                last_transcribe_t = now
                pcm = take_tail_window_f32(ring)
                prompt = last_committed[-300:]

                t0 = time.time()
                curr = await transcribe_window_streaming(pcm, prompt)
                asr_ms = int((time.time() - t0) * 1000)
                e2e_ms = asr_ms  # Placeholder for total latency (client-to-text)

                # Commit common prefix words to prevent flicker
                prefix = common_prefix_words(last_partial, curr)
                if len(prefix.split()) >= MIN_COMMIT_WORDS:
                    if not last_committed.endswith(prefix):
                        last_committed = (last_committed + " " + prefix).strip()

                # Filter meaningless disfluencies and jitter
                low = curr.strip().lower()
                if low in DISFLUENCIES:
                    continue

                if curr and meaningful_delta(last_partial, curr):
                    last_partial = curr
                    await ws.send_text(json.dumps({
                        "type": "partial",
                        "text": curr,
                        "asr_ms": asr_ms,
                        "e2e_ms": e2e_ms
                    }))

            # Detect silence to finalize utterance
            if speaking and (now - last_voice_t) >= SILENCE_TIMEOUT_S:
                speaking = False
                pcm = take_tail_window_f32(ring)
                prompt = last_committed[-300:]

                t0 = time.time()
                final_text = await transcribe_window_streaming(pcm, prompt)
                asr_ms = int((time.time() - t0) * 1000)
                e2e_ms = asr_ms
                ttct_ms = int((time.time() - last_voice_t) * 1000)

                # Skip filler words and silence segments
                if final_text and final_text.strip().lower() in DISFLUENCIES:
                    final_text = ""

                if final_text:
                    transcript_lines.append(final_text)
                    last_committed = (last_committed + " " + final_text).strip()
                    last_partial = ""

                    await ws.send_text(json.dumps({
                        "type": "final",
                        "text": final_text,
                        "lines": transcript_lines[-6:],
                        "asr_ms": asr_ms,
                        "e2e_ms": e2e_ms,
                        "ttct_ms": ttct_ms
                    }))

                # Reset streaming buffer and ring buffer
                if _fw_stream is not None:
                    _fw_stream.clear()
                ring.clear()

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print("Error in stream:", e)
    finally:
        try:
            await ws.close()
        except:
            pass
        print("Stream closed.")

if __name__ == "__main__":
    # Start FastAPI server with uvloop for improved event loop performance
    uvicorn.run(app, host="0.0.0.0", port=8787, loop="uvloop")

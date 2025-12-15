"""
ssm_server.py

Streaming Speech-to-Text server using Faster-Whisper + FastAPI + WebRTC-VAD.

Features:
  • CPU-only, low-latency adaptive chunking and overlap
  • Incremental decode on rolling audio buffer
  • Smart sentence-based prompt reuse
  • Early finalization for sub-100ms TTCT
  • Local prefix commit to stabilize partials
  • Single-concurrency decode with thread pinning
"""

import os
import asyncio
import json
import time
from collections import deque
from typing import Optional

import numpy as np
import uvicorn
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

# Thread pinning before BLAS loads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

 
# Configuration
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2

VAD_MODE = 2
SILENCE_TIMEOUT_S = 0.45
EARLY_FINAL_MS = 120

MIN_CHUNK_SEC = 0.5
MAX_CHUNK_SEC = 1.8
START_CHUNK_SEC = 0.8

MIN_OVERLAP_SEC = 0.05
MAX_OVERLAP_SEC = 0.18
START_OVERLAP_SEC = 0.10

RETRANSCRIBE_BASE_MS = 600
RETRANSCRIBE_FAST_MS = 350
RETRANSCRIBE_SLOW_MS = 900
MAX_PARTIAL_HZ = 4

TRIM_TRIGGER_SEC = 15.0
TRIM_KEEP_SEC = 10.0
MAX_BUFFER_SEC = 10

PROMPT_CHARS_MAX = 200
PROMPT_LOOKBACK_CHARS = 400
MIN_COMMIT_WORDS = 4
MIN_PARTIAL_DELTA_CH = 3
DISFLUENCIES = {"um", "uh", "erm", "hmm", "mm", "mmm", "uhh", "uhm"}

MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny.en")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")

 
# Model
print(f"Loading Faster-Whisper: {MODEL_NAME} ({COMPUTE_TYPE}, CPU)")
model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE)
print("Model ready.")

try:
    _ = model.transcribe(np.zeros(int(0.5 * SAMPLE_RATE), np.float32),
                         language="en", without_timestamps=True)
    print("Model warmed with 0.5s dummy audio.")
except Exception as e:
    print("Warmup skipped:", e)

 
# Utilities
dumps = json.dumps

def int16_to_float32(b: bytes) -> np.ndarray:
    """Convert PCM16 bytes to float32 array."""
    if not b:
        return np.zeros(0, np.float32)
    return np.frombuffer(b, np.int16).astype(np.float32) / 32768.0


def common_prefix_words(a: str, b: str) -> str:
    """Find common prefix by words."""
    aw, bw = a.split(), b.split()
    out = []
    for x, y in zip(aw, bw):
        if x == y:
            out.append(x)
        else:
            break
    return " ".join(out)


def meaningful_delta(prev: str, curr: str) -> bool:
    """Return True if new text differs enough to emit."""
    if not prev:
        return bool(curr)
    if abs(len(curr) - len(prev)) >= MIN_PARTIAL_DELTA_CH:
        return True
    if curr.endswith((".", "!", "?", "…", ":", ";", ",")):
        return True
    return False


def smart_prompt(committed: str) -> str:
    """Extract a sentence-aware suffix for prompt reuse."""
    if not committed:
        return ""
    tail = committed[-PROMPT_LOOKBACK_CHARS:]
    for sep in [".", "!", "?", "…"]:
        idx = tail.rfind(sep)
        if idx != -1 and idx + 1 < len(tail):
            tail = tail[idx + 1:].lstrip()
            break
    return tail[-PROMPT_CHARS_MAX:]


 
# Voice Activity Gate
class VoiceGate:
    """WebRTC-based speech activity tracker."""

    def __init__(self, mode: int = VAD_MODE):
        self.vad = webrtcvad.Vad(mode)
        self.is_voice_last = False
        self.silence_since: Optional[float] = None
        self.last_voice_t = time.time()
        self.in_utterance = False

    def feed(self, last_bytes: bytes) -> None:
        """Update speech state from latest PCM frame."""
        if len(last_bytes) < FRAME_BYTES:
            return
        frame = last_bytes[-FRAME_BYTES:]
        try:
            is_voice = self.vad.is_speech(frame, SAMPLE_RATE)
        except Exception:
            is_voice = True
        now = time.time()
        if is_voice:
            self.last_voice_t = now
            self.silence_since = None
            if not self.in_utterance:
                self.in_utterance = True
        else:
            if self.silence_since is None:
                self.silence_since = now
        self.is_voice_last = is_voice

    def silent_for(self) -> float:
        """Return duration of current silence."""
        return 0.0 if self.silence_since is None else (time.time() - self.silence_since)


 
# Streaming ASR
class StreamingASR:
    """Adaptive rolling buffer with dynamic chunk and overlap."""

    def __init__(self):
        self._f32 = np.zeros(0, np.float32)
        self.chunk_s = START_CHUNK_SEC
        self.overlap_s = START_OVERLAP_SEC
        self._lock = asyncio.Semaphore(1)

    def feed(self, chunk_b: bytes) -> None:
        """Append new PCM16 chunk to buffer."""
        pcm = int16_to_float32(chunk_b)
        if pcm.size == 0:
            return
        self._f32 = np.concatenate((self._f32, pcm))
        if self._f32.size > int(TRIM_TRIGGER_SEC * SAMPLE_RATE):
            keep = int(TRIM_KEEP_SEC * SAMPLE_RATE)
            self._f32 = self._f32[-keep:]

    def tail_window(self) -> np.ndarray:
        """Return last (chunk + overlap) seconds of audio."""
        if self._f32.size == 0:
            return np.zeros(0, np.float32)
        win = int((self.chunk_s + self.overlap_s) * SAMPLE_RATE)
        return self._f32[-win:] if self._f32.size > win else self._f32

    def adapt_chunk(self, speaking: bool, silence_s: float, changed: bool) -> None:
        """Adjust chunk length based on activity."""
        dc = self.chunk_s
        if speaking:
            dc = min(MAX_CHUNK_SEC, dc * (1.18 if changed else 0.97))
        else:
            dc = max(MIN_CHUNK_SEC, dc * (0.80 if silence_s >= (SILENCE_TIMEOUT_S / 2) else 0.90))
        self.chunk_s = dc

    def adapt_overlap(self, prefix_words: int, changed: bool) -> None:
        """Adjust overlap for stability."""
        ov = self.overlap_s
        if changed and prefix_words < MIN_COMMIT_WORDS:
            ov = min(MAX_OVERLAP_SEC, ov * 1.20)
        elif prefix_words >= MIN_COMMIT_WORDS * 2:
            ov = max(MIN_OVERLAP_SEC, ov * 0.9)
        self.overlap_s = ov

    async def transcribe_once(self, prompt: str = "") -> str:
        """Run a single model inference."""
        pcm = self.tail_window()
        if pcm.size == 0:
            return ""
        async with self._lock:
            segs, _ = model.transcribe(
                pcm,
                language="en",
                without_timestamps=True,
                initial_prompt=(prompt or None),
            )
        return "".join(s.text for s in segs).strip()

    def clear(self) -> None:
        """Reset buffer and parameters."""
        self._f32 = np.zeros(0, np.float32)
        self.chunk_s = START_CHUNK_SEC
        self.overlap_s = START_OVERLAP_SEC


 
# FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    """Main WebSocket endpoint for streaming ASR."""
    await ws.accept()
    print("Client connected.")

    ring = deque(maxlen=int(SAMPLE_RATE * MAX_BUFFER_SEC) * 2)
    vad = VoiceGate()
    asr = StreamingASR()

    transcript_lines: list[str] = []
    last_committed = ""
    last_partial = ""
    last_partial_sent_t = 0.0
    last_transcribe_t = time.time()
    last_partial_change_t = 0.0
    retranscribe_ms = RETRANSCRIBE_BASE_MS

    try:
        while True:
            msg = await ws.receive_bytes()
            ring.extend(msg)
            rb = bytes(ring)
            vad.feed(rb)
            asr.feed(msg)

            now = time.time()
            speaking = vad.in_utterance and vad.is_voice_last
            silence_s = vad.silent_for()
            due = (now - last_transcribe_t) * 1000 >= retranscribe_ms
            rate_ok = (now - last_partial_sent_t) >= (1.0 / MAX_PARTIAL_HZ)

            # Partial decode
            if speaking and due and rate_ok:
                last_transcribe_t = now
                prompt = smart_prompt(last_committed)
                t0 = time.time()
                curr = await asr.transcribe_once(prompt)
                asr_ms = int((time.time() - t0) * 1000)
                prefix = common_prefix_words(last_partial, curr)
                prefix_words = len(prefix.split())
                changed = curr and meaningful_delta(last_partial, curr) and (curr.strip().lower() not in DISFLUENCIES)

                asr.adapt_chunk(True, silence_s, bool(changed))
                asr.adapt_overlap(prefix_words, bool(changed))

                if prefix_words >= MIN_COMMIT_WORDS and not last_committed.endswith(prefix):
                    last_committed = (last_committed + " " + prefix).strip()

                retranscribe_ms = RETRANSCRIBE_FAST_MS if changed else RETRANSCRIBE_SLOW_MS

                if changed:
                    last_partial = curr
                    last_partial_change_t = now
                    last_partial_sent_t = now
                    await ws.send_text(dumps({
                        "type": "partial",
                        "text": curr,
                        "asr_ms": asr_ms,
                        "win_s": round(asr.chunk_s, 2),
                        "ovl_s": round(asr.overlap_s, 2)
                    }))

            # Early finalization
            if vad.in_utterance and not vad.is_voice_last:
                silent_ms = int(silence_s * 1000)
                since_change_ms = int((now - last_partial_change_t) * 1000)
                if silent_ms >= EARLY_FINAL_MS and since_change_ms >= EARLY_FINAL_MS:
                    prompt = smart_prompt(last_committed)
                    t0 = time.time()
                    final_text = await asr.transcribe_once(prompt)
                    asr_ms = int((time.time() - t0) * 1000)
                    ttct_ms = silent_ms
                    if final_text and final_text.strip().lower() not in DISFLUENCIES:
                        transcript_lines.append(final_text)
                        last_committed = (last_committed + " " + final_text).strip()
                        last_partial = ""
                        await ws.send_text(dumps({
                            "type": "final",
                            "text": final_text,
                            "lines": transcript_lines[-6:],
                            "asr_ms": asr_ms,
                            "ttct_ms": ttct_ms
                        }))
                    asr.clear()
                    ring.clear()
                    vad.in_utterance = False
                    vad.silence_since = None
                    retranscribe_ms = RETRANSCRIBE_BASE_MS
                    continue

            # Hard finalization
            if vad.in_utterance and silence_s >= SILENCE_TIMEOUT_S:
                prompt = smart_prompt(last_committed)
                t0 = time.time()
                final_text = await asr.transcribe_once(prompt)
                asr_ms = int((time.time() - t0) * 1000)
                ttct_ms = int(silence_s * 1000)
                if final_text and final_text.strip().lower() not in DISFLUENCIES:
                    transcript_lines.append(final_text)
                    last_committed = (last_committed + " " + final_text).strip()
                    last_partial = ""
                    await ws.send_text(dumps({
                        "type": "final",
                        "text": final_text,
                        "lines": transcript_lines[-6:],
                        "asr_ms": asr_ms,
                        "ttct_ms": ttct_ms
                    }))
                asr.clear()
                ring.clear()
                vad.in_utterance = False
                vad.silence_since = None
                retranscribe_ms = RETRANSCRIBE_BASE_MS

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Stream error: {e}")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        print("Stream closed.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8787, loop="uvloop", reload=False)

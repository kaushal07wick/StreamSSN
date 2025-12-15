"""
real_time_asr_optimized.py
CPU-only, low-latency streaming ASR with:
- dynamic chunking (short window + overlap)
- local agreement commit
- prompt context reuse
- tighter retranscribe cadence
- warmup + thread pinning
- optional Faster-Whisper streaming API (safe fallback)

API unchanged: /ws/stream emits {"type":"partial"|"final", ...}
"""

# ---- Thread pinning BEFORE imports that load BLAS ----
# ---- Controlled parallelism BEFORE imports that load BLAS ----
import os
import multiprocessing

# detect core count
num_cores = multiprocessing.cpu_count()
# limit to physical cores if hyperthreading is on (≈ half)
num_threads = max(1, num_cores // 2)

os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

print(f"⚙️ Using up to {num_threads} CPU threads for math kernels")

import asyncio, json, time, webrtcvad, uvicorn, numpy as np
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ===== Tunables (safe defaults for CPU) =====
SAMPLE_RATE            = 16000
FRAME_MS               = 20
FRAME_BYTES            = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 16-bit mono
MAX_BUFFER_SEC         = 12                 # headroom
SILENCE_TIMEOUT_S      = 0.35               # faster finals
RETRANSCRIBE_EVERY_MS  = 250                # smoother cadence (less churn)
CHUNK_SEC              = 1.0                # window length
OVERLAP_SEC            = 0.15               # tail overlap to stabilize boundaries
MIN_COMMIT_WORDS       = 4                  # local agreement threshold
MIN_PARTIAL_DELTA_CH   = 3                  # ignore tiny jitter
VAD_MODE               = 2                  # 0..3 (aggressive->relaxed)
MODEL_NAME             = os.getenv("WHISPER_MODEL", "tiny.en")   # keep CPU friendly
COMPUTE_TYPE           = "int8"             # CPU int8
USE_FW_STREAMING       = True               # try Faster-Whisper streaming API if available

# ===== Model =====
from faster_whisper import WhisperModel

print(f"🔄 Loading Faster-Whisper {MODEL_NAME} ({COMPUTE_TYPE}, CPU)...")
model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE)
print("✅ Faster-Whisper ready on CPU.")

# Warm the model once to avoid a slow first inference
try:
    _ = model.transcribe(np.zeros(int(0.5*SAMPLE_RATE), np.float32), language="en", without_timestamps=True)
    print("🔥 Warmed model with 0.5s dummy audio.")
except Exception as _e:
    print("⚠️ Warmup skipped:", _e)

# Try to prepare a streaming transcriber if supported (safe fallback)
_fw_stream = None
if USE_FW_STREAMING:
    try:
        # Not all versions expose a streaming helper.
        # We simulate a simple stream wrapper that accumulates audio
        # and reuses options; falls back to normal path if anything fails.
        class _FWStream:
            def __init__(self):
                self._buf = np.zeros((0,), dtype=np.float32)
            def feed(self, pcm_f32: np.ndarray):
                if pcm_f32.size:
                    self._buf = np.concatenate((self._buf, pcm_f32))[-int((CHUNK_SEC+OVERLAP_SEC)*SAMPLE_RATE):]
            def result(self, prompt: str) -> str:
                segs, _ = model.transcribe(self._buf, language="en", without_timestamps=True,
                                           initial_prompt=(prompt or None))
                return "".join([s.text for s in segs]).strip()
            def clear(self):
                self._buf = np.zeros((0,), dtype=np.float32)

        _fw_stream = _FWStream()
        print("🧪 Streaming facade enabled (buffer-reuse).")
    except Exception as _e:
        _fw_stream = None
        print("ℹ️ Streaming facade unavailable; using standard transcribe().", _e)

# ===== FastAPI =====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health(): return {"status": "ok"}

# ===== Helpers =====
VAD = webrtcvad.Vad(VAD_MODE)
DISFLUENCIES = {"um", "uh", "erm", "hmm", "mm", "mmm", "uhh", "uhm"}

def int16_to_float32(b: bytes) -> np.ndarray:
    return np.frombuffer(b, np.int16).astype(np.float32) / 32768.0

def take_tail_window_f32(ring: deque) -> np.ndarray:
    """Return sliding window with overlap from the current ring buffer as float32 PCM."""
    if not ring:
        return np.zeros((0,), dtype=np.float32)
    pcm = int16_to_float32(bytes(ring))
    max_len = int((CHUNK_SEC + OVERLAP_SEC) * SAMPLE_RATE)
    if pcm.size > max_len:
        pcm = pcm[-max_len:]
    return pcm

def common_prefix_words(a: str, b: str) -> str:
    aw, bw = a.split(), b.split()
    out = []
    for x, y in zip(aw, bw):
        if x == y: out.append(x)
        else: break
    return " ".join(out)

def meaningful_delta(prev: str, curr: str) -> bool:
    if not prev: return bool(curr)
    if abs(len(curr) - len(prev)) >= MIN_PARTIAL_DELTA_CH: return True
    if curr.endswith((".", "!", "?", "…", ":", ";", ",")): return True
    return False

async def transcribe_window(pcm_f32: np.ndarray, prompt_text: str = "") -> str:
    """Standard (non-stream) transcribe."""
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
    """Use streaming facade if available; otherwise fallback."""
    if _fw_stream is None:
        return await transcribe_window(pcm_f32, prompt_text)
    _fw_stream.feed(pcm_f32)
    return _fw_stream.result(prompt_text)

# ===== WebSocket =====
@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    print("🎧 stream connected")

    ring = deque(maxlen=int(SAMPLE_RATE * MAX_BUFFER_SEC) * 2)  # bytes
    transcript_lines: list[str] = []
    last_partial = ""
    last_committed = ""   # committed transcript (prompt context)
    last_voice_t = last_transcribe_t = time.time()
    speaking = False

    try:
        while True:
            msg = await ws.receive_bytes()
            now = time.time()
            ring.extend(msg)

            if len(ring) < FRAME_BYTES:
                continue

            # VAD over last 20 ms frame
            frame = bytes(list(ring)[-FRAME_BYTES:])
            is_voice = VAD.is_speech(frame, SAMPLE_RATE)
            if is_voice:
                speaking = True
                last_voice_t = now

            # periodic partials while speaking
            if speaking and (now - last_transcribe_t) * 1000 >= RETRANSCRIBE_EVERY_MS:
                last_transcribe_t = now
                pcm = take_tail_window_f32(ring)
                prompt = last_committed[-300:]

                t0 = time.time()
                curr = await transcribe_window_streaming(pcm, prompt)
                asr_ms = max(0, int((time.time() - t0) * 1000))
                e2e_ms = asr_ms

                # local agreement with previous partial
                prefix = common_prefix_words(last_partial, curr)
                if len(prefix.split()) >= MIN_COMMIT_WORDS:
                    if not last_committed.endswith(prefix):
                        last_committed = (last_committed + " " + prefix).strip()

                # De-noise disfluencies & jitter
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

            # silence → finalize
            if speaking and (now - last_voice_t) >= SILENCE_TIMEOUT_S:
                speaking = False
                pcm = take_tail_window_f32(ring)
                prompt = last_committed[-300:]

                t0 = time.time()
                final_text = await transcribe_window_streaming(pcm, prompt)
                asr_ms = max(0, int((time.time() - t0) * 1000))
                e2e_ms = asr_ms
                ttct_ms = max(0, int((time.time() - last_voice_t) * 1000))

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

                # reset streaming buffer if enabled
                if _fw_stream is not None:
                    _fw_stream.clear()
                ring.clear()

    except WebSocketDisconnect:
        print("🔌 client disconnected")
    except Exception as e:
        print(f"⚠️ {e}")
    finally:
        try:
            await ws.close()
        except:
            pass
        print("🧹 stream closed")

if __name__ == "__main__":
    # uvloop is fine; keep defaults if not installed
    uvicorn.run(app, host="0.0.0.0", port=8787, loop="uvloop")

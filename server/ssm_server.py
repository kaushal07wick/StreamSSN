"""
real_time_asr_optimized.py
CPU-only, low-latency streaming ASR with:
- dynamic chunking (sliding window + overlap)
- local agreement commit
- prompt context reuse
- tighter retranscribe cadence
API unchanged: /ws/stream sends {"type":"partial"| "final", ...}
"""

import os, asyncio, json, time, webrtcvad, uvicorn, numpy as np
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

# ===== Tunables (safe defaults) =====
SAMPLE_RATE            = 16000
FRAME_MS               = 20
MAX_BUFFER_SEC         = 12                 # extra headroom
SILENCE_TIMEOUT_S      = 0.35               # faster finals
RETRANSCRIBE_EVERY_MS  = 180                # snappier partials
CHUNK_SEC              = 1.6                # window length for dynamic chunk
OVERLAP_SEC            = 0.30               # overlap tail to stabilize boundaries
MIN_COMMIT_WORDS       = 4                  # local agreement threshold
MIN_PARTIAL_DELTA_CH   = 3                  # ignore tiny jitter
VAD_MODE               = 2                  # 0..3 (aggressive->relaxed)
MODEL_NAME             = os.getenv("WHISPER_MODEL", "tiny.en")  # keep CPU friendly
COMPUTE_TYPE           = "int8"             # CPU int8

VAD = webrtcvad.Vad(VAD_MODE)
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 16-bit mono bytes per frame

# ===== Model =====
print(f"🔄 Loading Faster-Whisper {MODEL_NAME} ({COMPUTE_TYPE}, CPU)...")
model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE)
print("✅ Faster-Whisper ready on CPU.")

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
def int16_to_float32(b: bytes) -> np.ndarray:
    return np.frombuffer(b, np.int16).astype(np.float32) / 32768.0

def take_tail_window(ring: deque) -> np.ndarray:
    """Return sliding window with overlap from the current ring buffer."""
    raw = bytes(ring)                        # full bytes in ring
    pcm = int16_to_float32(raw)
    if pcm.size == 0:
        return pcm
    max_len = int((CHUNK_SEC + OVERLAP_SEC) * SAMPLE_RATE)
    if pcm.size > max_len:
        pcm = pcm[-max_len:]
    return pcm

def common_prefix_words(a: str, b: str) -> str:
    """Word-level common prefix (exact match)."""
    aw, bw = a.split(), b.split()
    out = []
    for x, y in zip(aw, bw):
        if x == y:
            out.append(x)
        else:
            break
    return " ".join(out)

def meaningful_delta(prev: str, curr: str) -> bool:
    if not prev: return bool(curr)
    if abs(len(curr) - len(prev)) >= MIN_PARTIAL_DELTA_CH: return True
    # punctuation end helps commit
    if curr.endswith((".", "!", "?", "…", ":", ";", ",")): return True
    return False

DISFLUENCIES = {"um", "uh", "erm", "hmm", "mm", "mmm", "uhh", "uhm"}

async def transcribe_window(pcm: np.ndarray, prompt_text: str = "") -> str:
    if pcm.size == 0:
        return ""
    segs, _ = model.transcribe(
        pcm,
        language="en",
        without_timestamps=True,
        initial_prompt=(prompt_text or None),
    )
    return "".join([s.text for s in segs]).strip()

# ===== WebSocket =====
@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    print("🎧 stream connected")

    ring = deque(maxlen=int(SAMPLE_RATE * MAX_BUFFER_SEC) * 2)  # bytes buffer
    transcript_lines: list[str] = []
    last_partial = ""
    last_committed = ""   # running committed transcript (for prompt context)
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

                # dynamic chunk: overlap + window
                pcm = take_tail_window(ring)

                # prompt context: last ~300 chars of committed text
                prompt = last_committed[-300:]

                t0 = time.time()
                curr = await transcribe_window(pcm, prompt)
                asr_ms = max(0, int((time.time() - t0) * 1000))
                e2e_ms = asr_ms

                # local agreement with previous partial
                prefix = common_prefix_words(last_partial, curr)
                if len(prefix.split()) >= MIN_COMMIT_WORDS:
                    # Commit the stable prefix once. Keep the rest as live text.
                    # We don't emit a "final" here; we just advance last_committed silently.
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

                # finalize current window with overlap to stabilize tail
                pcm = take_tail_window(ring)
                prompt = last_committed[-300:]

                t0 = time.time()
                final_text = await transcribe_window(pcm, prompt)
                asr_ms = max(0, int((time.time() - t0) * 1000))
                e2e_ms = asr_ms
                ttct_ms = max(0, int((time.time() - last_voice_t) * 1000))

                # filter trivial finals
                if final_text:
                    # prune pure disfluency finals
                    if final_text.strip().lower() in DISFLUENCIES:
                        final_text = ""

                if final_text:
                    transcript_lines.append(final_text)
                    # update committed context
                    last_committed = (last_committed + " " + final_text).strip()
                    # reset rolling partial
                    last_partial = ""

                    await ws.send_text(json.dumps({
                        "type": "final",
                        "text": final_text,
                        "lines": transcript_lines[-6:],
                        "asr_ms": asr_ms,
                        "e2e_ms": e2e_ms,
                        "ttct_ms": ttct_ms
                    }))

                ring.clear()  # start fresh for next utterance

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
    # keep the same defaults you’re already using
    uvicorn.run(app, host="0.0.0.0", port=8787, loop="uvloop")

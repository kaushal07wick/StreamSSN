import asyncio, json, time, webrtcvad, uvicorn, numpy as np
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

# ===== Config =====
SAMPLE_RATE       = 16000
FRAME_MS          = 20
SILENCE_TIMEOUT_S = 1.0
RETRANSCRIBE_EVERY_MS = 700     # how often we re-run model
MAX_BUFFER_SEC    = 8

VAD = webrtcvad.Vad(2)
frame_bytes = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 16-bit mono

print("🔄 Loading Faster-Whisper model (tiny.en, int8)...")
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
print("✅ Faster-Whisper ready.")

# ===== FastAPI =====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health(): return {"status": "ok"}

def int16_to_float32(b: bytes) -> np.ndarray:
    """Convert 16-bit PCM -> float32"""
    return np.frombuffer(b, np.int16).astype(np.float32) / 32768.0

@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    print("🎧 stream connected")

    ring = deque(maxlen=int(SAMPLE_RATE * MAX_BUFFER_SEC) * 2)
    last_voice_t = last_transcribe_t = time.time()
    speaking = False
    transcript_lines = []
    last_text = ""

    try:
        while True:
            msg = await ws.receive_bytes()
            now = time.time()
            ring.extend(msg)
            if len(ring) < frame_bytes:
                continue

            frame = bytes(list(ring)[-frame_bytes:])
            is_voice = VAD.is_speech(frame, SAMPLE_RATE)
            if is_voice:
                speaking = True
                last_voice_t = now

            # retranscribe every N ms
            if (now - last_transcribe_t) * 1000 >= RETRANSCRIBE_EVERY_MS:
                last_transcribe_t = now
                pcm = int16_to_float32(bytes(ring))
                t0 = time.time()
                segs, _ = model.transcribe(pcm, language="en", without_timestamps=True)
                text = "".join([s.text for s in segs]).strip()
                asr_ms = max(0, int((time.time() - t0) * 1000))
                e2e_ms = max(0, asr_ms)
                if text and text != last_text:
                    last_text = text
                    await ws.send_text(json.dumps({
                        "type": "partial",
                        "text": text,
                        "asr_ms": asr_ms,
                        "e2e_ms": e2e_ms
                    }))

            # end of utterance (silence)
            if speaking and (now - last_voice_t) >= SILENCE_TIMEOUT_S:
                speaking = False
                pcm = int16_to_float32(bytes(ring))
                t0 = time.time()
                segs, _ = model.transcribe(pcm, language="en", without_timestamps=True)
                final_text = "".join([s.text for s in segs]).strip()
                asr_ms = max(0, int((time.time() - t0) * 1000))
                e2e_ms = max(0, asr_ms)
                if final_text:
                    transcript_lines.append(final_text)
                    await ws.send_text(json.dumps({
                        "type": "final",
                        "text": final_text,
                        "lines": transcript_lines[-6:],
                        "asr_ms": asr_ms,
                        "e2e_ms": e2e_ms
                    }))
                    last_text = ""
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
    uvicorn.run(app, host="0.0.0.0", port=8787)

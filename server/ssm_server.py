import asyncio, json, time, numpy as np, uvicorn, concurrent.futures, webrtcvad
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from inference import engine

# ===== Config =====
SAMPLE_RATE = 16000
FRAME_MS = 20
CHUNK_MS = 200                  # process every 0.2s for near real-time
VAD = webrtcvad.Vad(2)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# ===== FastAPI =====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

def int16_to_float32(b: bytes) -> np.ndarray:
    return np.frombuffer(b, np.int16).astype(np.float32) / 32768.0

async def run_transcribe(pcm_f32: np.ndarray):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: engine.transcribe(pcm_f32))

@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    print("🎧 stream connected")

    frame_bytes = int(SAMPLE_RATE * FRAME_MS / 1000) * 2
    chunk_bytes = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2
    ring = deque(maxlen=int(SAMPLE_RATE * 4) * 2)  # 4 s rolling window

    inflight = False
    buffer_time = time.time()

    async def process_loop():
        nonlocal inflight
        while True:
            await asyncio.sleep(CHUNK_MS / 1000.0)
            if inflight or len(ring) < SAMPLE_RATE * 0.5 * 2:
                continue
            inflight = True
            pcm = int16_to_float32(bytes(ring))
            t0 = time.time()
            text, asr_ms = await run_transcribe(pcm)
            inflight = False
            if text.strip():
                await ws.send_text(json.dumps({
                    "type": "partial",
                    "text": text.strip(),
                    "asr_ms": int((time.time() - t0) * 1000),
                }))

    asyncio.create_task(process_loop())

    try:
        while True:
            data = await ws.receive_bytes()
            ring.extend(data)

            # quick VAD to drop complete silence
            if len(data) >= frame_bytes:
                frame = data[-frame_bytes:]
                if not VAD.is_speech(frame, SAMPLE_RATE):
                    continue
    except WebSocketDisconnect:
        print("🔌 disconnected")
    except Exception as e:
        print(f"⚠️ error: {e}")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        print("🧹 closed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8787)
import asyncio, json, time, numpy as np, uvicorn, concurrent.futures, webrtcvad
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from inference import engine

# ===== Config =====
SAMPLE_RATE = 16000
FRAME_MS = 20
CHUNK_MS = 200                  # process every 0.2s for near real-time
VAD = webrtcvad.Vad(2)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# ===== FastAPI =====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

def int16_to_float32(b: bytes) -> np.ndarray:
    return np.frombuffer(b, np.int16).astype(np.float32) / 32768.0

async def run_transcribe(pcm_f32: np.ndarray):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: engine.transcribe(pcm_f32))

@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    print("🎧 stream connected")

    frame_bytes = int(SAMPLE_RATE * FRAME_MS / 1000) * 2
    chunk_bytes = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2
    ring = deque(maxlen=int(SAMPLE_RATE * 4) * 2)  # 4 s rolling window

    inflight = False
    buffer_time = time.time()

    async def process_loop():
        nonlocal inflight
        while True:
            await asyncio.sleep(CHUNK_MS / 1000.0)
            if inflight or len(ring) < SAMPLE_RATE * 0.5 * 2:
                continue
            inflight = True
            pcm = int16_to_float32(bytes(ring))
            t0 = time.time()
            text, asr_ms = await run_transcribe(pcm)
            inflight = False
            if text.strip():
                await ws.send_text(json.dumps({
                    "type": "partial",
                    "text": text.strip(),
                    "asr_ms": int((time.time() - t0) * 1000),
                }))

    asyncio.create_task(process_loop())

    try:
        while True:
            data = await ws.receive_bytes()
            ring.extend(data)

            # quick VAD to drop complete silence
            if len(data) >= frame_bytes:
                frame = data[-frame_bytes:]
                if not VAD.is_speech(frame, SAMPLE_RATE):
                    continue
    except WebSocketDisconnect:
        print("🔌 disconnected")
    except Exception as e:
        print(f"⚠️ error: {e}")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        print("🧹 closed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8787)

import asyncio, json, time, numpy as np, uvicorn, concurrent.futures, webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

SAMPLE_RATE = 16000
VAD = webrtcvad.Vad(2)

print("🔄 Loading Whisper model (small.en, int8)...", flush=True)
model = WhisperModel("small.en", device="cpu", compute_type="int8", num_workers=4)
print("✅ Model preloaded and ready.", flush=True)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

@app.get("/health")
def health():
    return {"status": "ok"}

async def async_transcribe(audio: np.ndarray):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: model.transcribe(audio, language="en", without_timestamps=True)
    )

@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    buf = bytearray()
    last = time.time()

    try:
        while True:
            msg = await ws.receive_bytes()
            recv_t = time.time()
            buf.extend(msg)

            if len(buf) > int(0.2 * SAMPLE_RATE) * 2 and (time.time() - last) > 0.2:
                audio = np.frombuffer(buf, np.int16).astype(np.float32) / 32768.0
                t0 = time.time()
                segs, _ = await async_transcribe(audio)
                asr_time = int((time.time() - t0) * 1000)
                text = "".join(s.text for s in segs).strip()
                buf = buf[-int(0.5 * SAMPLE_RATE) * 2:]
                last = time.time()
                e2e = int((time.time() - recv_t) * 1000)
                await ws.send_text(json.dumps({
                    "type": "partial",
                    "text": text,
                    "asr_ms": asr_time,
                    "e2e_ms": e2e
                }))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"⚠️ {e}", flush=True)
    finally:
        await ws.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8787)

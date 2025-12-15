# Streaming Whisper STT on Modal (GPU / H100 build)
# - Faster-Whisper on CUDA (float16 precision)
# - WebSocket /ws for streaming 16-bit PCM audio
# - Emits {"type": "partial"|"final", "asr_ms", "e2e_ms", "ttct_ms"}

import asyncio
import json
import multiprocessing
import os
import time
from collections import deque
from typing import Deque, Optional

import modal

APP_NAME = "stream-ssm-whisper-gpu"
app = modal.App(APP_NAME)

# ---------- Controlled parallelism ----------
NUM_CORES = multiprocessing.cpu_count()
NUM_THREADS = max(1, NUM_CORES // 2)
# ---------- Modal image ----------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "build-essential")
    .pip_install(
        "torch==2.5.1+cu124",
        "torchvision==0.20.1+cu124",
        "torchaudio==2.5.1+cu124",
        "fastapi==0.115.6",
        "uvicorn==0.32.1",
        "webrtcvad==2.0.10",
        "numpy==2.1.3",
        "faster-whisper==1.0.3",
        "requests",
        extra_options=["--extra-index-url", "https://download.pytorch.org/whl/cu124"],
    )
    .env(
        {
            "OMP_NUM_THREADS": str(NUM_THREADS),
            "OPENBLAS_NUM_THREADS": str(NUM_THREADS),
            "MKL_NUM_THREADS": str(NUM_THREADS),
            "NUMEXPR_NUM_THREADS": str(NUM_THREADS),
            "WHISPER_MODEL": os.environ.get("WHISPER_MODEL", "small.en"),
            "WHISPER_COMPUTE_TYPE": "float16",
        }
    )
)


# ---------- Tunables ----------
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2
MAX_BUFFER_SEC = 12
SILENCE_TIMEOUT_S = 0.35
RETRANSCRIBE_EVERY_MS = 250
CHUNK_SEC = 1.0
OVERLAP_SEC = 0.15
MIN_COMMIT_WORDS = 4
MIN_PARTIAL_DELTA_CH = 3
VAD_MODE = 2
DISFLUENCIES = {"um", "uh", "erm", "hmm", "mm", "mmm", "uhh", "uhm"}

# ============================================================

@app.cls(image=image, gpu="h100", timeout=600)
class WhisperSTT:
    """GPU-accelerated streaming ASR (Whisper) with WebSocket endpoint."""

    @modal.enter()
    def _load(self):
        """Load Faster-Whisper on GPU and initialize VAD."""
        from faster_whisper import WhisperModel
        import webrtcvad
        import numpy as np
        import torch

        self.model_name = os.getenv("WHISPER_MODEL", "small.en")
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading {self.model_name} ({self.compute_type}) on {device}...")
        t0 = time.time()
        self.model = WhisperModel(
            self.model_name,
            device=device,
            compute_type=self.compute_type,
        )
        print(f"✅ Model loaded in {time.time() - t0:.2f}s")

        # Warmup
        try:
            _ = self.model.transcribe(
                np.zeros(int(0.5 * SAMPLE_RATE), np.float32),
                language="en",
                without_timestamps=True,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            print("🔥 Warmed up model with 0.5 s dummy input.")
        except Exception as e:
            print("⚠️ Warmup skipped:", e)

        self.vad = webrtcvad.Vad(VAD_MODE)

    # ---------- Utility helpers ----------
    def _int16_to_float32(self, b: bytes):
        import numpy as np
        return np.frombuffer(b, np.int16).astype(np.float32) / 32768.0

    def _take_tail_window_f32(self, ring: Deque[int]):
        import numpy as np
        if not ring:
            return np.zeros((0,), dtype=np.float32)
        pcm = self._int16_to_float32(bytes(ring))
        max_len = int((CHUNK_SEC + OVERLAP_SEC) * SAMPLE_RATE)
        if pcm.size > max_len:
            pcm = pcm[-max_len:]
        return pcm

    def _common_prefix_words(self, a: str, b: str) -> str:
        aw, bw = a.split(), b.split()
        out = []
        for x, y in zip(aw, bw):
            if x == y:
                out.append(x)
            else:
                break
        return " ".join(out)

    def _meaningful_delta(self, prev: str, curr: str) -> bool:
        if not prev:
            return bool(curr)
        if abs(len(curr) - len(prev)) >= MIN_PARTIAL_DELTA_CH:
            return True
        if curr.endswith((".", "!", "?", "…", ":", ";", ",")):
            return True
        return False

    def _transcribe_once(self, pcm_f32, prompt_text: str = "") -> str:
        """One transcription pass."""
        segs, _ = self.model.transcribe(
            pcm_f32,
            language="en",
            without_timestamps=True,
            initial_prompt=(prompt_text or None),
        )
        return "".join([s.text for s in segs]).strip()

    # ---------- FastAPI WebSocket ----------
    @modal.asgi_app()
    def api(self):
        """Expose FastAPI app with /health and /ws endpoints."""
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import JSONResponse

        app = FastAPI()

        @app.get("/health")
        def health():
            return JSONResponse({"status": "ok"})

        @app.websocket("/ws")
        async def ws_stream(ws: WebSocket):
            await ws.accept()
            print("🎧 WS connected")

            ring: Deque[int] = deque(maxlen=int(SAMPLE_RATE * MAX_BUFFER_SEC) * 2)
            transcript_lines = []
            last_partial = ""
            last_committed = ""
            last_voice_t = last_transcribe_t = time.time()
            speaking = False

            try:
                while True:
                    data: Optional[bytes] = await ws.receive_bytes()
                    now = time.time()
                    ring.extend(data)

                    if len(ring) < FRAME_BYTES:
                        continue

                    frame = bytes(list(ring)[-FRAME_BYTES:])
                    is_voice = self.vad.is_speech(frame, SAMPLE_RATE)
                    if is_voice:
                        speaking = True
                        last_voice_t = now

                    # periodic partials
                    if speaking and (now - last_transcribe_t) * 1000 >= RETRANSCRIBE_EVERY_MS:
                        last_transcribe_t = now
                        recv_t = now
                        pcm = self._take_tail_window_f32(ring)
                        prompt = last_committed[-300:]

                        t0 = time.time()
                        curr = self._transcribe_once(pcm, prompt)
                        t1 = time.time()

                        asr_ms = int((t1 - t0) * 1000)
                        e2e_ms = int((t1 - recv_t) * 1000)

                        prefix = self._common_prefix_words(last_partial, curr)
                        if len(prefix.split()) >= MIN_COMMIT_WORDS:
                            if not last_committed.endswith(prefix):
                                last_committed = (last_committed + " " + prefix).strip()

                        if curr.strip().lower() in DISFLUENCIES:
                            continue

                        if curr and self._meaningful_delta(last_partial, curr):
                            last_partial = curr
                            await ws.send_text(
                                json.dumps(
                                    {
                                        "type": "partial",
                                        "text": curr,
                                        "asr_ms": asr_ms,
                                        "e2e_ms": e2e_ms,
                                    }
                                )
                            )

                    # silence → finalize
                    if speaking and (now - last_voice_t) >= SILENCE_TIMEOUT_S:
                        speaking = False
                        recv_t = now
                        pcm = self._take_tail_window_f32(ring)
                        prompt = last_committed[-300:]

                        t0 = time.time()
                        final_text = self._transcribe_once(pcm, prompt)
                        t1 = time.time()

                        asr_ms = int((t1 - t0) * 1000)
                        e2e_ms = int((t1 - recv_t) * 1000)
                        ttct_ms = int((t1 - last_voice_t) * 1000)

                        if final_text.strip().lower() in DISFLUENCIES:
                            final_text = ""

                        if final_text:
                            transcript_lines.append(final_text)
                            last_committed = (last_committed + " " + final_text).strip()
                            last_partial = ""

                            await ws.send_text(
                                json.dumps(
                                    {
                                        "type": "final",
                                        "text": final_text,
                                        "lines": transcript_lines[-6:],
                                        "asr_ms": asr_ms,
                                        "e2e_ms": e2e_ms,
                                        "ttct_ms": ttct_ms,
                                    }
                                )
                            )

                        ring.clear()

            except WebSocketDisconnect:
                print("🔌 WS disconnected")
                try:
                    await ws.close()
                except Exception:
                    pass
            except Exception as e:
                print("⚠️ WS error:", e)
                try:
                    await ws.close(code=1011)
                except Exception:
                    pass

            return app

        return app

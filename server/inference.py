# inference.py
import time
import numpy as np
from faster_whisper import WhisperModel

class InferenceEngine:
    def __init__(self, model_size="small.en"):
        print(f"🔄 Loading Whisper model ({model_size}, int8)...", flush=True)
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            num_workers=4,
        )
        print("✅ Model loaded and ready.", flush=True)

    def transcribe(self, pcm: np.ndarray, initial_prompt: str | None = None):
        """Transcribe a PCM segment (float32, 16 kHz)."""
        t0 = time.time()
        segments, _ = self.model.transcribe(
            pcm,
            language="en",
            beam_size=1,
            condition_on_previous_text=True,
            initial_prompt=initial_prompt or "",
            without_timestamps=True,
        )
        text = " ".join(s.text for s in segments).strip()
        return text, int((time.time() - t0) * 1000)

engine = InferenceEngine()

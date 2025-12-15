# **Stream-SSM: Real-Time Speech Stream Model**

A high-performance, **CPU-optimized real-time ASR backend** built on **FastAPI** and **Faster-Whisper**.
Stream-SSM runs dynamic chunked transcription, prompt reuse, and adaptive silence gating — achieving near-GPU latency **entirely on CPU**.

The system is engineered as an open counterpart to **Cartesia / Ink-Whisper** stacks, designed for **low-latency, incremental decoding** in live pipelines and research environments.

![image](stream.png)

---

## ⚙️ **Core Optimizations**

| Optimization                  | Description                                                                                                 |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Dynamic Chunking**          | Adaptive 1 s window with 150 ms overlap for stable text boundaries.                                         |
| **250 ms Retranscribe Cycle** | Continuous micro-window transcription for smooth partials.                                                  |
| **Local Agreement Commit**    | Emits only after multi-word prefix consensus to eliminate jitter.                                           |
| **Prompt Context Reuse**      | Carries recent transcript prefixes across chunks for contextual decoding.                                   |
| **VAD-Triggered Finals**      | Detects 350 ms of silence to finalize utterances automatically.                                             |
| **Thread-Controlled BLAS**    | Dynamically pins OpenBLAS/MKL threads to half CPU cores — prevents oversubscription and stabilizes latency. |
| **Int8 Inference Path**       | Uses Faster-Whisper INT8 compute type for 3-4× faster CPU throughput.                                       |
| **Streaming Facade**          | Simulated continuous decode buffer to reduce context resets.                                                |
| **Warm Model Boot**           | Pre-loads with 0.5 s dummy audio to eliminate first-inference lag.                                          |

---

## ⚡ **Measured Performance (Intel i3-12100, 8 Threads, 16 GB RAM)**

| Metric                 | Typical Value  | Description               |
| ---------------------- | -------------- | ------------------------- |
| **Partial Latency**    | 250–400 ms     | Speech → partial text     |
| **Finalization Delay** | ≤ 350 ms       | Silence → confirmed final |
| **Throughput**         | 6–8× real-time | On single physical core   |
| **RAM Usage**          | < 1 GB         | Including buffers + model |
| **Startup**            | 2–3 s          | Full FastAPI + model warm |

---

## 🛰️ **API**

**Endpoint:**
`ws://localhost:8787/ws/stream`

**Input:**
16-bit PCM (mono, 16 kHz) frames sent as binary chunks.

**Output:**
JSON events streamed back to the client:

```json
{"type": "partial", "text": "hello wor", "asr_ms": 280, "e2e_ms": 310}
{"type": "final", "text": "hello world", "asr_ms": 295, "e2e_ms": 340, "ttct_ms": 370}
```

**Fields:**

* `asr_ms`: model inference latency only
* `e2e_ms`: full path latency (audio → text emission)
* `ttct_ms`: silence-to-text closure time

---

## 🧰 **Run Locally**

```bash
pip install -r requirements.txt
python real_time_asr_optimized.py
```

Optional model selection:

```bash
export WHISPER_MODEL=tiny.en     # or base.en / small.en
```

Access the health endpoint at:

```bash
curl http://localhost:8787/health
```

---

## 🧩 **Integration**

Stream-SSM pairs seamlessly with a **Go gRPC gateway**, which:

* Handles thousands of concurrent WebSocket/gRPC streams
* Routes sessions to multiple Python ASR workers
* Aggregates and exports latency metrics (`asr_ms`, `e2e_ms`, `ttct_ms`) to Prometheus/Grafana

This separation of control (Go) and inference (Python) delivers **low-latency scaling** under real-time load.

---

## 🔭 **Use Cases**

* Benchmarking **CPU-only real-time ASR** on constrained edge hardware
* Researching **incremental decoding and context reuse** in Whisper models
* Building **speech-driven control systems** or live AI agents
* Comparing **transformer vs. state-space streaming architectures**

---

## 🧠 **Planned Extensions**

* GPU-accelerated modal runtime with adaptive batching
* Learned silence boundary predictor
* Real-time state-space hybrid inference (Stream-SSM v3)
* Cross-language incremental decoding support

---

> **Stream-SSM** — engineered for real-time, latency-tight, model-agnostic speech understanding.
> Built for edge. Tuned for speed. Designed for truth-to-speech precision.

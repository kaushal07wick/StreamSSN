"use client";
import { useRef, useState, useEffect } from "react";

export default function Page() {
  const [connected, setConn] = useState(false);
  const [text, setText] = useState("");
  const [asr, setAsr] = useState(0);
  const [e2e, setE2e] = useState(0);
  const [latencyHistory, setLatencyHistory] = useState<number[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // append new latency values
  useEffect(() => {
    if (e2e > 0 && connected) {
      setLatencyHistory((prev) => [...prev.slice(-39), e2e]);
    }
  }, [e2e]);

  async function start() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;
    const ctx = new AudioContext();
    ctxRef.current = ctx;
    await ctx.audioWorklet.addModule("/worklet-processor.js");

    const ws = new WebSocket("ws://localhost:8787/ws/stream");
    ws.binaryType = "arraybuffer";
    ws.onopen = () => setConn(true);
    ws.onclose = () => setConn(false);
    ws.onmessage = (e) => {
      try {
        const m = JSON.parse(e.data);
        if (m.type === "partial") {
          setText(m.text);
          setAsr(m.asr_ms);
          setE2e(m.e2e_ms);
        }
      } catch {}
    };
    wsRef.current = ws;

    const node = new AudioWorkletNode(ctx, "pcm-processor");
    node.port.onmessage = (event) => {
      if (wsRef.current && wsRef.current.readyState === 1)
        wsRef.current.send(event.data);
    };

    const src = ctx.createMediaStreamSource(stream);
    src.connect(node);
    node.connect(ctx.destination);
  }

  function stop() {
    wsRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    ctxRef.current?.close();
    setConn(false);
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-6">
      <div className="w-full max-w-2xl">
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-2 tracking-tight">
            StreamSSM
          </h1>
          <p className="text-purple-300 text-lg">
            Real-Time Speech Recognition
          </p>
        </div>

        <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 shadow-2xl border border-white/20">
          <div className="flex justify-center items-center mb-8">
            <VoicePulse
              connected={connected}
              audioCtx={ctxRef.current}
              stream={streamRef.current}
            />
          </div>

          <div className="text-center mb-6">
            <div
              className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${
                connected
                  ? "bg-green-500/20 text-green-300"
                  : "bg-gray-500/20 text-gray-300"
              }`}
            >
              <div
                className={`w-2 h-2 rounded-full ${
                  connected ? "bg-green-400 animate-pulse" : "bg-gray-400"
                }`}
              />
              <span className="text-sm font-medium">
                {connected ? "Listening" : "Standby"}
              </span>
            </div>
          </div>

          <div className="bg-black/20 rounded-2xl p-6 mb-6 min-h-[100px] flex items-center justify-center">
            <p className="text-white text-xl text-center leading-relaxed">
              {text || "Start speaking to see your words appear here..."}
            </p>
          </div>

          {/* Latency Graph — stays visible after stop */}
          {latencyHistory.length > 0 && (
            <div className="mb-6">
              <h3 className="text-white text-sm font-semibold mb-3 text-center">
                End-to-End Latency
              </h3>
              <LatencyGraph data={latencyHistory} />
            </div>
          )}

          <div className="grid grid-cols-2 gap-4 mb-6">
            <MetricCard label="ASR Latency" value={`${asr}ms`} color="blue" />
            <MetricCard label="E2E Latency" value={`${e2e}ms`} color="purple" />
          </div>

          <div className="flex justify-center">
            {!connected ? (
              <button
                onClick={start}
                className="px-8 py-4 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl font-semibold text-lg hover:from-green-600 hover:to-emerald-700 transition-all shadow-lg hover:shadow-xl transform hover:scale-105"
              >
                Start Recording
              </button>
            ) : (
              <button
                onClick={stop}
                className="px-8 py-4 bg-gradient-to-r from-red-500 to-rose-600 text-white rounded-xl font-semibold text-lg hover:from-red-600 hover:to-rose-700 transition-all shadow-lg hover:shadow-xl transform hover:scale-105"
              >
                Stop Recording
              </button>
            )}
          </div>
        </div>

        <div className="text-center mt-6 text-purple-300 text-sm">
          Powered by Whisper + WebGPU
        </div>
      </div>
    </main>
  );
}

/* === VOICE PULSE VISUALIZER === */
function VoicePulse({
  connected,
  audioCtx,
  stream,
}: {
  connected: boolean;
  audioCtx: AudioContext | null;
  stream: MediaStream | null;
}) {
  const [levels, setLevels] = useState<number[]>(new Array(10).fill(1));

  useEffect(() => {
    if (!connected || !audioCtx || !stream) return;
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 64;
    const src = audioCtx.createMediaStreamSource(stream);
    src.connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);
    let raf: number;

    const update = () => {
      analyser.getByteFrequencyData(data);
      const chunk = Math.floor(data.length / 10);
      const bars = Array.from({ length: 10 }, (_, i) =>
        data
          .slice(i * chunk, (i + 1) * chunk)
          .reduce((a, b) => a + b, 0) / chunk / 255
      );
      setLevels(bars);
      raf = requestAnimationFrame(update);
    };
    update();

    return () => cancelAnimationFrame(raf);
  }, [connected, audioCtx, stream]);

  return (
    <div className="flex items-end justify-center gap-1 h-10 mb-6">
      {levels.map((lvl, i) => (
        <div
          key={i}
          className="w-[5px] bg-purple-400 rounded-sm transition-all duration-100"
          style={{ height: `${Math.max(4, lvl * 100)}%` }}
        ></div>
      ))}
    </div>
  );
}

/* === LATENCY GRAPH === */
function LatencyGraph({ data }: { data: number[] }) {
  const maxValue = Math.max(...data, 100);
  const width = 100;
  const height = 80;

  const points = data
    .map((value, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = height - (value / maxValue) * height;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className="bg-black/20 rounded-xl p-4">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-24"
        preserveAspectRatio="none"
      >
        <line
          x1="0"
          y1={height / 2}
          x2={width}
          y2={height / 2}
          stroke="rgba(255,255,255,0.1)"
          strokeWidth="0.5"
        />
        <polygon
          points={`0,${height} ${points} ${width},${height}`}
          fill="url(#gradient)"
          opacity="0.3"
        />
        <polyline
          points={points}
          fill="none"
          stroke="#a78bfa"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <defs>
          <linearGradient id="gradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.5" />
            <stop offset="100%" stopColor="#a78bfa" stopOpacity="0" />
          </linearGradient>
        </defs>
      </svg>
      <div className="flex justify-between mt-2 text-xs text-gray-400">
        <span>0ms</span>
        <span>{Math.round(maxValue)}ms</span>
      </div>
    </div>
  );
}

/* === METRICS === */
function MetricCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  const colorClasses = {
    blue: "from-blue-500/20 to-blue-600/20 border-blue-500/30",
    purple: "from-purple-500/20 to-purple-600/20 border-purple-500/30",
  };

  return (
    <div
      className={`bg-gradient-to-br ${
        colorClasses[color as keyof typeof colorClasses]
      } border rounded-xl p-4 backdrop-blur-sm`}
    >
      <p className="text-gray-300 text-xs font-medium mb-1">{label}</p>
      <p className="text-white text-2xl font-bold">{value}</p>
    </div>
  );
}

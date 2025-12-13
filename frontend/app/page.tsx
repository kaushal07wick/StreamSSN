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
        } else if (m.type === "final") {
          setText(m.text);            // lock in final
          setAsr(m.asr_ms);
          setE2e(m.e2e_ms);
          // optionally push latencyHistory here
          setLatencyHistory(prev => [...prev.slice(-39), m.e2e_ms]);
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
    <main className="min-h-screen bg-[#e8e7e4] flex items-center justify-center px-6 py-10 font-mono">
      <div className="w-full max-w-3xl bg-[#101010] border border-[#2a2a2a] rounded-2xl p-10 shadow-[0_0_40px_rgba(0,0,0,0.3)]">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-5xl font-extrabold tracking-tight text-[#ff4400] uppercase mb-2">
            StreamSSM
          </h1>
          <p className="text-gray-400 text-sm tracking-widest uppercase">
            Defense Speech Interface — Rev 2.1
          </p>
        </div>

        {/* Pulse + Status */}
        <div className="flex flex-col items-center mb-8">
          <VoicePulse connected={connected} audioCtx={ctxRef.current} stream={streamRef.current} />
          <div
            className={`mt-4 inline-flex items-center gap-2 px-4 py-2 border ${
              connected ? "border-[#ff4400] text-[#ff4400]" : "border-gray-600 text-gray-500"
            } rounded-md uppercase text-xs tracking-widest`}
          >
            <div
              className={`w-2 h-2 rounded-full ${
                connected ? "bg-[#ff4400]" : "bg-gray-600"
              }`}
            />
            {connected ? "Live Feed Active" : "Standby"}
          </div>
        </div>

        {/* Transcript */}
        <div className="bg-[#181818] border border-[#333] rounded-xl p-6 mb-8 min-h-[120px] flex items-center justify-center">
          <p className="text-gray-100 text-xl leading-relaxed text-center">
            {text || "Awaiting transmission..."}
          </p>
        </div>

        {/* Latency Graph */}
        {latencyHistory.length > 0 && (
          <div className="mb-8">
            <h3 className="text-[#ff4400] text-xs uppercase tracking-widest mb-2 text-center">
              End-to-End Latency (ms)
            </h3>
            <LatencyGraph data={latencyHistory} />
          </div>
        )}

        {/* Metrics */}
        <div className="grid grid-cols-2 gap-4 mb-10">
          <MetricCard label="ASR Latency" value={`${asr} ms`} />
          <MetricCard label="E2E Latency" value={`${e2e} ms`} />
        </div>

        {/* Button */}
        <div className="flex justify-center">
          {!connected ? (
            <button
              onClick={start}
              className="px-8 py-4 bg-[#ff4400] text-black font-bold rounded-md hover:bg-[#ff6633] transition-all uppercase tracking-wider"
            >
              Engage
            </button>
          ) : (
            <button
              onClick={stop}
              className="px-8 py-4 bg-gray-900 text-[#ff4400] border border-[#ff4400] font-bold rounded-md hover:bg-[#1a1a1a] transition-all uppercase tracking-wider"
            >
              Terminate
            </button>
          )}
        </div>

        {/* Footer */}
        <p className="text-center text-xs text-gray-600 mt-8 tracking-widest uppercase">
          Engineered by Whisper + WebGPU Systems
        </p>
      </div>
    </main>
  );
}

/* === VoicePulse === */
function VoicePulse({ connected, audioCtx, stream }: { connected: boolean; audioCtx: AudioContext | null; stream: MediaStream | null }) {
  const [levels, setLevels] = useState<number[]>(new Array(14).fill(0.5));

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
      const chunk = Math.floor(data.length / 14);
      const bars = Array.from({ length: 14 }, (_, i) =>
        data.slice(i * chunk, (i + 1) * chunk).reduce((a, b) => a + b, 0) / chunk / 255
      );
      setLevels(bars);
      raf = requestAnimationFrame(update);
    };
    update();
    return () => cancelAnimationFrame(raf);
  }, [connected, audioCtx, stream]);

  return (
    <div className="flex items-end justify-center gap-1 h-12">
      {levels.map((lvl, i) => (
        <div
          key={i}
          className="w-[6px] bg-[#ff4400] transition-all duration-75"
          style={{ height: `${Math.max(8, lvl * 80)}%` }}
        ></div>
      ))}
    </div>
  );
}

/* === LatencyGraph === */
function LatencyGraph({ data }: { data: number[] }) {
  const maxValue = Math.max(...data, 100);
  const width = 100;
  const height = 60;
  const points = data.map((v, i) => `${(i / (data.length - 1)) * width},${height - (v / maxValue) * height}`).join(" ");

  return (
    <div className="bg-[#121212] border border-[#333] rounded-md p-3">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-20" preserveAspectRatio="none">
        <line x1="0" y1={height / 2} x2={width} y2={height / 2} stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />
        <polyline points={points} fill="none" stroke="#ff4400" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    </div>
  );
}

/* === Metric === */
function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-[#181818] border border-[#333] rounded-md p-4 text-center">
      <p className="text-gray-500 text-xs uppercase tracking-widest mb-1">{label}</p>
      <p className="text-[#ff4400] text-xl font-bold">{value}</p>
    </div>
  );
}

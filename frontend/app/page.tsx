"use client";
import { useRef, useState, useEffect } from "react";

export default function Page() {
  const [connected, setConnected] = useState(false);
  const [text, setText] = useState("");
  const [asr, setAsr] = useState(0);
  const [e2e, setE2e] = useState(0);
  const [latencyHistory, setLatencyHistory] = useState<number[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    if (connected && e2e > 0) {
      setLatencyHistory((prev) => [...prev.slice(-99), e2e]);
    }
  }, [e2e, connected]);

  async function start() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const ctx = new AudioContext();
      ctxRef.current = ctx;
      await ctx.audioWorklet.addModule("/worklet-processor.js");

      const ws = new WebSocket("ws://localhost:8787/ws/stream");
      ws.binaryType = "arraybuffer";

      ws.onopen = () => setConnected(true);
      ws.onclose = () => setConnected(false);
      ws.onerror = () => setConnected(false);

      ws.onmessage = (e) => {
        try {
          const m = JSON.parse(e.data);
          if (m.type === "partial" || m.type === "final") {
            setText(m.text || "");
            setAsr(m.asr_ms ?? 0);
            setE2e(m.e2e_ms ?? 0);
          }
        } catch {}
      };

      wsRef.current = ws;

      const node = new AudioWorkletNode(ctx, "pcm-processor");
      node.port.onmessage = (event) => {
        const data = event.data;
        if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(data);
      };

      const src = ctx.createMediaStreamSource(stream);
      src.connect(node);
      node.connect(ctx.destination);
    } catch (err) {
      console.error("Audio init failed:", err);
      setConnected(false);
    }
  }

  function stop() {
    wsRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    ctxRef.current?.close();
    setConnected(false);
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-[#050505] via-[#0a0a0a] to-[#0f0505] flex items-center justify-center px-6 py-10 font-mono text-gray-100">
      <div className="w-full max-w-4xl bg-black/60 backdrop-blur-sm border border-[#ff4400]/20 rounded-lg p-8 shadow-[0_0_80px_rgba(255,68,0,0.15)]">
        {/* Header with Grid Pattern */}
        <div className="relative text-center mb-8 pb-6 border-b border-[#ff4400]/20">
          <div className="absolute inset-0 opacity-5" style={{
            backgroundImage: `repeating-linear-gradient(0deg, transparent, transparent 2px, #ff4400 2px, #ff4400 3px),
                             repeating-linear-gradient(90deg, transparent, transparent 2px, #ff4400 2px, #ff4400 3px)`,
            backgroundSize: '20px 20px'
          }}></div>
          <h1 className="relative text-6xl font-black tracking-tighter text-[#ff4400] mb-1" style={{
            textShadow: '0 0 20px rgba(255,68,0,0.5), 0 0 40px rgba(255,68,0,0.2)'
          }}>
            STREAMSSM
          </h1>
          <p className="relative text-[#ff4400]/60 text-xs tracking-[0.3em] uppercase font-light">
            Defense Speech Interface <span className="text-[#ff4400]/40">—</span> Rev 2.1
          </p>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Left Column: Voice Pulse + Status */}
          <div className="lg:col-span-1 flex flex-col gap-4">
            <div className="bg-gradient-to-br from-[#1a0a00] to-[#0a0a0a] border border-[#ff4400]/30 rounded-lg p-6 flex flex-col items-center justify-center">
              <div className="mb-4 text-[#ff4400]/50 text-[10px] tracking-widest uppercase">
                Audio Input
              </div>
              <VoicePulse connected={connected} audioCtx={ctxRef.current} stream={streamRef.current} />
              <div className={`mt-6 inline-flex items-center gap-2 px-4 py-2 border ${
                connected ? "border-[#ff4400] text-[#ff4400] bg-[#ff4400]/5" : "border-gray-700 text-gray-600 bg-gray-900/20"
              } rounded-md uppercase text-[10px] tracking-widest transition-all duration-300`}>
                <div className={`w-2 h-2 rounded-full ${connected ? "bg-[#ff4400] animate-pulse" : "bg-gray-700"}`} />
                {connected ? "Live" : "Standby"}
              </div>
            </div>

            {/* Metrics Cards */}
            <div className="grid grid-cols-2 gap-3">
              <MetricCard label="ASR" value={`${asr}`} unit="ms" connected={connected} />
              <MetricCard label="E2E" value={`${e2e}`} unit="ms" connected={connected} />
            </div>
          </div>

          {/* Right Column: Transcript + Graph */}
          <div className="lg:col-span-2 flex flex-col gap-4">
            {/* Transcript */}
            <div className="bg-gradient-to-br from-[#0f0f0f] to-[#050505] border border-[#333] rounded-lg p-6 min-h-[140px] flex items-center justify-center relative overflow-hidden">
              <div className="absolute top-2 left-3 text-[#ff4400]/30 text-[9px] tracking-widest uppercase">
                Transcript
              </div>
              <p className="text-gray-100 text-lg leading-relaxed text-center whitespace-pre-wrap relative z-10 mt-4">
                {text || <span className="text-gray-600 italic">Awaiting transmission...</span>}
              </p>
              {connected && (
                <div className="absolute bottom-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#ff4400]/30 to-transparent"></div>
              )}
            </div>

            {/* Latency Graph */}
            <div className="bg-gradient-to-br from-[#0f0f0f] to-[#050505] border border-[#333] rounded-lg p-4 relative overflow-hidden">
              <div className="absolute top-2 left-3 text-[#ff4400]/30 text-[9px] tracking-widest uppercase">
                End-to-End Latency Timeline
              </div>
              <LatencyGraph data={latencyHistory} connected={connected} />
            </div>
          </div>
        </div>

        {/* Control Button */}
        <div className="flex justify-center pt-4">
          {!connected ? (
            <button
              onClick={start}
              className="group relative px-10 py-4 bg-[#ff4400] text-black font-bold rounded-md overflow-hidden transition-all duration-300 hover:shadow-[0_0_30px_rgba(255,68,0,0.6)] uppercase tracking-wider"
            >
              <span className="relative z-10">Engage</span>
              <div className="absolute inset-0 bg-gradient-to-r from-[#ff6600] to-[#ff4400] opacity-0 group-hover:opacity-100 transition-opacity"></div>
            </button>
          ) : (
            <button
              onClick={stop}
              className="group px-10 py-4 bg-transparent text-[#ff4400] border-2 border-[#ff4400] font-bold rounded-md hover:bg-[#ff4400]/10 transition-all duration-300 uppercase tracking-wider"
            >
              Terminate
            </button>
          )}
        </div>

        {/* Footer */}
        <p className="text-center text-[10px] text-gray-700 mt-8 tracking-widest uppercase">
          Powered by Whisper + WebGPU
        </p>
      </div>
    </main>
  );
}

/* === VoicePulse === */
function VoicePulse({
  connected,
  audioCtx,
  stream,
}: {
  connected: boolean;
  audioCtx: AudioContext | null;
  stream: MediaStream | null;
}) {
  const [levels, setLevels] = useState<number[]>(new Array(20).fill(0.3));

  useEffect(() => {
    if (!connected || !audioCtx || !stream) {
      setLevels(new Array(20).fill(0.3));
      return;
    }
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 128;
    analyser.smoothingTimeConstant = 0.7;
    const src = audioCtx.createMediaStreamSource(stream);
    src.connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);
    let raf: number;

    const update = () => {
      analyser.getByteFrequencyData(data);
      const chunk = Math.floor(data.length / 20);
      const bars = Array.from({ length: 20 }, (_, i) => {
        const avg = data.slice(i * chunk, (i + 1) * chunk).reduce((a, b) => a + b, 0) / chunk / 255;
        return Math.max(0.1, avg);
      });
      setLevels(bars);
      raf = requestAnimationFrame(update);
    };
    update();
    return () => cancelAnimationFrame(raf);
  }, [connected, audioCtx, stream]);

  return (
    <div className="flex items-end justify-center gap-[3px] h-20 w-full px-4">
      {levels.map((lvl, i) => (
        <div
          key={i}
          className="flex-1 rounded-t-sm transition-all duration-100 ease-out"
          style={{ 
            height: `${Math.max(10, lvl * 100)}%`, 
            background: connected 
              ? `linear-gradient(to top, #ff4400, #ff6600)`
              : 'rgba(100,100,100,0.3)',
            opacity: connected ? 0.8 + lvl * 0.2 : 0.4,
            boxShadow: connected && lvl > 0.5 ? `0 0 10px rgba(255,68,0,${lvl * 0.6})` : 'none'
          }}
        ></div>
      ))}
    </div>
  );
}

/* === Improved Streaming LatencyGraph === */
function LatencyGraph({ data, connected }: { data: number[]; connected: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [stats, setStats] = useState({ min: 0, max: 0, avg: 0 });

  useEffect(() => {
    if (data.length > 0) {
      const min = Math.min(...data);
      const max = Math.max(...data);
      const avg = data.reduce((a, b) => a + b, 0) / data.length;
      setStats({ min, max, avg });
    }
  }, [data]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = { top: 30, right: 20, bottom: 35, left: 50 };
    const graphWidth = width - padding.left - padding.right;
    const graphHeight = height - padding.top - padding.bottom;

    let anim: number;
    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      // Background
      const bg = ctx.createLinearGradient(0, 0, 0, height);
      bg.addColorStop(0, "#0a0a0a");
      bg.addColorStop(1, "#050505");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, width, height);

      // Determine Y-axis scale
      const maxLatency = data.length > 0 ? Math.max(...data, 100) : 100;
      const yScale = Math.ceil(maxLatency / 50) * 50; // Round up to nearest 50

      // Grid lines
      ctx.strokeStyle = "rgba(255,68,0,0.08)";
      ctx.lineWidth = 1;
      ctx.font = "10px monospace";
      ctx.fillStyle = "rgba(150,150,150,0.5)";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";

      const gridLines = 5;
      for (let i = 0; i <= gridLines; i++) {
        const y = padding.top + (graphHeight / gridLines) * i;
        const value = Math.round(yScale - (yScale / gridLines) * i);
        
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
        
        ctx.fillText(`${value}`, padding.left - 8, y);
      }

      // Y-axis label
      ctx.save();
      ctx.translate(15, height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = "rgba(255,68,0,0.5)";
      ctx.font = "11px monospace";
      ctx.textAlign = "center";
      ctx.fillText("Latency (ms)", 0, 0);
      ctx.restore();

      // X-axis
      ctx.strokeStyle = "rgba(255,68,0,0.2)";
      ctx.beginPath();
      ctx.moveTo(padding.left, height - padding.bottom);
      ctx.lineTo(width - padding.right, height - padding.bottom);
      ctx.stroke();

      // X-axis labels
      ctx.fillStyle = "rgba(150,150,150,0.5)";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const xTicks = 5;
      for (let i = 0; i <= xTicks; i++) {
        const x = padding.left + (graphWidth / xTicks) * i;
        const seconds = Math.round(((xTicks - i) / xTicks) * 10); // 10 seconds total
        ctx.fillText(`${seconds}s`, x, height - padding.bottom + 8);
      }

      // X-axis label
      ctx.fillStyle = "rgba(255,68,0,0.5)";
      ctx.font = "11px monospace";
      ctx.fillText("Time", width / 2, height - 10);

      // Draw data if available
      if (data.length > 1) {
        const points = data.slice(-100); // Last 100 points
        const xStep = graphWidth / Math.max(points.length - 1, 1);

        // Area fill
        ctx.beginPath();
        ctx.moveTo(padding.left, height - padding.bottom);
        points.forEach((val, i) => {
          const x = padding.left + i * xStep;
          const y = padding.top + graphHeight - (val / yScale) * graphHeight;
          if (i === 0) {
            ctx.lineTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        });
        ctx.lineTo(padding.left + (points.length - 1) * xStep, height - padding.bottom);
        ctx.closePath();
        
        const gradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
        gradient.addColorStop(0, "rgba(255,68,0,0.3)");
        gradient.addColorStop(1, "rgba(255,68,0,0.02)");
        ctx.fillStyle = gradient;
        ctx.fill();

        // Line with gradient opacity
        ctx.beginPath();
        points.forEach((val, i) => {
          const x = padding.left + i * xStep;
          const y = padding.top + graphHeight - (val / yScale) * graphHeight;
          
          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        });
        
        ctx.strokeStyle = "#ff4400";
        ctx.lineWidth = 2.5;
        ctx.shadowColor = "rgba(255,68,0,0.5)";
        ctx.shadowBlur = 4;
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Most recent point highlight
        if (points.length > 0) {
          const lastX = padding.left + (points.length - 1) * xStep;
          const lastY = padding.top + graphHeight - (points[points.length - 1] / yScale) * graphHeight;
          
          ctx.beginPath();
          ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
          ctx.fillStyle = "#ff4400";
          ctx.fill();
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      } else if (connected) {
        // "Waiting for data" message
        ctx.fillStyle = "rgba(100,100,100,0.5)";
        ctx.font = "12px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("Awaiting data stream...", width / 2, height / 2);
      } else {
        // Standby grid
        ctx.strokeStyle = "rgba(100,100,100,0.1)";
        ctx.lineWidth = 0.5;
        for (let i = 0; i < 10; i++) {
          const y = padding.top + (graphHeight / 10) * i;
          ctx.beginPath();
          ctx.moveTo(padding.left, y);
          ctx.lineTo(width - padding.right, y);
          ctx.stroke();
        }
      }

      anim = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(anim);
  }, [data, connected]);

  return (
    <div className="relative">
      <canvas ref={canvasRef} className="w-full h-48" style={{ display: 'block' }} />
      {data.length > 0 && (
        <div className="mt-3 flex justify-center gap-6 text-[10px] text-gray-500">
          <div>Min: <span className="text-[#ff4400]">{stats.min.toFixed(0)}ms</span></div>
          <div>Avg: <span className="text-[#ff4400]">{stats.avg.toFixed(0)}ms</span></div>
          <div>Max: <span className="text-[#ff4400]">{stats.max.toFixed(0)}ms</span></div>
        </div>
      )}
    </div>
  );
}

/* === MetricCard === */
function MetricCard({ label, value, unit, connected }: { label: string; value: string; unit: string; connected: boolean }) {
  return (
    <div className="bg-gradient-to-br from-[#1a0a00] to-[#0a0a0a] border border-[#ff4400]/30 rounded-lg p-4 text-center relative overflow-hidden">
      <div className="absolute inset-0 bg-[#ff4400]/5 opacity-0 transition-opacity duration-300" 
           style={{ opacity: connected ? 0.5 : 0 }}></div>
      <p className="relative text-[#ff4400]/50 text-[10px] uppercase tracking-widest mb-2">{label}</p>
      <p className="relative text-[#ff4400] text-2xl font-bold tabular-nums">
        {value}<span className="text-sm font-normal ml-1">{unit}</span>
      </p>
    </div>
  );
}
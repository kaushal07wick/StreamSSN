class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000; // deviceRate → 16k
    this.acc = 0;
    this.buffer = [];
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;

    // simple decimation
    for (let i = 0; i < input.length; i += this.ratio) {
      const s = Math.max(-1, Math.min(1, input[Math.floor(i)]));
      this.buffer.push(s);
    }

    // send roughly 20 ms (~320 samples @16 kHz) at a time
    if (this.buffer.length >= 320) {
      const chunk = this.buffer.splice(0, 320);
      const buf = new ArrayBuffer(chunk.length * 2);
      const view = new DataView(buf);
      for (let i = 0; i < chunk.length; i++) {
        view.setInt16(i * 2, chunk[i] < 0 ? chunk[i] * 0x8000 : chunk[i] * 0x7fff, true);
      }
      this.port.postMessage(buf);
    }
    return true;
  }
}
registerProcessor("pcm-processor", PCMProcessor);
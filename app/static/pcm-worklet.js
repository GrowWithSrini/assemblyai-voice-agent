// AudioWorklet processor: takes the mic's Float32 stream and emits
// 16 kHz mono PCM16 (signed little-endian) chunks back to the main thread.
// AssemblyAI realtime STT expects PCM16 mono at 16 kHz.
//
// The AudioContext is created with sampleRate: 16000 where the browser honors
// it (Chrome/Firefox). Safari ignores that hint, so we also accept the real
// input rate via processorOptions and linearly resample to 16 kHz.

const TARGET_RATE = 16000;

class PCMDownsampler extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.inputRate = (options.processorOptions && options.processorOptions.inputSampleRate) || sampleRate;
    this._carry = 0; // fractional read position carried across process() calls
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || channel.length === 0) return true;

    const ratio = this.inputRate / TARGET_RATE;

    if (ratio === 1) {
      this.port.postMessage(floatToPCM16(channel).buffer, [floatToPCM16(channel).buffer]);
      return true;
    }

    // Resample by linear interpolation.
    const outLength = Math.floor((channel.length - this._carry) / ratio);
    const out = new Float32Array(Math.max(outLength, 0));
    let pos = this._carry;
    for (let i = 0; i < out.length; i++) {
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const a = channel[idx] || 0;
      const b = channel[idx + 1] !== undefined ? channel[idx + 1] : a;
      out[i] = a + (b - a) * frac;
      pos += ratio;
    }
    this._carry = pos - channel.length;

    if (out.length > 0) {
      const pcm = floatToPCM16(out);
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}

function floatToPCM16(float32) {
  const pcm = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm;
}

registerProcessor('pcm-downsampler', PCMDownsampler);

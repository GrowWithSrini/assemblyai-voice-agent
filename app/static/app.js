// AssemblyAI Realtime Voice Agent — browser client (orchestration lives here).
//
// Pipeline:
//   mic --16kHz PCM16--> AssemblyAI streaming STT (wss /v3/ws)
//        --Turn(end_of_turn)--> POST /api/chat (LLM Gateway proxy, server-side key)
//        --reply text--> Web Speech API (speechSynthesis)
//
// Turn-taking / VAD comes from the STT stream (SpeechStarted + Turn.end_of_turn).
// Barge-in: user speech while the agent is talking cancels TTS + the LLM request.

const els = {
  toggle: document.getElementById('toggle'),
  btnLabel: document.getElementById('btnLabel'),
  status: document.getElementById('status'),
  statusText: document.getElementById('statusText'),
  transcript: document.getElementById('transcript'),
  voice: document.getElementById('voice'),
  lang: document.getElementById('lang'),
  langChip: document.getElementById('langChip'),
  viz: document.getElementById('viz'),
};

let cfg = { sample_rate: 16000, speech_model: 'universal-3-5-pro', mode: 'balanced' };

let ws = null;
let micStream = null;
let audioInCtx = null;
let workletNode = null;
let sourceNode = null;

// outbound audio batching: STT wants 50–1000 ms frames
let pcmBuffer = []; // Array<Int16Array>
let pcmBufferSamples = 0;
let FRAME_SAMPLES = 1600; // 100 ms @ 16 kHz

let running = false;
let sttReady = false;

// visualizer
let analyser = null;
let vizRAF = null;

// orchestration state
let sessionId = null; // LangGraph thread id — conversation memory lives server-side
let replyCount = 0;
let agentBusy = false; // awaiting LLM or speaking
let llmAbort = null;
const handledTurns = new Set(); // turn_order values already sent to the LLM (STT can re-emit a formatted end_of_turn)

// ---------- UI helpers ----------

function setStatus(text, kind = '') {
  els.statusText.textContent = text;
  els.status.className = `statusbar ${kind}`.trim();
}

// phase drives the orb/backdrop styling: idle | connecting | listening | thinking | speaking
function setPhase(phase) {
  document.body.dataset.phase = phase;
}

function setRunningUI(isRunning) {
  els.btnLabel.textContent = isRunning ? 'Tap to stop' : 'Tap to start';
  els.toggle.setAttribute('aria-label', isRunning ? 'Stop conversation' : 'Start conversation');
  els.voice.disabled = isRunning;
  els.lang.disabled = isRunning;
}

// ---------- mic visualizer ----------

function startViz() {
  const canvas = els.viz;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const size = 300;
  canvas.width = size * dpr;
  canvas.height = size * dpr;
  const c = canvas.getContext('2d');
  c.scale(dpr, dpr);

  const bins = analyser.frequencyBinCount;
  const freq = new Uint8Array(bins);
  const BARS = 72;
  const cx = size / 2;
  const cy = size / 2;
  const r0 = 96;

  const palette = { listening: '34, 211, 238', thinking: '139, 92, 246', speaking: '236, 72, 153' };

  const draw = () => {
    vizRAF = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(freq);
    c.clearRect(0, 0, size, size);

    const rgb = palette[document.body.dataset.phase] || '148, 163, 184';
    for (let i = 0; i < BARS; i++) {
      const bin = Math.floor((i / BARS) * (bins * 0.6));
      const v = freq[bin] / 255;
      const len = 3 + v * 46;
      const ang = (i / BARS) * Math.PI * 2 - Math.PI / 2;
      const cos = Math.cos(ang);
      const sin = Math.sin(ang);
      c.beginPath();
      c.moveTo(cx + cos * r0, cy + sin * r0);
      c.lineTo(cx + cos * (r0 + len), cy + sin * (r0 + len));
      c.strokeStyle = `rgba(${rgb}, ${0.25 + v * 0.65})`;
      c.lineWidth = 3;
      c.lineCap = 'round';
      c.stroke();
    }
  };
  draw();
}

function stopViz() {
  if (vizRAF) cancelAnimationFrame(vizRAF);
  vizRAF = null;
  const c = els.viz.getContext('2d');
  c && c.clearRect(0, 0, els.viz.width, els.viz.height);
}

const bubbles = new Map();

function upsertBubble(itemId, role, text) {
  let el = bubbles.get(itemId);
  if (!el) {
    el = document.createElement('div');
    el.className = `bubble ${role}`;
    const av = document.createElement('span');
    av.className = 'av';
    av.textContent = role === 'user' ? 'U' : 'A';
    const body = document.createElement('span');
    body.className = 'body';
    el.append(av, body);
    els.transcript.appendChild(el);
    bubbles.set(itemId, el);
  }
  el.querySelector('.body').textContent = text;
  els.transcript.scrollTop = els.transcript.scrollHeight;
  return el;
}

function logLine(text) {
  const el = document.createElement('div');
  el.className = 'bubble system';
  const body = document.createElement('span');
  body.className = 'body';
  body.textContent = text;
  el.appendChild(body);
  els.transcript.appendChild(el);
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

// ---------- Web Speech API (TTS) ----------

const synth = window.speechSynthesis;
let voices = [];

// Preferred en-US voices by OS/browser, best first. First match wins as the default.
const PREFERRED_VOICES = [
  'Google US English',
  'Microsoft Aria Online (Natural) - English (United States)',
  'Microsoft Ava Online (Natural) - English (United States)',
  'Samantha',
  'Ava (Premium)',
  'Aria',
  'Microsoft Zira - English (United States)',
  'Alex',
];

function pickDefaultVoice() {
  for (const name of PREFERRED_VOICES) {
    const hit = voices.find((v) => v.name.toLowerCase().includes(name.toLowerCase()));
    if (hit) return hit.name;
  }
  const enUs = voices.find((v) => v.lang === 'en-US');
  return (enUs || voices[0]).name;
}

function loadVoices() {
  voices = synth ? synth.getVoices() : [];
  if (!voices.length) return;
  const rank = (v) => (v.lang === 'en-US' ? 0 : v.lang.startsWith('en') ? 1 : 2);
  voices.sort((a, b) => rank(a) - rank(b) || a.name.localeCompare(b.name));
  const prev = els.voice.value;
  els.voice.innerHTML = '';
  for (const v of voices) {
    const opt = document.createElement('option');
    opt.value = v.name;
    opt.textContent = `${v.name} — ${v.lang}${v.default ? ' (system default)' : ''}`;
    els.voice.appendChild(opt);
  }
  els.voice.value = prev && voices.some((v) => v.name === prev) ? prev : pickDefaultVoice();
}

if (synth) {
  loadVoices();
  synth.onvoiceschanged = loadVoices;
} else {
  logLine('This browser has no speechSynthesis — the agent will reply in text only.');
}

function speak(text) {
  if (!synth) return;
  synth.cancel();
  const u = new SpeechSynthesisUtterance(text);
  const chosen = voices.find((v) => v.name === els.voice.value);
  if (chosen) {
    u.voice = chosen;
    u.lang = chosen.lang;
  }
  u.rate = 1.05;
  u.onend = () => {
    agentBusy = false;
    if (running) {
      setStatus('Listening', 'ok');
      setPhase('listening');
    }
  };
  u.onerror = () => {
    agentBusy = false;
  };
  agentBusy = true;
  setStatus('Agent speaking…', 'ok');
  setPhase('speaking');
  synth.speak(u);
}

function stopSpeaking() {
  if (synth) synth.cancel();
}

// ---------- barge-in ----------

function interruptAgent(reason) {
  if (!agentBusy) return;
  stopSpeaking();
  if (llmAbort) {
    llmAbort.abort();
    llmAbort = null;
  }
  agentBusy = false;
  logLine(`— interrupted (${reason}) —`);
  if (running) {
    setStatus('Listening', 'ok');
    setPhase('listening');
  }
}

// ---------- orchestration ----------

async function runAgentTurn(userText) {
  agentBusy = true;
  setStatus('Thinking…', 'ok');
  setPhase('thinking');

  llmAbort = new AbortController();
  let reply;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message: userText }),
      signal: llmAbort.signal,
    });
    if (res.status === 429) {
      const info = await res.json().catch(() => ({}));
      const wait = info.retry_after || 50;
      agentBusy = false;
      setStatus(`Free LLM rate limited — ~${wait}s until the next turn`, 'err');
      logLine(`rate limited: wait ~${wait}s (free model allows ~2 turns / 50s)`);
      if (running) setPhase('listening');
      return;
    }
    if (!res.ok) throw new Error(`chat ${res.status}: ${await res.text()}`);
    reply = (await res.json()).reply;
  } catch (err) {
    if (err.name === 'AbortError') return; // barge-in already handled
    agentBusy = false;
    setStatus('LLM error: ' + err.message, 'err');
    logLine('LLM error: ' + err.message);
    if (running) setPhase('listening');
    return;
  } finally {
    llmAbort = null;
  }

  upsertBubble('agent-' + ++replyCount, 'agent', reply);
  speak(reply);
}

// ---------- STT websocket ----------

function handleSttMessage(msg) {
  switch (msg.type) {
    case 'Begin':
      sttReady = true;
      setStatus('Listening', 'ok');
      setPhase('listening');
      break;

    case 'SpeechStarted':
      interruptAgent('you started talking');
      setPhase('listening');
      break;

    case 'Turn': {
      const text = msg.transcript || '';
      if (msg.language_code) {
        els.langChip.hidden = false;
        els.langChip.textContent = `lang: ${msg.language_code}` +
          (msg.language_confidence ? ` (${Math.round(msg.language_confidence * 100)}%)` : '');
      }
      const turnKey = `turn-${msg.turn_order ?? 'x'}`;

      if (!msg.end_of_turn) {
        if (text) {
          if (agentBusy) interruptAgent('you started talking');
          upsertBubble(turnKey, 'user', text);
        }
        return;
      }

      // end of turn — STT can emit this twice for one turn (raw, then formatted).
      // Update the bubble each time; hand a given turn to the LLM only once.
      if (!text) break;
      upsertBubble(turnKey, 'user', text);
      if (!handledTurns.has(turnKey)) {
        handledTurns.add(turnKey);
        runAgentTurn(text);
      }
      break;
    }

    case 'Termination':
      logLine(
        `Termination — ${Math.round(msg.audio_duration_seconds || 0)}s audio, ` +
          `${Math.round(msg.session_duration_seconds || 0)}s session`,
      );
      break;

    default:
      break;
  }
}

// ---------- mic capture ----------

function flushPcm(force = false) {
  if (!ws || ws.readyState !== WebSocket.OPEN || !sttReady) return;
  if (!force && pcmBufferSamples < FRAME_SAMPLES) return;
  if (pcmBufferSamples === 0) return;

  const merged = new Int16Array(pcmBufferSamples);
  let off = 0;
  for (const arr of pcmBuffer) {
    merged.set(arr, off);
    off += arr.length;
  }
  pcmBuffer = [];
  pcmBufferSamples = 0;
  ws.send(merged.buffer);
}

async function startMic() {
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });

  audioInCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: cfg.sample_rate });
  if (audioInCtx.state === 'suspended') await audioInCtx.resume();
  await audioInCtx.audioWorklet.addModule('/pcm-worklet.js');

  sourceNode = audioInCtx.createMediaStreamSource(micStream);
  workletNode = new AudioWorkletNode(audioInCtx, 'pcm-downsampler', {
    processorOptions: { inputSampleRate: audioInCtx.sampleRate },
  });
  workletNode.port.onmessage = (e) => {
    if (!running) return;
    const chunk = new Int16Array(e.data);
    pcmBuffer.push(chunk);
    pcmBufferSamples += chunk.length;
    flushPcm();
  };
  sourceNode.connect(workletNode);
  // Not connected to destination — we don't monitor our own mic.

  analyser = audioInCtx.createAnalyser();
  analyser.fftSize = 512;
  analyser.smoothingTimeConstant = 0.82;
  sourceNode.connect(analyser);
  startViz();
}

// ---------- lifecycle ----------

async function start() {
  running = true;
  sttReady = false;
  agentBusy = false;
  sessionId = (crypto.randomUUID && crypto.randomUUID()) || `sess-${Date.now()}-${Math.random()}`;
  replyCount = 0;
  bubbles.clear();
  handledTurns.clear();
  pcmBuffer = [];
  pcmBufferSamples = 0;
  els.transcript.innerHTML = '';
  els.langChip.hidden = true;
  setRunningUI(true);
  setPhase('connecting');
  setStatus('Loading config…');

  try {
    cfg = await fetch('/api/config').then((r) => r.json());
    FRAME_SAMPLES = Math.round(cfg.sample_rate / 10);
  } catch (err) {
    setStatus('Config error: ' + err.message, 'err');
    return stop();
  }

  setStatus('Requesting token…');
  let token;
  try {
    const res = await fetch('/api/stt-token');
    if (!res.ok) throw new Error(`token ${res.status}: ${await res.text()}`);
    token = (await res.json()).token;
  } catch (err) {
    setStatus('Token error: ' + err.message, 'err');
    return stop();
  }

  const url = new URL(cfg.stt_ws_url);
  url.searchParams.set('token', token);
  url.searchParams.set('sample_rate', String(cfg.sample_rate));
  url.searchParams.set('speech_model', cfg.speech_model);
  url.searchParams.set('mode', cfg.mode);
  url.searchParams.set('format_turns', 'true');
  url.searchParams.set('language_detection', 'true');
  if (els.lang.value && els.lang.value !== 'auto') {
    url.searchParams.set('language_codes', els.lang.value);
  }

  setStatus('Connecting…');
  ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';

  ws.onopen = async () => {
    try {
      await startMic();
    } catch (err) {
      setStatus('Mic error: ' + err.message, 'err');
      stop();
    }
  };
  ws.onmessage = (e) => {
    let msg;
    try {
      msg = JSON.parse(e.data);
    } catch (_) {
      return;
    }
    handleSttMessage(msg);
  };
  ws.onerror = () => setStatus('WebSocket error', 'err');
  ws.onclose = (e) => {
    if (running) {
      const codes = {
        1008: 'unauthorized (token)',
        3005: 'server error',
        3007: 'bad audio chunk',
        3008: 'session timeout',
      };
      setStatus(`Disconnected — ${e.code} ${codes[e.code] || ''}`.trim(), e.code === 1000 ? '' : 'err');
      stop();
    }
  };
}

function stop() {
  running = false;
  sttReady = false;
  agentBusy = false;
  setPhase('idle');
  stopSpeaking();
  stopViz();
  analyser = null;
  if (llmAbort) {
    llmAbort.abort();
    llmAbort = null;
  }

  if (ws && ws.readyState === WebSocket.OPEN) {
    flushPcm(true);
    try {
      ws.send(JSON.stringify({ type: 'Terminate' }));
    } catch (_) {}
    setTimeout(() => {
      try {
        ws.close(1000);
      } catch (_) {}
      ws = null;
    }, 200);
  } else {
    ws = null;
  }

  if (workletNode) {
    workletNode.port.onmessage = null;
    workletNode.disconnect();
    workletNode = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (audioInCtx) {
    audioInCtx.close();
    audioInCtx = null;
  }
  if (micStream) {
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
  }

  setRunningUI(false);
  if (!els.status.classList.contains('err')) setStatus('Idle');
}

els.toggle.addEventListener('click', () => (running ? stop() : start()));
window.addEventListener('beforeunload', () => {
  if (running) stop();
});

setRunningUI(false);
setPhase('idle');
setStatus('Idle');

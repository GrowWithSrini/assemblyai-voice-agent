# AssemblyAI Realtime Voice Agent

A voice agent where **AssemblyAI does only realtime speech-to-text**, a **LangGraph**
agent handles the conversation, and the browser drives audio in/out. Bring-your-own
LLM (via LLM Gateway) and TTS.

```
mic ──16 kHz PCM16──▶ AssemblyAI Streaming STT (wss /v3/ws, Universal-3.5 Pro)
                          │  Turn(end_of_turn) → user utterance
                          ▼
     POST /api/chat ──▶ LangGraph agent ──▶ AssemblyAI LLM Gateway (OpenAI-compatible)
                          │  (server-side memory per session_id)
                          ▼  reply text
                     Web Speech API (speechSynthesis) speaks the reply
```

Turn-taking / VAD comes from the STT stream (`SpeechStarted` + `Turn.end_of_turn`).
Barge-in: talking over the agent cancels both the TTS and the in-flight LLM request.

```
app/
  main.py              app assembly + static hosting (create_app)
  settings.py          env-driven configuration (Settings, get_settings)
  schemas.py           request models (ChatRequest: session_id + message)
  assemblyai.py        direct AssemblyAI calls: mint_stt_token()
  agent.py             LangGraph agent: StateGraph + checkpointer, run_agent()
  api.py               HTTP routes: /api/config, /api/stt-token, /api/chat, /favicon.*
  static/
    index.html         UI
    app.js             STT WebSocket, orchestration loop, TTS, barge-in
    pcm-worklet.js     Float32 → 16 kHz mono PCM16 AudioWorklet
```

The backend is thin: `api.py` validates requests and delegates — token mints to
`assemblyai.py`, conversation turns to `agent.py`. The API key is read only in those
two modules; `settings.py` is the single source of config. There is **no inbound
WebSocket** — the browser opens the STT socket directly to AssemblyAI with the minted
token, so the server is plain request/response.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then put your key in .env
# key (used for BOTH STT and LLM Gateway): https://www.assemblyai.com/dashboard/api-keys
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000>, click the mic, allow it, and talk. The agent replies out
loud; talk over it to interrupt. `localhost` is a secure origin so no HTTPS is needed
locally; to reach it from another device, use an HTTPS tunnel (ngrok / Cloudflare Tunnel).

## How it works

| Piece | Where | Detail |
|---|---|---|
| STT token | `GET /api/stt-token` (server) | `GET https://streaming.assemblyai.com/v3/token?expires_in_seconds=60` — **raw key**, no `Bearer`. Single-use. |
| STT stream | browser | `wss://streaming.assemblyai.com/v3/ws?token=…&sample_rate=16000&speech_model=universal-3-5-pro&mode=balanced&format_turns=true&language_detection=true`. Mic → AudioWorklet → 16 kHz mono PCM16 → 100 ms binary frames. |
| Turn-taking | browser | `Turn` with `end_of_turn:false` → live partial; `end_of_turn:true` → final utterance → LLM. `SpeechStarted` drives barge-in. |
| Multilingual | — | Universal-3.5 Pro code-switches across 18 languages natively. The language selector is "Auto" by default; pick one to pin `language_codes`. `language_detection=true` shows the detected language chip. |
| Agent | `POST /api/chat` (server) | LangGraph `StateGraph` → `ChatOpenAI` pointed at `https://llm-gateway.assemblyai.com/v1` (**raw key**, OpenAI-compatible). The client sends only `{session_id, message}`; conversation memory is a LangGraph checkpointer keyed on `session_id`. |
| TTS | browser | Web Speech API (`speechSynthesis`). Voice list is populated from the OS; `en-US` sorted first. No key, no network. |
| Barge-in | browser | `SpeechStarted` or a non-empty partial while the agent is busy → `speechSynthesis.cancel()` + `AbortController` on the `/api/chat` fetch. |
| Termination | browser | Sends `{"type":"Terminate"}` on Stop and `beforeunload`, then closes. |

### Agent (LangGraph)

`agent.py` builds a `StateGraph` — one `call_model` node today (`START → call_model → END`).
`run_agent(session_id, message)` invokes it with `thread_id = session_id`, so the
**server** keeps the running message history via a `MemorySaver` checkpointer; the
browser is stateless and sends one message per turn.

- **Tools:** the free model reports `tools=False`, so there's no `ToolNode` yet. On a
  tool-capable model (`claude-haiku-4-5-20251001`, `gemini-2.5-flash-lite`, …) uncomment
  the marked block in `agent.py` and it becomes a ReAct agent.
- **`MemorySaver` is in-process.** One instance is fine; a multi-instance deployment
  needs a shared checkpointer (`langgraph-checkpoint-postgres`/`-redis`) or sticky
  sessions. Relevant for the Azure step.
- System prompt: `SYSTEM_PROMPT` in `agent.py`.

### LLM model

`LLM_MODEL` in `.env`. This account currently only has access to **`qwen3.5-4b-32k-fast`**,
the free model.

> ⚠️ **Free-model rate limit: ~2 requests per 50 s** (`x-ratelimit-limit: 2`). That's
> ~1 conversational turn every 25 s — usable for a quick demo, not a fluent conversation.
> On a 429 the server returns immediately with `retry_after`; the UI shows the wait and
> drops that turn (it doesn't queue or hammer).

**The real fix** is enabling paid model access on the AssemblyAI account, then setting
`LLM_MODEL` to `claude-haiku-4-5-20251001` (cheap + fast, good for voice),
`gemini-2.5-flash`, `claude-sonnet-4-6`, etc. Model IDs are exact versioned strings — see
the [LLM Gateway overview](https://www.assemblyai.com/docs/llm-gateway/overview).
`GET https://llm-gateway.assemblyai.com/v1/models` lists IDs (existence, not entitlement).
To skip the Gateway entirely, point `LLM_BASE` at another OpenAI-compatible endpoint
(and its key) where `ChatOpenAI` is constructed in `agent.py`.

### Swapping TTS

Replace `speak()` / `stopSpeaking()` in `app.js`. For a cloud voice (ElevenLabs, Cartesia,
OpenAI), add a `POST /api/tts` route in `api.py` plus a helper in `assemblyai.py` (keeps
that key server-side too) that returns audio, and play it through an `AudioContext` —
mirror the barge-in `cancel()` path.

### Knobs

- STT: `STT_MODE` (`min_latency` / `balanced` / `max_accuracy`), `STT_SAMPLE_RATE` — env, parsed in `settings.py`.
- LLM: `LLM_MODEL`, `LLM_MAX_TOKENS` — env; graph and `SYSTEM_PROMPT` in `agent.py`.
- TTS: voice dropdown; `u.rate` in `app.js`.

## Verified

Against the live API on 2026-09-03: `/v3/token` mint, WS connect with the full param set
(`Begin` → clean `Termination`), 100 ms PCM16 framing accepted, and a LangGraph
round-trip through `/api/chat` on `qwen3.5-4b-32k-fast` — including cross-turn memory
(“what hobby did I mention?” answered from the checkpointer, client sending one message).

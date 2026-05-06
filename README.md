# council-proxy

> **Multi-LLM ensemble proxy at the OpenAI-compatible transport layer.**
> Drop-in for any client that speaks `POST /v1/chat/completions`. Routes a single request to *N* models in parallel, then has a Chairman LLM synthesize the best response.

```
┌─ Client (any OpenAI-compatible) ─┐
│   set base_url to council-proxy  │
└──────────────┬───────────────────┘
               ↓
        ┌──────────────┐
        │ council-proxy│  ←── this repo
        │   :3460/v1   │
        └──┬────┬────┬─┘
           ↓    ↓    ↓
        ┌──┴┐ ┌─┴┐ ┌─┴┐ ┌──┐
   Router│  │ │A │ │C │ │..│   ← N council members in parallel
        └─┬┘ └─┬┘ └─┬┘ └─┬┘
          └────┴────┴────┘
                ↓
         ┌──────┴──────┐
         │  Chairman   │   ← fuses N responses into one answer
         │ (LLM model) │
         └──────┬──────┘
                ↓
            response
```

## Why?

A single LLM has bias, hallucinations, and silent failure modes. Multi-LLM ensembles reduce these by getting *independent* perspectives — but most existing solutions sit at the application layer (LangGraph, AutoGen) and require client-side rewrites.

**council-proxy lives at the transport layer.** Any client that already calls `POST /v1/chat/completions` — Hermes, OpenAI SDK, LangChain, Continue.dev, LiteLLM, raw curl — gets ensemble behavior by changing one URL.

### What's different from existing tools

| Tool | Layer | What it does |
|---|---|---|
| LiteLLM | Transport | Proxy + routing (one model per request) |
| LangGraph / AutoGen | Application | Multi-agent orchestration (requires client code) |
| portkey | Transport | Gateway (routing, observability) |
| **council-proxy** | **Transport** | **Multi-LLM fan-out + Chairman fusion + protocol adaptation** |

## Features

- **N-way fan-out** — Router decides per-request whether to call tools or dispatch to all council members in parallel
- **Chairman fusion** — A dedicated LLM synthesizes the N responses into a single answer with a citation/scoring footer (`📊 council eval #NNN`)
- **Protocol adaptation for strict providers** — Built-in handling for DeepSeek V4's protocol quirks (8 known traps, see `dsv4*` functions): `thinking:disabled` enforcement, `image_url` stripping, `reasoning_content` round-trip handling, `assistant→tool` message sequencing, automatic `tool_call` field completion
- **Tool-call loop with stuck detection** — Up to 10 rounds; auto-falls-through to consensus when 5 consecutive search-class tool calls fail
- **Composite search** — Tavily/Exa primary + SearXNG fallback, with a state machine that auto-switches when quota/auth errors are detected
- **Permanent archive** — Every council turn writes a JSON file with all member responses, skipped members + reasons, and Chairman output. Users can reference past turns with `#NNN` to inject historical context (bypasses upstream client's context window compression)
- **Single API key option** — Use OpenRouter to drive Grok / Qwen / Gemini / GPT / Claude / DeepSeek with one key, or mix direct provider connections for advanced control

## Quick start

```bash
# 1. Clone
git clone https://github.com/arthurthestupid/council-proxy.git
cd council-proxy

# 2. Configure
cp .env.example .env
# Edit .env: at minimum set OPENROUTER_API_KEY (sign up at openrouter.ai)

# 3. Run
node council-proxy.mjs
# Listens on 127.0.0.1:3460
```

Then in your client, set `base_url` (or equivalent) to `http://127.0.0.1:3460/v1`. Any API key string works (the proxy uses its own outbound keys from `.env`).

### Verify

```bash
# Discover the proxy's model list
curl http://127.0.0.1:3460/v1/models

# Send a test request
curl -X POST http://127.0.0.1:3460/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "council-v1",
    "messages": [{"role":"user","content":"Say hello in 3 different ways."}]
  }'
```

## Configuration

### Single key (recommended for getting started)

```env
OPENROUTER_API_KEY=sk-or-v1-...
```

This drives all council members and (with one tweak in `council-proxy.mjs`) the Chairman as well. Get a key at [openrouter.ai](https://openrouter.ai).

### Multi-provider (for production / cost optimization)

```env
OPENROUTER_API_KEY=sk-or-v1-...    # for council members
DEEPSEEK_API_KEY=sk-...            # for Chairman, direct connection
```

Direct connection to DeepSeek for the Chairman can be 5-10x faster than OpenRouter under load (OpenRouter occasionally has capacity-driven 40s timeouts).

### Member lineup

Council members are defined in `council-proxy.mjs` near the top (`const MEMBERS = [...]`). The reference lineup ships with 4 active members chosen for cross-vendor diversity:

| Code | Model | Provider | Notes |
|---|---|---|---|
| **A** | Grok 4.1 Fast | xAI | Reasoning, vision, **2M context** |
| **C** | Qwen 3.5 Flash | Alibaba | Strong Chinese, **vision + video**, **1M context** |
| **H** | Gemini 3.1 Flash Lite Preview | Google | **Full multimodal (image/audio/video)**, 1M context |
| **J** | GPT-5.4 Nano | OpenAI | Fast, follows instructions reliably, 400K context |

> Specs verified against [OpenRouter API](https://openrouter.ai/api/v1/models) and provider documentation as of 2026-05.

**You can — and should — swap these for your own preferences.** Other validated members (kept disabled in the source for reference): Step 3.5 Flash, GLM-4.7, MiMo-V2, Gemini 2.5. Member IDs (A, B, C, ...) are stable and never reused even after a model is removed, so historical archive references stay correct.

The Chairman defaults to DeepSeek V4 Flash (direct connection). Fallback is Grok 4.1 Fast via OpenRouter.

## Use cases

### As a Hermes (NousResearch) provider

```yaml
# In your hermes config.yaml
custom_providers:
  - name: council-v1
    base_url: http://127.0.0.1:3460/v1
    api_key: dummy
agent:
  model: council-v1
```

### As an OpenAI Python SDK target

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:3460/v1", api_key="dummy")
response = client.chat.completions.create(
    model="council-v1",
    messages=[{"role": "user", "content": "..."}],
)
```

### As a LangChain LLM

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(base_url="http://127.0.0.1:3460/v1", api_key="dummy", model="council-v1")
```

### As a Continue.dev / Aider / Cody backend

Set the model's `base_url` (or `apiBase`) to `http://127.0.0.1:3460/v1` in the client's config. No other code changes needed.

## Architecture

See [council-proxy.mjs](council-proxy.mjs) for the full implementation (~1700 lines, single file, no build step).

Key functions to read for understanding:

| Function | Role |
|---|---|
| `frontEndRoute` | Entry point — handles archive `#NNN` injection, search-state-machine SearXNG injection, dispatch to Router or Members |
| `routeRequest` | Router decision (tool call vs consensus) using a single-agent LLM with the client's tools schema |
| `buildMemberSystemPrompt` | Per-member prompt construction (knowledge boundary, no fabrication of times/numbers) |
| `buildFusionUserMessage` | Chairman input builder (all member responses + skipped reasons + eval-line format) |
| `callChairman` | Chairman invocation with fallback model |
| `writeCouncilArchive` | Persist turn to `archive/NNN.json` |
| `dsv4PrepareMessages` / `dsv4SanitizeBody` / `dsv4AttemptRecovery` | DeepSeek protocol adaptation (3-layer: outbound clean, outbound param strip, inbound error recovery) |

## Limitations

Be honest with yourself before deploying this:

- **30-90s latency per turn** — Dominated by the slowest member (4 in parallel + Chairman serial). Fine for chat-paced workflows (Discord, async agents); poor for streaming UIs that expect <2s first token.
- **5 calls per turn, but cheap unit cost** — Each turn invokes 5 LLM calls (4 members + Chairman). The reference lineup uses fast/flash/nano-tier models, so each turn typically costs **$0.005–$0.02 USD**. Even at 100+ turns/day, monthly cost lands around **$15–30** — comparable to a single frontier-model call. The 5x multiplier doesn't matter when each individual call is cheap. *If you swap members for frontier models (Claude Opus, GPT-5 Pro), the multiplier becomes painful — keep the lineup quick.*
- **Member homogenization risk** — same prompt + same context to 4 LLMs may produce surface-level "different perspectives" that converge under the hood. Diversity comes from genuinely different training corpora; cross-vendor selection helps but doesn't fully solve this.
- **Configuration complexity** — at minimum 1 API key (OpenRouter); for full feature set 2+ keys (OpenRouter + DeepSeek direct).
- **No native multi-LLM self-awareness in members** — each member sees only its own context and doesn't know it's part of a council. They can hallucinate "why other members didn't respond" when asked. The proxy mitigates this by injecting `skipped_members` reasons in archive references, but it's not a complete fix. See `buildArchiveInjectionMessage` for the current approach.

## Status

Personal project, best-effort maintenance. Issues and PRs welcome but no SLA on response time. The codebase is production-grade for the author's own use; treat it as a reference implementation and adapt to your needs.

## License

MIT — see [LICENSE](LICENSE).

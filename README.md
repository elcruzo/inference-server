# Inference server (Rust)

HTTP/1.1 inference with a **from-scratch** character bigram language model and a continuous-batching-lite scheduler. Primary code is Rust (`std` only). `python_server.py` is a fallback if `rustc` is missing; `cargo test` / `cargo build` are the real suite.

## Papers / systems

- OpenAI Completions / Chat Completions HTTP shapes.
- Yu et al., *Orca* / vLLM continuous batching — iteration-level scheduling, not request-level.
- Bengio et al. / classic n-gram LMs — the generator here is a real bigram with unigram backoff, not an echo.

## Routes

| Method | Path | Body |
|---|---|---|
| GET | `/health` | — |
| POST | `/v1/completions` | `{prompt, max_tokens, temperature, top_p, seed, stream?}` |
| POST | `/v1/chat/completions` | `{messages:[{role,content}], max_tokens, ...}` |

Errors: `{"error":{"message","type","code"}}`. Invalid JSON → **400**. Generation budget **30s** → **408**. `max_tokens` is clamped to **256**.

SSE (`stream: true`): `data: {"token":"x"}\n\n` then `data: [DONE]\n\n`.

## Language model

Character-level **bigram counts** from a baked-in English corpus, plus unigram backoff and add-ε. Sampling: temperature (0 = greedy argmax) and nucleus `top_p`. A `seed` drives an LCG. Tokens are generated from `P(next | prev)` — the completion is **not** a copy of the prompt.

## Continuous-batching-lite queue

Single-threaded scheduler, shared by all connections:

1. **waiting** — FIFO of accepted HTTP jobs.
2. **active** — up to `max_batch` (8) jobs in decode.
3. Each **tick**: admit from waiting, then emit **one token per active job** (interleaved in one pass), retire finished jobs onto their `mpsc` / queue.

HTTP threads only enqueue and wait on that channel. Timeouts live at the HTTP layer.

## Run

```bash
cargo test
cargo run                          # LISTEN=127.0.0.1:3003
python python_server.py 3003       # fallback
python -m pytest test_server.py -q
python demo.py
```

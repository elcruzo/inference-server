# Inference server (Rust)

HTTP/1.1 inference with a **tiny decoder-only Transformer** and **Orca-style continuous batching**. Primary code is Rust (`std` only, handwritten matmuls). `python_server.py` is the named Python backend when `rustc`/`cargo` is unavailable; `cargo test` / `cargo build` are the real suite.

## Implements

- Tiny GPT (char-level): token + learned pos embeddings, Pre-LN causal MHA, GELU MLP, weight-tied `lm_head`
- Iteration-level continuous batching: waiting FIFO → active set (`max_batch=8`) → one decode step per tick with per-job KV cache
- OpenAI-shaped `/v1/completions` and `/v1/chat/completions` (+ SSE stream)
- `/metrics` — request counts, tokens, mean latency, tokens/sec, scheduler steps
- Named device paths for training/Python: `cpu` and `mps` raise if the requested device is unavailable

## Papers / systems

- Yu et al., *Orca* (OSDI 2022) — iteration-level scheduling. This server admits from a waiting FIFO, runs **one model step per active job per tick**, and keeps a **dense per-job KV** cache.
- Kwon et al., *vLLM* / PagedAttention (2023) — continuous batching + paged KV. This server implements the iteration-level queue with dense KV.
- Radford / nanoGPT — decoder-only LM shape for the tiny generator.

## Routes

| Method | Path | Body |
|---|---|---|
| GET | `/health` | — |
| GET | `/metrics` | — |
| POST | `/v1/completions` | `{prompt, max_tokens, temperature, top_p, seed, stream?}` |
| POST | `/v1/chat/completions` | `{messages:[{role,content}], max_tokens, ...}` |

Errors: `{"error":{"message","type","code"}}`. Invalid JSON → **400**. Generation budget **30s** → **408**. `max_tokens` is clamped to **256**. No auth.

SSE (`stream: true`): `data: {"token":"x"}\n\n` then `data: [DONE]\n\n`.

## Language model

Weights live in `model/tiny_gpt.bin` (embedded into the Rust binary via `include_bytes!`). Train/export:

```bash
python train_export.py --device mps --steps 250   # or --device cpu
```

Architecture: `d_model=32`, `n_heads=4`, `n_layers=2`, `d_ff=128`, `max_len=64`, char vocab from `corpus.txt`. Sampling: temperature (0 = greedy) + nucleus `top_p`. A `seed` drives an LCG.

## Continuous batching queue

Single-threaded scheduler, shared by all connections:

1. **waiting** — FIFO of accepted HTTP jobs.
2. **active** — up to `max_batch` (8) jobs in decode.
3. Each **tick**: admit from waiting; for each active job run **one model iteration** (prefill on first token, else KV-cached decode); retire finished jobs onto their `mpsc` / queue.

HTTP threads only enqueue and wait on that channel. Timeouts live at the HTTP layer.

## Papers on disk

- [`papers/yu-orca-2022.pdf`](papers/yu-orca-2022.pdf) — Yu et al. Orca continuous batching (OSDI 2022)
- [`papers/kwon-vllm-pagedattention-2023.pdf`](papers/kwon-vllm-pagedattention-2023.pdf) — Kwon et al. vLLM / PagedAttention (2023) ([arXiv:2309.06180](https://arxiv.org/abs/2309.06180))

## Compared to vLLM / Orca

**What you learn here:**
- Iteration-level continuous batching (waiting FIFO → active ≤ `max_batch`)
- Per-job KV cache + one model step per scheduler tick
- OpenAI-shaped `/v1/completions` with `/metrics` (tok/s, latency)

| | This repo | vLLM / Orca |
|---|---|---|
| Engine | Tiny char GPT (Rust/`python_server`) | PagedAttention + continuous batch |
| Batching | Dense KV, max_batch=8 | Block-table KV, high concurrency |
| Model | d=32, 2 layers | Llama-scale GPUs |

### Numbers (2026-08-16, Darwin 25.5.0 arm64 / Apple M5)

| Metric | This repo | Baseline | Source |
|---|---|---|---|
| Single-stream tok/s | ~4070 (CPU, 64 gen) | ~71 tok/s Llama-3.1-8B FP16 (1 user, RTX 4090) | `python main.py`; SitePoint 2026 (different model/HW) |
| Cont. batch tok/s | ~4030 (8×32, max_batch=8) | ~920 tok/s @ 50 users (same 8B FP16) | `python main.py`; SitePoint 2026 |
| Caveat | Toy CPU LM | GPU production | not apples-to-apples |

```bash
python main.py
```

## Run

```bash
cargo test
cargo run                          # LISTEN=127.0.0.1:3003
python train_export.py --device cpu # regenerate weights if needed
python python_server.py 3003       # named Python backend (CPU path)
python -m pytest test_server.py -q
python main.py
```

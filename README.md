# Inference server (Rust)

HTTP/1.1 inference with a **from-scratch tiny decoder-only Transformer** and **Orca-style continuous batching**. Primary code is Rust (`std` only: hand matmuls, no candle/tch/vLLM). `python_server.py` is the named Python backend when `rustc`/`cargo` is unavailable; `cargo test` / `cargo build` are the real suite.

## Implements

- Tiny GPT (char-level): token + learned pos embeddings, Pre-LN causal MHA, GELU MLP, weight-tied `lm_head`
- Iteration-level continuous batching: waiting FIFO → active set (`max_batch=8`) → one decode step per tick with per-job KV cache
- OpenAI-shaped `/v1/completions` and `/v1/chat/completions` (+ SSE stream)
- `/metrics` — request counts, tokens, mean latency, tokens/sec, scheduler steps
- Named device paths for training/Python: `cpu` and `mps` (explicit; no silent remap)

Does **not** wrap vLLM, llama.cpp, or HuggingFace `generate` as the engine.

## Papers / systems

- Yu et al., *Orca* (OSDI 2022) — iteration-level scheduling / selective batching.
- Kwon et al., *vLLM* / PagedAttention (2023) — continuous batching + paged KV (we implement scheduling + dense KV, not the pager).
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

## Run

```bash
cargo test
cargo run                          # LISTEN=127.0.0.1:3003
python train_export.py --device cpu # regenerate weights if needed
python python_server.py 3003       # named Python backend (CPU path)
python -m pytest test_server.py -q
python demo.py
```

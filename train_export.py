#!/usr/bin/env python3
"""Train a tiny char-level GPT and export float32 weights for the Rust server.

Named device paths: --device cpu | mps. Unavailable explicit `mps` raises.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

# Keep in sync with Rust Config / engine.py
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
MAX_LEN = 64


def pick_device(name: str) -> torch.device:
    """Named device path. Explicit `mps` / `cpu` must resolve; unavailable `mps` raises."""
    key = name.strip().lower()
    if key == "cpu":
        return torch.device("cpu")
    if key == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("requested device 'mps' but torch.backends.mps.is_available() is False")
        return torch.device("mps")
    raise ValueError(f"unknown device {name!r}; use 'cpu' or 'mps'")


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, kv_cache=None):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        q_len, k_len = q.size(2), k.size(2)
        past = k_len - q_len
        keep = torch.ones(q_len, k_len, dtype=torch.bool, device=x.device).tril(diagonal=past)
        scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
        w = torch.softmax(scores, dim=-1)
        out = (w @ v).transpose(1, 2).contiguous().view(b, t, -1)
        return self.wo(out), (k, v)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, d_ff)
        self.proj = nn.Linear(d_ff, d_model)

    def forward(self, x, kv_cache=None):
        a, cache = self.attn(self.ln1(x), kv_cache)
        x = x + a
        x = x + self.proj(gelu_tanh(self.fc(self.ln2(x))))
        return x, cache


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = MAX_LEN
        self.tok = nn.Embedding(vocab_size, D_MODEL)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        self.blocks = nn.ModuleList([Block(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight

    def forward(self, idx: torch.Tensor, kv_caches=None):
        b, t = idx.shape
        if kv_caches is not None and kv_caches[0] is not None:
            past = kv_caches[0][0].size(2)
            pos = torch.arange(past, past + t, device=idx.device)
        else:
            pos = torch.arange(t, device=idx.device)
        pos = pos % self.max_len
        x = self.tok(idx) + self.pos(pos)
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = None if kv_caches is None else kv_caches[i]
            x, cache = block(x, cache)
            new_caches.append(cache)
        logits = self.lm_head(self.ln_f(x))
        return logits, new_caches


def _f32(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().float().numpy().tobytes()


def export_bin(model: TinyGPT, itos: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vocab = "".join(itos).encode("utf-8")
    assert len(vocab) == model.vocab_size
    chunks: list[bytes] = []
    chunks.append(b"TGPT")
    chunks.append(struct.pack("<I", 1))  # version
    chunks.append(
        struct.pack(
            "<IIIIII",
            model.vocab_size,
            D_MODEL,
            N_HEADS,
            N_LAYERS,
            D_FF,
            MAX_LEN,
        )
    )
    chunks.append(struct.pack("<I", len(vocab)))
    chunks.append(vocab)
    # Weight order must match Rust loader.
    chunks.append(_f32(model.tok.weight))
    chunks.append(_f32(model.pos.weight))
    for block in model.blocks:
        chunks.append(_f32(block.ln1.weight))
        chunks.append(_f32(block.ln1.bias))
        chunks.append(_f32(block.attn.wq.weight))
        chunks.append(_f32(block.attn.wq.bias))
        chunks.append(_f32(block.attn.wk.weight))
        chunks.append(_f32(block.attn.wk.bias))
        chunks.append(_f32(block.attn.wv.weight))
        chunks.append(_f32(block.attn.wv.bias))
        chunks.append(_f32(block.attn.wo.weight))
        chunks.append(_f32(block.attn.wo.bias))
        chunks.append(_f32(block.ln2.weight))
        chunks.append(_f32(block.ln2.bias))
        chunks.append(_f32(block.fc.weight))
        chunks.append(_f32(block.fc.bias))
        chunks.append(_f32(block.proj.weight))
        chunks.append(_f32(block.proj.bias))
    chunks.append(_f32(model.ln_f.weight))
    chunks.append(_f32(model.ln_f.bias))
    path.write_bytes(b"".join(chunks))
    meta = {
        "vocab_size": model.vocab_size,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "max_len": MAX_LEN,
        "itos": itos,
    }
    (path.parent / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def train(device: torch.device, steps: int, seed: int) -> tuple[TinyGPT, list[str], list[float]]:
    torch.manual_seed(seed)
    text = (ROOT / "corpus.txt").read_text(encoding="utf-8")
    itos = sorted(set(text))
    stoi = {c: i for i, c in enumerate(itos)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=device)
    model = TinyGPT(len(itos)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses: list[float] = []
    model.train()
    n = data.numel() - MAX_LEN - 1
    batch = 16
    for _ in range(steps):
        ix = torch.randint(0, n, (batch,), device=device)
        x = torch.stack([data[i : i + MAX_LEN] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + MAX_LEN] for i in ix])
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(itos)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return model.cpu(), itos, losses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"], help="named device path")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = pick_device(args.device)
    print(f"training on device={device} steps={args.steps}")
    model, itos, losses = train(device, args.steps, args.seed)
    uniform = math.log(len(itos))
    print(f"vocab={len(itos)} first_loss={losses[0]:.3f} last_loss={losses[-1]:.3f} uniform={uniform:.3f}")
    if losses[-1] >= uniform - 0.2:
        raise SystemExit("training did not beat uniform baseline; refuse to export a weak model")
    out = MODEL_DIR / "tiny_gpt.bin"
    export_bin(model, itos, out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

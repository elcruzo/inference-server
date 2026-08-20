"""Python reference: tiny GPT + continuous-batching scheduler + metrics.

Primary implementation is Rust. This mirrors architecture for the named Python HTTP backend and demos.
Named device paths: pick_device('cpu' | 'mps').
"""

from __future__ import annotations

import json
import math
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
MODEL_BIN = ROOT / "model" / "tiny_gpt.bin"
MODEL_CFG = ROOT / "model" / "config.json"

D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
MAX_LEN = 64
MAX_TOKENS_CAP = 256
DEFAULT_MAX_TOKENS = 16


def pick_device(name: str) -> torch.device:
    """Named device path. Explicit `mps` / `cpu` must resolve; unavailable `mps` raises."""
    key = name.strip().lower()
    if key == "cpu":
        return torch.device("cpu")
    if key == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("requested device 'mps' but MPS is unavailable")
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


class Lcg:
    def __init__(self, seed: int):
        self.state = int(seed) | 1

    def next_u64(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1) & ((1 << 64) - 1)
        return self.state

    def uniform(self) -> float:
        return (self.next_u64() >> 11) / float(1 << 53)


class LanguageModel:
    """Loads exported tiny GPT weights; runs on a named device path."""

    def __init__(self, device: str = "cpu"):
        self.device = pick_device(device)
        cfg = json.loads(MODEL_CFG.read_text(encoding="utf-8"))
        self.itos = cfg["itos"]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.model = TinyGPT(len(self.itos))
        self._load_bin(MODEL_BIN)
        self.model.to(self.device)
        self.model.eval()

    def _load_bin(self, path: Path) -> None:
        raw = path.read_bytes()
        assert raw[:4] == b"TGPT"
        ver, vs, d, nh, nl, dff, ml = struct.unpack_from("<IIIIIII", raw, 4)
        assert ver == 1 and vs == len(self.itos) and d == D_MODEL
        assert nh == N_HEADS and nl == N_LAYERS and dff == D_FF and ml == MAX_LEN
        off = 4 + 4 * 7
        vlen = struct.unpack_from("<I", raw, off)[0]
        off += 4
        vocab = raw[off : off + vlen]
        off += vlen
        assert list(vocab) == [ord(c) for c in self.itos]

        def take(n: int) -> torch.Tensor:
            nonlocal off
            nbytes = n * 4
            t = torch.frombuffer(bytearray(raw[off : off + nbytes]), dtype=torch.float32).clone()
            off += nbytes
            return t

        with torch.no_grad():
            self.model.tok.weight.copy_(take(vs * d).view(vs, d))
            self.model.pos.weight.copy_(take(ml * d).view(ml, d))
            for block in self.model.blocks:
                block.ln1.weight.copy_(take(d))
                block.ln1.bias.copy_(take(d))
                block.attn.wq.weight.copy_(take(d * d).view(d, d))
                block.attn.wq.bias.copy_(take(d))
                block.attn.wk.weight.copy_(take(d * d).view(d, d))
                block.attn.wk.bias.copy_(take(d))
                block.attn.wv.weight.copy_(take(d * d).view(d, d))
                block.attn.wv.bias.copy_(take(d))
                block.attn.wo.weight.copy_(take(d * d).view(d, d))
                block.attn.wo.bias.copy_(take(d))
                block.ln2.weight.copy_(take(d))
                block.ln2.bias.copy_(take(d))
                block.fc.weight.copy_(take(dff * d).view(dff, d))
                block.fc.bias.copy_(take(dff))
                block.proj.weight.copy_(take(d * dff).view(d, dff))
                block.proj.bias.copy_(take(d))
            self.model.ln_f.weight.copy_(take(d))
            self.model.ln_f.bias.copy_(take(d))
        assert off == len(raw)

    def encode(self, text: str) -> list[int]:
        ids = [self.stoi.get(c, self.stoi.get(" ", 0)) for c in text]
        return ids or [0]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    @torch.no_grad()
    def prefill(self, ids: list[int]):
        if len(ids) > MAX_LEN:
            ids = ids[-MAX_LEN:]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits, caches = self.model(idx)
        return logits[0, -1].detach().cpu(), caches

    @torch.no_grad()
    def decode_step(self, token_id: int, caches):
        idx = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        logits, caches = self.model(idx, caches)
        return logits[0, -1].detach().cpu(), caches

    def sample_logits(self, logits: torch.Tensor, temperature: float, top_p: float, rng: Lcg) -> int:
        w = logits.float().clone()
        n = w.numel()
        if temperature <= 0:
            return int(w.argmax().item())
        w = w / temperature
        w = torch.softmax(w, dim=-1)
        probs = w.tolist()
        p_cut = 1.0 if top_p <= 0 else min(top_p, 1.0)
        order = sorted(range(n), key=lambda i: probs[i], reverse=True)
        kept: list[int] = []
        cum = 0.0
        for i in order:
            kept.append(i)
            cum += probs[i]
            if cum >= p_cut:
                break
        z2 = sum(probs[i] for i in kept)
        u = rng.uniform() * z2
        acc = 0.0
        for i in kept:
            acc += probs[i]
            if u <= acc:
                return i
        return kept[-1]

    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float, seed: int) -> str:
        rng = Lcg(seed)
        ids = self.encode(prompt)
        logits, caches = self.prefill(ids)
        out: list[int] = []
        for _ in range(max_tokens):
            nxt = self.sample_logits(logits, temperature, top_p, rng)
            ids.append(nxt)
            out.append(nxt)
            logits, caches = self.decode_step(nxt, caches)
        return self.decode(out)


@dataclass
class Job:
    ids: list[int]
    max_new: int
    temperature: float
    top_p: float
    rng: Lcg
    tx: Queue
    n_new: int = 0
    caches: list | None = None
    last_logits: torch.Tensor | None = None


@dataclass
class Metrics:
    requests_total: int = 0
    requests_ok: int = 0
    tokens_total: int = 0
    latency_ms_total: float = 0.0
    scheduler_steps: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_ok(self, tokens: int, started: float) -> None:
        ms = (time.time() - started) * 1000.0
        with self._lock:
            self.requests_total += 1
            self.requests_ok += 1
            self.tokens_total += tokens
            self.latency_ms_total += ms

    def record_error(self) -> None:
        with self._lock:
            self.requests_total += 1

    def tick_sched(self) -> None:
        with self._lock:
            self.scheduler_steps += 1

    def snapshot(self) -> dict:
        with self._lock:
            ok = self.requests_ok
            lat = self.latency_ms_total
            tokens = self.tokens_total
            return {
                "requests_total": self.requests_total,
                "requests_ok": ok,
                "tokens_total": tokens,
                "latency_ms_total": lat,
                "mean_latency_ms": 0.0 if ok == 0 else lat / ok,
                "tokens_per_sec": 0.0 if lat <= 0 else tokens / (lat / 1000.0),
                "scheduler_steps": self.scheduler_steps,
            }


class Scheduler:
    """FIFO wait queue + active set. Each step emits one token per active job."""

    def __init__(self, lm: LanguageModel, max_batch: int = 8, metrics: Metrics | None = None):
        self.lm = lm
        self.waiting: deque[Job] = deque()
        self.active: list[Job] = []
        self.max_batch = max(1, max_batch)
        self.cv = threading.Condition()
        self.metrics = metrics or Metrics()

    def enqueue(self, job: Job) -> None:
        with self.cv:
            self.waiting.append(job)
            self.cv.notify()

    def has_work(self) -> bool:
        return bool(self.waiting or self.active)

    def step(self) -> bool:
        while len(self.active) < self.max_batch and self.waiting:
            self.active.append(self.waiting.popleft())
        if not self.active:
            return False
        still: list[Job] = []
        for job in self.active:
            if job.max_new == 0:
                job.tx.put(("", True))
                continue
            if job.caches is None:
                logits, caches = self.lm.prefill(job.ids)
                job.last_logits = logits
                job.caches = caches
            nxt = self.lm.sample_logits(job.last_logits, job.temperature, job.top_p, job.rng)
            job.ids.append(nxt)
            job.n_new += 1
            done = job.n_new >= job.max_new
            ch = self.lm.itos[nxt]
            job.tx.put((ch, done))
            if not done:
                logits, caches = self.lm.decode_step(nxt, job.caches)
                job.last_logits = logits
                job.caches = caches
                still.append(job)
        self.active = still
        self.metrics.tick_sched()
        return True

    def run_forever(self) -> None:
        while True:
            with self.cv:
                while not self.has_work():
                    self.cv.wait()
                self.step()

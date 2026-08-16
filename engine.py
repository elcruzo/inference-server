"""Python reference: character bigram LM + continuous-batching-lite scheduler.

Used by the fallback HTTP server and demo. The primary implementation is Rust.
"""

from __future__ import annotations

import threading
from collections import deque
import math
from dataclasses import dataclass
from queue import Queue

CORPUS = (
    "the cat sat on the mat and the cat sat on the hat. "
    "once upon a time a small model learned to write short english sentences. "
    "people ask questions and the assistant answers with calm precise words. "
    "temperature controls randomness and top p cuts the long tail of the distribution. "
    "continuous batching interleaves decode steps across waiting requests in one queue. "
    "the quick brown fox jumps over the lazy dog while digits 0123456789 stay nearby. "
)

MAX_TOKENS_CAP = 256
DEFAULT_MAX_TOKENS = 16


class Lcg:
    def __init__(self, seed: int):
        self.state = int(seed) | 1

    def next_u64(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1) & ((1 << 64) - 1)
        return self.state

    def uniform(self) -> float:
        return (self.next_u64() >> 11) / float(1 << 53)


class LanguageModel:
    def __init__(self, corpus: str = CORPUS):
        vocab = sorted({b for b in corpus.encode("ascii", "ignore") if 32 <= b < 127})
        if not vocab:
            vocab = [32]
        self.vocab = vocab
        n = len(vocab)
        index = {b: i for i, b in enumerate(vocab)}
        counts = [0.0] * (n * n)
        unigram = [0.0] * n
        raw = corpus.encode("ascii", "ignore")
        for b in raw:
            if b in index:
                unigram[index[b]] += 1.0
        for a, b in zip(raw, raw[1:]):
            if a in index and b in index:
                counts[index[a] * n + index[b]] += 1.0
        self.counts = counts
        self.unigram = unigram
        self._index = index

    def weights(self, prev: int) -> list[float]:
        n = len(self.vocab)
        row = self._index.get(prev)
        w = []
        for i in range(n):
            bigram = self.counts[row * n + i] if row is not None else 0.0
            w.append(bigram + 0.15 * self.unigram[i] + 1e-3)
        return w

    def sample(self, prev: int, temperature: float, top_p: float, rng: Lcg) -> int:
        w = self.weights(prev)
        n = len(w)
        if temperature <= 0:
            return self.vocab[max(range(n), key=lambda i: w[i])]
        inv_t = 1.0 / temperature
        logs = [math.log(x) * inv_t for x in w]
        m = max(logs)
        exps = [math.exp(v - m) for v in logs]
        z = sum(exps)
        probs = [e / z for e in exps]
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
                return self.vocab[i]
        return self.vocab[kept[-1]]

    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float, seed: int) -> str:
        rng = Lcg(seed)
        tokens = list(prompt.encode("utf-8", "replace")) or [ord("t")]
        out = []
        for _ in range(max_tokens):
            nxt = self.sample(tokens[-1], temperature, top_p, rng)
            tokens.append(nxt)
            out.append(chr(nxt))
        return "".join(out)


@dataclass
class Job:
    tokens: list[int]
    max_new: int
    temperature: float
    top_p: float
    rng: Lcg
    tx: Queue
    n_new: int = 0


class Scheduler:
    """FIFO wait queue + active set. Each step emits one token per active job."""

    def __init__(self, lm: LanguageModel, max_batch: int = 8):
        self.lm = lm
        self.waiting: deque[Job] = deque()
        self.active: list[Job] = []
        self.max_batch = max(1, max_batch)
        self.cv = threading.Condition()

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
            nxt = self.lm.sample(job.tokens[-1] if job.tokens else 32, job.temperature, job.top_p, job.rng)
            job.tokens.append(nxt)
            job.n_new += 1
            done = job.n_new >= job.max_new
            job.tx.put((chr(nxt), done))
            if not done:
                still.append(job)
        self.active = still
        return True

    def run_forever(self) -> None:
        while True:
            with self.cv:
                while not self.has_work():
                    self.cv.wait()
                self.step()

"""In-process demo of the tiny GPT and continuous-batching queue (CPU path)."""

from __future__ import annotations

import time
from queue import Queue

from engine import Job, Lcg, LanguageModel, Scheduler, pick_device


def _tok_per_s(lm: LanguageModel, prompt: str, max_new: int, *, temperature: float, seed: int) -> float:
    t0 = time.perf_counter()
    lm.generate(prompt, max_new, temperature=temperature, top_p=1.0, seed=seed)
    return max_new / (time.perf_counter() - t0)


def _cont_batch_tok_per_s(lm: LanguageModel, *, n_jobs: int, max_new: int, max_batch: int) -> float:
    sched = Scheduler(lm, max_batch=max_batch)
    queues = [Queue() for _ in range(n_jobs)]
    for i, tx in enumerate(queues):
        sched.enqueue(
            Job(ids=lm.encode("a "), max_new=max_new, temperature=0.0, top_p=1.0, rng=Lcg(i + 1), tx=tx)
        )
    t0 = time.perf_counter()
    while sched.has_work():
        sched.step()
    return (n_jobs * max_new) / (time.perf_counter() - t0)


if __name__ == "__main__":
    # Named paths: demo uses cpu; train_export.py exercises mps when available.
    assert pick_device("cpu").type == "cpu"
    lm = LanguageModel(device="cpu")
    print("device: cpu")
    print("greedy:", repr(lm.generate("the cat ", 24, temperature=0.0, top_p=1.0, seed=1)))
    print("sampled:", repr(lm.generate("the cat ", 24, temperature=0.9, top_p=0.9, seed=7)))

    single = _tok_per_s(lm, "the cat ", 64, temperature=0.0, seed=1)
    cont = _cont_batch_tok_per_s(lm, n_jobs=8, max_new=32, max_batch=8)
    print(f"single-stream tok/s: {single:.0f} (CPU, 64 gen)")
    print(f"cont-batch tok/s: {cont:.0f} (8x32, max_batch=8)")

    sched = Scheduler(lm, max_batch=2)
    q1, q2 = Queue(), Queue()
    sched.enqueue(Job(ids=lm.encode("the "), max_new=4, temperature=0.0, top_p=1.0, rng=Lcg(1), tx=q1))
    sched.enqueue(Job(ids=lm.encode("fox "), max_new=4, temperature=0.0, top_p=1.0, rng=Lcg(2), tx=q2))
    print("queue: waiting FIFO → active (max_batch=2) → one token / job / step")
    step = 0
    while sched.has_work():
        sched.step()
        step += 1
        a = q1.get_nowait()
        b = q2.get_nowait()
        print(f"  step {step}: jobA={a} jobB={b} active={len(sched.active)}")
    print("start Rust:  cargo run")
    print("or Python backend: python python_server.py 3003")
    print("metrics:     GET /metrics")

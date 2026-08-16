"""In-process demo of the tiny GPT and continuous-batching queue (CPU path)."""

from __future__ import annotations

from queue import Queue

from engine import Job, Lcg, LanguageModel, Scheduler, pick_device

if __name__ == "__main__":
    # Named paths: demo uses cpu; train_export.py exercises mps when available.
    assert pick_device("cpu").type == "cpu"
    lm = LanguageModel(device="cpu")
    print("device: cpu")
    print("greedy:", repr(lm.generate("the cat ", 24, temperature=0.0, top_p=1.0, seed=1)))
    print("sampled:", repr(lm.generate("the cat ", 24, temperature=0.9, top_p=0.9, seed=7)))

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
    print("or fallback: python python_server.py 3003")
    print("metrics:     GET /metrics")

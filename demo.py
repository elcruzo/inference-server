"""In-process demo of the bigram LM and the continuous-batching queue."""

from __future__ import annotations

from queue import Queue

from engine import Job, Lcg, LanguageModel, Scheduler

if __name__ == "__main__":
    lm = LanguageModel()
    print("greedy:", repr(lm.generate("the cat ", 24, temperature=0.0, top_p=1.0, seed=1)))
    print("sampled:", repr(lm.generate("the cat ", 24, temperature=0.9, top_p=0.9, seed=7)))

    sched = Scheduler(lm, max_batch=2)
    q1, q2 = Queue(), Queue()
    sched.enqueue(Job(tokens=list(b"the "), max_new=4, temperature=0.0, top_p=1.0, rng=Lcg(1), tx=q1))
    sched.enqueue(Job(tokens=list(b"fox "), max_new=4, temperature=0.0, top_p=1.0, rng=Lcg(2), tx=q2))
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

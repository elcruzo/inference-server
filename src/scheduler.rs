//! Continuous batching (Orca-style iteration-level scheduling).
//!
//! Queue:
//!   waiting  — accepted HTTP jobs, FIFO
//!   active   — up to `max_batch` jobs being decoded
//! Each `step`:
//!   1. retire finished / admit from waiting until `max_batch`
//!   2. for each active job: one model iteration (prefill on first token, else decode)
//!   3. emit one token event per job; swap-remove finished jobs
//!
//! This is iteration-level scheduling (not request-level static batching). KV cache
//! is per-job; the engine does not wrap vLLM.

use std::collections::VecDeque;
use std::sync::mpsc::Sender;

use crate::lm::{KvCache, LanguageModel, Lcg};

#[derive(Debug, Clone)]
pub struct TokenEvent {
    pub token: String,
    pub done: bool,
}

pub struct Job {
    pub ids: Vec<usize>,
    pub max_new: usize,
    pub n_new: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub rng: Lcg,
    pub caches: Option<Vec<KvCache>>,
    pub last_logits: Option<Vec<f32>>,
    pub tx: Sender<TokenEvent>,
}

pub struct Scheduler {
    pub waiting: VecDeque<Job>,
    pub active: Vec<Job>,
    pub max_batch: usize,
}

impl Scheduler {
    pub fn new(max_batch: usize) -> Self {
        Self {
            waiting: VecDeque::new(),
            active: Vec::new(),
            max_batch: max_batch.max(1),
        }
    }

    pub fn enqueue(&mut self, job: Job) {
        self.waiting.push_back(job);
    }

    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.active.is_empty()
    }

    /// One iteration tick across the active set.
    pub fn step(&mut self, lm: &LanguageModel) -> bool {
        while self.active.len() < self.max_batch {
            match self.waiting.pop_front() {
                Some(j) => self.active.push(j),
                None => break,
            }
        }
        if self.active.is_empty() {
            return false;
        }
        let mut i = 0;
        while i < self.active.len() {
            let done = {
                let job = &mut self.active[i];
                if job.max_new == 0 {
                    let _ = job.tx.send(TokenEvent {
                        token: String::new(),
                        done: true,
                    });
                    true
                } else {
                    if job.caches.is_none() {
                        let (logits, caches) = lm.prefill(&job.ids);
                        job.last_logits = Some(logits);
                        job.caches = Some(caches);
                    }
                    let logits = job.last_logits.as_ref().expect("logits after prefill");
                    let nxt = lm.sample_logits(logits, job.temperature, job.top_p, &mut job.rng);
                    job.ids.push(nxt);
                    job.n_new += 1;
                    let token = (lm.decode_byte(nxt) as char).to_string();
                    let finished = job.n_new >= job.max_new;
                    if !finished {
                        let caches = job.caches.as_mut().unwrap();
                        job.last_logits = Some(lm.decode_step(nxt, caches));
                    }
                    let _ = job.tx.send(TokenEvent {
                        token,
                        done: finished,
                    });
                    finished
                }
            };
            if done {
                self.active.swap_remove(i);
            } else {
                i += 1;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::LanguageModel;
    use std::sync::mpsc;

    #[test]
    fn interleaves_two_jobs() {
        let lm = LanguageModel::default_model();
        let mut sched = Scheduler::new(2);
        let (tx1, rx1) = mpsc::channel();
        let (tx2, rx2) = mpsc::channel();
        sched.enqueue(Job {
            ids: lm.encode("the "),
            max_new: 3,
            n_new: 0,
            temperature: 0.0,
            top_p: 1.0,
            rng: Lcg::new(1),
            caches: None,
            last_logits: None,
            tx: tx1,
        });
        sched.enqueue(Job {
            ids: lm.encode("cat "),
            max_new: 3,
            n_new: 0,
            temperature: 0.0,
            top_p: 1.0,
            rng: Lcg::new(2),
            caches: None,
            last_logits: None,
            tx: tx2,
        });
        sched.step(&lm);
        assert_eq!(rx1.try_iter().count(), 1);
        assert_eq!(rx2.try_iter().count(), 1);
        assert_eq!(sched.active.len(), 2);
        sched.step(&lm);
        sched.step(&lm);
        assert!(sched.active.is_empty());
        assert_eq!(rx1.try_iter().filter(|e| e.done).count(), 1);
        assert_eq!(rx2.try_iter().filter(|e| e.done).count(), 1);
    }
}

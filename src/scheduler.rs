//! Continuous-batching-lite: FIFO wait queue + active set, one decode step each tick.
//!
//! Queue:
//!   waiting  — accepted HTTP jobs, FIFO
//!   active   — up to `max_batch` jobs being decoded
//! Each `step` fills `active` from `waiting`, then generates **one token per
//! active job** (interleaved in a single-threaded pass), then retires finished
//! jobs onto their completion channel. Fairness is FIFO; no preemption except
//! the HTTP-layer timeout.

use std::collections::VecDeque;
use std::sync::mpsc::Sender;

use crate::lm::{LanguageModel, Lcg};

#[derive(Debug, Clone)]
pub struct TokenEvent {
    pub token: String,
    pub done: bool,
}

pub struct Job {
    pub tokens: Vec<u8>,
    pub max_new: usize,
    pub n_new: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub rng: Lcg,
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

    /// One decode tick: admit up to `max_batch`, emit one token per active job.
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
            let job = &mut self.active[i];
            if job.max_new == 0 {
                let _ = job.tx.send(TokenEvent { token: String::new(), done: true });
                self.active.swap_remove(i);
                continue;
            }
            let prev = *job.tokens.last().unwrap_or(&b' ');
            let nxt = lm.sample(prev, job.temperature, job.top_p, &mut job.rng);
            job.tokens.push(nxt);
            job.n_new += 1;
            let done = job.n_new >= job.max_new;
            let _ = job.tx.send(TokenEvent {
                token: (nxt as char).to_string(),
                done,
            });
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
            tokens: b"the ".to_vec(),
            max_new: 3,
            n_new: 0,
            temperature: 0.0,
            top_p: 1.0,
            rng: Lcg::new(1),
            tx: tx1,
        });
        sched.enqueue(Job {
            tokens: b"cat ".to_vec(),
            max_new: 3,
            n_new: 0,
            temperature: 0.0,
            top_p: 1.0,
            rng: Lcg::new(2),
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

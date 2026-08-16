//! Character-level bigram LM with unigram backoff. Generates tokens; does not echo.

use std::collections::BTreeSet;

/// Seed corpus — counts are estimated from this text (plus a tiny additive smoother).
pub const CORPUS: &str = "\
the cat sat on the mat and the cat sat on the hat. \
once upon a time a small model learned to write short english sentences. \
people ask questions and the assistant answers with calm precise words. \
temperature controls randomness and top p cuts the long tail of the distribution. \
continuous batching interleaves decode steps across waiting requests in one queue. \
the quick brown fox jumps over the lazy dog while digits 0123456789 stay nearby. \
";

#[derive(Clone, Debug)]
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }

    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

#[derive(Clone, Debug)]
pub struct LanguageModel {
    pub vocab: Vec<u8>,
    counts: Vec<f64>,
    unigram: Vec<f64>,
}

impl LanguageModel {
    pub fn from_corpus(text: &str) -> Self {
        let bytes = text.as_bytes();
        let mut set = BTreeSet::new();
        for &b in bytes {
            if b >= 32 && b < 127 {
                set.insert(b);
            }
        }
        if set.is_empty() {
            set.insert(b' ');
        }
        let vocab: Vec<u8> = set.into_iter().collect();
        let n = vocab.len();
        let mut index = [usize::MAX; 256];
        for (i, &b) in vocab.iter().enumerate() {
            index[b as usize] = i;
        }
        let mut counts = vec![0.0; n * n];
        let mut unigram = vec![0.0; n];
        for &b in bytes {
            if index[b as usize] != usize::MAX {
                unigram[index[b as usize]] += 1.0;
            }
        }
        for w in bytes.windows(2) {
            let a = index[w[0] as usize];
            let b = index[w[1] as usize];
            if a != usize::MAX && b != usize::MAX {
                counts[a * n + b] += 1.0;
            }
        }
        Self { vocab, counts, unigram }
    }

    pub fn default_model() -> Self {
        Self::from_corpus(CORPUS)
    }

    fn n(&self) -> usize {
        self.vocab.len()
    }

    fn id(&self, b: u8) -> Option<usize> {
        self.vocab.iter().position(|&x| x == b)
    }

    /// Unnormalized weights P(next | prev) with unigram backoff + add-1.
    pub fn weights(&self, prev: u8) -> Vec<f64> {
        let n = self.n();
        let row = self.id(prev);
        let mut w = vec![0.0; n];
        for i in 0..n {
            let bigram = row.map(|r| self.counts[r * n + i]).unwrap_or(0.0);
            w[i] = bigram + 0.15 * self.unigram[i] + 1e-3;
        }
        w
    }

    pub fn sample(&self, prev: u8, temperature: f64, top_p: f64, rng: &mut Lcg) -> u8 {
        let mut w = self.weights(prev);
        let n = w.len();
        if temperature <= 0.0 {
            let mut best = 0;
            for i in 1..n {
                if w[i] > w[best] {
                    best = i;
                }
            }
            return self.vocab[best];
        }
        let inv_t = 1.0 / temperature;
        let mut m = w[0].ln() * inv_t;
        for &x in &w[1..] {
            let v = x.ln() * inv_t;
            if v > m {
                m = v;
            }
        }
        let mut z = 0.0;
        for x in &mut w {
            *x = (*x).ln().mul_add(inv_t, -m).exp();
            z += *x;
        }
        for x in &mut w {
            *x /= z;
        }
        let p_cut = if top_p <= 0.0 { 1.0 } else { top_p.min(1.0) };
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| w[b].partial_cmp(&w[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0.0;
        let mut keep = n;
        for (k, &i) in idx.iter().enumerate() {
            cum += w[i];
            if cum >= p_cut {
                keep = k + 1;
                break;
            }
        }
        idx.truncate(keep.max(1));
        let z2: f64 = idx.iter().map(|&i| w[i]).sum();
        let u = rng.uniform() * z2;
        let mut acc = 0.0;
        for &i in &idx {
            acc += w[i];
            if u <= acc {
                return self.vocab[i];
            }
        }
        self.vocab[*idx.last().unwrap()]
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize, temperature: f64, top_p: f64, seed: u64) -> String {
        let mut rng = Lcg::new(seed);
        let mut tokens: Vec<u8> = prompt.as_bytes().to_vec();
        if tokens.is_empty() {
            tokens.push(b't');
        }
        let mut out = String::new();
        for _ in 0..max_tokens {
            let prev = *tokens.last().unwrap();
            let nxt = self.sample(prev, temperature, top_p, &mut rng);
            tokens.push(nxt);
            out.push(nxt as char);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_new_text() {
        let lm = LanguageModel::default_model();
        let prompt = "hello";
        let text = lm.generate(prompt, 8, 0.0, 1.0, 1);
        assert_eq!(text.len(), 8);
        assert_ne!(text, prompt);
        assert!(text.chars().all(|c| c.is_ascii()));
    }

    #[test]
    fn respects_max_tokens() {
        let lm = LanguageModel::default_model();
        assert_eq!(lm.generate("the ", 3, 0.8, 0.9, 42).len(), 3);
        assert_eq!(lm.generate("the ", 0, 1.0, 1.0, 1).len(), 0);
    }
}

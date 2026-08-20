//! Tiny decoder-only GPT: Pre-LN causal MHA + GELU MLP, weight-tied lm_head.
//! Weights from `model/tiny_gpt.bin` (`train_export.py`). Char-level, d=32, handwritten matmuls.

use crate::tensor::{add_inplace, gelu_tanh, layernorm, linear, softmax_inplace};

const WEIGHTS: &[u8] = include_bytes!("../model/tiny_gpt.bin");

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
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_len: usize,
}

#[derive(Clone, Debug)]
struct Linear {
    weight: Vec<f32>, // [out, in]
    bias: Option<Vec<f32>>,
    in_f: usize,
    out_f: usize,
}

impl Linear {
    fn apply(&self, x: &[f32], y: &mut [f32]) {
        linear(x, &self.weight, self.bias.as_deref(), self.in_f, self.out_f, y);
    }
}

#[derive(Clone, Debug)]
struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LayerNorm {
    fn apply(&self, x: &[f32], y: &mut [f32]) {
        layernorm(x, &self.weight, &self.bias, 1e-5, y);
    }
}

#[derive(Clone, Debug)]
struct Block {
    ln1: LayerNorm,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    ln2: LayerNorm,
    fc: Linear,
    proj: Linear,
}

/// Per-layer KV cache: keys/values as [n_heads * seq * d_k] row-major (h, t, d).
#[derive(Clone, Debug, Default)]
pub struct KvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub seq: usize,
}

#[derive(Clone, Debug)]
pub struct LanguageModel {
    pub config: Config,
    pub vocab: Vec<u8>,
    tok: Vec<f32>, // [V, D]
    pos: Vec<f32>, // [T, D]
    blocks: Vec<Block>,
    ln_f: LayerNorm,
}

struct Cursor<'a> {
    data: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, i: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.i + n > self.data.len() {
            return Err("truncated weight file".into());
        }
        let s = &self.data[self.i..self.i + n];
        self.i += n;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let bytes = self.take(n * 4)?;
        let mut out = Vec::with_capacity(n);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    }

    fn linear(&mut self, in_f: usize, out_f: usize, bias: bool) -> Result<Linear, String> {
        let weight = self.f32s(out_f * in_f)?;
        let bias = if bias { Some(self.f32s(out_f)?) } else { None };
        Ok(Linear { weight, bias, in_f, out_f })
    }

    fn ln(&mut self, d: usize) -> Result<LayerNorm, String> {
        Ok(LayerNorm {
            weight: self.f32s(d)?,
            bias: self.f32s(d)?,
        })
    }
}

impl LanguageModel {
    pub fn load_bytes(data: &[u8]) -> Result<Self, String> {
        let mut c = Cursor::new(data);
        let magic = c.take(4)?;
        if magic != b"TGPT" {
            return Err("bad magic".into());
        }
        let ver = c.u32()?;
        if ver != 1 {
            return Err(format!("unsupported version {ver}"));
        }
        let vocab_size = c.u32()? as usize;
        let d_model = c.u32()? as usize;
        let n_heads = c.u32()? as usize;
        let n_layers = c.u32()? as usize;
        let d_ff = c.u32()? as usize;
        let max_len = c.u32()? as usize;
        if d_model % n_heads != 0 {
            return Err("d_model not divisible by n_heads".into());
        }
        let vlen = c.u32()? as usize;
        if vlen != vocab_size {
            return Err("vocab length mismatch".into());
        }
        let vocab = c.take(vlen)?.to_vec();
        let tok = c.f32s(vocab_size * d_model)?;
        let pos = c.f32s(max_len * d_model)?;
        let mut blocks = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let ln1 = c.ln(d_model)?;
            let wq = c.linear(d_model, d_model, true)?;
            let wk = c.linear(d_model, d_model, true)?;
            let wv = c.linear(d_model, d_model, true)?;
            let wo = c.linear(d_model, d_model, true)?;
            let ln2 = c.ln(d_model)?;
            let fc = c.linear(d_model, d_ff, true)?;
            let proj = c.linear(d_ff, d_model, true)?;
            blocks.push(Block { ln1, wq, wk, wv, wo, ln2, fc, proj });
        }
        let ln_f = c.ln(d_model)?;
        if c.i != data.len() {
            return Err(format!("trailing {} bytes in weight file", data.len() - c.i));
        }
        Ok(Self {
            config: Config {
                vocab_size,
                d_model,
                n_heads,
                n_layers,
                d_ff,
                max_len,
            },
            vocab,
            tok,
            pos,
            blocks,
            ln_f,
        })
    }

    pub fn default_model() -> Self {
        Self::load_bytes(WEIGHTS).expect("embedded tiny_gpt.bin")
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        for b in text.bytes() {
            if let Some(i) = self.vocab.iter().position(|&c| c == b) {
                ids.push(i);
            } else if let Some(i) = self.vocab.iter().position(|&c| c == b' ') {
                ids.push(i);
            }
        }
        if ids.is_empty() {
            ids.push(0);
        }
        ids
    }

    pub fn decode_byte(&self, id: usize) -> u8 {
        self.vocab[id.min(self.vocab.len() - 1)]
    }

    fn emb_tok(&self, id: usize) -> &[f32] {
        let d = self.config.d_model;
        &self.tok[id * d..(id + 1) * d]
    }

    fn emb_pos(&self, pos: usize) -> &[f32] {
        let d = self.config.d_model;
        let p = pos % self.config.max_len;
        &self.pos[p * d..(p + 1) * d]
    }

    /// Prefill full prompt; returns last-position logits and filled KV caches.
    pub fn prefill(&self, ids: &[usize]) -> (Vec<f32>, Vec<KvCache>) {
        let cfg = &self.config;
        let d = cfg.d_model;
        let t = ids.len().min(cfg.max_len);
        let ids = &ids[ids.len().saturating_sub(t)..];
        let mut x = vec![0.0f32; t * d];
        for (i, &id) in ids.iter().enumerate() {
            let tok = self.emb_tok(id % cfg.vocab_size);
            let pos = self.emb_pos(i);
            for j in 0..d {
                x[i * d + j] = tok[j] + pos[j];
            }
        }
        let mut caches = vec![KvCache::default(); cfg.n_layers];
        for (li, block) in self.blocks.iter().enumerate() {
            x = self.block_forward(block, &x, t, 0, &mut caches[li]);
        }
        // final LN + tied lm_head on last token
        let last = &x[(t - 1) * d..t * d];
        let mut normed = vec![0.0f32; d];
        self.ln_f.apply(last, &mut normed);
        let logits = self.lm_logits(&normed);
        (logits, caches)
    }

    /// One decode step given the new token id and mutable caches.
    pub fn decode_step(&self, id: usize, caches: &mut [KvCache]) -> Vec<f32> {
        let cfg = &self.config;
        let d = cfg.d_model;
        let past = caches[0].seq;
        let pos = past; // absolute position before append
        let mut x = vec![0.0f32; d];
        let tok = self.emb_tok(id % cfg.vocab_size);
        let pe = self.emb_pos(pos);
        for j in 0..d {
            x[j] = tok[j] + pe[j];
        }
        for (li, block) in self.blocks.iter().enumerate() {
            let mut seq = vec![0.0f32; d];
            seq.copy_from_slice(&x);
            let out = self.block_forward(block, &seq, 1, past, &mut caches[li]);
            x.copy_from_slice(&out);
        }
        let mut normed = vec![0.0f32; d];
        self.ln_f.apply(&x, &mut normed);
        self.lm_logits(&normed)
    }

    fn lm_logits(&self, h: &[f32]) -> Vec<f32> {
        // tied: logits[v] = h · tok[v]
        let d = self.config.d_model;
        let mut logits = vec![0.0f32; self.config.vocab_size];
        for v in 0..self.config.vocab_size {
            let row = &self.tok[v * d..(v + 1) * d];
            let mut s = 0.0f32;
            for j in 0..d {
                s += h[j] * row[j];
            }
            logits[v] = s;
        }
        logits
    }

    fn block_forward(&self, block: &Block, x: &[f32], t: usize, past: usize, cache: &mut KvCache) -> Vec<f32> {
        let cfg = &self.config;
        let d = cfg.d_model;
        let nh = cfg.n_heads;
        let dk = d / nh;
        let mut out = vec![0.0f32; t * d];

        // Attn on LN1(x)
        let mut ln = vec![0.0f32; t * d];
        for i in 0..t {
            block.ln1.apply(&x[i * d..(i + 1) * d], &mut ln[i * d..(i + 1) * d]);
        }
        let mut q = vec![0.0f32; t * d];
        let mut k_new = vec![0.0f32; t * d];
        let mut v_new = vec![0.0f32; t * d];
        for i in 0..t {
            block.wq.apply(&ln[i * d..(i + 1) * d], &mut q[i * d..(i + 1) * d]);
            block.wk.apply(&ln[i * d..(i + 1) * d], &mut k_new[i * d..(i + 1) * d]);
            block.wv.apply(&ln[i * d..(i + 1) * d], &mut v_new[i * d..(i + 1) * d]);
        }
        // append K/V to cache in [h, seq, dk] layout
        let old_seq = cache.seq;
        let new_seq = old_seq + t;
        let mut k_all = vec![0.0f32; nh * new_seq * dk];
        let mut v_all = vec![0.0f32; nh * new_seq * dk];
        for h in 0..nh {
            for s in 0..old_seq {
                let src = ((h * old_seq + s) * dk)..((h * old_seq + s + 1) * dk);
                let dst = ((h * new_seq + s) * dk)..((h * new_seq + s + 1) * dk);
                k_all[dst.clone()].copy_from_slice(&cache.k[src.clone()]);
                v_all[dst].copy_from_slice(&cache.v[src]);
            }
            for i in 0..t {
                let s = old_seq + i;
                for j in 0..dk {
                    // q/k/v token layout is [t, heads, dk] interleaved in d: token i head h
                    k_all[(h * new_seq + s) * dk + j] = k_new[i * d + h * dk + j];
                    v_all[(h * new_seq + s) * dk + j] = v_new[i * d + h * dk + j];
                }
            }
        }
        cache.k = k_all.clone();
        cache.v = v_all.clone();
        cache.seq = new_seq;

        let scale = 1.0 / (dk as f32).sqrt();
        let mut attn_out = vec![0.0f32; t * d];
        for h in 0..nh {
            for qi in 0..t {
                let abs_q = past + qi;
                let mut scores = vec![0.0f32; new_seq];
                for s in 0..new_seq {
                    if s > abs_q {
                        scores[s] = f32::NEG_INFINITY;
                        continue;
                    }
                    let mut dot = 0.0f32;
                    for j in 0..dk {
                        let qv = q[qi * d + h * dk + j];
                        let kv = k_all[(h * new_seq + s) * dk + j];
                        dot += qv * kv;
                    }
                    scores[s] = dot * scale;
                }
                softmax_inplace(&mut scores);
                for j in 0..dk {
                    let mut acc = 0.0f32;
                    for s in 0..new_seq {
                        acc += scores[s] * v_all[(h * new_seq + s) * dk + j];
                    }
                    attn_out[qi * d + h * dk + j] = acc;
                }
            }
        }
        let mut proj = vec![0.0f32; t * d];
        for i in 0..t {
            block.wo.apply(&attn_out[i * d..(i + 1) * d], &mut proj[i * d..(i + 1) * d]);
        }
        out.copy_from_slice(x);
        for i in 0..t * d {
            out[i] += proj[i];
        }

        // MLP
        let mut ln2 = vec![0.0f32; t * d];
        for i in 0..t {
            block.ln2.apply(&out[i * d..(i + 1) * d], &mut ln2[i * d..(i + 1) * d]);
        }
        let mut mid = vec![0.0f32; t * cfg.d_ff];
        for i in 0..t {
            block.fc.apply(&ln2[i * d..(i + 1) * d], &mut mid[i * cfg.d_ff..(i + 1) * cfg.d_ff]);
            for j in 0..cfg.d_ff {
                mid[i * cfg.d_ff + j] = gelu_tanh(mid[i * cfg.d_ff + j]);
            }
        }
        let mut mlp = vec![0.0f32; t * d];
        for i in 0..t {
            block.proj.apply(&mid[i * cfg.d_ff..(i + 1) * cfg.d_ff], &mut mlp[i * d..(i + 1) * d]);
        }
        add_inplace(&mut out, &mlp);
        out
    }

    pub fn sample_logits(&self, logits: &[f32], temperature: f64, top_p: f64, rng: &mut Lcg) -> usize {
        let n = logits.len();
        if temperature <= 0.0 {
            let mut best = 0;
            for i in 1..n {
                if logits[i] > logits[best] {
                    best = i;
                }
            }
            return best;
        }
        let inv_t = 1.0 / temperature as f32;
        let mut w: Vec<f32> = logits.iter().map(|x| x * inv_t).collect();
        softmax_inplace(&mut w);
        let p_cut = if top_p <= 0.0 { 1.0 } else { top_p.min(1.0) } as f32;
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| w[b].partial_cmp(&w[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0.0f32;
        let mut keep = n;
        for (k, &i) in idx.iter().enumerate() {
            cum += w[i];
            if cum >= p_cut {
                keep = k + 1;
                break;
            }
        }
        idx.truncate(keep.max(1));
        let z2: f32 = idx.iter().map(|&i| w[i]).sum();
        let u = rng.uniform() as f32 * z2;
        let mut acc = 0.0f32;
        for &i in &idx {
            acc += w[i];
            if u <= acc {
                return i;
            }
        }
        *idx.last().unwrap()
    }

    /// Prefill + decode `max_tokens` new characters.
    pub fn generate(&self, prompt: &str, max_tokens: usize, temperature: f64, top_p: f64, seed: u64) -> String {
        let mut rng = Lcg::new(seed);
        let mut ids = self.encode(prompt);
        if ids.len() > self.config.max_len {
            ids = ids[ids.len() - self.config.max_len..].to_vec();
        }
        let (mut logits, mut caches) = self.prefill(&ids);
        let mut out = String::new();
        for _ in 0..max_tokens {
            let nxt = self.sample_logits(&logits, temperature, top_p, &mut rng);
            ids.push(nxt);
            out.push(self.decode_byte(nxt) as char);
            logits = self.decode_step(nxt, &mut caches);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_embedded_weights() {
        let lm = LanguageModel::default_model();
        assert_eq!(lm.config.d_model, 32);
        assert_eq!(lm.config.n_layers, 2);
        assert!(lm.vocab.len() >= 26);
    }

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

    #[test]
    fn causal_prefill_changes_under_prefix() {
        let lm = LanguageModel::default_model();
        let (a, _) = lm.prefill(&lm.encode("the cat"));
        let (b, _) = lm.prefill(&lm.encode("the dog"));
        let diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 1e-3, "different prefixes must change last logits");
    }
}

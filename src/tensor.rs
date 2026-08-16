//! Small f32 tensor helpers for the from-scratch tiny GPT (no BLAS / candle / tch).

#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// y = x @ W^T + b  with W shaped [out, in] (PyTorch Linear layout).
pub fn linear(x: &[f32], w: &[f32], b: Option<&[f32]>, in_f: usize, out_f: usize, y: &mut [f32]) {
    debug_assert_eq!(x.len(), in_f);
    debug_assert_eq!(w.len(), out_f * in_f);
    debug_assert_eq!(y.len(), out_f);
    for o in 0..out_f {
        let mut s = 0.0f32;
        let row = &w[o * in_f..(o + 1) * in_f];
        for i in 0..in_f {
            s += x[i] * row[i];
        }
        if let Some(bias) = b {
            s += bias[o];
        }
        y[o] = s;
    }
}

pub fn layernorm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, y: &mut [f32]) {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| {
        let d = *v - mean;
        d * d
    }).sum::<f32>()
        / n as f32;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        y[i] = (x[i] - mean) * inv * weight[i] + bias[i];
    }
}

pub fn softmax_inplace(logits: &mut [f32]) {
    let mut m = logits[0];
    for &v in &logits[1..] {
        if v > m {
            m = v;
        }
    }
    let mut z = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - m).exp();
        z += *v;
    }
    for v in logits.iter_mut() {
        *v /= z;
    }
}

pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += *y;
    }
}

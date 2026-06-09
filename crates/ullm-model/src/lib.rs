//! A minimal, correctness-first CPU runtime for the Llama architecture.
//!
//! Phase 0 scope: F32 weights, a single sequence, greedy decoding. The goal is a
//! readable numerical reference — the oracle the Metal backend will be validated
//! against — not speed. Quantized weights, batching, and SIMD come later.

use ullm_core::{Error, Result};
use ullm_gguf::GgufModel;

/// Hyperparameters describing a Llama model.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub vocab_size: usize,
    pub n_ctx: usize,
    pub rope_theta: f32,
    pub eps: f32,
}

/// Per-layer weights (row-major `[out, in]`, as produced by the GGUF loader).
struct LayerWeights {
    attn_norm: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    ffn_norm: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
}

/// A loaded Llama model with its KV cache.
pub struct LlamaModel {
    pub config: LlamaConfig,
    tok_embeddings: Vec<f32>, // [vocab, n_embd]
    layers: Vec<LayerWeights>,
    final_norm: Vec<f32>, // [n_embd]
    output: Vec<f32>,     // [vocab, n_embd]
    key_cache: Vec<f32>,  // [n_layer * n_ctx * kv_dim]
    val_cache: Vec<f32>,
}

impl LlamaModel {
    /// Load a Llama model from a parsed GGUF file (F32 weights only for now).
    pub fn from_gguf(model: &GgufModel) -> Result<Self> {
        let arch = model.architecture().unwrap_or_default();
        if arch != "llama" {
            return Err(Error::Unsupported(format!(
                "architecture '{arch}' is not supported yet (Phase 0 is llama-only)"
            )));
        }

        let req_u = |key: &str| -> Result<u64> {
            model
                .metadata_get(key)
                .and_then(|v| v.to_u64())
                .ok_or_else(|| Error::Format(format!("missing metadata '{key}'")))
        };
        let opt_u = |key: &str| model.metadata_get(key).and_then(|v| v.to_u64());
        let opt_f = |key: &str| model.metadata_get(key).and_then(|v| v.to_f32());

        let n_embd = req_u("llama.embedding_length")? as usize;
        let n_head = req_u("llama.attention.head_count")? as usize;
        let n_layer = req_u("llama.block_count")? as usize;
        let n_kv_head = opt_u("llama.attention.head_count_kv").unwrap_or(n_head as u64) as usize;
        let n_ff = req_u("llama.feed_forward_length")? as usize;
        let n_ctx = opt_u("llama.context_length").unwrap_or(2048) as usize;
        let head_dim = opt_u("llama.attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(n_embd / n_head);
        let rope_theta = opt_f("llama.rope.freq_base").unwrap_or(10000.0);
        let eps = opt_f("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let vocab_size = model.model_spec().vocab_size as usize;

        let config = LlamaConfig {
            n_embd,
            n_layer,
            n_head,
            n_kv_head,
            head_dim,
            n_ff,
            vocab_size,
            n_ctx,
            rope_theta,
            eps,
        };

        let tok_embeddings = tensor_f32(model, "token_embd.weight")?;
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            layers.push(LayerWeights {
                attn_norm: tensor_f32(model, &format!("blk.{i}.attn_norm.weight"))?,
                wq: tensor_f32(model, &format!("blk.{i}.attn_q.weight"))?,
                wk: tensor_f32(model, &format!("blk.{i}.attn_k.weight"))?,
                wv: tensor_f32(model, &format!("blk.{i}.attn_v.weight"))?,
                wo: tensor_f32(model, &format!("blk.{i}.attn_output.weight"))?,
                ffn_norm: tensor_f32(model, &format!("blk.{i}.ffn_norm.weight"))?,
                w_gate: tensor_f32(model, &format!("blk.{i}.ffn_gate.weight"))?,
                w_up: tensor_f32(model, &format!("blk.{i}.ffn_up.weight"))?,
                w_down: tensor_f32(model, &format!("blk.{i}.ffn_down.weight"))?,
            });
        }
        let final_norm = tensor_f32(model, "output_norm.weight")?;
        // Some models tie the output projection to the input embeddings.
        let output = tensor_f32(model, "output.weight").unwrap_or_else(|_| tok_embeddings.clone());

        let kv_dim = n_kv_head * head_dim;
        let cache = vec![0.0f32; n_layer * n_ctx * kv_dim];

        Ok(LlamaModel {
            config,
            tok_embeddings,
            layers,
            final_norm,
            output,
            key_cache: cache.clone(),
            val_cache: cache,
        })
    }

    /// Run one decoding step for `token` at sequence position `pos`, returning
    /// the logits over the vocabulary. Updates the KV cache in place.
    pub fn forward(&mut self, token: u32, pos: usize) -> Vec<f32> {
        let c = &self.config;
        let dim = c.n_embd;
        let head_dim = c.head_dim;
        let kv_dim = c.n_kv_head * head_dim;
        let q_dim = c.n_head * head_dim;
        let kv_mul = c.n_head / c.n_kv_head;
        let n_ctx = c.n_ctx;
        let n_layer = c.n_layer;
        let n_head = c.n_head;
        let n_ff = c.n_ff;
        let eps = c.eps;
        let rope_theta = c.rope_theta;
        let scale = (head_dim as f32).sqrt().recip();

        let base = token as usize * dim;
        let mut x = self.tok_embeddings[base..base + dim].to_vec();

        for l in 0..n_layer {
            let lw = &self.layers[l];

            // --- attention ---
            let xb = rmsnorm(&x, &lw.attn_norm, eps);
            let mut q = matvec(&lw.wq, &xb, q_dim, dim);
            let mut k = matvec(&lw.wk, &xb, kv_dim, dim);
            let v = matvec(&lw.wv, &xb, kv_dim, dim);
            rope(&mut q, n_head, head_dim, pos, rope_theta);
            rope(&mut k, c.n_kv_head, head_dim, pos, rope_theta);

            let kv_base = l * n_ctx * kv_dim;
            let off = kv_base + pos * kv_dim;
            self.key_cache[off..off + kv_dim].copy_from_slice(&k);
            self.val_cache[off..off + kv_dim].copy_from_slice(&v);

            let mut att_out = vec![0.0f32; q_dim];
            for h in 0..n_head {
                let kvh = h / kv_mul;
                let qh = &q[h * head_dim..h * head_dim + head_dim];

                let mut scores: Vec<f32> = (0..=pos)
                    .map(|t| {
                        let ko = kv_base + t * kv_dim + kvh * head_dim;
                        let kt = &self.key_cache[ko..ko + head_dim];
                        qh.iter().zip(kt).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                softmax(&mut scores);

                let oo = h * head_dim;
                for (t, &a) in scores.iter().enumerate() {
                    let vo = kv_base + t * kv_dim + kvh * head_dim;
                    let vt = &self.val_cache[vo..vo + head_dim];
                    for (dst, &vv) in att_out[oo..oo + head_dim].iter_mut().zip(vt) {
                        *dst += a * vv;
                    }
                }
            }

            let attn = matvec(&lw.wo, &att_out, dim, q_dim);
            for (xi, ai) in x.iter_mut().zip(&attn) {
                *xi += ai;
            }

            // --- feed-forward (SwiGLU) ---
            let xb = rmsnorm(&x, &lw.ffn_norm, eps);
            let gate = matvec(&lw.w_gate, &xb, n_ff, dim);
            let up = matvec(&lw.w_up, &xb, n_ff, dim);
            let hidden: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
            let down = matvec(&lw.w_down, &hidden, dim, n_ff);
            for (xi, di) in x.iter_mut().zip(&down) {
                *xi += di;
            }
        }

        let x = rmsnorm(&x, &self.final_norm, eps);
        matvec(&self.output, &x, c.vocab_size, dim)
    }

    /// Greedily generate up to `max_new` tokens after `prompt`, stopping at `eos`.
    pub fn generate(&mut self, prompt: &[u32], max_new: usize, eos: Option<u32>) -> Vec<u32> {
        let mut generated = Vec::new();
        let mut pos = 0usize;
        let mut logits: Vec<f32> = Vec::new();

        for &tok in prompt {
            if pos >= self.config.n_ctx {
                break;
            }
            logits = self.forward(tok, pos);
            pos += 1;
        }

        for _ in 0..max_new {
            if pos >= self.config.n_ctx || logits.is_empty() {
                break;
            }
            let next = argmax(&logits) as u32;
            if Some(next) == eos {
                break;
            }
            generated.push(next);
            logits = self.forward(next, pos);
            pos += 1;
        }
        generated
    }
}

/// Load a tensor as a freshly-allocated `f32` vector, dequantizing as needed.
fn tensor_f32(model: &GgufModel, name: &str) -> Result<Vec<f32>> {
    let info = model
        .tensors
        .get(name)
        .ok_or_else(|| Error::Format(format!("missing tensor '{name}'")))?;
    let n: usize = info.shape.iter().product();
    let bytes = model
        .tensor_data(name)
        .ok_or_else(|| Error::Format(format!("no data for tensor '{name}'")))?;
    ullm_core::dequant::dequantize(info.dtype, bytes, n)
}

/// `y[o] = sum_i w[o*in + i] * x[i]`, with `w` stored row-major as `[out, in]`.
fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    (0..out_dim)
        .map(|o| {
            let row = &w[o * in_dim..o * in_dim + in_dim];
            row.iter().zip(x).map(|(a, b)| a * b).sum()
        })
        .collect()
}

/// RMS normalization with a learned per-channel gain.
fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps;
    let inv = ms.sqrt().recip();
    x.iter().zip(weight).map(|(xi, wi)| xi * inv * wi).collect()
}

/// In-place numerically-stable softmax.
fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = sum.recip();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// SiLU / swish activation: `x * sigmoid(x)`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Rotary position embedding (interleaved / ggml "NORM" convention), applied to
/// each head independently in place.
fn rope(vec: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    for h in 0..n_heads {
        let off = h * head_dim;
        let mut i = 0;
        while i + 1 < head_dim {
            let freq = theta.powf(i as f32 / head_dim as f32).recip();
            let (sin, cos) = (pos as f32 * freq).sin_cos();
            let (a, b) = (vec[off + i], vec[off + i + 1]);
            vec[off + i] = a * cos - b * sin;
            vec[off + i + 1] = a * sin + b * cos;
            i += 2;
        }
    }
}

/// Index of the largest element (first on ties).
fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_normalizes_to_unit_rms() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let y = rmsnorm(&x, &w, 0.0);
        let rms = (y.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_is_a_distribution() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn matvec_identity() {
        let w = vec![1.0, 0.0, 0.0, 1.0];
        assert_eq!(matvec(&w, &[3.0, 5.0], 2, 2), vec![3.0, 5.0]);
    }

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
    }
}

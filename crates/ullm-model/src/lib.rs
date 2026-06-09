//! A correctness-first CPU runtime for the Llama architecture.
//!
//! Weights stay in their quantized GGUF form and are dequantized one row at a
//! time during each matmul (in parallel over rows), so the model uses ~4-7x less
//! memory than f32 and starts with no up-front dequantization. It remains the
//! numerical reference the Metal backend is validated against.

use std::cmp::Ordering;

use rayon::prelude::*;
use ullm_core::{DType, Error, Result};
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

/// Sampling parameters for text generation.
#[derive(Debug, Clone)]
pub struct SampleParams {
    /// Softmax temperature. `<= 0` means greedy (argmax).
    pub temperature: f32,
    /// Keep only the top-k highest logits (`0` disables).
    pub top_k: usize,
    /// Nucleus sampling: keep the smallest set with cumulative prob >= `top_p`.
    pub top_p: f32,
    /// RNG seed (`0` uses a fixed default for reproducibility).
    pub seed: u64,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        }
    }
}

/// A weight matrix kept in its quantized GGUF form (`[out, cols]`, row-major).
/// Rows are dequantized on demand during the matmul.
#[derive(Clone)]
struct QWeight {
    data: Vec<u8>,
    dtype: DType,
    out: usize,
    cols: usize,
}

impl QWeight {
    fn row_bytes(&self) -> usize {
        (self.cols / self.dtype.block_size()) * self.dtype.type_size()
    }
}

/// Per-layer weights: norms in f32 (tiny), projections kept quantized.
struct LayerWeights {
    attn_norm: Vec<f32>,
    wq: QWeight,
    wk: QWeight,
    wv: QWeight,
    wo: QWeight,
    ffn_norm: Vec<f32>,
    w_gate: QWeight,
    w_up: QWeight,
    w_down: QWeight,
}

/// A loaded Llama model with its KV cache.
pub struct LlamaModel {
    pub config: LlamaConfig,
    tok_embeddings: QWeight, // [vocab, n_embd]
    layers: Vec<LayerWeights>,
    final_norm: Vec<f32>, // [n_embd]
    output: QWeight,      // [vocab, n_embd]
    key_cache: Vec<f32>,  // [n_layer * n_ctx * kv_dim]
    val_cache: Vec<f32>,
}

impl LlamaModel {
    /// Load a Llama model from a parsed GGUF file (any supported quantization).
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

        let tok_embeddings = qweight(model, "token_embd.weight")?;
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            layers.push(LayerWeights {
                attn_norm: tensor_f32(model, &format!("blk.{i}.attn_norm.weight"))?,
                wq: qweight(model, &format!("blk.{i}.attn_q.weight"))?,
                wk: qweight(model, &format!("blk.{i}.attn_k.weight"))?,
                wv: qweight(model, &format!("blk.{i}.attn_v.weight"))?,
                wo: qweight(model, &format!("blk.{i}.attn_output.weight"))?,
                ffn_norm: tensor_f32(model, &format!("blk.{i}.ffn_norm.weight"))?,
                w_gate: qweight(model, &format!("blk.{i}.ffn_gate.weight"))?,
                w_up: qweight(model, &format!("blk.{i}.ffn_up.weight"))?,
                w_down: qweight(model, &format!("blk.{i}.ffn_down.weight"))?,
            });
        }
        let final_norm = tensor_f32(model, "output_norm.weight")?;
        // Some models tie the output projection to the input embeddings.
        let output = qweight(model, "output.weight").unwrap_or_else(|_| tok_embeddings.clone());

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
        let eps = c.eps;
        let rope_theta = c.rope_theta;
        let scale = (head_dim as f32).sqrt().recip();

        let mut x = vec![0.0f32; dim];
        {
            let w = &self.tok_embeddings;
            let rb = w.row_bytes();
            let o = token as usize;
            ullm_core::dequant::dequantize_into(w.dtype, &w.data[o * rb..o * rb + rb], &mut x)
                .expect("dequantize embedding row");
        }

        for l in 0..n_layer {
            let lw = &self.layers[l];

            // --- attention ---
            let xb = rmsnorm(&x, &lw.attn_norm, eps);
            let mut q = matvec_q(&lw.wq, &xb);
            let mut k = matvec_q(&lw.wk, &xb);
            let v = matvec_q(&lw.wv, &xb);
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

            let attn = matvec_q(&lw.wo, &att_out);
            for (xi, ai) in x.iter_mut().zip(&attn) {
                *xi += ai;
            }

            // --- feed-forward (SwiGLU) ---
            let xb = rmsnorm(&x, &lw.ffn_norm, eps);
            let gate = matvec_q(&lw.w_gate, &xb);
            let up = matvec_q(&lw.w_up, &xb);
            let hidden: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
            let down = matvec_q(&lw.w_down, &hidden);
            for (xi, di) in x.iter_mut().zip(&down) {
                *xi += di;
            }
        }

        let x = rmsnorm(&x, &self.final_norm, eps);
        matvec_q(&self.output, &x)
    }

    /// Greedily generate up to `max_new` tokens after `prompt`, stopping at `eos`.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        eos: Option<u32>,
        params: &SampleParams,
    ) -> Vec<u32> {
        let mut generated = Vec::new();
        let mut pos = 0usize;
        let mut logits: Vec<f32> = Vec::new();
        let mut rng = if params.seed == 0 {
            0x853c_49e6_748f_ea9b
        } else {
            params.seed
        };

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
            let next = sample_token(&logits, params, &mut rng);
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

/// Load a small tensor (a norm) as a freshly-allocated `f32` vector.
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

/// Load a weight matrix, keeping its quantized bytes (a copy of the mmap slice).
fn qweight(model: &GgufModel, name: &str) -> Result<QWeight> {
    let info = model
        .tensors
        .get(name)
        .ok_or_else(|| Error::Format(format!("missing tensor '{name}'")))?;
    let (out, cols) = match info.shape.as_slice() {
        [o, c] => (*o, *c),
        [c] => (1, *c),
        _ => {
            return Err(Error::Unsupported(format!(
                "weight '{name}' has rank {}",
                info.shape.len()
            )));
        }
    };
    let bytes = model
        .tensor_data(name)
        .ok_or_else(|| Error::Format(format!("no data for tensor '{name}'")))?;
    Ok(QWeight {
        data: bytes.to_vec(),
        dtype: info.dtype,
        out,
        cols,
    })
}

/// `y[o] = sum_i W[o, i] * x[i]`, dequantizing each row of `w` on the fly into a
/// per-thread reused buffer. Memory-bound work, parallel over output rows.
fn matvec_q(w: &QWeight, x: &[f32]) -> Vec<f32> {
    let rb = w.row_bytes();
    let cols = w.cols;
    (0..w.out)
        .into_par_iter()
        .map_init(
            || vec![0.0f32; cols],
            |buf, o| {
                ullm_core::dequant::dequantize_into(w.dtype, &w.data[o * rb..o * rb + rb], buf)
                    .expect("dequantize weight row");
                buf.iter().zip(x).map(|(a, b)| a * b).sum()
            },
        )
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

/// Sample a token id from `logits` according to `params`.
fn sample_token(logits: &[f32], params: &SampleParams, rng: &mut u64) -> u32 {
    if params.temperature <= 0.0 {
        return argmax(logits) as u32;
    }

    let mut cand: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, l / params.temperature))
        .collect();
    cand.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    if params.top_k > 0 && params.top_k < cand.len() {
        cand.truncate(params.top_k);
    }

    let max = cand[0].1;
    let mut probs: Vec<f32> = cand.iter().map(|(_, l)| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    let mut cutoff = probs.len();
    if params.top_p < 1.0 {
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= params.top_p {
                cutoff = i + 1;
                break;
            }
        }
    }

    let total: f32 = probs[..cutoff].iter().sum();
    let r = next_f32(rng) * total;
    let mut acc = 0.0;
    for (&p, c) in probs[..cutoff].iter().zip(&cand[..cutoff]) {
        acc += p;
        if r < acc {
            return c.0 as u32;
        }
    }
    cand[cutoff - 1].0 as u32
}

/// One step of a SplitMix64 RNG.
fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform `f32` in `[0, 1)`.
fn next_f32(state: &mut u64) -> f32 {
    (next_u64(state) >> 40) as f32 / (1u64 << 24) as f32
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
    fn matvec_q_f32_identity() {
        let data: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let w = QWeight {
            data,
            dtype: DType::F32,
            out: 2,
            cols: 2,
        };
        assert_eq!(matvec_q(&w, &[3.0, 5.0]), vec![3.0, 5.0]);
    }

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
    }
}

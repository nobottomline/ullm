//! A correctness-first CPU runtime for the Llama architecture.
//!
//! Weights stay in their quantized GGUF form and are dequantized one row at a
//! time during each matmul (in parallel over rows), so the model uses ~4-7x less
//! memory than f32 and starts with no up-front dequantization. It remains the
//! numerical reference the Metal backend is validated against.

mod config;
mod math;
mod sample;
mod weights;

use ullm_core::{DType, Error, Result};
use ullm_gguf::GgufModel;
use ullm_safetensors::SafeTensorsModel;

pub use config::{Arch, LlamaConfig};
pub use sample::SampleParams;

use math::{add_bias, gelu, matvec_q, rmsnorm, rope, rope_neox, silu, softmax};
use sample::sample_token;
use weights::{qweight, tensor_f32};

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

/// Per-layer weights: norms in f32 (tiny), projections kept quantized. Q/K/V
/// biases are present on some architectures (e.g. Qwen2).
struct LayerWeights {
    attn_norm: Vec<f32>,
    wq: QWeight,
    wk: QWeight,
    wv: QWeight,
    wo: QWeight,
    q_bias: Option<Vec<f32>>,
    k_bias: Option<Vec<f32>>,
    v_bias: Option<Vec<f32>>,
    // Gemma-style extras (None on Llama/Qwen).
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    post_attn_norm: Option<Vec<f32>>,
    post_ffn_norm: Option<Vec<f32>>,
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
    /// RoPE convention: NeoX (rotate-half) for un-permuted weights loaded from
    /// SafeTensors / HF; interleaved for GGUF (whose Llama weights are permuted).
    rope_neox: bool,
}

impl LlamaModel {
    /// Load a Llama model from a parsed GGUF file (any supported quantization).
    pub fn from_gguf(model: &GgufModel) -> Result<Self> {
        let arch_str = model.architecture().unwrap_or_default().to_string();
        let arch = match arch_str.as_str() {
            "llama" => Arch::Llama,
            "qwen2" => Arch::Qwen2,
            "gemma3" => Arch::Gemma3,
            other => {
                return Err(Error::Unsupported(format!(
                    "architecture '{other}' is not supported yet (llama, qwen2, gemma3)"
                )));
            }
        };

        // Metadata keys are namespaced by architecture, e.g. `llama.block_count`.
        let req_u = |suffix: &str| -> Result<u64> {
            let k = format!("{arch_str}.{suffix}");
            model
                .metadata_get(&k)
                .and_then(|v| v.to_u64())
                .ok_or_else(|| Error::Format(format!("missing metadata '{k}'")))
        };
        let opt_u = |suffix: &str| {
            model
                .metadata_get(&format!("{arch_str}.{suffix}"))
                .and_then(|v| v.to_u64())
        };
        let opt_f = |suffix: &str| {
            model
                .metadata_get(&format!("{arch_str}.{suffix}"))
                .and_then(|v| v.to_f32())
        };

        let n_embd = req_u("embedding_length")? as usize;
        let n_head = req_u("attention.head_count")? as usize;
        let n_layer = req_u("block_count")? as usize;
        let n_kv_head = opt_u("attention.head_count_kv").unwrap_or(n_head as u64) as usize;
        let n_ff = req_u("feed_forward_length")? as usize;
        let n_ctx = (opt_u("context_length").unwrap_or(2048) as usize).min(8192);
        let head_dim = opt_u("attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(n_embd / n_head);
        let rope_theta = opt_f("rope.freq_base").unwrap_or(10000.0);
        let eps = opt_f("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let vocab_size = model.model_spec().vocab_size as usize;

        let config = LlamaConfig {
            arch,
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
                q_bias: tensor_f32(model, &format!("blk.{i}.attn_q.bias")).ok(),
                k_bias: tensor_f32(model, &format!("blk.{i}.attn_k.bias")).ok(),
                v_bias: tensor_f32(model, &format!("blk.{i}.attn_v.bias")).ok(),
                q_norm: tensor_f32(model, &format!("blk.{i}.attn_q_norm.weight")).ok(),
                k_norm: tensor_f32(model, &format!("blk.{i}.attn_k_norm.weight")).ok(),
                post_attn_norm: tensor_f32(model, &format!("blk.{i}.post_attention_norm.weight"))
                    .ok(),
                post_ffn_norm: tensor_f32(model, &format!("blk.{i}.post_ffw_norm.weight")).ok(),
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
            // GGUF Llama/Qwen weights are permuted for interleaved RoPE.
            rope_neox: false,
        })
    }

    /// Load a model from a Hugging Face SafeTensors directory (`config.json` +
    /// `*.safetensors`). Weights are kept in their stored BF16/F16/F32 form and
    /// dequantized per row, exactly as for GGUF.
    pub fn from_safetensors(model: &SafeTensorsModel) -> Result<Self> {
        let arch_name = model
            .config()
            .get("architectures")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let arch = match arch_name {
            "Qwen3ForCausalLM" => Arch::Qwen3,
            other => {
                return Err(Error::Unsupported(format!(
                    "SafeTensors architecture '{other}' is not supported yet (Qwen3)"
                )));
            }
        };

        let req = |k: &str| -> Result<usize> {
            model
                .config_usize(k)
                .ok_or_else(|| Error::Format(format!("config.json: missing '{k}'")))
        };
        let n_embd = req("hidden_size")?;
        let n_head = req("num_attention_heads")?;
        let n_layer = req("num_hidden_layers")?;
        let n_kv_head = model.config_usize("num_key_value_heads").unwrap_or(n_head);
        let n_ff = req("intermediate_size")?;
        let head_dim = model
            .config_usize("head_dim")
            .unwrap_or(n_embd / n_head);
        let n_ctx = model
            .config_usize("max_position_embeddings")
            .unwrap_or(8192)
            .min(8192);
        let rope_theta = model.config_f32("rope_theta").unwrap_or(1_000_000.0);
        let eps = model.config_f32("rms_norm_eps").unwrap_or(1e-6);
        let vocab_size = req("vocab_size")?;

        let config = LlamaConfig {
            arch,
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

        let tok_embeddings = qweight(model, "model.embed_tokens.weight")?;
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = format!("model.layers.{i}");
            layers.push(LayerWeights {
                attn_norm: tensor_f32(model, &format!("{p}.input_layernorm.weight"))?,
                wq: qweight(model, &format!("{p}.self_attn.q_proj.weight"))?,
                wk: qweight(model, &format!("{p}.self_attn.k_proj.weight"))?,
                wv: qweight(model, &format!("{p}.self_attn.v_proj.weight"))?,
                wo: qweight(model, &format!("{p}.self_attn.o_proj.weight"))?,
                q_bias: tensor_f32(model, &format!("{p}.self_attn.q_proj.bias")).ok(),
                k_bias: tensor_f32(model, &format!("{p}.self_attn.k_proj.bias")).ok(),
                v_bias: tensor_f32(model, &format!("{p}.self_attn.v_proj.bias")).ok(),
                q_norm: tensor_f32(model, &format!("{p}.self_attn.q_norm.weight")).ok(),
                k_norm: tensor_f32(model, &format!("{p}.self_attn.k_norm.weight")).ok(),
                post_attn_norm: None,
                post_ffn_norm: None,
                ffn_norm: tensor_f32(model, &format!("{p}.post_attention_layernorm.weight"))?,
                w_gate: qweight(model, &format!("{p}.mlp.gate_proj.weight"))?,
                w_up: qweight(model, &format!("{p}.mlp.up_proj.weight"))?,
                w_down: qweight(model, &format!("{p}.mlp.down_proj.weight"))?,
            });
        }
        let final_norm = tensor_f32(model, "model.norm.weight")?;
        // Tied embeddings when there is no separate lm_head.
        let output = qweight(model, "lm_head.weight").unwrap_or_else(|_| tok_embeddings.clone());

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
            // HF weights are not permuted: use NeoX RoPE.
            rope_neox: true,
        })
    }

    /// Run one decoding step for `token` at sequence position `pos`, returning
    /// the logits over the vocabulary. Updates the KV cache in place.
    pub fn forward(&mut self, token: u32, pos: usize) -> Vec<f32> {
        if self.config.arch == Arch::Gemma3 {
            self.forward_gemma(token, pos)
        } else {
            self.forward_llama(token, pos)
        }
    }

    fn forward_llama(&mut self, token: u32, pos: usize) -> Vec<f32> {
        let use_neox = self.rope_neox;
        let c = &self.config;
        let dim = c.n_embd;
        let head_dim = c.head_dim;
        let kv_dim = c.n_kv_head * head_dim;
        let q_dim = c.n_head * head_dim;
        let kv_mul = c.n_head / c.n_kv_head;
        let n_ctx = c.n_ctx;
        let n_layer = c.n_layer;
        let n_head = c.n_head;
        let n_kv_head = c.n_kv_head;
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
            let mut v = matvec_q(&lw.wv, &xb);
            if let Some(b) = &lw.q_bias {
                add_bias(&mut q, b);
            }
            if let Some(b) = &lw.k_bias {
                add_bias(&mut k, b);
            }
            if let Some(b) = &lw.v_bias {
                add_bias(&mut v, b);
            }

            // Qwen3-style per-head Q/K RMSNorm, applied before RoPE.
            if let Some(qn) = &lw.q_norm {
                for h in 0..n_head {
                    let s = h * head_dim;
                    let normed = rmsnorm(&q[s..s + head_dim], qn, eps);
                    q[s..s + head_dim].copy_from_slice(&normed);
                }
            }
            if let Some(kn) = &lw.k_norm {
                for h in 0..n_kv_head {
                    let s = h * head_dim;
                    let normed = rmsnorm(&k[s..s + head_dim], kn, eps);
                    k[s..s + head_dim].copy_from_slice(&normed);
                }
            }

            if use_neox {
                rope_neox(&mut q, n_head, head_dim, pos, rope_theta);
                rope_neox(&mut k, n_kv_head, head_dim, pos, rope_theta);
            } else {
                rope(&mut q, n_head, head_dim, pos, rope_theta);
                rope(&mut k, n_kv_head, head_dim, pos, rope_theta);
            }

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

    /// Gemma-3 forward: scaled embeddings, `(1+w)` RMSNorm, per-head Q/K-norm,
    /// NeoX RoPE, GeGLU, and sandwich (post-attention / post-FFN) norms. Sliding
    /// window attention is treated as full attention (correct for short context).
    fn forward_gemma(&mut self, token: u32, pos: usize) -> Vec<f32> {
        let c = &self.config;
        let dim = c.n_embd;
        let head_dim = c.head_dim;
        let kv_dim = c.n_kv_head * head_dim;
        let q_dim = c.n_head * head_dim;
        let kv_mul = c.n_head / c.n_kv_head;
        let n_ctx = c.n_ctx;
        let n_layer = c.n_layer;
        let n_head = c.n_head;
        let n_kv_head = c.n_kv_head;
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
        // Gemma scales the input embeddings by sqrt(n_embd).
        let emb_scale = (dim as f32).sqrt();
        for xi in x.iter_mut() {
            *xi *= emb_scale;
        }

        for l in 0..n_layer {
            let lw = &self.layers[l];

            // --- attention (pre-norm + sandwich post-norm) ---
            let xb = rmsnorm(&x, &lw.attn_norm, eps);
            let mut q = matvec_q(&lw.wq, &xb);
            let mut k = matvec_q(&lw.wk, &xb);
            let v = matvec_q(&lw.wv, &xb);

            // Per-head Q/K RMSNorm before RoPE.
            if let Some(qn) = &lw.q_norm {
                for h in 0..n_head {
                    let s = h * head_dim;
                    let normed = rmsnorm(&q[s..s + head_dim], qn, eps);
                    q[s..s + head_dim].copy_from_slice(&normed);
                }
            }
            if let Some(kn) = &lw.k_norm {
                for h in 0..n_kv_head {
                    let s = h * head_dim;
                    let normed = rmsnorm(&k[s..s + head_dim], kn, eps);
                    k[s..s + head_dim].copy_from_slice(&normed);
                }
            }

            // Gemma GGUF weights are not permuted: use NeoX (rotate-half) RoPE.
            rope_neox(&mut q, n_head, head_dim, pos, rope_theta);
            rope_neox(&mut k, n_kv_head, head_dim, pos, rope_theta);

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

            let mut attn = matvec_q(&lw.wo, &att_out);
            if let Some(w) = &lw.post_attn_norm {
                attn = rmsnorm(&attn, w, eps);
            }
            for (xi, ai) in x.iter_mut().zip(&attn) {
                *xi += ai;
            }

            // --- feed-forward (GeGLU + sandwich post-norm) ---
            let xb = rmsnorm(&x, &lw.ffn_norm, eps);
            let gate = matvec_q(&lw.w_gate, &xb);
            let up = matvec_q(&lw.w_up, &xb);
            let hidden: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| gelu(*g) * u).collect();
            let mut down = matvec_q(&lw.w_down, &hidden);
            if let Some(w) = &lw.post_ffn_norm {
                down = rmsnorm(&down, w, eps);
            }
            for (xi, di) in x.iter_mut().zip(&down) {
                *xi += di;
            }
        }

        let x = rmsnorm(&x, &self.final_norm, eps);
        matvec_q(&self.output, &x)
    }

    /// Generate tokens after `prompt`, invoking `on_token` for each new token.
    /// Stops at EOS, `max_new`, the context limit, or when `on_token` returns
    /// `false`.
    pub fn generate_stream<F: FnMut(u32) -> bool>(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        eos: Option<u32>,
        params: &SampleParams,
        mut on_token: F,
    ) {
        let mut pos = 0usize;
        let mut logits: Vec<f32> = Vec::new();
        let mut rng = if params.seed == 0 {
            0x853c_49e6_748f_ea9b
        } else {
            params.seed
        };

        for &tok in prompt {
            if pos >= self.config.n_ctx {
                return;
            }
            logits = self.forward(tok, pos);
            pos += 1;
        }

        for _ in 0..max_new {
            if pos >= self.config.n_ctx || logits.is_empty() {
                break;
            }
            let next = sample_token(&logits, params, &mut rng);
            if Some(next) == eos || !on_token(next) {
                break;
            }
            logits = self.forward(next, pos);
            pos += 1;
        }
    }

    /// Collect generated token ids (a convenience wrapper over `generate_stream`).
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        eos: Option<u32>,
        params: &SampleParams,
    ) -> Vec<u32> {
        let mut out = Vec::new();
        self.generate_stream(prompt, max_new, eos, params, |t| {
            out.push(t);
            true
        });
        out
    }
}

//! A correctness-first CPU runtime for the Llama architecture.
//!
//! Weights stay in their quantized GGUF form and are dequantized one row at a
//! time during each matmul (in parallel over rows), so the model uses ~4-7x less
//! memory than f32 and starts with no up-front dequantization. It remains the
//! numerical reference the Metal backend is validated against.

mod config;
mod math;
mod mlx;
mod sample;
mod weights;

use ullm_core::{DType, Error, Result};
use ullm_gguf::GgufModel;
use ullm_metal::{GpuExperts, GpuForward, GpuLayerInput, GpuModelInput, GpuParams, GpuWeight};
use ullm_safetensors::SafeTensorsModel;

pub use config::{Arch, LlamaConfig};
pub use sample::SampleParams;

use math::{
    add_bias, dequant_mlx_row, gelu, matmul_q, matvec_q, rmsnorm, rope, rope_neox, silu, softmax,
};
use sample::sample_token;
use weights::{qweight, tensor_f32};

/// Side tables for an MLX 4-bit weight kept resident (group scale + bias),
/// instead of pre-dequantizing it to a wider type.
#[derive(Clone)]
struct MlxQuant {
    scales: Vec<f32>,
    biases: Vec<f32>,
    group_size: usize,
}

/// A weight matrix kept in its quantized form (`[out, cols]`, row-major). Rows
/// are dequantized on demand during the matmul. `mlx` is set for MLX 4-bit
/// weights (whose scales/biases live in separate tensors).
#[derive(Clone)]
struct QWeight {
    data: Vec<u8>,
    dtype: DType,
    out: usize,
    cols: usize,
    mlx: Option<MlxQuant>,
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
    // Mixture-of-experts FFN: when `moe_gate` is set, the dense `w_*` above are
    // placeholders and the router selects among the per-expert weights instead.
    moe_gate: Option<QWeight>,
    experts_gate: Vec<QWeight>,
    experts_up: Vec<QWeight>,
    experts_down: Vec<QWeight>,
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
    /// Optional resident GPU forward; when present, `forward` runs on the GPU.
    gpu: Option<GpuForward>,
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
        // Gemma-3 uses sliding-window attention on its local layers.
        let sliding_window = if arch == Arch::Gemma3 {
            opt_u("attention.sliding_window").unwrap_or(1024) as usize
        } else {
            0
        };

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
            n_experts: 0,
            n_experts_used: 0,
            moe_inter: 0,
            sliding_window,
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
                moe_gate: None,
                experts_gate: Vec::new(),
                experts_up: Vec::new(),
                experts_down: Vec::new(),
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
            // GGUF Llama/Qwen weights are permuted for interleaved RoPE; Gemma
            // GGUF weights are not, so Gemma uses NeoX even from GGUF.
            rope_neox: arch == Arch::Gemma3,
            gpu: None,
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
        let head_dim = model.config_usize("head_dim").unwrap_or(n_embd / n_head);
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
            n_experts: 0,
            n_experts_used: 0,
            moe_inter: 0,
            sliding_window: 0,
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
                moe_gate: None,
                experts_gate: Vec::new(),
                experts_up: Vec::new(),
                experts_down: Vec::new(),
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
            gpu: None,
        })
    }

    /// Run one decoding step for `token` at sequence position `pos`, returning
    /// the logits over the vocabulary. Updates the KV cache in place.
    pub fn forward(&mut self, token: u32, pos: usize) -> Vec<f32> {
        if self.gpu.is_some() {
            let x_init = self.embed(token);
            return self
                .gpu
                .as_ref()
                .unwrap()
                .forward(&x_init, pos)
                .expect("gpu forward");
        }
        if self.config.arch == Arch::Gemma3 {
            self.forward_gemma(token, pos)
        } else {
            self.forward_llama(token, pos)
        }
    }

    /// Move the model's weights onto the GPU and route `forward` through the
    /// single-command-buffer Metal forward pass. Falls back to CPU if no GPU.
    pub fn enable_gpu(&mut self) -> Result<()> {
        let gpu = self.build_gpu()?;
        self.gpu = Some(gpu);
        Ok(())
    }

    /// Whether the GPU forward path is active.
    pub fn gpu_enabled(&self) -> bool {
        self.gpu.is_some()
    }

    /// Step through GPU layer 0 op-by-op, reporting NaN/inf/max (debugging).
    pub fn gpu_forward_debug(&self, token: u32, pos: usize) {
        if let Some(gpu) = &self.gpu {
            let x = self.embed(token);
            gpu.forward_debug(&x, pos);
        }
    }

    /// The (already scaled, for Gemma) input embedding for `token`.
    fn embed(&self, token: u32) -> Vec<f32> {
        let w = &self.tok_embeddings;
        let o = token as usize;
        let mut x = vec![0.0f32; self.config.n_embd];
        if let Some(mlx) = &w.mlx {
            dequant_mlx_row(&w.data, mlx, o, w.cols, &mut x);
        } else {
            let rb = w.row_bytes();
            ullm_core::dequant::dequantize_into(w.dtype, &w.data[o * rb..o * rb + rb], &mut x)
                .expect("dequantize embedding row");
        }
        if self.config.arch == Arch::Gemma3 {
            let s = (self.config.n_embd as f32).sqrt();
            for xi in &mut x {
                *xi *= s;
            }
        }
        x
    }

    fn build_gpu(&self) -> Result<GpuForward> {
        let c = &self.config;
        let qk_norm = self.layers.first().is_some_and(|l| l.q_norm.is_some());
        let sandwich_norm = self
            .layers
            .first()
            .is_some_and(|l| l.post_attn_norm.is_some());
        let params = GpuParams {
            n_embd: c.n_embd,
            n_layer: c.n_layer,
            n_head: c.n_head,
            n_kv_head: c.n_kv_head,
            head_dim: c.head_dim,
            n_ff: c.n_ff,
            n_ctx: c.n_ctx,
            vocab: c.vocab_size,
            rope_theta: c.rope_theta,
            eps: c.eps,
            rope_neox: self.rope_neox,
            qk_norm,
            sandwich_norm,
            geglu: c.arch == Arch::Gemma3,
            n_experts: c.n_experts,
            n_experts_used: c.n_experts_used,
            moe_inter: c.moe_inter,
            sliding_window: c.sliding_window,
        };
        fn gw(w: &QWeight) -> GpuWeight<'_> {
            GpuWeight {
                dtype: w.dtype,
                bytes: &w.data,
                out: w.out,
                cols: w.cols,
                mlx_scales: w.mlx.as_ref().map(|m| m.scales.as_slice()),
                mlx_biases: w.mlx.as_ref().map(|m| m.biases.as_slice()),
                mlx_group: w.mlx.as_ref().map_or(0, |m| m.group_size),
            }
        }
        // Concatenate a layer's per-expert weights into one stacked upload.
        fn experts(es: &[QWeight]) -> GpuExperts {
            let mut bytes = Vec::new();
            let mut scales = Vec::new();
            let mut biases = Vec::new();
            for e in es {
                bytes.extend_from_slice(&e.data);
                let m = e.mlx.as_ref().expect("MoE expert must be MLX-quantized");
                scales.extend_from_slice(&m.scales);
                biases.extend_from_slice(&m.biases);
            }
            let first = &es[0];
            GpuExperts {
                bytes,
                scales,
                biases,
                n_experts: es.len(),
                out: first.out,
                cols: first.cols,
                group: first.mlx.as_ref().map_or(64, |m| m.group_size),
            }
        }
        let layers = self
            .layers
            .iter()
            .map(|l| GpuLayerInput {
                attn_norm: &l.attn_norm,
                wq: gw(&l.wq),
                wk: gw(&l.wk),
                wv: gw(&l.wv),
                wo: gw(&l.wo),
                q_bias: l.q_bias.as_deref(),
                k_bias: l.k_bias.as_deref(),
                v_bias: l.v_bias.as_deref(),
                q_norm: l.q_norm.as_deref(),
                k_norm: l.k_norm.as_deref(),
                post_attn_norm: l.post_attn_norm.as_deref(),
                post_ffn_norm: l.post_ffn_norm.as_deref(),
                ffn_norm: &l.ffn_norm,
                w_gate: gw(&l.w_gate),
                w_up: gw(&l.w_up),
                w_down: gw(&l.w_down),
                moe_gate: l.moe_gate.as_ref().map(gw),
                experts_gate: (!l.experts_gate.is_empty()).then(|| experts(&l.experts_gate)),
                experts_up: (!l.experts_up.is_empty()).then(|| experts(&l.experts_up)),
                experts_down: (!l.experts_down.is_empty()).then(|| experts(&l.experts_down)),
            })
            .collect();
        let input = GpuModelInput {
            params,
            output: gw(&self.output),
            final_norm: &self.final_norm,
            layers,
        };
        GpuForward::new(&input)
    }

    fn forward_llama(&mut self, token: u32, pos: usize) -> Vec<f32> {
        let use_neox = self.rope_neox;
        let c = &self.config;
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

        let mut x = self.embed(token);

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

            // --- feed-forward (SwiGLU, or mixture-of-experts) ---
            let xb = rmsnorm(&x, &lw.ffn_norm, eps);
            let down = if let Some(gate_w) = &lw.moe_gate {
                moe_ffn(lw, gate_w, &xb, c.n_experts_used)
            } else {
                let gate = matvec_q(&lw.w_gate, &xb);
                let up = matvec_q(&lw.w_up, &xb);
                let hidden: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
                matvec_q(&lw.w_down, &hidden)
            };
            for (xi, di) in x.iter_mut().zip(&down) {
                *xi += di;
            }
        }

        let x = rmsnorm(&x, &self.final_norm, eps);
        matvec_q(&self.output, &x)
    }

    /// Batched prefill for the Llama family (Llama / Qwen2 / Qwen3 / Qwen3-MoE):
    /// run all `tokens` (at positions `start_pos .. start_pos + tokens.len()`)
    /// through ONE forward, reading each attention/FFN weight only once via
    /// `matmul_q` instead of once per token. Fills the KV cache for every
    /// position and returns the LAST token's logits (all a sampler needs).
    ///
    /// Numerically identical to calling [`forward_llama`] token-by-token — the
    /// causal mask is exactly the per-token `0..=pos` score window, so position
    /// `s` never sees a future key. Gemma (sandwich norms / sliding window) and
    /// the GPU path keep their own token-by-token routes.
    fn forward_batch(&mut self, tokens: &[u32], start_pos: usize) -> Vec<f32> {
        let use_neox = self.rope_neox;
        let c = &self.config;
        let head_dim = c.head_dim;
        let kv_dim = c.n_kv_head * head_dim;
        let q_dim = c.n_head * head_dim;
        let kv_mul = c.n_head / c.n_kv_head;
        let n_ctx = c.n_ctx;
        let n_layer = c.n_layer;
        let n_head = c.n_head;
        let n_kv_head = c.n_kv_head;
        let n_embd = c.n_embd;
        let eps = c.eps;
        let rope_theta = c.rope_theta;
        let n_used = c.n_experts_used;
        let scale = (head_dim as f32).sqrt().recip();
        let s_len = tokens.len();

        // Token-major activations: row `s` is `x[s * n_embd .. (s + 1) * n_embd]`.
        let mut x = vec![0.0f32; s_len * n_embd];
        for (s, &tok) in tokens.iter().enumerate() {
            x[s * n_embd..(s + 1) * n_embd].copy_from_slice(&self.embed(tok));
        }

        for l in 0..n_layer {
            let lw = &self.layers[l];

            // --- attention: norm -> Q/K/V (batched matmul) ---
            let mut xb = vec![0.0f32; s_len * n_embd];
            for s in 0..s_len {
                let n = rmsnorm(&x[s * n_embd..(s + 1) * n_embd], &lw.attn_norm, eps);
                xb[s * n_embd..(s + 1) * n_embd].copy_from_slice(&n);
            }
            let mut q = matmul_q(&lw.wq, &xb, s_len);
            let mut k = matmul_q(&lw.wk, &xb, s_len);
            let mut v = matmul_q(&lw.wv, &xb, s_len);

            // Per-token bias, Q/K-norm, RoPE, then write the KV cache for every
            // position BEFORE any attention reads it.
            for s in 0..s_len {
                let pos = start_pos + s;
                let qs = &mut q[s * q_dim..s * q_dim + q_dim];
                let ks = &mut k[s * kv_dim..s * kv_dim + kv_dim];
                let vs = &mut v[s * kv_dim..s * kv_dim + kv_dim];
                if let Some(b) = &lw.q_bias {
                    add_bias(qs, b);
                }
                if let Some(b) = &lw.k_bias {
                    add_bias(ks, b);
                }
                if let Some(b) = &lw.v_bias {
                    add_bias(vs, b);
                }
                if let Some(qn) = &lw.q_norm {
                    for h in 0..n_head {
                        let hs = h * head_dim;
                        let normed = rmsnorm(&qs[hs..hs + head_dim], qn, eps);
                        qs[hs..hs + head_dim].copy_from_slice(&normed);
                    }
                }
                if let Some(kn) = &lw.k_norm {
                    for h in 0..n_kv_head {
                        let hs = h * head_dim;
                        let normed = rmsnorm(&ks[hs..hs + head_dim], kn, eps);
                        ks[hs..hs + head_dim].copy_from_slice(&normed);
                    }
                }
                if use_neox {
                    rope_neox(qs, n_head, head_dim, pos, rope_theta);
                    rope_neox(ks, n_kv_head, head_dim, pos, rope_theta);
                } else {
                    rope(qs, n_head, head_dim, pos, rope_theta);
                    rope(ks, n_kv_head, head_dim, pos, rope_theta);
                }
                let off = l * n_ctx * kv_dim + pos * kv_dim;
                self.key_cache[off..off + kv_dim].copy_from_slice(ks);
                self.val_cache[off..off + kv_dim].copy_from_slice(vs);
            }

            // Causal attention per token: query `pos` attends to keys `0..=pos`.
            let mut att_out = vec![0.0f32; s_len * q_dim];
            for s in 0..s_len {
                let pos = start_pos + s;
                let qrow = &q[s * q_dim..s * q_dim + q_dim];
                let orow = &mut att_out[s * q_dim..s * q_dim + q_dim];
                for h in 0..n_head {
                    let kvh = h / kv_mul;
                    let qh = &qrow[h * head_dim..h * head_dim + head_dim];
                    let mut scores: Vec<f32> = (0..=pos)
                        .map(|t| {
                            let ko = l * n_ctx * kv_dim + t * kv_dim + kvh * head_dim;
                            let kt = &self.key_cache[ko..ko + head_dim];
                            qh.iter().zip(kt).map(|(a, b)| a * b).sum::<f32>() * scale
                        })
                        .collect();
                    softmax(&mut scores);
                    let oo = h * head_dim;
                    for (t, &a) in scores.iter().enumerate() {
                        let vo = l * n_ctx * kv_dim + t * kv_dim + kvh * head_dim;
                        let vt = &self.val_cache[vo..vo + head_dim];
                        for (dst, &vv) in orow[oo..oo + head_dim].iter_mut().zip(vt) {
                            *dst += a * vv;
                        }
                    }
                }
            }

            let attn = matmul_q(&lw.wo, &att_out, s_len);
            for (xi, ai) in x.iter_mut().zip(&attn) {
                *xi += ai;
            }

            // --- feed-forward (SwiGLU, or per-token mixture-of-experts) ---
            let mut xb2 = vec![0.0f32; s_len * n_embd];
            for s in 0..s_len {
                let n = rmsnorm(&x[s * n_embd..(s + 1) * n_embd], &lw.ffn_norm, eps);
                xb2[s * n_embd..(s + 1) * n_embd].copy_from_slice(&n);
            }
            let down = if let Some(gate_w) = &lw.moe_gate {
                let mut d = vec![0.0f32; s_len * n_embd];
                for s in 0..s_len {
                    let o = moe_ffn(lw, gate_w, &xb2[s * n_embd..(s + 1) * n_embd], n_used);
                    d[s * n_embd..(s + 1) * n_embd].copy_from_slice(&o);
                }
                d
            } else {
                let gate = matmul_q(&lw.w_gate, &xb2, s_len);
                let up = matmul_q(&lw.w_up, &xb2, s_len);
                let hidden: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
                matmul_q(&lw.w_down, &hidden, s_len)
            };
            for (xi, di) in x.iter_mut().zip(&down) {
                *xi += di;
            }
        }

        // Only the last token's logits matter for the next sample.
        let last = (s_len - 1) * n_embd;
        let xf = rmsnorm(&x[last..last + n_embd], &self.final_norm, eps);
        matvec_q(&self.output, &xf)
    }

    /// Gemma-3 forward: scaled embeddings, `(1+w)` RMSNorm, per-head Q/K-norm,
    /// NeoX RoPE, GeGLU, and sandwich (post-attention / post-FFN) norms. Sliding
    /// window attention is treated as full attention (correct for short context).
    fn forward_gemma(&mut self, token: u32, pos: usize) -> Vec<f32> {
        let c = &self.config;
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
        let sliding_window = c.sliding_window;
        let scale = (head_dim as f32).sqrt().recip();

        // Embedding (with Gemma's sqrt(n_embd) scale) is computed in `embed`.
        let mut x = self.embed(token);

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

            // Gemma local layers (every 6th is global) use sliding-window
            // attention: only the most recent `sliding_window` keys are visible.
            let attn_start = if sliding_window > 0 && l % 6 != 5 && pos + 1 > sliding_window {
                pos + 1 - sliding_window
            } else {
                0
            };
            let mut att_out = vec![0.0f32; q_dim];
            for h in 0..n_head {
                let kvh = h / kv_mul;
                let qh = &q[h * head_dim..h * head_dim + head_dim];
                let mut scores: Vec<f32> = (attn_start..=pos)
                    .map(|t| {
                        let ko = kv_base + t * kv_dim + kvh * head_dim;
                        let kt = &self.key_cache[ko..ko + head_dim];
                        qh.iter().zip(kt).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                softmax(&mut scores);
                let oo = h * head_dim;
                for (idx, &a) in scores.iter().enumerate() {
                    let t = attn_start + idx;
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

        // Prefill: on CPU for the Llama family, run the whole prompt through one
        // batched forward (each weight read once) instead of token-by-token. The
        // GPU path and Gemma keep their per-token routes.
        let batched = !self.gpu_enabled() && self.config.arch != Arch::Gemma3;
        if batched && !prompt.is_empty() {
            let take = prompt.len().min(self.config.n_ctx);
            logits = self.forward_batch(&prompt[..take], 0);
            pos = take;
        } else {
            for &tok in prompt {
                if pos >= self.config.n_ctx {
                    return;
                }
                logits = self.forward(tok, pos);
                pos += 1;
            }
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

    /// Debug: run `prompt` through the batched prefill and the token-by-token
    /// CPU forward, then compare the final-position logits. They must be
    /// numerically identical up to floating-point reduction order. Also times
    /// both paths to surface the batched-prefill speedup.
    pub fn prefill_check(&mut self, prompt: &[u32]) -> PrefillCheck {
        let t0 = std::time::Instant::now();
        let lb = self.forward_batch(prompt, 0);
        let batch_ms = t0.elapsed().as_secs_f64() * 1e3;

        let t1 = std::time::Instant::now();
        let mut ls = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            ls = self.forward_llama(tok, pos);
        }
        let seq_ms = t1.elapsed().as_secs_f64() * 1e3;

        let max_diff = lb
            .iter()
            .zip(&ls)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                    if x > bv { (i, x) } else { (bi, bv) }
                })
                .0 as u32
        };
        let (batch_argmax, seq_argmax) = (argmax(&lb), argmax(&ls));
        PrefillCheck {
            max_diff,
            agree: batch_argmax == seq_argmax,
            batch_argmax,
            seq_argmax,
            batch_ms,
            seq_ms,
        }
    }
}

/// Result of [`LlamaModel::prefill_check`]: a correctness comparison plus the
/// wall-clock time of each prefill path.
pub struct PrefillCheck {
    pub max_diff: f32,
    pub agree: bool,
    pub batch_argmax: u32,
    pub seq_argmax: u32,
    pub batch_ms: f64,
    pub seq_ms: f64,
}

/// Mixture-of-experts feed-forward: route `xb` to the top-k experts by router
/// logit, then sum their SwiGLU outputs weighted by the renormalized softmax.
fn moe_ffn(lw: &LayerWeights, gate_w: &QWeight, xb: &[f32], n_used: usize) -> Vec<f32> {
    let logits = matvec_q(gate_w, xb);
    // Top-k experts by router logit.
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    idx.truncate(n_used);
    // Softmax over the selected logits (== softmax over all then renormalize).
    let max = idx
        .iter()
        .map(|&i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut wsum = 0.0f32;
    let mut weights: Vec<f32> = idx
        .iter()
        .map(|&i| {
            let e = (logits[i] - max).exp();
            wsum += e;
            e
        })
        .collect();
    for w in &mut weights {
        *w /= wsum;
    }
    // Accumulate the weighted expert outputs.
    let dim = lw.experts_down.first().map(|q| q.out).unwrap_or(0);
    let mut out = vec![0.0f32; dim];
    for (k, &e) in idx.iter().enumerate() {
        let gate = matvec_q(&lw.experts_gate[e], xb);
        let up = matvec_q(&lw.experts_up[e], xb);
        let hidden: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
        let down = matvec_q(&lw.experts_down[e], &hidden);
        let wk = weights[k];
        for (o, d) in out.iter_mut().zip(&down) {
            *o += wk * d;
        }
    }
    out
}

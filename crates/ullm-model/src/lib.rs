//! A correctness-first CPU runtime for the Llama family — Llama 2/3, Qwen2,
//! Qwen3, Qwen3-MoE and Gemma-3 — loaded from GGUF, SafeTensors or MLX.
//!
//! Weights stay in their quantized form and are dequantized one row at a time
//! during each matmul (in parallel over rows), so the model uses ~4-7x less
//! memory than f32 and starts with no up-front dequantization. It remains the
//! numerical reference the Metal backend is validated against, and hosts the
//! sampling loop and grammar-constraint hook (see [`constraint`]).

mod config;
mod constraint;
mod deltanet;
mod math;
mod mlx;
mod sample;
mod weights;

use ullm_core::{DType, Error, Result};
use ullm_gguf::GgufModel;
use ullm_metal::{GpuExperts, GpuForward, GpuLayerInput, GpuModelInput, GpuParams, GpuWeight};
use ullm_safetensors::SafeTensorsModel;

pub use config::{Arch, LlamaConfig};
pub use constraint::{GrammarConstraint, LogitConstraint};
pub use sample::SampleParams;
// Re-export the grammar types so callers build constraints without a direct dep.
pub use ullm_grammar::{Grammar, GrammarDfa, GrammarState, TokenTrie};

use deltanet::{DeltaNetDims, DeltaNetState, deltanet_step};
use math::{
    add_bias, dequant_mlx_row, gelu, matmul_q, matvec_q, rmsnorm, rope, rope_neox, silu, softmax,
};
use sample::{apply_no_repeat_ngram, apply_repetition_penalty, sample_token};
use weights::{qweight, stacked_experts, tensor_f32};

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
    /// Always-on shared expert (Qwen3-Next MoE): added to the routed experts,
    /// scaled by `sigmoid(shared_gate · x)`.
    shared: Option<SharedExpert>,
    /// Gated-DeltaNet (linear attention) block. When `Some`, this layer replaces
    /// softmax attention with the SSM recurrence (Qwen3.5 / Qwen3-Next hybrids)
    /// and the `wq`/`wk`/`wv`/`wo` fields above are unused placeholders.
    linear: Option<LinearAttn>,
}

/// The softmax-attention projections of a layer, grouped so the loader can swap
/// the whole block for a linear-attention one.
struct FullAttn {
    wq: QWeight,
    wk: QWeight,
    wv: QWeight,
    wo: QWeight,
    q_bias: Option<Vec<f32>>,
    k_bias: Option<Vec<f32>>,
    v_bias: Option<Vec<f32>>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
}

/// An empty, never-read `QWeight` placeholder (for the dense fields a MoE layer
/// doesn't use, or the attention fields a linear layer doesn't use).
fn empty_qweight() -> QWeight {
    QWeight {
        data: Vec::new(),
        dtype: DType::F32,
        out: 0,
        cols: 0,
        mlx: None,
    }
}

impl FullAttn {
    /// Unused placeholder for a linear-attention layer (its `wq..wo` are never read).
    fn placeholder() -> Self {
        Self {
            wq: empty_qweight(),
            wk: empty_qweight(),
            wv: empty_qweight(),
            wo: empty_qweight(),
            q_bias: None,
            k_bias: None,
            v_bias: None,
            q_norm: None,
            k_norm: None,
        }
    }
}

/// The feed-forward weights of a layer — dense SwiGLU or a sparse MoE.
struct Ffn {
    w_gate: QWeight,
    w_up: QWeight,
    w_down: QWeight,
    moe_gate: Option<QWeight>,
    experts_gate: Vec<QWeight>,
    experts_up: Vec<QWeight>,
    experts_down: Vec<QWeight>,
    shared: Option<SharedExpert>,
}

/// Always-on shared expert of a Qwen3-Next MoE layer (a SwiGLU MLP plus a
/// scalar gate applied to its output).
struct SharedExpert {
    gate: QWeight,
    up: QWeight,
    down: QWeight,
    gate_logit: QWeight, // [1, hidden] -> one logit; sigmoid scales the output
}

/// Weights of a Gated-DeltaNet (linear attention) block. Projections are kept
/// quantized like the rest of the model; the small per-head tables stay f32.
struct LinearAttn {
    in_proj_qkv: QWeight,
    in_proj_z: QWeight,
    in_proj_b: QWeight,
    in_proj_a: QWeight,
    out_proj: QWeight,
    conv1d: Vec<f32>,  // [conv_dim, conv_kernel]
    a_log: Vec<f32>,   // [n_v_heads]
    dt_bias: Vec<f32>, // [n_v_heads]
    norm: Vec<f32>,    // [head_v]
    dims: DeltaNetDims,
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
    /// Per-layer recurrent state for linear-attention (Gated-DeltaNet) layers —
    /// `Some` only on those layers. Reset at sequence position 0. A non-empty
    /// `Some` makes this a hybrid model (CPU-only for now).
    linear_states: Vec<Option<DeltaNetState>>,
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
            rotary_dim: head_dim,
            attn_gated: false,
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
                shared: None,
                linear: None,
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
            linear_states: Vec::new(),
        })
    }

    /// Load a model from a Hugging Face SafeTensors directory (`config.json` +
    /// `*.safetensors`). Weights are kept in their stored BF16/F16/F32 form and
    /// dequantized per row, exactly as for GGUF.
    pub fn from_safetensors(model: &SafeTensorsModel) -> Result<Self> {
        // Hyperparameters live under `text_config` for multimodal models, at the
        // root for plain LLMs. Architecture is keyed off `model_type` (with the
        // `architectures` list as a fallback); optional features (bias, Q/K-norm)
        // are detected from the tensors below, not from the architecture name.
        let tc = model.text_config();
        let model_type = tc
            .get("model_type")
            .and_then(|v| v.as_str())
            .or_else(|| model.config().get("model_type").and_then(|v| v.as_str()));
        let arch_name = model
            .config()
            .get("architectures")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str());
        let arch = Arch::detect(model_type, arch_name).ok_or_else(|| {
            Error::Unsupported(format!(
                "SafeTensors model_type '{}' / architecture '{}' is not supported yet \
                 (Llama, Mistral, Qwen2, Qwen3 and Qwen3 multimodal text decoders)",
                model_type.unwrap_or("?"),
                arch_name.unwrap_or("?")
            ))
        })?;
        // Hybrid linear/full models (Qwen3.5 / Qwen3-Next): linear-attention on
        // most layers, output-gated full attention on the rest, and a sparse MoE
        // FFN (routed experts + a shared expert) when `num_experts` is set.
        let hybrid = tc.get("linear_num_value_heads").is_some();
        let n_experts = tc.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let is_moe = n_experts > 0;
        if is_moe && !hybrid {
            return Err(Error::Unsupported(
                "non-hybrid mixture-of-experts from SafeTensors is not wired yet — run the \
                 MLX 4-bit build of this model, or convert it to GGUF"
                    .into(),
            ));
        }
        if arch == Arch::Gemma3 {
            return Err(Error::Unsupported(
                "Gemma-3 from SafeTensors is not wired yet — use the GGUF build".into(),
            ));
        }

        let req = |k: &str| -> Result<usize> {
            tc.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| Error::Format(format!("config.json: missing '{k}'")))
        };
        let cu = |k: &str| tc.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
        let cf = |k: &str| tc.get(k).and_then(|v| v.as_f64()).map(|v| v as f32);
        let n_embd = req("hidden_size")?;
        let n_head = req("num_attention_heads")?;
        let n_layer = req("num_hidden_layers")?;
        let n_kv_head = cu("num_key_value_heads").unwrap_or(n_head);
        // Dense FFN width; absent on pure-MoE models (they use moe_intermediate_size).
        let n_ff = cu("intermediate_size").unwrap_or(0);
        let head_dim = cu("head_dim").unwrap_or(n_embd / n_head);
        let n_ctx = cu("max_position_embeddings").unwrap_or(8192).min(8192);
        // rope_theta may sit at the root or inside `rope_parameters` (transformers
        // 5.x). Partial rotary (Qwen3.5) rotates only `head_dim * factor` of each
        // head.
        let rope_params = tc.get("rope_parameters");
        let rope_theta = cf("rope_theta")
            .or_else(|| {
                rope_params
                    .and_then(|v| v.get("rope_theta"))
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
            })
            .unwrap_or(1_000_000.0);
        let partial = cf("partial_rotary_factor")
            .or_else(|| {
                rope_params
                    .and_then(|v| v.get("partial_rotary_factor"))
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
            })
            .unwrap_or(1.0);
        let rotary_dim = (head_dim as f32 * partial) as usize;
        let eps = cf("rms_norm_eps").unwrap_or(1e-6);
        let vocab_size = req("vocab_size")?;
        let n_experts_used = cu("num_experts_per_tok").unwrap_or(0);
        let moe_inter = cu("moe_intermediate_size").unwrap_or(0);

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
            rotary_dim,
            attn_gated: hybrid,
            n_experts,
            n_experts_used,
            moe_inter,
            sliding_window: 0,
        };

        // Multimodal models prefix the text decoder (e.g. `model.language_model.`);
        // plain LLMs use `model.`. Probe for the layer-0 norm to find the prefix.
        let prefix = ["model.language_model.", "language_model.model.", "model."]
            .into_iter()
            .find(|p| model.has_tensor(&format!("{p}layers.0.input_layernorm.weight")))
            .ok_or_else(|| {
                Error::Format("could not locate the decoder layers in the SafeTensors".into())
            })?;

        // Per-layer Gated-DeltaNet geometry; `layer_types` (or
        // `full_attention_interval`) decides which layers are linear.
        let ldims = hybrid.then(|| DeltaNetDims {
            hidden: n_embd,
            n_v_heads: cu("linear_num_value_heads").unwrap_or(0),
            n_k_heads: cu("linear_num_key_heads").unwrap_or(0),
            head_k: cu("linear_key_head_dim").unwrap_or(0),
            head_v: cu("linear_value_head_dim").unwrap_or(0),
            conv_kernel: cu("linear_conv_kernel_dim").unwrap_or(4),
            eps,
        });
        let layer_types = tc.get("layer_types").and_then(|v| v.as_array()).cloned();
        let full_interval = cu("full_attention_interval");
        let is_linear = |i: usize| -> bool {
            if let Some(lt) = &layer_types {
                lt.get(i).and_then(|v| v.as_str()) == Some("linear_attention")
            } else {
                full_interval.is_some_and(|fi| (i + 1) % fi != 0)
            }
        };

        // Standard models must have q/k/v/o projections; fail clearly otherwise.
        if !hybrid && !model.has_tensor(&format!("{prefix}layers.0.self_attn.q_proj.weight")) {
            return Err(Error::Unsupported(
                "this model has no standard self_attn.q_proj and is not a recognized \
                 linear-attention hybrid — unsupported attention block"
                    .into(),
            ));
        }

        // Qwen3-Next RMSNorm applies `(1 + weight)` (Gemma-style), unlike plain
        // Qwen3. Fold the +1 into the regular norm weights at load so the shared
        // plain RMSNorm is exact. The DeltaNet gated norm stays plain (it is
        // `weight *`, not `(1+weight) *`), so it is NOT folded.
        let plus1 = |mut v: Vec<f32>| -> Vec<f32> {
            if hybrid {
                v.iter_mut().for_each(|x| *x += 1.0);
            }
            v
        };

        let tok_embeddings = qweight(model, &format!("{prefix}embed_tokens.weight"))?;
        let mut layers = Vec::with_capacity(n_layer);
        let mut linear_states = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = format!("{prefix}layers.{i}");
            // Norms and the (dense) FFN are common to both attention kinds.
            let attn_norm = plus1(tensor_f32(model, &format!("{p}.input_layernorm.weight"))?);
            let ffn_norm = plus1(tensor_f32(
                model,
                &format!("{p}.post_attention_layernorm.weight"),
            )?);
            // Feed-forward: dense SwiGLU, or a sparse MoE (routed experts split
            // out of the stacked, gate/up-fused tensors, plus a shared expert).
            let mp = format!("{p}.mlp");
            let ffn = if is_moe {
                Ffn {
                    w_gate: empty_qweight(),
                    w_up: empty_qweight(),
                    w_down: empty_qweight(),
                    moe_gate: Some(qweight(model, &format!("{mp}.gate.weight"))?),
                    experts_gate: stacked_experts(
                        model,
                        &format!("{mp}.experts.gate_up_proj"),
                        n_experts,
                        2 * moe_inter,
                        0,
                        moe_inter,
                        n_embd,
                    )?,
                    experts_up: stacked_experts(
                        model,
                        &format!("{mp}.experts.gate_up_proj"),
                        n_experts,
                        2 * moe_inter,
                        moe_inter,
                        moe_inter,
                        n_embd,
                    )?,
                    experts_down: stacked_experts(
                        model,
                        &format!("{mp}.experts.down_proj"),
                        n_experts,
                        n_embd,
                        0,
                        n_embd,
                        moe_inter,
                    )?,
                    shared: Some(SharedExpert {
                        gate: qweight(model, &format!("{mp}.shared_expert.gate_proj.weight"))?,
                        up: qweight(model, &format!("{mp}.shared_expert.up_proj.weight"))?,
                        down: qweight(model, &format!("{mp}.shared_expert.down_proj.weight"))?,
                        gate_logit: qweight(model, &format!("{mp}.shared_expert_gate.weight"))?,
                    }),
                }
            } else {
                Ffn {
                    w_gate: qweight(model, &format!("{mp}.gate_proj.weight"))?,
                    w_up: qweight(model, &format!("{mp}.up_proj.weight"))?,
                    w_down: qweight(model, &format!("{mp}.down_proj.weight"))?,
                    moe_gate: None,
                    experts_gate: Vec::new(),
                    experts_up: Vec::new(),
                    experts_down: Vec::new(),
                    shared: None,
                }
            };

            let (attn, linear, state) = if hybrid && is_linear(i) {
                let d = ldims.unwrap();
                let lp = format!("{p}.linear_attn");
                let lin = LinearAttn {
                    in_proj_qkv: qweight(model, &format!("{lp}.in_proj_qkv.weight"))?,
                    in_proj_z: qweight(model, &format!("{lp}.in_proj_z.weight"))?,
                    in_proj_b: qweight(model, &format!("{lp}.in_proj_b.weight"))?,
                    in_proj_a: qweight(model, &format!("{lp}.in_proj_a.weight"))?,
                    out_proj: qweight(model, &format!("{lp}.out_proj.weight"))?,
                    conv1d: tensor_f32(model, &format!("{lp}.conv1d.weight"))?,
                    a_log: tensor_f32(model, &format!("{lp}.A_log"))?,
                    dt_bias: tensor_f32(model, &format!("{lp}.dt_bias"))?,
                    norm: tensor_f32(model, &format!("{lp}.norm.weight"))?,
                    dims: d,
                };
                (
                    FullAttn::placeholder(),
                    Some(lin),
                    Some(DeltaNetState::new(&d)),
                )
            } else {
                let sp = format!("{p}.self_attn");
                let full = FullAttn {
                    wq: qweight(model, &format!("{sp}.q_proj.weight"))?,
                    wk: qweight(model, &format!("{sp}.k_proj.weight"))?,
                    wv: qweight(model, &format!("{sp}.v_proj.weight"))?,
                    wo: qweight(model, &format!("{sp}.o_proj.weight"))?,
                    q_bias: tensor_f32(model, &format!("{sp}.q_proj.bias")).ok(),
                    k_bias: tensor_f32(model, &format!("{sp}.k_proj.bias")).ok(),
                    v_bias: tensor_f32(model, &format!("{sp}.v_proj.bias")).ok(),
                    q_norm: tensor_f32(model, &format!("{sp}.q_norm.weight"))
                        .ok()
                        .map(&plus1),
                    k_norm: tensor_f32(model, &format!("{sp}.k_norm.weight"))
                        .ok()
                        .map(&plus1),
                };
                (full, None, None)
            };
            linear_states.push(state);
            layers.push(LayerWeights {
                attn_norm,
                wq: attn.wq,
                wk: attn.wk,
                wv: attn.wv,
                wo: attn.wo,
                q_bias: attn.q_bias,
                k_bias: attn.k_bias,
                v_bias: attn.v_bias,
                q_norm: attn.q_norm,
                k_norm: attn.k_norm,
                post_attn_norm: None,
                post_ffn_norm: None,
                ffn_norm,
                w_gate: ffn.w_gate,
                w_up: ffn.w_up,
                w_down: ffn.w_down,
                moe_gate: ffn.moe_gate,
                experts_gate: ffn.experts_gate,
                experts_up: ffn.experts_up,
                experts_down: ffn.experts_down,
                shared: ffn.shared,
                linear,
            });
        }
        let final_norm = plus1(tensor_f32(model, &format!("{prefix}norm.weight"))?);
        // The output projection is `lm_head.weight` (at the root or under the
        // decoder prefix); tied to the input embeddings when absent.
        let output = qweight(model, "lm_head.weight")
            .or_else(|_| qweight(model, &format!("{prefix}lm_head.weight")))
            .unwrap_or_else(|_| tok_embeddings.clone());

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
            linear_states,
        })
    }

    /// Run one decoding step for `token` at sequence position `pos`, returning
    /// the logits over the vocabulary. Updates the KV cache in place.
    pub fn forward(&mut self, token: u32, pos: usize) -> Vec<f32> {
        if let Some(gpu) = self.gpu.as_ref() {
            let x_init = self.embed(token);
            return gpu.forward(&x_init, pos).expect("gpu forward");
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
        if self.is_hybrid() {
            return Err(Error::Unsupported(
                "linear-attention (Qwen3.5 / Qwen3-Next) models run on CPU only — the GPU \
                 forward has no state-space path yet"
                    .into(),
            ));
        }
        let gpu = self.build_gpu()?;
        self.gpu = Some(gpu);
        Ok(())
    }

    /// Whether this is a hybrid linear-attention model (CPU-only; any layer is a
    /// Gated-DeltaNet block).
    pub fn is_hybrid(&self) -> bool {
        self.linear_states.iter().any(Option::is_some)
    }

    /// Whether the GPU forward path is active.
    pub fn gpu_enabled(&self) -> bool {
        self.gpu.is_some()
    }

    /// The model's context window (KV-cache length): the hard cap on prompt +
    /// generated tokens.
    pub fn context_len(&self) -> usize {
        self.config.n_ctx
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
        let rotary_dim = c.rotary_dim;
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

        // Reset the linear-attention recurrent state at the start of a sequence.
        if pos == 0 {
            for s in self.linear_states.iter_mut().flatten() {
                s.reset();
            }
        }

        let mut x = self.embed(token);

        for l in 0..n_layer {
            let lw = &self.layers[l];

            // --- attention: Gated-DeltaNet (linear) or standard softmax ---
            let xb = rmsnorm(&x, &lw.attn_norm, eps);
            let attn = if let Some(lin) = &lw.linear {
                let qkv = matvec_q(&lin.in_proj_qkv, &xb);
                let zz = matvec_q(&lin.in_proj_z, &xb);
                let bb = matvec_q(&lin.in_proj_b, &xb);
                let aa = matvec_q(&lin.in_proj_a, &xb);
                let state = self.linear_states[l].as_mut().expect("linear state");
                let normed = deltanet_step(
                    state,
                    &qkv,
                    &zz,
                    &bb,
                    &aa,
                    &lin.conv1d,
                    &lin.a_log,
                    &lin.dt_bias,
                    &lin.norm,
                    lin.dims,
                );
                matvec_q(&lin.out_proj, &normed)
            } else {
                let mut q = matvec_q(&lw.wq, &xb);
                // Qwen3-Next output-gated attention: q_proj emits [query | gate]
                // per head. Split off the gate; it multiplies the attention output.
                let gate = if c.attn_gated {
                    let mut query = vec![0.0f32; q_dim];
                    let mut gate = vec![0.0f32; q_dim];
                    for h in 0..n_head {
                        let src = h * head_dim * 2;
                        let dst = h * head_dim;
                        query[dst..dst + head_dim].copy_from_slice(&q[src..src + head_dim]);
                        gate[dst..dst + head_dim]
                            .copy_from_slice(&q[src + head_dim..src + 2 * head_dim]);
                    }
                    q = query;
                    Some(gate)
                } else {
                    None
                };
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
                    rope_neox(&mut q, n_head, head_dim, rotary_dim, pos, rope_theta);
                    rope_neox(&mut k, n_kv_head, head_dim, rotary_dim, pos, rope_theta);
                } else {
                    rope(&mut q, n_head, head_dim, rotary_dim, pos, rope_theta);
                    rope(&mut k, n_kv_head, head_dim, rotary_dim, pos, rope_theta);
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

                // Apply the attention output gate (Qwen3-Next), if any.
                if let Some(gate) = &gate {
                    for (a, g) in att_out.iter_mut().zip(gate) {
                        *a *= 1.0 / (1.0 + (-g).exp());
                    }
                }
                matvec_q(&lw.wo, &att_out)
            };
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
        let rotary_dim = c.rotary_dim;
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
                    rope_neox(qs, n_head, head_dim, rotary_dim, pos, rope_theta);
                    rope_neox(ks, n_kv_head, head_dim, rotary_dim, pos, rope_theta);
                } else {
                    rope(qs, n_head, head_dim, rotary_dim, pos, rope_theta);
                    rope(ks, n_kv_head, head_dim, rotary_dim, pos, rope_theta);
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
        let rotary_dim = c.rotary_dim;
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
            rope_neox(&mut q, n_head, head_dim, rotary_dim, pos, rope_theta);
            rope_neox(&mut k, n_kv_head, head_dim, rotary_dim, pos, rope_theta);

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
    /// `false`. An optional [`LogitConstraint`] (e.g. a grammar) restricts every
    /// sampled token to keep the *generated* output valid — the prompt is never
    /// constrained, only the response.
    pub fn generate_stream<F: FnMut(u32) -> bool>(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        eos: Option<u32>,
        params: &SampleParams,
        mut constraint: Option<&mut dyn LogitConstraint>,
        mut on_token: F,
    ) {
        let mut pos = 0usize;
        let mut logits: Vec<f32> = Vec::new();
        let mut rng = if params.seed == 0 {
            0x853c_49e6_748f_ea9b
        } else {
            params.seed
        };

        // Prefill: read each weight once for the whole prompt instead of
        // token-by-token. The GPU uses the batched Metal matmul (BF16 / MLX /
        // Q4_K / Q6_K dense models); the CPU Llama family uses the batched
        // forward. MoE GPU models fall back to the per-token route. Short prompts
        // skip the GPU batch — its read-once win only pays off past a few dozen
        // tokens, and below that per-token is already a few ms.
        const GPU_BATCH_PREFILL_MIN: usize = 64;
        if !prompt.is_empty() {
            let take = prompt.len().min(self.config.n_ctx);
            let did_batch = if let Some(gpu) = self.gpu.as_ref() {
                if gpu.supports_batched_prefill() && take >= GPU_BATCH_PREFILL_MIN {
                    let n = self.config.n_embd;
                    let mut embeds = Vec::with_capacity(take * n);
                    for &tok in &prompt[..take] {
                        embeds.extend_from_slice(&self.embed(tok));
                    }
                    logits = gpu.forward_batch(&embeds, take).expect("gpu forward_batch");
                    pos = take;
                    true
                } else {
                    false
                }
            } else if self.config.arch != Arch::Gemma3 && !self.is_hybrid() {
                logits = self.forward_batch(&prompt[..take], 0);
                pos = take;
                true
            } else {
                false
            };
            if !did_batch {
                for &tok in &prompt[..take] {
                    if pos >= self.config.n_ctx {
                        return;
                    }
                    logits = self.forward(tok, pos);
                    pos += 1;
                }
            }
        }

        // Recent-token window for the repetition penalty (prompt + generated).
        let mut history: Vec<u32> = prompt.to_vec();
        for _ in 0..max_new {
            if pos >= self.config.n_ctx || logits.is_empty() {
                break;
            }
            let start = history.len().saturating_sub(params.repeat_last_n);
            apply_repetition_penalty(&mut logits, &history[start..], params.repeat_penalty);
            // Block verbatim n-gram loops — but not under a grammar constraint,
            // where banning a token could leave no valid continuation.
            if constraint.is_none() {
                apply_no_repeat_ngram(&mut logits, &history, params.no_repeat_ngram);
            }
            if let Some(c) = constraint.as_mut() {
                c.constrain(&mut logits);
            }
            let next = sample_token(&logits, params, &mut rng);
            if Some(next) == eos || !on_token(next) {
                break;
            }
            if let Some(c) = constraint.as_mut() {
                c.accept(next);
            }
            history.push(next);
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
        constraint: Option<&mut dyn LogitConstraint>,
    ) -> Vec<u32> {
        let mut out = Vec::new();
        self.generate_stream(prompt, max_new, eos, params, constraint, |t| {
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

    /// GPU analogue of [`prefill_check`](Self::prefill_check): compare the GPU
    /// batched prefill against the GPU per-token forward at the final position,
    /// and time both. Returns `None` if the GPU path is inactive or the model
    /// has no batched-prefill kernel (e.g. an MoE model or an unsupported dtype).
    pub fn gpu_prefill_check(&self, prompt: &[u32]) -> Option<PrefillCheck> {
        let gpu = self.gpu.as_ref()?;
        if !gpu.supports_batched_prefill() || prompt.is_empty() {
            return None;
        }
        let n = self.config.n_embd;
        let mut embeds = Vec::with_capacity(prompt.len() * n);
        for &tok in prompt {
            embeds.extend_from_slice(&self.embed(tok));
        }

        let t0 = std::time::Instant::now();
        let lb = gpu
            .forward_batch(&embeds, prompt.len())
            .expect("gpu forward_batch");
        let batch_ms = t0.elapsed().as_secs_f64() * 1e3;

        // Per-token forward refills the same KV cache positions deterministically,
        // so its final-position logits are the reference for the batched pass.
        let t1 = std::time::Instant::now();
        let mut ls = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            let x = self.embed(tok);
            ls = gpu.forward(&x, pos).expect("gpu forward");
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
        Some(PrefillCheck {
            max_diff,
            agree: batch_argmax == seq_argmax,
            batch_argmax,
            seq_argmax,
            batch_ms,
            seq_ms,
        })
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
    // Always-on shared expert (Qwen3-Next): a SwiGLU MLP scaled by a sigmoid gate.
    if let Some(s) = &lw.shared {
        let gate = matvec_q(&s.gate, xb);
        let up = matvec_q(&s.up, xb);
        let hidden: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
        let shared_out = matvec_q(&s.down, &hidden);
        let g = 1.0 / (1.0 + (-matvec_q(&s.gate_logit, xb)[0]).exp());
        for (o, d) in out.iter_mut().zip(&shared_out) {
            *o += g * d;
        }
    }
    out
}

//! Loading Apple MLX models (4-bit group quant + Qwen3-MoE).
//!
//! MLX stores each quantized linear as three tensors (`weight` u32-packed,
//! `scales`, `biases`); we dequantize to BF16 at load and reuse the existing
//! BF16 matvec path. Mixture-of-experts layers store all experts stacked in one
//! `switch_mlp.*` tensor, sliced per expert here.

use ullm_core::dequant::{bf16_to_f32, dequantize_mlx_q4};
use ullm_core::ir::WeightSource;
use ullm_core::{DType, Error, Result};
use ullm_safetensors::SafeTensorsModel;

use crate::config::{Arch, LlamaConfig};
use crate::{LayerWeights, LlamaModel, QWeight};

/// Read a BF16 tensor as `f32`.
fn read_bf16(st: &SafeTensorsModel, name: &str) -> Result<Vec<f32>> {
    let bytes = st
        .tensor_data(name)
        .ok_or_else(|| Error::Format(format!("missing tensor '{name}'")))?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

/// Read a U32 tensor as `u32` words.
fn read_u32(st: &SafeTensorsModel, name: &str) -> Result<Vec<u32>> {
    let bytes = st
        .tensor_data(name)
        .ok_or_else(|| Error::Format(format!("missing tensor '{name}'")))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Pack an `f32` weight matrix into a BF16 [`QWeight`].
fn bf16_qweight(vals: &[f32], out: usize, cols: usize) -> QWeight {
    let mut data = Vec::with_capacity(vals.len() * 2);
    for v in vals {
        let bf = (v.to_bits() >> 16) as u16;
        data.extend_from_slice(&bf.to_le_bytes());
    }
    QWeight {
        data,
        dtype: DType::BF16,
        out,
        cols,
    }
}

/// Dequantize one MLX 4-bit linear (`<name>.weight/scales/biases`) to BF16.
fn mlx_qweight(st: &SafeTensorsModel, name: &str, group_size: usize) -> Result<QWeight> {
    let info = st
        .tensor_bag()
        .get(&format!("{name}.weight"))
        .ok_or_else(|| Error::Format(format!("missing '{name}.weight'")))?;
    let out = info.shape[0];
    let in_dim = info.shape[1] * 8;
    let words = read_u32(st, &format!("{name}.weight"))?;
    let scales = read_bf16(st, &format!("{name}.scales"))?;
    let biases = read_bf16(st, &format!("{name}.biases"))?;
    let vals = dequantize_mlx_q4(&words, &scales, &biases, out, in_dim, group_size);
    Ok(bf16_qweight(&vals, out, in_dim))
}

/// Dequantize a stacked MLX expert tensor `[n_experts, out, in/8]` into one
/// BF16 [`QWeight`] per expert.
fn mlx_experts(
    st: &SafeTensorsModel,
    name: &str,
    n_experts: usize,
    out: usize,
    in_dim: usize,
    group_size: usize,
) -> Result<Vec<QWeight>> {
    let words = read_u32(st, &format!("{name}.weight"))?;
    let scales = read_bf16(st, &format!("{name}.scales"))?;
    let biases = read_bf16(st, &format!("{name}.biases"))?;
    let wpe = out * (in_dim / 8); // words per expert
    let spe = out * (in_dim / group_size); // scales/biases per expert
    let mut experts = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let vals = dequantize_mlx_q4(
            &words[e * wpe..(e + 1) * wpe],
            &scales[e * spe..(e + 1) * spe],
            &biases[e * spe..(e + 1) * spe],
            out,
            in_dim,
            group_size,
        );
        experts.push(bf16_qweight(&vals, out, in_dim));
    }
    Ok(experts)
}

fn placeholder() -> QWeight {
    QWeight {
        data: Vec::new(),
        dtype: DType::F32,
        out: 0,
        cols: 0,
    }
}

impl LlamaModel {
    /// Load an Apple MLX model directory (4-bit, Qwen3-MoE). Weights are
    /// dequantized to BF16 at load.
    pub fn from_mlx(st: &SafeTensorsModel) -> Result<Self> {
        let arch_name = st
            .config()
            .get("architectures")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if arch_name != "Qwen3MoeForCausalLM" {
            return Err(Error::Unsupported(format!(
                "MLX architecture '{arch_name}' is not supported yet (Qwen3MoeForCausalLM)"
            )));
        }
        let gs = st
            .config()
            .get("quantization")
            .and_then(|q| q.get("group_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;

        let req = |k: &str| -> Result<usize> {
            st.config_usize(k)
                .ok_or_else(|| Error::Format(format!("config.json: missing '{k}'")))
        };
        let n_embd = req("hidden_size")?;
        let n_head = req("num_attention_heads")?;
        let n_layer = req("num_hidden_layers")?;
        let n_kv_head = st.config_usize("num_key_value_heads").unwrap_or(n_head);
        let head_dim = st.config_usize("head_dim").unwrap_or(n_embd / n_head);
        let n_experts = req("num_experts")?;
        let n_experts_used = req("num_experts_per_tok")?;
        let moe_inter = req("moe_intermediate_size")?;
        let n_ctx = st
            .config_usize("max_position_embeddings")
            .unwrap_or(8192)
            .min(8192);
        let rope_theta = st.config_f32("rope_theta").unwrap_or(1_000_000.0);
        let eps = st.config_f32("rms_norm_eps").unwrap_or(1e-6);
        let vocab_size = req("vocab_size")?;

        let config = LlamaConfig {
            arch: Arch::Qwen3Moe,
            n_embd,
            n_layer,
            n_head,
            n_kv_head,
            head_dim,
            n_ff: 0,
            vocab_size,
            n_ctx,
            rope_theta,
            eps,
            n_experts,
            n_experts_used,
            moe_inter,
        };

        let tok_embeddings = mlx_qweight(st, "model.embed_tokens", gs)?;
        let output = mlx_qweight(st, "lm_head", gs)?;
        let final_norm = read_bf16(st, "model.norm.weight")?;

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = format!("model.layers.{i}");
            layers.push(LayerWeights {
                attn_norm: read_bf16(st, &format!("{p}.input_layernorm.weight"))?,
                wq: mlx_qweight(st, &format!("{p}.self_attn.q_proj"), gs)?,
                wk: mlx_qweight(st, &format!("{p}.self_attn.k_proj"), gs)?,
                wv: mlx_qweight(st, &format!("{p}.self_attn.v_proj"), gs)?,
                wo: mlx_qweight(st, &format!("{p}.self_attn.o_proj"), gs)?,
                q_bias: None,
                k_bias: None,
                v_bias: None,
                q_norm: Some(read_bf16(st, &format!("{p}.self_attn.q_norm.weight"))?),
                k_norm: Some(read_bf16(st, &format!("{p}.self_attn.k_norm.weight"))?),
                post_attn_norm: None,
                post_ffn_norm: None,
                ffn_norm: read_bf16(st, &format!("{p}.post_attention_layernorm.weight"))?,
                w_gate: placeholder(),
                w_up: placeholder(),
                w_down: placeholder(),
                moe_gate: Some(mlx_qweight(st, &format!("{p}.mlp.gate"), gs)?),
                experts_gate: mlx_experts(
                    st,
                    &format!("{p}.mlp.switch_mlp.gate_proj"),
                    n_experts,
                    moe_inter,
                    n_embd,
                    gs,
                )?,
                experts_up: mlx_experts(
                    st,
                    &format!("{p}.mlp.switch_mlp.up_proj"),
                    n_experts,
                    moe_inter,
                    n_embd,
                    gs,
                )?,
                experts_down: mlx_experts(
                    st,
                    &format!("{p}.mlp.switch_mlp.down_proj"),
                    n_experts,
                    n_embd,
                    moe_inter,
                    gs,
                )?,
            });
        }

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
            rope_neox: true,
            gpu: None,
        })
    }
}

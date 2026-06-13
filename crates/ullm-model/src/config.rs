//! Model architecture and hyperparameters.

/// Architecture variant — selects the forward-pass behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// Llama-2 / Llama-3 and any plain-Llama GGUF.
    Llama,
    /// Qwen2 / Qwen2.5 — adds Q/K/V attention biases.
    Qwen2,
    /// Qwen3 — per-head Q/K RMSNorm before RoPE, no attention bias.
    Qwen3,
    /// Qwen3-MoE — Qwen3 attention + a top-k mixture-of-experts feed-forward.
    Qwen3Moe,
    /// Gemma 3 — `sqrt(n_embd)`-scaled embeddings, Q/K-norm, sandwich
    /// (post-attention / post-FFN) RMSNorms, GeGLU, and NeoX-style RoPE.
    /// Note: the GGUF converter folds Gemma's `(1 + w)` norm gain into the
    /// stored weights, so we apply a plain RMSNorm here.
    Gemma3,
}

impl Arch {
    /// Map a Hugging Face `model_type` (preferred) or `architectures[0]` string to
    /// an internal architecture. This is the single registry to extend for model
    /// coverage: a whole family shares one forward, and multimodal text-decoder
    /// variants (`qwen3_5`, `qwen3_vl`, `*_text`, …) collapse onto their base.
    /// Optional per-model features (attention bias, Q/K-norm, sandwich norms) are
    /// detected from the actual tensors at load time, not encoded here. Returns
    /// `None` for an unrecognized family.
    pub fn detect(model_type: Option<&str>, architectures: Option<&str>) -> Option<Arch> {
        let mt = model_type.unwrap_or("").to_ascii_lowercase();
        let an = architectures.unwrap_or("").to_ascii_lowercase();
        let is = |needle: &str| mt.contains(needle) || an.contains(needle);
        // Non-text-generation modalities share a base family name (e.g.
        // `qwen3_tts`) — reject them up front so they get a clear "unsupported"
        // message instead of being mistaken for a text LLM.
        if ["tts", "audio", "whisper", "m2m", "ocr", "yolo"]
            .iter()
            .any(|m| is(m))
        {
            return None;
        }
        // Mixture-of-experts variants share the base family name, so test first.
        if is("qwen3") && is("moe") {
            return Some(Arch::Qwen3Moe);
        }
        if is("gemma3") || is("gemma_3") {
            return Some(Arch::Gemma3);
        }
        // Qwen3 family — per-head Q/K RMSNorm, SwiGLU (qwen3, qwen3_5, qwen3_vl…).
        if is("qwen3") {
            return Some(Arch::Qwen3);
        }
        if is("qwen2") {
            return Some(Arch::Qwen2);
        }
        // Plain Llama-family decoders (Llama 2/3, Mistral, …).
        if is("llama") || is("mistral") {
            return Some(Arch::Llama);
        }
        None
    }
}

/// Hyperparameters describing a model.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub arch: Arch,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    /// Rotary dimensions of each head (== `head_dim` for full RoPE; smaller for
    /// partial rotary, e.g. Qwen3.5 rotates only the first quarter).
    pub rotary_dim: usize,
    /// Output-gated attention (Qwen3-Next / Qwen3.5): `q_proj` emits twice the
    /// query width — half query, half a gate that multiplies the attention output.
    pub attn_gated: bool,
    pub n_ff: usize,
    pub vocab_size: usize,
    pub n_ctx: usize,
    pub rope_theta: f32,
    pub eps: f32,
    /// Mixture-of-experts (0 for dense models).
    pub n_experts: usize,
    /// Experts selected per token (top-k routing).
    pub n_experts_used: usize,
    /// Per-expert feed-forward width.
    pub moe_inter: usize,
    /// Sliding-window attention span (0 = full attention on every layer). For
    /// Gemma-3, local layers attend to the last `sliding_window` tokens and
    /// every 6th layer uses full attention.
    pub sliding_window: usize,
}

#[cfg(test)]
mod tests {
    use super::Arch;

    #[test]
    fn detect_maps_families() {
        let mt = |s: &str| Arch::detect(Some(s), None);
        // Qwen3 family, including multimodal text-decoder variants.
        assert_eq!(mt("qwen3"), Some(Arch::Qwen3));
        assert_eq!(mt("qwen3_5"), Some(Arch::Qwen3));
        assert_eq!(mt("qwen3_5_text"), Some(Arch::Qwen3));
        assert_eq!(mt("qwen3_vl_text"), Some(Arch::Qwen3));
        // MoE variants resolve to the MoE arch even though they contain "qwen3".
        assert_eq!(mt("qwen3_moe"), Some(Arch::Qwen3Moe));
        assert_eq!(mt("qwen3_5_moe_text"), Some(Arch::Qwen3Moe));
        // Other families.
        assert_eq!(mt("qwen2"), Some(Arch::Qwen2));
        assert_eq!(mt("llama"), Some(Arch::Llama));
        assert_eq!(mt("mistral"), Some(Arch::Llama));
        assert_eq!(mt("gemma3_text"), Some(Arch::Gemma3));
        // Falls back to the architectures[0] string when model_type is unknown.
        assert_eq!(
            Arch::detect(None, Some("Qwen3ForCausalLM")),
            Some(Arch::Qwen3)
        );
        // Unrecognized families and non-text modalities return None (the caller
        // emits a clear "unsupported" error).
        assert_eq!(mt("whisper"), None);
        assert_eq!(mt("m2m_100"), None);
        assert_eq!(mt("qwen3_tts"), None); // audio, despite containing "qwen3"
        assert_eq!(mt("glm_ocr_text"), None);
    }
}

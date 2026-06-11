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

/// Hyperparameters describing a model.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub arch: Arch,
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

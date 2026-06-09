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
}

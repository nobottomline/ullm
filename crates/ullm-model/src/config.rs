//! Llama hyperparameters.

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

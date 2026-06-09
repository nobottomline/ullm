//! The container-agnostic intermediate representation.
//!
//! Every loader (GGUF, SafeTensors, PyTorch) populates these same structures, so
//! the runtime never branches on where the weights came from. This is a Phase 0
//! skeleton; fields will grow as loaders and the runtime are implemented.

use std::collections::BTreeMap;

use crate::DType;

/// Metadata plus the location of one tensor's bytes within a model container.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Canonical tensor name (normalized across source formats).
    pub name: String,
    /// Element / block type of the stored data.
    pub dtype: DType,
    /// Logical shape, outermost dimension first.
    pub shape: Vec<usize>,
    /// Byte offset of this tensor's data within its container.
    pub offset: u64,
}

impl TensorInfo {
    /// Number of elements (product of the shape dimensions).
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A flat, name-addressed collection of a model's tensors (weights).
#[derive(Debug, Default)]
pub struct TensorBag {
    pub tensors: BTreeMap<String, TensorInfo>,
}

impl TensorBag {
    /// Create an empty bag.
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a tensor by canonical name.
    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Number of tensors in the bag.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the bag holds no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Normalized model description, independent of the source file format.
#[derive(Debug, Default, Clone)]
pub struct ModelSpec {
    /// Architecture id, e.g. "llama", "qwen3".
    pub architecture: String,
    /// Context length the model was trained / served with.
    pub context_length: u32,
    /// Embedding (hidden) dimension.
    pub hidden_size: u32,
    /// Number of transformer blocks.
    pub num_layers: u32,
    /// Number of attention (query) heads.
    pub num_heads: u32,
    /// Number of key/value heads (equals `num_heads` unless GQA/MQA).
    pub num_kv_heads: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
}

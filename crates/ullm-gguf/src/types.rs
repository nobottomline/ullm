//! Mapping from GGUF/ggml tensor type ids to uLLM's [`DType`].

use ullm_core::{DType, Error, Result};

/// Translate a ggml tensor type id (as stored in a GGUF tensor-info record)
/// into a [`DType`]. Unsupported quantizations return `Error::Unsupported`.
pub fn dtype_from_ggml(id: u32) -> Result<DType> {
    Ok(match id {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::Q4_0,
        3 => DType::Q4_1,
        6 => DType::Q5_0,
        7 => DType::Q5_1,
        8 => DType::Q8_0,
        10 => DType::Q2K,
        11 => DType::Q3K,
        12 => DType::Q4K,
        13 => DType::Q5K,
        14 => DType::Q6K,
        15 => DType::Q8K,
        30 => DType::BF16,
        other => {
            return Err(Error::Unsupported(format!(
                "ggml tensor type id {other} is not yet supported"
            )));
        }
    })
}

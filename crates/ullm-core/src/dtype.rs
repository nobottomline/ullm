//! Element and block data types for weights and activations.
//!
//! Floating types are stored element-wise. Quantized types store fixed-size
//! blocks of weights with shared scales (and sometimes minimums), following the
//! GGUF block layout. Bits-per-weight values are *nominal*.

/// A tensor element / block type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    F32,
    F16,
    BF16,
    // Legacy GGUF quants.
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    // GGUF k-quants.
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl DType {
    /// Nominal bits per weight for this type.
    pub fn bits_per_weight(self) -> f32 {
        use DType::*;
        match self {
            F32 => 32.0,
            F16 | BF16 => 16.0,
            Q8_0 | Q8K => 8.5,
            Q6K => 6.5625,
            Q5_0 | Q5_1 | Q5K => 5.5,
            Q4_0 | Q4_1 | Q4K => 4.5,
            Q3K => 3.4375,
            Q2K => 2.625,
        }
    }

    /// Whether this is a quantized (block) type.
    pub fn is_quantized(self) -> bool {
        !matches!(self, DType::F32 | DType::F16 | DType::BF16)
    }
}

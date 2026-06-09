//! Loading model weights from any [`WeightSource`] (GGUF, SafeTensors):
//! matrices kept as raw bytes, norms expanded to f32.

use ullm_core::ir::WeightSource;
use ullm_core::{Error, Result};

use crate::QWeight;

/// Load a small tensor (a norm) as a freshly-allocated `f32` vector.
pub(crate) fn tensor_f32(model: &dyn WeightSource, name: &str) -> Result<Vec<f32>> {
    let info = model
        .tensor_bag()
        .get(name)
        .ok_or_else(|| Error::Format(format!("missing tensor '{name}'")))?;
    let n: usize = info.shape.iter().product();
    let bytes = model
        .tensor_data(name)
        .ok_or_else(|| Error::Format(format!("no data for tensor '{name}'")))?;
    ullm_core::dequant::dequantize(info.dtype, bytes, n)
}

/// Load a weight matrix, keeping its stored bytes (a copy of the mmap slice).
pub(crate) fn qweight(model: &dyn WeightSource, name: &str) -> Result<QWeight> {
    let info = model
        .tensor_bag()
        .get(name)
        .ok_or_else(|| Error::Format(format!("missing tensor '{name}'")))?;
    let (out, cols) = match info.shape.as_slice() {
        [o, c] => (*o, *c),
        [c] => (1, *c),
        _ => {
            return Err(Error::Unsupported(format!(
                "weight '{name}' has rank {}",
                info.shape.len()
            )));
        }
    };
    let bytes = model
        .tensor_data(name)
        .ok_or_else(|| Error::Format(format!("no data for tensor '{name}'")))?;
    Ok(QWeight {
        data: bytes.to_vec(),
        dtype: info.dtype,
        out,
        cols,
        mlx: None,
    })
}

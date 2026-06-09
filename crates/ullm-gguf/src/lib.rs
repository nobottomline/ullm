//! GGUF model-file loader.
//!
//! GGUF is the single-file format used by the llama.cpp / ggml ecosystem: one
//! file carries the weights, all hyperparameters, and the tokenizer as typed
//! key/value metadata. This crate memory-maps a `.gguf` file and parses it into
//! uLLM's container-agnostic IR ([`TensorBag`] + [`ModelSpec`]) without copying
//! the weight data.

mod reader;
mod types;
pub mod value;

use std::collections::BTreeMap;
use std::path::Path;

use ullm_core::ir::{ModelSpec, TensorBag, TensorInfo};
use ullm_core::{DType, Error, Result};

use reader::Reader;
use types::dtype_from_ggml;
pub use value::Value;

/// "GGUF" as a little-endian u32.
const GGUF_MAGIC: u32 = 0x4655_4747;
/// Default tensor-data alignment when `general.alignment` is absent.
const DEFAULT_ALIGNMENT: u64 = 32;

/// Backing storage for a loaded model: either a memory map (the normal path) or
/// an owned buffer (used in tests and for in-memory models).
enum Backing {
    Mmap(memmap2::Mmap),
    Bytes(Vec<u8>),
}

impl Backing {
    fn as_slice(&self) -> &[u8] {
        match self {
            Backing::Mmap(m) => m,
            Backing::Bytes(b) => b,
        }
    }
}

/// A parsed GGUF model: metadata, tensor directory, and access to tensor bytes.
pub struct GgufModel {
    backing: Backing,
    /// GGUF format version (2 or 3).
    pub version: u32,
    /// All metadata key/value pairs, keyed by their dotted names.
    pub metadata: BTreeMap<String, Value>,
    /// Directory of tensors (names, dtypes, shapes, absolute byte offsets).
    pub tensors: TensorBag,
}

impl GgufModel {
    /// Open and parse a GGUF file by memory-mapping it.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: a read-only mmap of a local model file. We never write through
        // the mapping; concurrent external truncation is out of scope, as it is
        // for every mmap-based loader (llama.cpp included).
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Self::parse(Backing::Mmap(mmap))
    }

    /// Parse a GGUF model already resident in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::parse(Backing::Bytes(bytes))
    }

    fn parse(backing: Backing) -> Result<Self> {
        let buf = backing.as_slice();
        let mut r = Reader::new(buf);

        let magic = r.u32()?;
        if magic != GGUF_MAGIC {
            return Err(Error::Format(format!(
                "not a GGUF file (bad magic {magic:#010x})"
            )));
        }
        let version = r.u32()?;
        if version != 2 && version != 3 {
            return Err(Error::Unsupported(format!(
                "GGUF version {version} (only 2 and 3 are supported)"
            )));
        }
        let tensor_count = r.u64()? as usize;
        let metadata_count = r.u64()? as usize;

        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = r.gguf_string()?;
            let type_id = r.u32()?;
            let val = value::read_value(&mut r, type_id)?;
            metadata.insert(key, val);
        }

        // A tensor-info record before its data offset is resolved.
        struct Raw {
            name: String,
            dtype: DType,
            shape: Vec<usize>,
            rel_offset: u64,
        }
        let mut raws = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.gguf_string()?;
            let n_dims = r.u32()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(r.u64()? as usize);
            }
            let dtype = dtype_from_ggml(r.u32()?)?;
            let rel_offset = r.u64()?;
            raws.push(Raw {
                name,
                dtype,
                shape,
                rel_offset,
            });
        }

        // Tensor data begins after the info section, padded up to `alignment`.
        let alignment = metadata
            .get("general.alignment")
            .and_then(Value::to_u64)
            .unwrap_or(DEFAULT_ALIGNMENT)
            .max(1);
        let info_end = r.position() as u64;
        let data_start = info_end.div_ceil(alignment) * alignment;

        let mut tensors = TensorBag::new();
        for raw in raws {
            // GGUF stores the fastest-varying dimension first; reverse to a
            // row-major (outermost-first) shape for the IR.
            let mut shape = raw.shape;
            shape.reverse();

            let abs_offset = data_start + raw.rel_offset;
            let len = tensor_byte_len(raw.dtype, &shape)?;
            let end = abs_offset
                .checked_add(len as u64)
                .ok_or_else(|| Error::Format("tensor offset overflow".into()))?;
            if end > buf.len() as u64 {
                return Err(Error::Format(format!(
                    "tensor '{}' extends past end of file",
                    raw.name
                )));
            }

            tensors.tensors.insert(
                raw.name.clone(),
                TensorInfo {
                    name: raw.name,
                    dtype: raw.dtype,
                    shape,
                    offset: abs_offset,
                },
            );
        }

        Ok(GgufModel {
            backing,
            version,
            metadata,
            tensors,
        })
    }

    /// Look up a metadata value by its dotted key.
    pub fn metadata_get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    /// The model's architecture id (e.g. "llama", "qwen3"), if present.
    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(Value::as_str)
    }

    /// The raw bytes of a tensor's data within the backing memory.
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensors.get(name)?;
        let start = info.offset as usize;
        let len = tensor_byte_len(info.dtype, &info.shape).ok()?;
        self.backing.as_slice().get(start..start + len)
    }

    /// Build a normalized [`ModelSpec`] from the architecture-prefixed metadata.
    pub fn model_spec(&self) -> ModelSpec {
        let arch = self.architecture().unwrap_or_default().to_string();
        let field = |suffix: &str| -> Option<u64> {
            self.metadata
                .get(&format!("{arch}.{suffix}"))
                .and_then(Value::to_u64)
        };

        // Resolve every field before moving `arch` into the result, so the
        // closure's borrow of `arch` has ended.
        let context_length = field("context_length").unwrap_or(0) as u32;
        let hidden_size = field("embedding_length").unwrap_or(0) as u32;
        let num_layers = field("block_count").unwrap_or(0) as u32;
        let num_heads = field("attention.head_count").unwrap_or(0) as u32;
        let num_kv_heads = field("attention.head_count_kv").unwrap_or(num_heads as u64) as u32;
        let vocab_size = field("vocab_size")
            .or_else(|| {
                self.metadata
                    .get("tokenizer.ggml.tokens")
                    .and_then(Value::as_array)
                    .map(|a| a.len() as u64)
            })
            .unwrap_or(0) as u32;

        ModelSpec {
            architecture: arch,
            context_length,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
        }
    }
}

/// Byte length of a tensor's data, given its dtype and row-major shape.
fn tensor_byte_len(dtype: DType, shape: &[usize]) -> Result<usize> {
    let n: usize = shape.iter().product();
    let block = dtype.block_size();
    if block == 0 || n % block != 0 {
        return Err(Error::Format(format!(
            "element count {n} is not a multiple of block size {block}"
        )));
    }
    Ok((n / block) * dtype.type_size())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// A minimal but valid GGUF v3 file with two metadata keys and one F32
    /// tensor of four elements.
    fn build_minimal_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&2u64.to_le_bytes()); // metadata count

        // general.architecture = "llama"
        push_str(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes()); // string
        push_str(&mut b, "llama");

        // llama.block_count = 1 (u32)
        push_str(&mut b, "llama.block_count");
        b.extend_from_slice(&4u32.to_le_bytes()); // u32
        b.extend_from_slice(&1u32.to_le_bytes());

        // tensor info: token_embd.weight, 1-D [4], F32, offset 0
        push_str(&mut b, "token_embd.weight");
        b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&4u64.to_le_bytes()); // dim 0
        b.extend_from_slice(&0u32.to_le_bytes()); // ggml type F32
        b.extend_from_slice(&0u64.to_le_bytes()); // relative offset

        // pad to 32-byte alignment
        while b.len() % 32 != 0 {
            b.push(0);
        }

        // tensor data: 4 × f32
        for x in [1.0f32, 2.0, 3.0, 4.0] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }

    #[test]
    fn parses_minimal_gguf() {
        let model = GgufModel::from_bytes(build_minimal_gguf()).unwrap();
        assert_eq!(model.version, 3);
        assert_eq!(model.architecture(), Some("llama"));

        let spec = model.model_spec();
        assert_eq!(spec.architecture, "llama");
        assert_eq!(spec.num_layers, 1);

        assert_eq!(model.tensors.len(), 1);
        let data = model.tensor_data("token_embd.weight").unwrap();
        assert_eq!(data.len(), 16);
        let first = f32::from_le_bytes(data[0..4].try_into().unwrap());
        assert_eq!(first, 1.0);
    }

    #[test]
    fn rejects_bad_magic() {
        let result = GgufModel::from_bytes(vec![0u8; 64]);
        assert!(matches!(result, Err(Error::Format(_))));
    }
}

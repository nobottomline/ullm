//! SafeTensors / Hugging Face model loader.
//!
//! SafeTensors is a simple, safe container: an 8-byte little-endian header
//! length, a JSON header describing every tensor (`dtype`, `shape`,
//! `data_offsets`), then the raw tensor bytes laid out row-major. A Hugging Face
//! model directory pairs one or more `*.safetensors` files (optionally indexed by
//! `model.safetensors.index.json`) with a `config.json` and a `tokenizer.json`.
//!
//! This crate memory-maps the shards and exposes them through uLLM's
//! container-agnostic [`WeightSource`] trait, so the runtime loads HF models with
//! the exact same code path it uses for GGUF.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde_json::Value;
use ullm_core::ir::{TensorBag, TensorInfo, WeightSource};
use ullm_core::{DType, Error, Result};

/// A parsed SafeTensors model: memory-mapped shards plus the HF `config.json`.
pub struct SafeTensorsModel {
    shards: Vec<Mmap>,
    tensors: TensorBag,
    /// Which shard each tensor's bytes live in (index into `shards`).
    shard_of: BTreeMap<String, usize>,
    /// Parsed `config.json` (an empty object when loaded from a bare file).
    config: Value,
    /// The model directory, used to locate `tokenizer.json`.
    dir: Option<PathBuf>,
}

impl SafeTensorsModel {
    /// Open a model from a Hugging Face directory or a single `.safetensors` file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            Self::open_dir(path)
        } else {
            Self::open_file(path)
        }
    }

    /// Open a single `.safetensors` file (no `config.json`).
    pub fn open_file(path: impl AsRef<Path>) -> Result<Self> {
        let mut model = Self {
            shards: Vec::new(),
            tensors: TensorBag::new(),
            shard_of: BTreeMap::new(),
            config: Value::Object(Default::default()),
            dir: None,
        };
        model.add_shard(path.as_ref())?;
        Ok(model)
    }

    /// Open a Hugging Face model directory: `config.json` + one or more shards
    /// (resolved via `model.safetensors.index.json` when present).
    pub fn open_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config = match std::fs::read(dir.join("config.json")) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| Error::Format(format!("config.json: {e}")))?,
            Err(_) => Value::Object(Default::default()),
        };

        let mut model = Self {
            shards: Vec::new(),
            tensors: TensorBag::new(),
            shard_of: BTreeMap::new(),
            config,
            dir: Some(dir.to_path_buf()),
        };

        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            // Sharded: the index maps each tensor to a shard filename.
            let bytes = std::fs::read(&index_path)?;
            let index: Value = serde_json::from_slice(&bytes)
                .map_err(|e| Error::Format(format!("index.json: {e}")))?;
            let map = index
                .get("weight_map")
                .and_then(Value::as_object)
                .ok_or_else(|| Error::Format("index.json: missing weight_map".into()))?;
            let mut files: Vec<String> = map
                .values()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect();
            files.sort();
            files.dedup();
            for file in files {
                model.add_shard(&dir.join(file))?;
            }
        } else {
            // Single file.
            let single = dir.join("model.safetensors");
            if !single.exists() {
                return Err(Error::Format(
                    "no model.safetensors or index.json in directory".into(),
                ));
            }
            model.add_shard(&single)?;
        }
        Ok(model)
    }

    /// Memory-map one shard and register its tensors.
    fn add_shard(&mut self, path: &Path) -> Result<()> {
        let file = std::fs::File::open(path)
            .map_err(|e| Error::Format(format!("open {}: {e}", path.display())))?;
        // SAFETY: read-only mmap of a local model file; never written through.
        let mmap = unsafe { Mmap::map(&file)? };
        let shard_idx = self.shards.len();

        let buf: &[u8] = &mmap;
        if buf.len() < 8 {
            return Err(Error::Format("safetensors file too small".into()));
        }
        let header_len = u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize;
        let data_start = 8 + header_len;
        let header_bytes = buf
            .get(8..data_start)
            .ok_or_else(|| Error::Format("safetensors header out of bounds".into()))?;
        let header: Value = serde_json::from_slice(header_bytes)
            .map_err(|e| Error::Format(format!("safetensors header: {e}")))?;
        let obj = header
            .as_object()
            .ok_or_else(|| Error::Format("safetensors header is not an object".into()))?;

        for (name, info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = parse_dtype(
                info.get("dtype")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::Format(format!("tensor '{name}': missing dtype")))?,
            )?;
            let shape: Vec<usize> = info
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Format(format!("tensor '{name}': missing shape")))?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            let offsets = info
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Format(format!("tensor '{name}': missing data_offsets")))?;
            let begin = offsets.first().and_then(Value::as_u64).unwrap_or(0) as usize;
            let abs_offset = (data_start + begin) as u64;

            self.tensors.tensors.insert(
                name.clone(),
                TensorInfo {
                    name: name.clone(),
                    dtype,
                    shape,
                    offset: abs_offset,
                },
            );
            self.shard_of.insert(name.clone(), shard_idx);
        }

        self.shards.push(mmap);
        Ok(())
    }

    /// The parsed `config.json`.
    pub fn config(&self) -> &Value {
        &self.config
    }

    /// The text-decoder config. Multimodal models (vision/audio + LLM) nest the
    /// language-model hyperparameters under `text_config`; plain LLMs keep them at
    /// the root. Returns the nested object when present, else the root config.
    pub fn text_config(&self) -> &Value {
        self.config.get("text_config").unwrap_or(&self.config)
    }

    /// Read a `usize` field from `config.json`.
    pub fn config_usize(&self, key: &str) -> Option<usize> {
        self.config
            .get(key)
            .and_then(Value::as_u64)
            .map(|v| v as usize)
    }

    /// Read an `f32` field from `config.json`.
    pub fn config_f32(&self, key: &str) -> Option<f32> {
        self.config
            .get(key)
            .and_then(Value::as_f64)
            .map(|v| v as f32)
    }

    /// Read a string field from `config.json`.
    pub fn config_str(&self, key: &str) -> Option<&str> {
        self.config.get(key).and_then(Value::as_str)
    }

    /// Whether a tensor with this exact name is present in the shards.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.get(name).is_some()
    }

    /// Path to `tokenizer.json` in the model directory, if present.
    pub fn tokenizer_json_path(&self) -> Option<PathBuf> {
        let p = self.dir.as_ref()?.join("tokenizer.json");
        p.exists().then_some(p)
    }
}

impl WeightSource for SafeTensorsModel {
    fn tensor_bag(&self) -> &TensorBag {
        &self.tensors
    }

    fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let shard = *self.shard_of.get(name)?;
        let info = self.tensors.get(name)?;
        let start = info.offset as usize;
        let len = byte_len(info.dtype, &info.shape)?;
        self.shards[shard].get(start..start + len)
    }
}

/// Map a SafeTensors dtype string to uLLM's [`DType`].
fn parse_dtype(s: &str) -> Result<DType> {
    Ok(match s {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "U32" | "I32" => DType::U32, // MLX packed 4-bit weights
        other => {
            return Err(Error::Unsupported(format!(
                "SafeTensors dtype '{other}' (only F32/F16/BF16/U32 are supported)"
            )));
        }
    })
}

/// Byte length of a contiguous (non-quantized) tensor.
fn byte_len(dtype: DType, shape: &[usize]) -> Option<usize> {
    let n: usize = shape.iter().product();
    let elsize = match dtype {
        DType::F32 | DType::U32 => 4,
        DType::F16 | DType::BF16 => 2,
        _ => return None,
    };
    Some(n * elsize)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a one-tensor SafeTensors blob: an F32 `w` of shape `[2, 2]`.
    fn synthetic() -> Vec<u8> {
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let header =
            r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"__metadata__":{"x":"y"}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        for f in data {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out
    }

    #[test]
    fn parses_header_and_reads_tensor() {
        let dir = std::env::temp_dir().join("ullm_st_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("one.safetensors");
        std::fs::write(&path, synthetic()).unwrap();

        let model = SafeTensorsModel::open_file(&path).unwrap();
        assert_eq!(model.tensor_bag().len(), 1, "__metadata__ must be skipped");
        let info = model.tensor_bag().get("w").unwrap();
        assert_eq!(info.dtype, DType::F32);
        assert_eq!(info.shape, vec![2, 2]);

        let bytes = model.tensor_data("w").unwrap();
        assert_eq!(bytes.len(), 16);
        let first = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let last = f32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!((first, last), (1.0, 4.0));

        std::fs::remove_file(&path).ok();
    }
}

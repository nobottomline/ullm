//! Tokenizers for uLLM, built from a GGUF file's `tokenizer.ggml.*` metadata.
//!
//! Two algorithms share one [`Tokenizer`] type:
//! - **SentencePiece (SPM)** — Llama-2 family (see [`spm`]).
//! - **byte-level BPE (GPT-2 style)** — Llama-3 / Qwen / SmolLM family (see [`bpe`]).

mod bpe;
mod hf;
mod spm;

use std::collections::HashMap;

use bpe::BpeData;

/// The category of a vocabulary token (mirrors GGUF's `token_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Undefined,
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenType {
    fn from_i32(v: i32) -> TokenType {
        match v {
            1 => TokenType::Normal,
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Unused,
            6 => TokenType::Byte,
            _ => TokenType::Undefined,
        }
    }
}

/// A tokenizer — SentencePiece or byte-level BPE — built from GGUF metadata.
/// Constructors live in the [`spm`] and [`bpe`] modules.
pub struct Tokenizer {
    tokens: Vec<String>,
    scores: Vec<f32>,
    types: Vec<TokenType>,
    token_to_id: HashMap<String, u32>,
    byte_to_id: Vec<Option<u32>>, // SPM byte-fallback tokens (length 256)
    id_to_byte: Vec<Option<u8>>,  // SPM, aligned to `tokens`
    bos: Option<u32>,
    eos: Option<u32>,
    add_bos: bool,
    add_space_prefix: bool,
    bpe: Option<BpeData>,
}

impl Tokenizer {
    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// The beginning-of-sequence token id, if any.
    pub fn bos_id(&self) -> Option<u32> {
        self.bos
    }

    /// The end-of-sequence token id, if any.
    pub fn eos_id(&self) -> Option<u32> {
        self.eos
    }

    /// Encode `text` into token ids. When `add_special` is set and the model is
    /// configured for it, a BOS token is prepended.
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_special && self.add_bos {
            if let Some(bos) = self.bos {
                out.push(bos);
            }
        }
        if self.bpe.is_some() {
            self.encode_bpe(text, &mut out);
        } else {
            self.encode_spm(text, &mut out);
        }
        out
    }

    /// Decode token ids back into text.
    pub fn decode(&self, ids: &[u32]) -> String {
        if self.bpe.is_some() {
            self.decode_bpe(ids)
        } else {
            self.decode_spm(ids)
        }
    }
}

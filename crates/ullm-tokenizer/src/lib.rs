//! Tokenizers for uLLM, built from a GGUF file's `tokenizer.ggml.*` metadata.
//!
//! Two algorithms share one [`Tokenizer`] type:
//! - **SentencePiece (SPM)** — Llama-2 family (see [`spm`]).
//! - **byte-level BPE (GPT-2 style)** — Llama-3 / Qwen / SmolLM family (see [`bpe`]).

mod bpe;
pub mod chat;
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
    /// Control / user-defined tokens (e.g. `<|im_start|>`, `<start_of_turn>`)
    /// matched verbatim in the input before sub-word tokenization, longest
    /// first. Needed so chat-template markers map to their single token id.
    specials: Vec<(String, u32)>,
}

/// Collect the bracketed control / user-defined tokens for verbatim matching.
pub(crate) fn collect_specials(tokens: &[String], types: &[TokenType]) -> Vec<(String, u32)> {
    let mut v: Vec<(String, u32)> = tokens
        .iter()
        .enumerate()
        .filter(|(i, t)| {
            !t.is_empty()
                && (t.starts_with('<') || t.starts_with('['))
                && matches!(
                    types.get(*i),
                    Some(TokenType::Control) | Some(TokenType::UserDefined)
                )
        })
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();
    v.sort_by_key(|s| std::cmp::Reverse(s.0.len()));
    v
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
        self.encode_segments(text, &mut out);
        out
    }

    /// Split the input on verbatim special tokens (each emitted as its own id),
    /// sub-word-tokenizing the spans in between.
    fn encode_segments(&self, text: &str, out: &mut Vec<u32>) {
        if self.specials.is_empty() {
            return self.encode_raw(text, out);
        }
        let mut seg_start = 0;
        let mut i = 0;
        while i < text.len() {
            let b = text.as_bytes()[i];
            let mut matched = None;
            if b == b'<' || b == b'[' {
                for (s, id) in &self.specials {
                    if text[i..].starts_with(s.as_str()) {
                        matched = Some((s.len(), *id));
                        break;
                    }
                }
            }
            if let Some((len, id)) = matched {
                if seg_start < i {
                    self.encode_raw(&text[seg_start..i], out);
                }
                out.push(id);
                i += len;
                seg_start = i;
            } else {
                i += 1;
                while i < text.len() && !text.is_char_boundary(i) {
                    i += 1;
                }
            }
        }
        if seg_start < text.len() {
            self.encode_raw(&text[seg_start..], out);
        }
    }

    fn encode_raw(&self, text: &str, out: &mut Vec<u32>) {
        if self.bpe.is_some() {
            self.encode_bpe(text, out);
        } else {
            self.encode_spm(text, out);
        }
    }

    /// Decode token ids back into text.
    pub fn decode(&self, ids: &[u32]) -> String {
        if self.bpe.is_some() {
            self.decode_bpe(ids)
        } else {
            self.decode_spm(ids)
        }
    }

    /// Raw bytes each token contributes to the output, indexed by id — the table
    /// that drives grammar-constrained decoding. Control / special tokens, which
    /// carry no literal text, get an empty piece (the grammar engine treats an
    /// empty piece as "not a text token" and never allows it).
    pub fn token_pieces(&self) -> Vec<Vec<u8>> {
        let has_bpe = self.bpe.is_some();
        (0..self.tokens.len() as u32)
            .map(|id| {
                if has_bpe {
                    self.piece_bytes_bpe(id)
                } else {
                    self.piece_bytes_spm(id)
                }
            })
            .collect()
    }
}

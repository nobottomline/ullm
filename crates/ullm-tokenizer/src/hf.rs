//! Hugging Face `tokenizer.json` loader (the `tokenizers` library format).
//!
//! Handles byte-level BPE tokenizers (Qwen / Llama-3 / GPT-2 family): the
//! `model.vocab` / `model.merges` tables plus the `added_tokens` (special)
//! tokens, reusing the same byte-level BPE machinery as the GGUF path. The
//! pre-tokenization regex is taken verbatim from the file's `pre_tokenizer`
//! `Split` rule, so digit grouping and contraction handling match the model
//! exactly.

use serde_json::Value;
use ullm_core::{Error, Result};

use crate::{TokenType, Tokenizer};

impl Tokenizer {
    /// Build a tokenizer from the bytes of a HF `tokenizer.json`.
    pub fn from_hf_json(
        bytes: &[u8],
        bos: Option<u32>,
        eos: Option<u32>,
        add_bos: bool,
    ) -> Result<Self> {
        let v: Value = serde_json::from_slice(bytes)
            .map_err(|e| Error::Format(format!("tokenizer.json: {e}")))?;
        let model = v
            .get("model")
            .ok_or_else(|| Error::Format("tokenizer.json: missing model".into()))?;
        if model.get("type").and_then(Value::as_str) != Some("BPE") {
            return Err(Error::Unsupported(format!(
                "tokenizer.json model type {:?} (only BPE is supported)",
                model.get("type").and_then(Value::as_str)
            )));
        }
        let vocab = model
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::Format("tokenizer.json: missing model.vocab".into()))?;

        // Gather (token, id) and the highest id (including special tokens), so
        // the vocab vector covers every id the model might emit.
        let mut max_id = 0u32;
        let entries: Vec<(&String, u32)> = vocab
            .iter()
            .map(|(tok, idv)| {
                let id = idv.as_u64().unwrap_or(0) as u32;
                max_id = max_id.max(id);
                (tok, id)
            })
            .collect();

        let added: Vec<(u32, &str, bool)> = v
            .get("added_tokens")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|a| {
                        let id = a.get("id").and_then(Value::as_u64)? as u32;
                        let content = a.get("content").and_then(Value::as_str)?;
                        let special = a.get("special").and_then(Value::as_bool).unwrap_or(true);
                        Some((id, content, special))
                    })
                    .collect()
            })
            .unwrap_or_default();
        for &(id, _, _) in &added {
            max_id = max_id.max(id);
        }

        let n = max_id as usize + 1;
        let mut tokens = vec![String::new(); n];
        let mut types = vec![TokenType::Undefined; n];
        for (tok, id) in entries {
            tokens[id as usize] = tok.clone();
            types[id as usize] = TokenType::Normal;
        }
        for (id, content, special) in added {
            tokens[id as usize] = content.to_string();
            types[id as usize] = if special {
                TokenType::Control
            } else {
                TokenType::UserDefined
            };
        }

        // Merges: newer files store `[["a","b"], ...]`, older ones `["a b", ...]`.
        let merges_v = model
            .get("merges")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Format("tokenizer.json: missing model.merges".into()))?;
        let mut pairs: Vec<(String, String)> = Vec::with_capacity(merges_v.len());
        for m in merges_v {
            if let Some(arr) = m.as_array() {
                if let (Some(a), Some(b)) = (
                    arr.first().and_then(Value::as_str),
                    arr.get(1).and_then(Value::as_str),
                ) {
                    pairs.push((a.to_string(), b.to_string()));
                }
            } else if let Some(s) = m.as_str() {
                let mut it = s.splitn(2, ' ');
                if let (Some(a), Some(b)) = (it.next(), it.next()) {
                    pairs.push((a.to_string(), b.to_string()));
                }
            }
        }

        let pattern = extract_split_regex(&v).ok_or_else(|| {
            Error::Unsupported("tokenizer.json: no Split pre-tokenizer regex found".into())
        })?;
        let regex = fancy_regex::Regex::new(&pattern)
            .map_err(|e| Error::Format(format!("tokenizer.json pre-tokenizer regex: {e}")))?;

        Self::from_bpe_parts(tokens, pairs, types, bos, eos, add_bos, regex)
    }
}

/// Pull the byte-level `Split` regex out of a `pre_tokenizer` (either a single
/// `Split` rule or a `Sequence` containing one).
fn extract_split_regex(v: &Value) -> Option<String> {
    fn from_node(n: &Value) -> Option<String> {
        if n.get("type").and_then(Value::as_str) == Some("Split") {
            return n
                .get("pattern")
                .and_then(|p| p.get("Regex"))
                .and_then(Value::as_str)
                .map(str::to_string);
        }
        None
    }
    let pt = v.get("pre_tokenizer")?;
    if let Some(s) = from_node(pt) {
        return Some(s);
    }
    if pt.get("type").and_then(Value::as_str) == Some("Sequence") {
        for sub in pt.get("pretokenizers")?.as_array()? {
            if let Some(s) = from_node(sub) {
                return Some(s);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::byte_to_unicode;

    /// A minimal byte-level `tokenizer.json`: all 256 base-byte tokens, two
    /// merges (`h`+`e` -> `he`, `Ġ`+`w` -> `Ġw`), and a GPT-2 Split regex.
    fn synthetic_json() -> Vec<u8> {
        let b2u = byte_to_unicode();
        let mut vocab = serde_json::Map::new();
        for (i, ch) in b2u.iter().enumerate() {
            vocab.insert(ch.to_string(), serde_json::json!(i));
        }
        let he = format!("{}{}", b2u[b'h' as usize], b2u[b'e' as usize]);
        let gw = format!("{}{}", b2u[b' ' as usize], b2u[b'w' as usize]);
        vocab.insert(he.clone(), serde_json::json!(256));
        vocab.insert(gw.clone(), serde_json::json!(257));
        let merges = serde_json::json!([
            [
                b2u[b'h' as usize].to_string(),
                b2u[b'e' as usize].to_string()
            ],
            [
                b2u[b' ' as usize].to_string(),
                b2u[b'w' as usize].to_string()
            ],
        ]);
        let doc = serde_json::json!({
            "added_tokens": [{"id": 258, "content": "<|end|>", "special": true}],
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Split", "pattern": {"Regex": r"'s| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"}},
                    {"type": "ByteLevel", "add_prefix_space": false}
                ]
            },
            "model": {"type": "BPE", "vocab": vocab, "merges": merges}
        });
        serde_json::to_vec(&doc).unwrap()
    }

    #[test]
    fn hf_bpe_merges_and_roundtrips() {
        let tk = Tokenizer::from_hf_json(&synthetic_json(), None, Some(258), false).unwrap();
        // "he" merges into token 256; " w" merges into 257.
        assert_eq!(tk.encode("he", false), vec![256]);
        let s = "he went";
        assert_eq!(tk.decode(&tk.encode(s, false)), s);
        // The special token decodes to nothing (Control type).
        assert_eq!(tk.decode(&[258]), "");
    }

    #[test]
    fn extracts_nested_split_regex() {
        let v: Value = serde_json::from_slice(&synthetic_json()).unwrap();
        assert!(extract_split_regex(&v).unwrap().contains(r"\p{L}"));
    }

    #[test]
    fn special_tokens_matched_verbatim() {
        let tk = Tokenizer::from_hf_json(&synthetic_json(), None, Some(258), false).unwrap();
        // "he" merges to 256; the special "<|end|>" is one id (258); "he" -> 256.
        assert_eq!(tk.encode("he<|end|>he", false), vec![256, 258, 256]);
    }
}

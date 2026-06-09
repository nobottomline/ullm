//! Byte-level BPE (GPT-2 style) tokenization: regex pre-tokenization, GPT-2
//! byte encoding, and rank-ordered merges. Used by Llama-3 / Qwen / SmolLM.

use std::collections::HashMap;

use ullm_core::{Error, Result};

use crate::{TokenType, Tokenizer};

/// Per-tokenizer data for byte-level BPE.
pub(crate) struct BpeData {
    /// `(left_id, right_id)` -> `(rank, merged_id)`.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// byte value -> base token id (its GPT-2 byte-encoded single char).
    byte_to_base: [u32; 256],
    /// GPT-2 byte-encoder char -> byte value.
    byte_decoder: HashMap<char, u8>,
    /// Pre-tokenization regex.
    regex: fancy_regex::Regex,
}

impl Tokenizer {
    /// Build a byte-level BPE tokenizer (GPT-2 style) from GGUF arrays.
    pub fn from_bpe(
        tokens: Vec<String>,
        merges: Vec<String>,
        token_types: Vec<i32>,
        bos: Option<u32>,
        eos: Option<u32>,
        add_bos: bool,
        pre: &str,
    ) -> Result<Self> {
        let n = tokens.len();
        let types: Vec<TokenType> = (0..n)
            .map(|i| {
                token_types
                    .get(i)
                    .map(|&v| TokenType::from_i32(v))
                    .unwrap_or(TokenType::Normal)
            })
            .collect();
        // GGUF stores merges as space-separated "A B" strings.
        let pairs: Vec<(String, String)> = merges
            .iter()
            .filter_map(|m| {
                let mut it = m.splitn(2, ' ');
                Some((it.next()?.to_string(), it.next()?.to_string()))
            })
            .collect();
        let regex = build_pretokenizer(pre)?;
        Self::from_bpe_parts(tokens, pairs, types, bos, eos, add_bos, regex)
    }

    /// Assemble a byte-level BPE tokenizer from already-parsed parts and an
    /// explicit pre-tokenization regex. Shared by the GGUF and HF-`tokenizer.json`
    /// constructors.
    pub(crate) fn from_bpe_parts(
        tokens: Vec<String>,
        merge_pairs: Vec<(String, String)>,
        types: Vec<TokenType>,
        bos: Option<u32>,
        eos: Option<u32>,
        add_bos: bool,
        regex: fancy_regex::Regex,
    ) -> Result<Self> {
        if tokens.is_empty() {
            return Err(Error::Format("tokenizer has no tokens".into()));
        }
        let n = tokens.len();
        let mut token_to_id = HashMap::with_capacity(n);
        for (id, piece) in tokens.iter().enumerate() {
            if !piece.is_empty() {
                token_to_id.entry(piece.clone()).or_insert(id as u32);
            }
        }

        // GPT-2 byte <-> printable-unicode-char tables.
        let b2u = byte_to_unicode();
        let mut byte_to_base = [0u32; 256];
        let mut byte_decoder = HashMap::with_capacity(256);
        for (b, &ch) in b2u.iter().enumerate() {
            byte_decoder.insert(ch, b as u8);
            byte_to_base[b] = *token_to_id
                .get(&ch.to_string())
                .ok_or_else(|| Error::Format(format!("BPE vocab missing base byte char {ch:?}")))?;
        }

        // merge pairs -> (rank, merged id), keyed by the ids of the two sides.
        let mut merge_map: HashMap<(u32, u32), (u32, u32)> =
            HashMap::with_capacity(merge_pairs.len());
        for (rank, (a, b)) in merge_pairs.iter().enumerate() {
            if let (Some(&la), Some(&rb)) = (token_to_id.get(a), token_to_id.get(b)) {
                if let Some(&mid) = token_to_id.get(&format!("{a}{b}")) {
                    merge_map.entry((la, rb)).or_insert((rank as u32, mid));
                }
            }
        }

        Ok(Tokenizer {
            tokens,
            scores: Vec::new(),
            types,
            token_to_id,
            byte_to_id: Vec::new(),
            id_to_byte: Vec::new(),
            bos,
            eos,
            add_bos,
            add_space_prefix: false,
            bpe: Some(BpeData {
                merges: merge_map,
                byte_to_base,
                byte_decoder,
                regex,
            }),
        })
    }

    pub(crate) fn encode_bpe(&self, text: &str, out: &mut Vec<u32>) {
        let bpe = self.bpe.as_ref().expect("bpe data");
        for m in bpe.regex.find_iter(text) {
            let chunk = match m {
                Ok(mm) => mm.as_str(),
                Err(_) => continue,
            };
            let mut ids: Vec<u32> = chunk
                .bytes()
                .map(|b| bpe.byte_to_base[b as usize])
                .collect();
            // Repeatedly merge the adjacent pair with the lowest merge rank.
            loop {
                let mut best: Option<(usize, u32, u32)> = None;
                for i in 0..ids.len().saturating_sub(1) {
                    if let Some(&(rank, mid)) = bpe.merges.get(&(ids[i], ids[i + 1])) {
                        if best.is_none_or(|(_, br, _)| rank < br) {
                            best = Some((i, rank, mid));
                        }
                    }
                }
                match best {
                    Some((i, _, mid)) => {
                        ids[i] = mid;
                        ids.remove(i + 1);
                    }
                    None => break,
                }
            }
            out.extend(ids);
        }
    }

    pub(crate) fn decode_bpe(&self, ids: &[u32]) -> String {
        let bpe = self.bpe.as_ref().expect("bpe data");
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let id = id as usize;
            if id >= self.tokens.len() {
                continue;
            }
            if matches!(
                self.types[id],
                TokenType::Control | TokenType::Unknown | TokenType::Unused | TokenType::Undefined
            ) {
                continue;
            }
            for ch in self.tokens[id].chars() {
                if let Some(&b) = bpe.byte_decoder.get(&ch) {
                    bytes.push(b);
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// GPT-2's reversible byte -> printable-unicode-char mapping.
pub(crate) fn byte_to_unicode() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut assigned = [false; 256];
    for &(lo, hi) in &[(0x21u32, 0x7E), (0xA1, 0xAC), (0xAE, 0xFF)] {
        for b in lo..=hi {
            map[b as usize] = char::from_u32(b).unwrap();
            assigned[b as usize] = true;
        }
    }
    let mut n = 0u32;
    for (b, slot) in map.iter_mut().enumerate() {
        if !assigned[b] {
            *slot = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    map
}

/// Build the pre-tokenization regex for a GGUF `tokenizer.ggml.pre` value.
fn build_pretokenizer(pre: &str) -> Result<fancy_regex::Regex> {
    let pat = match pre {
        "llama3" | "llama-bpe" | "smaug-bpe" => {
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        }
        _ => r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    };
    fancy_regex::Regex::new(pat).map_err(|e| Error::Format(format!("bad pre-tokenizer regex: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_bpe() -> Tokenizer {
        let b2u = byte_to_unicode();
        let mut tokens: Vec<String> = b2u.iter().map(|c| c.to_string()).collect();
        let he = format!("{}{}", b2u[b'h' as usize], b2u[b'e' as usize]);
        tokens.push(he); // merged "he" at id 256
        let merges = vec![format!("{} {}", b2u[b'h' as usize], b2u[b'e' as usize])];
        let types = vec![1i32; tokens.len()];
        Tokenizer::from_bpe(tokens, merges, types, None, None, false, "gpt2").unwrap()
    }

    #[test]
    fn bpe_merges_and_roundtrips() {
        let tk = toy_bpe();
        assert_eq!(tk.encode("he", false), vec![256]); // 'h' + 'e' merge into "he"
        assert_eq!(tk.decode(&[256]), "he");
        // Spaces and punctuation round-trip via GPT-2 byte encoding.
        let s = "hello world!";
        assert_eq!(tk.decode(&tk.encode(s, false)), s);
    }
}

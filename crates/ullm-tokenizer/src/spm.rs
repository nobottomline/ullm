//! SentencePiece (Llama-2) tokenization: a score-based greedy merge over a
//! linked symbol list, with `<0xNN>` byte fallback on decode.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use ullm_core::{Error, Result};

use crate::{TokenType, Tokenizer};

/// SentencePiece marker that stands in for a space (U+2581, "▁").
const SPACE: char = '\u{2581}';

impl Tokenizer {
    /// Build a SentencePiece tokenizer from raw GGUF arrays.
    #[allow(clippy::too_many_arguments)]
    pub fn from_sentencepiece(
        tokens: Vec<String>,
        scores: Vec<f32>,
        token_types: Vec<i32>,
        bos: Option<u32>,
        eos: Option<u32>,
        _unk: Option<u32>,
        add_bos: bool,
        add_space_prefix: bool,
    ) -> Result<Self> {
        if tokens.is_empty() {
            return Err(Error::Format("tokenizer has no tokens".into()));
        }
        if scores.len() != tokens.len() {
            return Err(Error::Format(format!(
                "tokenizer scores ({}) do not match token count ({})",
                scores.len(),
                tokens.len()
            )));
        }

        let n = tokens.len();
        let types: Vec<TokenType> = (0..n)
            .map(|i| {
                token_types
                    .get(i)
                    .map(|&v| TokenType::from_i32(v))
                    .unwrap_or(TokenType::Normal)
            })
            .collect();

        let mut token_to_id = HashMap::with_capacity(n);
        let mut byte_to_id = vec![None; 256];
        let mut id_to_byte = vec![None; n];
        for (id, piece) in tokens.iter().enumerate() {
            token_to_id.entry(piece.clone()).or_insert(id as u32);
            if types[id] == TokenType::Byte {
                if let Some(b) = parse_byte_piece(piece) {
                    byte_to_id[b as usize] = Some(id as u32);
                    id_to_byte[id] = Some(b);
                }
            }
        }

        Ok(Tokenizer {
            tokens,
            scores,
            types,
            token_to_id,
            byte_to_id,
            id_to_byte,
            bos,
            eos,
            add_bos,
            add_space_prefix,
            bpe: None,
        })
    }

    pub(crate) fn encode_spm(&self, text: &str, out: &mut Vec<u32>) {
        // Normalize: optional leading space, then map every space to ▁.
        let mut normalized = String::with_capacity(text.len() + SPACE.len_utf8());
        if self.add_space_prefix {
            normalized.push(SPACE);
        }
        for ch in text.chars() {
            normalized.push(if ch == ' ' { SPACE } else { ch });
        }
        if normalized.is_empty() {
            return;
        }

        // Initial symbols: one UTF-8 character each, in a doubly-linked list.
        let mut symbols: Vec<Symbol> = Vec::new();
        for (start, ch) in normalized.char_indices() {
            let idx = symbols.len() as i64;
            symbols.push(Symbol {
                start,
                len: ch.len_utf8(),
                prev: idx - 1,
                next: idx + 1,
            });
        }
        if let Some(last) = symbols.last_mut() {
            last.next = -1;
        }

        // Seed every adjacent bigram that exists in the vocabulary.
        let mut queue: BinaryHeap<Bigram> = BinaryHeap::new();
        for i in 1..symbols.len() {
            self.try_bigram(&normalized, &symbols, (i - 1) as i64, i as i64, &mut queue);
        }

        // Greedily merge the highest-scoring adjacent pair until none remain.
        while let Some(b) = queue.pop() {
            let (li, ri) = (b.left as usize, b.right as usize);
            if symbols[li].len == 0 || symbols[ri].len == 0 {
                continue;
            }
            if symbols[li].len + symbols[ri].len != b.size {
                continue; // one side was already merged into something else
            }
            symbols[li].len += symbols[ri].len;
            symbols[ri].len = 0;
            let after = symbols[ri].next;
            symbols[li].next = after;
            if after >= 0 {
                symbols[after as usize].prev = b.left;
            }
            let prev = symbols[li].prev;
            let next = symbols[li].next;
            self.try_bigram(&normalized, &symbols, prev, b.left, &mut queue);
            self.try_bigram(&normalized, &symbols, b.left, next, &mut queue);
        }

        // Emit surviving symbols; fall back to byte tokens for unknown pieces.
        let mut i = 0i64;
        while i >= 0 {
            let sym = &symbols[i as usize];
            if sym.len > 0 {
                let piece = &normalized[sym.start..sym.start + sym.len];
                if let Some(&id) = self.token_to_id.get(piece) {
                    out.push(id);
                } else {
                    for &byte in piece.as_bytes() {
                        if let Some(id) = self.byte_to_id[byte as usize] {
                            out.push(id);
                        }
                    }
                }
            }
            i = sym.next;
        }
    }

    fn try_bigram(
        &self,
        text: &str,
        symbols: &[Symbol],
        left: i64,
        right: i64,
        queue: &mut BinaryHeap<Bigram>,
    ) {
        if left < 0 || right < 0 {
            return;
        }
        let (l, r) = (left as usize, right as usize);
        let piece = &text[symbols[l].start..symbols[r].start + symbols[r].len];
        if let Some(&id) = self.token_to_id.get(piece) {
            queue.push(Bigram {
                left,
                right,
                score: self.scores[id as usize],
                size: piece.len(),
            });
        }
    }

    pub(crate) fn decode_spm(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let id = id as usize;
            if id >= self.tokens.len() {
                continue;
            }
            match self.types[id] {
                TokenType::Control
                | TokenType::Unknown
                | TokenType::Unused
                | TokenType::Undefined => {}
                TokenType::Byte => {
                    if let Some(b) = self.id_to_byte[id] {
                        bytes.push(b);
                    }
                }
                TokenType::Normal | TokenType::UserDefined => {
                    bytes.extend_from_slice(self.tokens[id].as_bytes());
                }
            }
        }
        let mut text = String::from_utf8_lossy(&bytes).replace(SPACE, " ");
        if self.add_space_prefix {
            if let Some(stripped) = text.strip_prefix(' ') {
                text = stripped.to_string();
            }
        }
        text
    }
}

/// A span of the normalized text in the merge linked-list.
struct Symbol {
    start: usize,
    len: usize,
    prev: i64,
    next: i64,
}

/// A candidate merge of two adjacent symbols.
struct Bigram {
    left: i64,
    right: i64,
    score: f32,
    size: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Bigram {}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> Ordering {
        // Highest score wins; ties are broken by the smaller left index.
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
    }
}

/// Parse a byte-fallback piece of the form `<0xNN>` into its byte value.
fn parse_byte_piece(piece: &str) -> Option<u8> {
    if piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>') {
        u8::from_str_radix(&piece[3..5], 16).ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy SPM vocabulary: specials, one byte token for 'x', the letters
    /// a/b/c, and the merge "ab" with a better score than the single letters.
    fn toy() -> Tokenizer {
        let tokens = vec![
            "<unk>".to_string(),
            "<s>".into(),
            "</s>".into(),
            "<0x78>".into(), // 'x'
            "a".into(),
            "b".into(),
            "c".into(),
            "ab".into(),
        ];
        let scores = vec![0.0, 0.0, 0.0, 0.0, -3.0, -3.0, -3.0, -1.0];
        let types = vec![2, 3, 3, 6, 1, 1, 1, 1];
        Tokenizer::from_sentencepiece(
            tokens,
            scores,
            types,
            Some(1),
            Some(2),
            Some(0),
            false,
            false,
        )
        .unwrap()
    }

    #[test]
    fn merges_by_score() {
        let tk = toy();
        let ids = tk.encode("abc", false);
        assert_eq!(ids, vec![7, 6]); // "ab", "c"
        assert_eq!(tk.decode(&ids), "abc");
    }

    #[test]
    fn byte_fallback() {
        let tk = toy();
        let ids = tk.encode("x", false); // 'x' only exists as a byte token
        assert_eq!(ids, vec![3]);
        assert_eq!(tk.decode(&ids), "x");
    }

    #[test]
    fn bos_prepended() {
        let tokens = vec!["<unk>".to_string(), "<s>".into(), "</s>".into(), "a".into()];
        let tk = Tokenizer::from_sentencepiece(
            tokens,
            vec![0.0; 4],
            vec![2, 3, 3, 1],
            Some(1),
            Some(2),
            Some(0),
            true,
            false,
        )
        .unwrap();
        assert_eq!(tk.encode("a", true), vec![1, 3]); // <s>, "a"
    }
}

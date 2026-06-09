//! Tokenizers for uLLM, built from a GGUF file's `tokenizer.ggml.*` metadata.
//!
//! Two algorithms share one [`Tokenizer`] type:
//! - **SentencePiece (SPM)** — Llama-2 family: a score-based greedy merge with
//!   byte fallback.
//! - **byte-level BPE (GPT-2 style)** — Llama-3 / Qwen / SmolLM family: regex
//!   pre-tokenization, GPT-2 byte encoding, and rank-ordered merges.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use ullm_core::{Error, Result};

/// SentencePiece marker that stands in for a space (U+2581, "▁").
const SPACE: char = '\u{2581}';

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

/// A SentencePiece (Llama-style) tokenizer.
pub struct Tokenizer {
    tokens: Vec<String>,
    scores: Vec<f32>,
    types: Vec<TokenType>,
    token_to_id: HashMap<String, u32>,
    byte_to_id: Vec<Option<u32>>, // length 256
    id_to_byte: Vec<Option<u8>>,  // aligned to `tokens`
    bos: Option<u32>,
    eos: Option<u32>,
    add_bos: bool,
    add_space_prefix: bool,
    bpe: Option<BpeData>,
}

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

    fn encode_spm(&self, text: &str, out: &mut Vec<u32>) {
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

    /// Decode token ids back into text.
    pub fn decode(&self, ids: &[u32]) -> String {
        if self.bpe.is_some() {
            self.decode_bpe(ids)
        } else {
            self.decode_spm(ids)
        }
    }

    /// SPM decode: byte-fallback tokens become raw bytes, `▁` becomes a space,
    /// and the leading space from the space prefix is removed.
    fn decode_spm(&self, ids: &[u32]) -> String {
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
        if tokens.is_empty() {
            return Err(Error::Format("tokenizer has no tokens".into()));
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
        for (id, piece) in tokens.iter().enumerate() {
            token_to_id.entry(piece.clone()).or_insert(id as u32);
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

        // merges: "A B" -> (rank, merged id).
        let mut merge_map: HashMap<(u32, u32), (u32, u32)> = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let mut it = m.splitn(2, ' ');
            if let (Some(a), Some(b)) = (it.next(), it.next()) {
                if let (Some(&la), Some(&rb)) = (token_to_id.get(a), token_to_id.get(b)) {
                    if let Some(&mid) = token_to_id.get(&format!("{a}{b}")) {
                        merge_map.entry((la, rb)).or_insert((rank as u32, mid));
                    }
                }
            }
        }

        let regex = build_pretokenizer(pre)?;

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

    fn encode_bpe(&self, text: &str, out: &mut Vec<u32>) {
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

    fn decode_bpe(&self, ids: &[u32]) -> String {
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

/// Per-tokenizer data for byte-level BPE.
struct BpeData {
    /// `(left_id, right_id)` -> `(rank, merged_id)`.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// byte value -> base token id (its GPT-2 byte-encoded single char).
    byte_to_base: [u32; 256],
    /// GPT-2 byte-encoder char -> byte value.
    byte_decoder: HashMap<char, u8>,
    /// Pre-tokenization regex.
    regex: fancy_regex::Regex,
}

/// GPT-2's reversible byte -> printable-unicode-char mapping.
fn byte_to_unicode() -> [char; 256] {
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

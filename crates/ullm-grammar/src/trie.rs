//! A byte trie over the token vocabulary — built once, reused every step to
//! compute the allowed-token mask fast.
//!
//! The naive mask simulates every token's bytes through the grammar separately:
//! O(vocab x token-length) with a lot of repeated work, since thousands of
//! tokens share prefixes (`"`, `"a`, `"ab`, ...). The trie shares that work: one
//! walk advances the grammar one byte per *trie edge*, and a byte the grammar
//! rejects prunes the whole subtree (every token with that prefix) at once.

/// A trie node: byte-keyed children and the ids of tokens that end here.
pub(crate) struct Node {
    pub(crate) children: Vec<(u8, u32)>,
    pub(crate) tokens: Vec<u32>,
}

/// A trie over token byte-pieces. Owns the pieces so it can also answer
/// `piece(id)` for advancing the matcher after a token is chosen.
pub struct TokenTrie {
    nodes: Vec<Node>,
    pieces: Vec<Vec<u8>>,
}

impl TokenTrie {
    /// Build the trie from per-token byte pieces (`Tokenizer::token_pieces`).
    /// Empty pieces (control / special tokens) carry no text and are skipped.
    pub fn new(pieces: Vec<Vec<u8>>) -> Self {
        let mut nodes = vec![Node {
            children: Vec::new(),
            tokens: Vec::new(),
        }];
        for (id, piece) in pieces.iter().enumerate() {
            if piece.is_empty() {
                continue;
            }
            let mut cur = 0usize;
            for &b in piece {
                let existing = nodes[cur]
                    .children
                    .iter()
                    .find(|(c, _)| *c == b)
                    .map(|&(_, n)| n as usize);
                cur = match existing {
                    Some(n) => n,
                    None => {
                        let n = nodes.len();
                        nodes.push(Node {
                            children: Vec::new(),
                            tokens: Vec::new(),
                        });
                        nodes[cur].children.push((b, n as u32));
                        n
                    }
                };
            }
            nodes[cur].tokens.push(id as u32);
        }
        TokenTrie { nodes, pieces }
    }

    /// The raw bytes token `id` contributes (empty for control / special tokens).
    pub fn piece(&self, id: u32) -> &[u8] {
        self.pieces
            .get(id as usize)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Vocabulary size (the mask length).
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    pub(crate) fn node(&self, idx: usize) -> &Node {
        &self.nodes[idx]
    }
}

//! Constrained sampling — restrict the next token to a grammar-valid set.
//!
//! This is the mechanism behind uLLM's guarantee: before each token is sampled,
//! the logits of every token that would break the contract are set to `-inf`, so
//! they can never be chosen. The output is structurally valid by construction —
//! no retries, no JSON-repair.

use ullm_grammar::{Grammar, GrammarState, TokenTrie};

/// A constraint applied to the logits in place before sampling.
pub trait LogitConstraint {
    /// Set the logit of every currently-disallowed token to `-inf`.
    fn constrain(&mut self, logits: &mut [f32]);
    /// Record that `token` was emitted, advancing the internal state.
    fn accept(&mut self, token: u32);
}

/// A [`LogitConstraint`] driven by a GBNF [`Grammar`]: it holds the live matcher,
/// a prebuilt [`TokenTrie`] over the vocabulary (for fast masking), and the EOS
/// id (allowed only when the grammar may legally terminate).
pub struct GrammarConstraint<'a> {
    state: GrammarState<'a>,
    trie: &'a TokenTrie,
    eos: Option<u32>,
    allowed: Vec<bool>,
}

impl<'a> GrammarConstraint<'a> {
    /// `trie` is built once from `Tokenizer::token_pieces()` and reused across
    /// requests; `eos` is the end-of-sequence id, if any.
    pub fn new(grammar: &'a Grammar, trie: &'a TokenTrie, eos: Option<u32>) -> Self {
        Self {
            state: GrammarState::new(grammar),
            trie,
            eos,
            allowed: vec![false; trie.vocab_size()],
        }
    }
}

impl LogitConstraint for GrammarConstraint<'_> {
    fn constrain(&mut self, logits: &mut [f32]) {
        self.state.allowed_mask_trie(self.trie, &mut self.allowed);
        let can_end = self.state.can_end();
        for (i, l) in logits.iter_mut().enumerate() {
            let ok = self.allowed.get(i).copied().unwrap_or(false)
                || (can_end && self.eos == Some(i as u32));
            if !ok {
                *l = f32::NEG_INFINITY;
            }
        }
        // Never hand the sampler an all -inf distribution: if the grammar is
        // stuck, fall back to stopping cleanly on EOS.
        if let Some(eos) = self.eos {
            if logits.iter().all(|l| !l.is_finite()) {
                logits[eos as usize] = 0.0;
            }
        }
    }

    fn accept(&mut self, token: u32) {
        let piece = self.trie.piece(token);
        if !piece.is_empty() {
            self.state.accept_token(piece);
        }
    }
}

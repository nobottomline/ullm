//! Constrained sampling — restrict the next token to a grammar-valid set.
//!
//! This is the mechanism behind uLLM's guarantee: before each token is sampled,
//! the logits of every token that would break the contract are set to `-inf`, so
//! they can never be chosen. The output is structurally valid by construction —
//! no retries, no JSON-repair.

use ullm_grammar::{Grammar, GrammarDfa, TokenTrie};

/// A constraint applied to the logits in place before sampling.
pub trait LogitConstraint {
    /// Set the logit of every currently-disallowed token to `-inf`.
    fn constrain(&mut self, logits: &mut [f32]);
    /// Record that `token` was emitted, advancing the internal state.
    fn accept(&mut self, token: u32);
}

/// A [`LogitConstraint`] driven by a GBNF [`Grammar`]. It owns a [`GrammarDfa`]
/// (a persistent matcher with per-state mask caching, built over a prebuilt
/// [`TokenTrie`]) and the EOS id (allowed only when the grammar may terminate).
pub struct GrammarConstraint<'a> {
    dfa: GrammarDfa<'a>,
    eos: Option<u32>,
}

impl<'a> GrammarConstraint<'a> {
    /// `trie` is built once from `Tokenizer::token_pieces()` and reused across
    /// requests; `eos` is the end-of-sequence id, if any.
    pub fn new(grammar: &'a Grammar, trie: &'a TokenTrie, eos: Option<u32>) -> Self {
        Self {
            dfa: GrammarDfa::new(grammar, trie),
            eos,
        }
    }
}

impl LogitConstraint for GrammarConstraint<'_> {
    fn constrain(&mut self, logits: &mut [f32]) {
        let can_end = self.dfa.can_end();
        let eos = self.eos;
        let allowed = self.dfa.allowed_mask();
        for (i, l) in logits.iter_mut().enumerate() {
            let ok = allowed.get(i).copied().unwrap_or(false) || (can_end && eos == Some(i as u32));
            if !ok {
                *l = f32::NEG_INFINITY;
            }
        }
        // Never hand the sampler an all -inf distribution: if the grammar is
        // stuck, fall back to stopping cleanly on EOS.
        if let Some(eos) = eos {
            if logits.iter().all(|l| !l.is_finite()) {
                logits[eos as usize] = 0.0;
            }
        }
    }

    fn accept(&mut self, token: u32) {
        self.dfa.accept(token);
    }
}

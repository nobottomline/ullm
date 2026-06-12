//! Grammar-constrained decoding — uLLM's structured-output guarantee.
//!
//! A [`Grammar`] (parsed from GBNF, the llama.cpp-compatible grammar notation)
//! drives a byte-level non-deterministic pushdown automaton. At every decode
//! step the engine computes exactly which next tokens keep the output on a path
//! the grammar can complete; everything else is masked to `-inf` and becomes
//! impossible to sample. The result is output that is *guaranteed* to match the
//! grammar — valid JSON, a value from a fixed set, a well-formed tool call —
//! with no retries and no post-hoc repair.
//!
//! The automaton works on raw bytes (token pieces, including byte-fallback
//! tokens, are byte strings), so multi-byte UTF-8 string *content* flows through
//! negated character classes naturally. Character *ranges* in the grammar are
//! interpreted over bytes, so ASCII ranges work directly; non-ASCII ranges are
//! a known limitation (see `docs/`).

mod parser;

use ullm_core::Result;

/// A set of byte ranges, optionally negated — one character class `[...]`.
#[derive(Clone, Debug, Default)]
pub struct CharSet {
    /// Inclusive `[lo, hi]` byte ranges; a single byte is `(b, b)`.
    ranges: Vec<(u8, u8)>,
    /// `[^...]`: match one byte that is in *none* of the ranges.
    negated: bool,
}

impl CharSet {
    fn matches(&self, b: u8) -> bool {
        let hit = self.ranges.iter().any(|&(lo, hi)| lo <= b && b <= hi);
        hit ^ self.negated
    }
}

/// One element of a rule alternative.
#[derive(Clone, Debug)]
enum Elem {
    /// Consume one byte matching the class.
    Char(CharSet),
    /// Expand another rule (by index) before continuing.
    Rule(usize),
}

/// A grammar rule: an alternation of sequences (`alts[0] | alts[1] | ...`). An
/// empty sequence is the epsilon production (matches nothing).
#[derive(Clone, Debug, Default)]
struct Rule {
    alts: Vec<Vec<Elem>>,
}

/// A compiled GBNF grammar.
#[derive(Clone, Debug)]
pub struct Grammar {
    rules: Vec<Rule>,
    root: usize,
}

/// A position inside the grammar: element `elem` of alternative `alt` of `rule`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Pos {
    rule: u32,
    alt: u32,
    elem: u32,
}

/// A stack of pending positions; the last entry is the top (next to match).
type Stack = Vec<Pos>;

const MAX_DEPTH: usize = 1024;

impl Grammar {
    /// Compile a grammar from GBNF text (the `root ::= ...` notation).
    pub fn from_gbnf(text: &str) -> Result<Grammar> {
        parser::parse(text)
    }

    /// The built-in JSON grammar: any syntactically valid JSON value.
    pub fn json() -> Grammar {
        Grammar::from_gbnf(JSON_GBNF).expect("built-in JSON grammar is valid")
    }

    fn elem_at(&self, p: Pos) -> &Elem {
        &self.rules[p.rule as usize].alts[p.alt as usize][p.elem as usize]
    }

    /// The position just after `p` within its sequence, or `None` at the end.
    fn next_pos(&self, p: Pos) -> Option<Pos> {
        let len = self.rules[p.rule as usize].alts[p.alt as usize].len() as u32;
        (p.elem + 1 < len).then_some(Pos {
            elem: p.elem + 1,
            ..p
        })
    }

    /// Epsilon-closure: expand `stack` until its top is a `Char` (a terminal we
    /// can match a byte against) or the stack is empty (an accept state). A
    /// stack with a `Rule` on top is replaced by one stack per alternative.
    fn close(&self, stack: Stack, out: &mut Vec<Stack>, depth: usize) {
        if depth > MAX_DEPTH {
            return;
        }
        match stack.last() {
            None => out.push(stack),
            Some(&top) => match self.elem_at(top) {
                Elem::Char(_) => out.push(stack),
                Elem::Rule(sub) => {
                    let sub = *sub;
                    let mut base = stack.clone();
                    base.pop();
                    if let Some(n) = self.next_pos(top) {
                        base.push(n);
                    }
                    self.expand_rule(sub, &base, out, depth + 1);
                }
            },
        }
    }

    /// Push, for each alternative of `rule`, the closure of `base` with that
    /// alternative's start on top (epsilon alternatives just close `base`).
    fn expand_rule(&self, rule: usize, base: &Stack, out: &mut Vec<Stack>, depth: usize) {
        for (ai, alt) in self.rules[rule].alts.iter().enumerate() {
            if alt.is_empty() {
                self.close(base.clone(), out, depth + 1);
            } else {
                let mut s = base.clone();
                s.push(Pos {
                    rule: rule as u32,
                    alt: ai as u32,
                    elem: 0,
                });
                self.close(s, out, depth + 1);
            }
        }
    }

    /// The set of closed stacks reachable from the root — the initial state.
    fn initial(&self) -> Vec<Stack> {
        let mut out = Vec::new();
        self.expand_rule(self.root, &Vec::new(), &mut out, 0);
        dedup(&mut out);
        out
    }

    /// Advance every stack by one input byte and re-close, yielding the next
    /// state. An empty result means no path accepts `b`.
    fn accept_byte(&self, stacks: &[Stack], b: u8) -> Vec<Stack> {
        let mut out = Vec::new();
        for st in stacks {
            let Some(&top) = st.last() else { continue };
            if let Elem::Char(set) = self.elem_at(top) {
                if set.matches(b) {
                    let mut adv = st.clone();
                    match self.next_pos(top) {
                        Some(n) => *adv.last_mut().unwrap() = n,
                        None => {
                            adv.pop();
                        }
                    }
                    self.close(adv, &mut out, 0);
                }
            }
        }
        dedup(&mut out);
        out
    }
}

fn dedup(stacks: &mut Vec<Stack>) {
    stacks.sort();
    stacks.dedup();
}

/// The live matcher for one generation: the current automaton state plus the
/// methods a sampler needs (which tokens are allowed, may we stop, advance).
pub struct GrammarState<'g> {
    grammar: &'g Grammar,
    stacks: Vec<Stack>,
}

impl<'g> GrammarState<'g> {
    /// Start at the grammar's root.
    pub fn new(grammar: &'g Grammar) -> Self {
        GrammarState {
            stacks: grammar.initial(),
            grammar,
        }
    }

    /// Whether the grammar may terminate here (so EOS is allowed).
    pub fn can_end(&self) -> bool {
        self.stacks.iter().any(|s| s.is_empty())
    }

    /// Whether the grammar is fully satisfied with nothing left to match.
    pub fn is_complete(&self) -> bool {
        !self.stacks.is_empty() && self.stacks.iter().all(|s| s.is_empty())
    }

    /// The bytes that could legally come next (a cheap pre-filter for masking).
    fn first_bytes(&self) -> [bool; 256] {
        let mut fb = [false; 256];
        for st in &self.stacks {
            if let Some(&top) = st.last() {
                if let Elem::Char(set) = self.grammar.elem_at(top) {
                    for (b, slot) in fb.iter_mut().enumerate() {
                        if set.matches(b as u8) {
                            *slot = true;
                        }
                    }
                }
            }
        }
        fb
    }

    /// Could the token with these raw bytes be appended without leaving the
    /// grammar? Empty pieces (control/special tokens) are never grammar tokens.
    fn accepts_piece(&self, piece: &[u8]) -> bool {
        if piece.is_empty() {
            return false;
        }
        let mut cur = self.grammar.accept_byte(&self.stacks, piece[0]);
        for &b in &piece[1..] {
            if cur.is_empty() {
                return false;
            }
            cur = self.grammar.accept_byte(&cur, b);
        }
        !cur.is_empty()
    }

    /// Fill `allowed[id] = true` for every token whose bytes keep the output on
    /// a grammar-valid path. `allowed` must be sized to the vocabulary.
    pub fn allowed_mask(&self, pieces: &[Vec<u8>], allowed: &mut [bool]) {
        let fb = self.first_bytes();
        for (id, piece) in pieces.iter().enumerate() {
            allowed[id] = match piece.first() {
                Some(&b0) if fb[b0 as usize] => self.accepts_piece(piece),
                _ => false,
            };
        }
    }

    /// Advance the state by a chosen token's bytes. Returns `false` (and leaves
    /// the state unchanged) if the token would leave the grammar — which cannot
    /// happen for a token that passed [`allowed_mask`].
    pub fn accept_token(&mut self, piece: &[u8]) -> bool {
        let mut cur = self.stacks.clone();
        for &b in piece {
            cur = self.grammar.accept_byte(&cur, b);
            if cur.is_empty() {
                return false;
            }
        }
        self.stacks = cur;
        true
    }
}

/// A syntactically-complete JSON value, with whitespace. Mirrors the canonical
/// llama.cpp `json.gbnf`.
pub const JSON_GBNF: &str = r#"
root   ::= ws value
value  ::= object | array | string | number | ("true" | "false" | "null")
object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array  ::= "[" ws ( value ("," ws value)* )? "]" ws
string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\"" ws
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws     ::= [ \t\n\r]*
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces_for(s: &[&str]) -> Vec<Vec<u8>> {
        s.iter().map(|t| t.as_bytes().to_vec()).collect()
    }

    /// Walk a grammar over a byte string one token-piece at a time, asserting
    /// each step is allowed; return whether it ends in an acceptable state.
    fn run(g: &Grammar, steps: &[&str]) -> bool {
        let mut st = GrammarState::new(g);
        let pieces = pieces_for(steps);
        for (i, p) in pieces.iter().enumerate() {
            let mut mask = vec![false; pieces.len()];
            st.allowed_mask(&pieces, &mut mask);
            assert!(mask[i], "step {i:?} ({:?}) should be allowed", p);
            assert!(st.accept_token(p), "accept {:?}", p);
        }
        st.can_end()
    }

    #[test]
    fn json_accepts_valid_object() {
        let g = Grammar::json();
        assert!(run(&g, &["{", "\"a\"", ":", "1", "}"]));
        assert!(run(&g, &["[", "true", ",", "false", "]"]));
        assert!(run(&g, &["\"hi\""]));
    }

    #[test]
    fn json_masks_invalid_continuation() {
        let g = Grammar::json();
        let mut st = GrammarState::new(&g);
        // After "{", a value/`}` cannot start — only a string key (or `}`).
        let pieces = pieces_for(&["{", "}", "1", "\"k\""]);
        assert!(st.accept_token(b"{"));
        let mut mask = vec![false; pieces.len()];
        st.allowed_mask(&pieces, &mut mask);
        assert!(mask[3], "a string key is allowed after {{");
        assert!(mask[1], "an empty object {{}} is allowed");
        assert!(!mask[2], "a bare number is NOT a valid key");
    }

    #[test]
    fn simple_alternation_and_repeat() {
        // A grammar over a fixed label set.
        let g = Grammar::from_gbnf(r#"root ::= "yes" | "no" | "maybe""#).unwrap();
        assert!(run(&g, &["yes"]));
        assert!(run(&g, &["no"]));
        let st = GrammarState::new(&g);
        let pieces = pieces_for(&["yes", "no", "x"]);
        let mut mask = vec![false; pieces.len()];
        st.allowed_mask(&pieces, &mut mask);
        assert!(mask[0] && mask[1] && !mask[2]);
    }

    #[test]
    fn cannot_end_mid_token() {
        let g = Grammar::from_gbnf(r#"root ::= "ab""#).unwrap();
        let mut st = GrammarState::new(&g);
        assert!(!st.can_end(), "empty output does not satisfy \"ab\"");
        assert!(st.accept_token(b"a"));
        assert!(!st.can_end(), "\"a\" alone does not satisfy \"ab\"");
        assert!(st.accept_token(b"b"));
        assert!(st.can_end(), "\"ab\" satisfies the grammar");
    }
}

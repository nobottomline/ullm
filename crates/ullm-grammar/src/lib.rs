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
mod schema;
mod trie;

pub use trie::TokenTrie;
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
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
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

    /// The built-in JSON grammar: any syntactically valid JSON value. (Equivalent
    /// to the permissive schema `true`; digit runs are bounded — see `schema`.)
    pub fn json() -> Grammar {
        Grammar::from_json_schema(&serde_json::Value::Bool(true))
            .expect("built-in JSON grammar is valid")
    }

    /// Compile a JSON Schema into a grammar that accepts exactly the JSON
    /// documents it describes (the right keys, types, `enum` values, ...). See
    /// the `schema` module for the supported keyword subset.
    pub fn from_json_schema(schema: &serde_json::Value) -> Result<Grammar> {
        Grammar::from_gbnf(&schema::schema_to_gbnf(schema)?)
    }

    /// Compile a JSON Schema given as a JSON string.
    pub fn from_json_schema_str(json: &str) -> Result<Grammar> {
        let schema: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| ullm_core::Error::Format(format!("JSON Schema: {e}")))?;
        Grammar::from_json_schema(&schema)
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

/// One token-trie walk, paired with the grammar, computing the allowed-token
/// mask. Distinct stack-sets are interned to ids and `(state, byte)` transitions
/// are memoized, so a permissive loop (e.g. inside a string, where one state
/// recurs across the whole subtree) computes each transition once instead of
/// once per trie edge.
struct Walk<'a> {
    g: &'a Grammar,
    states: Vec<Vec<Stack>>,
    ids: std::collections::HashMap<Vec<Stack>, u32>,
    edge: std::collections::HashMap<(u32, u8), u32>,
}

/// Sentinel transition target meaning "the byte is rejected here".
const DEAD: u32 = u32::MAX;

impl Walk<'_> {
    fn intern(&mut self, s: Vec<Stack>) -> u32 {
        if let Some(&id) = self.ids.get(&s) {
            return id;
        }
        let id = self.states.len() as u32;
        self.states.push(s.clone());
        self.ids.insert(s, id);
        id
    }

    /// The state after consuming byte `b` in state `id`, or `None` if rejected.
    fn step(&mut self, id: u32, b: u8) -> Option<u32> {
        if let Some(&n) = self.edge.get(&(id, b)) {
            return (n != DEAD).then_some(n);
        }
        let next = self.g.accept_byte(&self.states[id as usize], b);
        let r = if next.is_empty() {
            DEAD
        } else {
            self.intern(next)
        };
        self.edge.insert((id, b), r);
        (r != DEAD).then_some(r)
    }

    // `trie` is a separate parameter (not a field) so iterating a node's children
    // doesn't borrow `self`, leaving `self` free for the memoized `step`.
    fn descend(&mut self, trie: &TokenTrie, node: usize, id: u32, allowed: &mut [bool]) {
        let n = trie.node(node);
        for &tid in &n.tokens {
            allowed[tid as usize] = true;
        }
        for &(byte, child) in &n.children {
            if let Some(nid) = self.step(id, byte) {
                self.descend(trie, child as usize, nid, allowed);
            }
        }
    }
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
    /// a grammar-valid path. `allowed` must be sized to the vocabulary. O(vocab)
    /// per call — see [`allowed_mask_trie`](Self::allowed_mask_trie) for the fast
    /// path used in generation.
    pub fn allowed_mask(&self, pieces: &[Vec<u8>], allowed: &mut [bool]) {
        let fb = self.first_bytes();
        for (id, piece) in pieces.iter().enumerate() {
            allowed[id] = match piece.first() {
                Some(&b0) if fb[b0 as usize] => self.accepts_piece(piece),
                _ => false,
            };
        }
    }

    /// Like [`allowed_mask`](Self::allowed_mask) but driven by a prebuilt
    /// [`TokenTrie`]: one walk shares prefix work across the whole vocabulary and
    /// prunes grammar-dead subtrees, so the cost tracks the number of *valid*
    /// prefixes rather than the vocabulary size.
    pub fn allowed_mask_trie(&self, trie: &TokenTrie, allowed: &mut [bool]) {
        for a in allowed.iter_mut() {
            *a = false;
        }
        let mut w = Walk {
            g: self.grammar,
            states: vec![self.stacks.clone()],
            ids: std::collections::HashMap::from([(self.stacks.clone(), 0u32)]),
            edge: std::collections::HashMap::new(),
        };
        w.descend(trie, 0, 0, allowed);
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

    #[test]
    fn trie_mask_equals_naive_mask() {
        // A vocabulary with shared prefixes and special (empty-piece) tokens.
        let pieces = pieces_for(&[
            "{", "}", "\"", "\"a", "\"ab", "true", "tru", "1", "12", ":", ",", " ",
        ]);
        let trie = TokenTrie::new(pieces.clone());
        let g = Grammar::json();
        // Drive a few steps and check the two mask paths agree at each one.
        for prefix in [
            vec![],
            vec![b"{".to_vec()],
            vec![b"{".to_vec(), b"\"a".to_vec()],
        ] {
            let mut st = GrammarState::new(&g);
            for p in &prefix {
                st.accept_token(p);
            }
            let mut naive = vec![false; pieces.len()];
            let mut fast = vec![false; pieces.len()];
            st.allowed_mask(&pieces, &mut naive);
            st.allowed_mask_trie(&trie, &mut fast);
            assert_eq!(naive, fast, "trie mask must match naive mask");
        }
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

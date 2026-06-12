//! A recursive-descent parser for GBNF (the llama.cpp grammar notation).
//!
//! Supports: `name ::= production`, alternation `|`, sequencing (whitespace),
//! string literals `"..."`, character classes `[...]` / `[^...]` with ranges and
//! escapes (`\n \r \t \\ \" \xNN \uNNNN`), grouping `( ... )`, the postfix
//! repetition operators `* + ?`, rule references, and `#` comments. Repetitions
//! and groups are desugared into anonymous helper rules. Newlines are treated as
//! ordinary whitespace; a rule ends where the next `name ::=` begins.

use std::collections::HashMap;

use ullm_core::{Error, Result};

use crate::{CharSet, Elem, Grammar, Rule};

pub(crate) fn parse(text: &str) -> Result<Grammar> {
    let mut p = Parser {
        src: text.as_bytes(),
        pos: 0,
        rules: Vec::new(),
        names: HashMap::new(),
    };
    p.parse_grammar()?;
    let root = *p
        .names
        .get("root")
        .ok_or_else(|| Error::Format("grammar has no `root` rule".into()))?;
    // Every referenced rule must have been defined (non-empty alts, or it was a
    // forward reference that got filled in).
    for (name, &idx) in &p.names {
        if p.rules[idx].alts.is_empty() {
            return Err(Error::Format(format!(
                "rule `{name}` is referenced but never defined"
            )));
        }
    }
    Ok(Grammar {
        rules: p.rules,
        root,
    })
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
    rules: Vec<Rule>,
    names: HashMap<String, usize>,
}

impl Parser<'_> {
    fn parse_grammar(&mut self) -> Result<()> {
        loop {
            self.skip_ws();
            if self.pos >= self.src.len() {
                return Ok(());
            }
            let name = self.parse_ident()?;
            self.skip_ws();
            self.expect(b"::=")?;
            let alts = self.parse_alternates()?;
            let idx = self.rule_id(&name);
            self.rules[idx].alts = alts;
        }
    }

    /// Get the index of a (possibly not-yet-defined) named rule.
    fn rule_id(&mut self, name: &str) -> usize {
        if let Some(&i) = self.names.get(name) {
            return i;
        }
        let i = self.rules.len();
        self.rules.push(Rule::default());
        self.names.insert(name.to_string(), i);
        i
    }

    /// Push a freshly-built anonymous rule (for groups / repetitions).
    fn add_rule(&mut self, alts: Vec<Vec<Elem>>) -> usize {
        let i = self.rules.len();
        self.rules.push(Rule { alts });
        i
    }

    fn parse_alternates(&mut self) -> Result<Vec<Vec<Elem>>> {
        let mut alts = vec![self.parse_sequence()?];
        loop {
            self.skip_ws();
            if self.peek() == Some(b'|') {
                self.pos += 1;
                alts.push(self.parse_sequence()?);
            } else {
                return Ok(alts);
            }
        }
    }

    fn parse_sequence(&mut self) -> Result<Vec<Elem>> {
        let mut seq = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                None => break,
                Some(b'|') | Some(b')') => break,
                // A bare identifier that is the start of the next rule
                // (`name ::=`) ends this one.
                Some(c) if is_ident_start(c) && self.at_rule_start() => break,
                _ => {}
            }
            let item = self.parse_item()?;
            seq.extend(item);
        }
        Ok(seq)
    }

    /// An atom plus an optional postfix repetition operator.
    fn parse_item(&mut self) -> Result<Vec<Elem>> {
        let body = self.parse_atom()?;
        self.skip_ws_inline();
        match self.peek() {
            Some(b'*') => {
                self.pos += 1;
                // S ::= body S | ε
                let s = self.add_rule(vec![Vec::new()]); // placeholder, filled below
                let mut rep = body.clone();
                rep.push(Elem::Rule(s));
                self.rules[s].alts = vec![rep, Vec::new()];
                Ok(vec![Elem::Rule(s)])
            }
            Some(b'+') => {
                self.pos += 1;
                // S ::= body S | body
                let s = self.add_rule(vec![Vec::new()]);
                let mut rep = body.clone();
                rep.push(Elem::Rule(s));
                self.rules[s].alts = vec![rep, body];
                Ok(vec![Elem::Rule(s)])
            }
            Some(b'?') => {
                self.pos += 1;
                // S ::= body | ε
                let s = self.add_rule(vec![body, Vec::new()]);
                Ok(vec![Elem::Rule(s)])
            }
            _ => Ok(body),
        }
    }

    fn parse_atom(&mut self) -> Result<Vec<Elem>> {
        self.skip_ws();
        match self.peek() {
            Some(b'"') => self.parse_string(),
            Some(b'[') => Ok(vec![Elem::Char(self.parse_class()?)]),
            Some(b'(') => {
                self.pos += 1;
                let alts = self.parse_alternates()?;
                self.skip_ws();
                self.expect(b")")?;
                let g = self.add_rule(alts);
                Ok(vec![Elem::Rule(g)])
            }
            Some(c) if is_ident_start(c) => {
                let name = self.parse_ident()?;
                Ok(vec![Elem::Rule(self.rule_id(&name))])
            }
            other => Err(Error::Format(format!(
                "unexpected character {:?} at byte {}",
                other.map(|b| b as char),
                self.pos
            ))),
        }
    }

    /// `"..."` -> one exact-byte `Char` element per literal byte.
    fn parse_string(&mut self) -> Result<Vec<Elem>> {
        self.pos += 1; // opening quote
        let mut out = Vec::new();
        loop {
            match self.peek() {
                None => return Err(Error::Format("unterminated string literal".into())),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    for b in self.parse_escape()? {
                        out.push(Elem::Char(exact(b)));
                    }
                }
                Some(c) => {
                    self.pos += 1;
                    out.push(Elem::Char(exact(c)));
                }
            }
        }
    }

    /// `[...]` / `[^...]` -> a `CharSet`.
    fn parse_class(&mut self) -> Result<CharSet> {
        self.pos += 1; // '['
        let negated = self.peek() == Some(b'^');
        if negated {
            self.pos += 1;
        }
        let mut ranges = Vec::new();
        loop {
            match self.peek() {
                None => return Err(Error::Format("unterminated character class".into())),
                Some(b']') => {
                    self.pos += 1;
                    return Ok(CharSet { ranges, negated });
                }
                Some(_) => {
                    let lo = self.parse_class_byte()?;
                    // A range `a-b` (but a trailing `-` before `]` is literal).
                    if self.peek() == Some(b'-') && self.peek_at(1) != Some(b']') {
                        self.pos += 1;
                        let hi = self.parse_class_byte()?;
                        ranges.push((lo.min(hi), lo.max(hi)));
                    } else {
                        ranges.push((lo, lo));
                    }
                }
            }
        }
    }

    /// One byte inside a character class (literal or escape).
    fn parse_class_byte(&mut self) -> Result<u8> {
        if self.peek() == Some(b'\\') {
            self.pos += 1;
            let bytes = self.parse_escape()?;
            // Class members are single bytes; a multi-byte `\u` escape in a
            // class is not representable at the byte level.
            if bytes.len() != 1 {
                return Err(Error::Format(
                    "multi-byte escape in a character class is unsupported".into(),
                ));
            }
            Ok(bytes[0])
        } else {
            let c = self.peek().unwrap();
            self.pos += 1;
            Ok(c)
        }
    }

    /// Decode one escape sequence (the `\` already consumed) into bytes.
    fn parse_escape(&mut self) -> Result<Vec<u8>> {
        let c = self
            .peek()
            .ok_or_else(|| Error::Format("dangling escape".into()))?;
        self.pos += 1;
        Ok(match c {
            b'n' => vec![b'\n'],
            b'r' => vec![b'\r'],
            b't' => vec![b'\t'],
            b'\\' => vec![b'\\'],
            b'"' => vec![b'"'],
            b'\'' => vec![b'\''],
            b']' => vec![b']'],
            b'[' => vec![b'['],
            b'/' => vec![b'/'],
            b'x' => {
                let hi = self.hex_digit()?;
                let lo = self.hex_digit()?;
                vec![hi * 16 + lo]
            }
            b'u' => {
                let mut cp = 0u32;
                for _ in 0..4 {
                    cp = cp * 16 + self.hex_digit()? as u32;
                }
                let ch = char::from_u32(cp)
                    .ok_or_else(|| Error::Format(format!("invalid \\u{cp:04x}")))?;
                ch.to_string().into_bytes()
            }
            other => return Err(Error::Format(format!("unknown escape \\{}", other as char))),
        })
    }

    fn hex_digit(&mut self) -> Result<u8> {
        let c = self
            .peek()
            .ok_or_else(|| Error::Format("truncated hex escape".into()))?;
        self.pos += 1;
        match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            b'A'..=b'F' => Ok(c - b'A' + 10),
            _ => Err(Error::Format(format!("bad hex digit {:?}", c as char))),
        }
    }

    fn parse_ident(&mut self) -> Result<String> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if is_ident_continue(c) {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(Error::Format(format!(
                "expected an identifier at byte {start}"
            )));
        }
        Ok(String::from_utf8_lossy(&self.src[start..self.pos]).into_owned())
    }

    /// Does an `IDENT ::=` begin at the current position? (Lookahead only.)
    fn at_rule_start(&self) -> bool {
        let mut i = self.pos;
        while i < self.src.len() && is_ident_continue(self.src[i]) {
            i += 1;
        }
        if i == self.pos {
            return false;
        }
        while i < self.src.len() && is_inline_ws(self.src[i]) {
            i += 1;
        }
        self.src[i..].starts_with(b"::=")
    }

    fn expect(&mut self, tok: &[u8]) -> Result<()> {
        if self.src[self.pos..].starts_with(tok) {
            self.pos += tok.len();
            Ok(())
        } else {
            Err(Error::Format(format!(
                "expected {:?} at byte {}",
                String::from_utf8_lossy(tok),
                self.pos
            )))
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek_at(&self, n: usize) -> Option<u8> {
        self.src.get(self.pos + n).copied()
    }

    /// Skip whitespace (newlines included) and `#` comments.
    fn skip_ws(&mut self) {
        loop {
            match self.peek() {
                Some(c) if c.is_ascii_whitespace() => self.pos += 1,
                Some(b'#') => {
                    while let Some(c) = self.peek() {
                        self.pos += 1;
                        if c == b'\n' {
                            break;
                        }
                    }
                }
                _ => return,
            }
        }
    }

    /// Skip only spaces/tabs (used before a postfix operator, so a newline ends
    /// the item rather than gluing a `*` from the next line onto it).
    fn skip_ws_inline(&mut self) {
        while let Some(c) = self.peek() {
            if is_inline_ws(c) {
                self.pos += 1;
            } else {
                return;
            }
        }
    }
}

fn exact(b: u8) -> CharSet {
    CharSet {
        ranges: vec![(b, b)],
        negated: false,
    }
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

fn is_ident_continue(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'-'
}

fn is_inline_ws(c: u8) -> bool {
    c == b' ' || c == b'\t'
}

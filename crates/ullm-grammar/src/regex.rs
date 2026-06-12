//! Regex -> GBNF compiler.
//!
//! A regular expression describes a regular language, which is exactly what a
//! right-linear grammar expresses — so we translate the regex to GBNF text and
//! reuse the whole grammar engine (classes, groups, alternation, `* + ?`) rather
//! than building a second automaton. `{n}` / `{n,m}` / `{n,}` are expanded into
//! repetitions. Everything becomes a GBNF character class `[..]` (over bytes,
//! emitted as `\xNN`) or a group `( .. )`, so quantifiers always bind to a single
//! atom.
//!
//! Supported: literals, `.`, classes `[...]`/`[^...]` with ranges, the escapes
//! `\d \w \s \D \W \S \n \r \t \f \v \xNN`, groups `(...)` / `(?:...)`,
//! alternation `|`, and the quantifiers `* + ? {n} {n,m} {n,}` (a trailing lazy
//! `?` is accepted and ignored). Anchors `^` `$` are ignored (the whole output is
//! constrained). Lookaround, backreferences and boundaries are rejected.

use ullm_core::{Error, Result};

/// Compile a regex into a GBNF fragment (a single parenthesised atom).
pub(crate) fn regex_to_gbnf(pattern: &str) -> Result<String> {
    let mut p = RegexParser {
        s: pattern.chars().collect(),
        i: 0,
    };
    let body = p.alternation()?;
    if p.i != p.s.len() {
        return Err(Error::Unsupported(format!(
            "regex: unexpected {:?} at position {}",
            p.s[p.i], p.i
        )));
    }
    Ok(format!("( {body} )"))
}

/// Cap on `{n,m}` expansion, so a pathological count can't blow up the grammar.
const MAX_REPEAT: usize = 10_000;

struct RegexParser {
    s: Vec<char>,
    i: usize,
}

impl RegexParser {
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }

    fn alternation(&mut self) -> Result<String> {
        let mut alts = vec![self.concat()?];
        while self.peek() == Some('|') {
            self.i += 1;
            alts.push(self.concat()?);
        }
        Ok(alts.join(" | "))
    }

    fn concat(&mut self) -> Result<String> {
        let mut parts = Vec::new();
        loop {
            match self.peek() {
                None | Some('|') | Some(')') => break,
                Some('^') | Some('$') => {
                    self.i += 1; // anchors: ignored
                }
                _ => parts.push(self.quantified()?),
            }
        }
        if parts.is_empty() {
            // An empty branch (e.g. `a|`) matches the empty string.
            Ok("\"\"".to_string())
        } else {
            Ok(parts.join(" "))
        }
    }

    fn quantified(&mut self) -> Result<String> {
        let atom = self.atom()?;
        let out = match self.peek() {
            Some('*') => {
                self.i += 1;
                format!("{atom}*")
            }
            Some('+') => {
                self.i += 1;
                format!("{atom}+")
            }
            Some('?') => {
                self.i += 1;
                format!("{atom}?")
            }
            Some('{') => self.repeat(&atom)?,
            _ => return Ok(atom),
        };
        if self.peek() == Some('?') {
            self.i += 1; // lazy quantifier: same language, ignore
        }
        Ok(out)
    }

    /// Expand `{n}` / `{n,m}` / `{n,}` of `atom` into a sequence.
    fn repeat(&mut self, atom: &str) -> Result<String> {
        self.i += 1; // '{'
        let n = self.number()?;
        let (lo, hi) = if self.peek() == Some(',') {
            self.i += 1;
            if self.peek() == Some('}') {
                (n, None) // {n,}
            } else {
                (n, Some(self.number()?)) // {n,m}
            }
        } else {
            (n, Some(n)) // {n}
        };
        if self.peek() != Some('}') {
            return Err(Error::Unsupported(
                "regex: malformed `{...}` quantifier".into(),
            ));
        }
        self.i += 1; // '}'
        if lo > MAX_REPEAT || hi.unwrap_or(0) > MAX_REPEAT {
            return Err(Error::Unsupported("regex: `{n,m}` count too large".into()));
        }
        let mut out = Vec::new();
        for _ in 0..lo {
            out.push(atom.to_string());
        }
        match hi {
            None => out.push(format!("{atom}*")),
            Some(hi) => {
                for _ in lo..hi {
                    out.push(format!("{atom}?"));
                }
            }
        }
        if out.is_empty() {
            Ok("\"\"".to_string())
        } else {
            Ok(out.join(" "))
        }
    }

    fn number(&mut self) -> Result<usize> {
        let start = self.i;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.i += 1;
        }
        if self.i == start {
            return Err(Error::Unsupported("regex: expected a number".into()));
        }
        self.s[start..self.i]
            .iter()
            .collect::<String>()
            .parse()
            .map_err(|_| Error::Unsupported("regex: bad number".into()))
    }

    /// One atom, returned as a single GBNF atom (`[..]` or `( .. )`).
    fn atom(&mut self) -> Result<String> {
        match self.peek() {
            Some('(') => {
                self.i += 1;
                // Skip a `(?:` non-capturing marker (other `(?...)` are rejected).
                if self.peek() == Some('?') {
                    if self.s.get(self.i + 1) == Some(&':') {
                        self.i += 2;
                    } else {
                        return Err(Error::Unsupported(
                            "regex: lookaround / flags `(?...)` unsupported".into(),
                        ));
                    }
                }
                let inner = self.alternation()?;
                if self.peek() != Some(')') {
                    return Err(Error::Unsupported("regex: unbalanced `(`".into()));
                }
                self.i += 1;
                Ok(format!("( {inner} )"))
            }
            Some('[') => self.class(),
            Some('.') => {
                self.i += 1;
                Ok(class_gbnf(&[(0x0a, 0x0a)], true)) // any byte except newline
            }
            Some('\\') => {
                self.i += 1;
                self.escape_atom()
            }
            Some(c) => {
                self.i += 1;
                Ok(literal_atom(c))
            }
            None => Err(Error::Unsupported("regex: unexpected end".into())),
        }
    }

    /// A `\X` escape used as an atom (outside a class).
    fn escape_atom(&mut self) -> Result<String> {
        let c = self
            .peek()
            .ok_or_else(|| Error::Unsupported("regex: dangling escape".into()))?;
        self.i += 1;
        if let Some((ranges, negated)) = class_escape(c) {
            return Ok(class_gbnf(&ranges, negated));
        }
        let byte = match c {
            'n' => 0x0a,
            'r' => 0x0d,
            't' => 0x09,
            'f' => 0x0c,
            'v' => 0x0b,
            '0' => 0x00,
            'x' => self.hex2()?,
            'b' | 'B' => {
                return Err(Error::Unsupported(
                    "regex: word boundary `\\b` unsupported".into(),
                ));
            }
            other if other.is_ascii() => other as u8,
            other => return Ok(literal_atom(other)), // multi-byte literal
        };
        Ok(class_gbnf(&[(byte, byte)], false))
    }

    fn hex2(&mut self) -> Result<u8> {
        let mut v = 0u8;
        for _ in 0..2 {
            let c = self
                .peek()
                .and_then(|c| c.to_digit(16))
                .ok_or_else(|| Error::Unsupported("regex: bad \\x escape".into()))?;
            v = v * 16 + c as u8;
            self.i += 1;
        }
        Ok(v)
    }

    /// A `[...]` / `[^...]` character class.
    fn class(&mut self) -> Result<String> {
        self.i += 1; // '['
        let negated = self.peek() == Some('^');
        if negated {
            self.i += 1;
        }
        let mut ranges: Vec<(u8, u8)> = Vec::new();
        loop {
            match self.peek() {
                None => return Err(Error::Unsupported("regex: unterminated class".into())),
                Some(']') => {
                    self.i += 1;
                    return Ok(class_gbnf(&ranges, negated));
                }
                _ => {
                    // `\d \w \s` inside a class contribute their ranges.
                    if self.peek() == Some('\\') {
                        if let Some(&n) = self.s.get(self.i + 1) {
                            if let Some((r, false)) = class_escape(n) {
                                self.i += 2;
                                ranges.extend(r);
                                continue;
                            }
                        }
                    }
                    let lo = self.class_byte()?;
                    if self.peek() == Some('-') && self.s.get(self.i + 1) != Some(&']') {
                        self.i += 1;
                        let hi = self.class_byte()?;
                        ranges.push((lo.min(hi), lo.max(hi)));
                    } else {
                        ranges.push((lo, lo));
                    }
                }
            }
        }
    }

    /// One byte inside a class (a literal or a single-byte escape).
    fn class_byte(&mut self) -> Result<u8> {
        if self.peek() == Some('\\') {
            self.i += 1;
            let c = self
                .peek()
                .ok_or_else(|| Error::Unsupported("regex: dangling class escape".into()))?;
            self.i += 1;
            Ok(match c {
                'n' => 0x0a,
                'r' => 0x0d,
                't' => 0x09,
                'f' => 0x0c,
                'v' => 0x0b,
                '0' => 0x00,
                'x' => self.hex2()?,
                other if other.is_ascii() => other as u8,
                _ => return Err(Error::Unsupported("regex: non-ASCII class escape".into())),
            })
        } else {
            let c = self.peek().unwrap();
            if !c.is_ascii() {
                return Err(Error::Unsupported(
                    "regex: non-ASCII class members unsupported".into(),
                ));
            }
            self.i += 1;
            Ok(c as u8)
        }
    }
}

/// `\d \w \s \D \W \S` -> (byte ranges, negated).
fn class_escape(c: char) -> Option<(Vec<(u8, u8)>, bool)> {
    let digit = vec![(0x30u8, 0x39u8)];
    let word = vec![(0x30, 0x39), (0x41, 0x5a), (0x61, 0x7a), (0x5f, 0x5f)];
    let space = vec![(0x09, 0x0d), (0x20, 0x20)];
    Some(match c {
        'd' => (digit, false),
        'D' => (digit, true),
        'w' => (word, false),
        'W' => (word, true),
        's' => (space, false),
        'S' => (space, true),
        _ => return None,
    })
}

/// Emit a GBNF character class over byte `ranges` (hex-escaped, so no GBNF
/// metacharacter ever needs special handling).
fn class_gbnf(ranges: &[(u8, u8)], negated: bool) -> String {
    let mut out = String::from("[");
    if negated {
        out.push('^');
    }
    for &(lo, hi) in ranges {
        if lo == hi {
            out.push_str(&format!("\\x{lo:02x}"));
        } else {
            out.push_str(&format!("\\x{lo:02x}-\\x{hi:02x}"));
        }
    }
    out.push(']');
    out
}

/// A literal character as a single GBNF atom: one byte -> a class, a multi-byte
/// UTF-8 char -> a group of byte classes (so a quantifier binds to the whole).
fn literal_atom(c: char) -> String {
    let mut buf = [0u8; 4];
    let bytes = c.encode_utf8(&mut buf).as_bytes();
    if bytes.len() == 1 {
        class_gbnf(&[(bytes[0], bytes[0])], false)
    } else {
        let inner: Vec<String> = bytes
            .iter()
            .map(|&b| class_gbnf(&[(b, b)], false))
            .collect();
        format!("( {} )", inner.join(" "))
    }
}

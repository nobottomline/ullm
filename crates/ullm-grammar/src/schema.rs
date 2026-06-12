//! JSON Schema -> GBNF compiler.
//!
//! Turns a (subset of) JSON Schema into a GBNF grammar that accepts exactly the
//! JSON documents the schema describes — so an agent hands uLLM a schema and
//! gets back output guaranteed to conform: the right keys, the right types, a
//! value from an `enum`, no extra properties.
//!
//! We emit GBNF *text* and feed it back through the parser, reusing all of its
//! machinery. Supported keywords: `type` (object/array/string/integer/number/
//! boolean/null, or an array of types), `properties`, `required`,
//! `additionalProperties` (objects are closed unless it is truthy), `items`,
//! `minItems`, `enum`, `const`, `anyOf`/`oneOf`, and local `$ref` into `$defs` /
//! `definitions` (including self-recursive schemas). Unsupported keywords are
//! ignored (the result is a valid superset constraint), and an untyped schema
//! falls back to "any JSON value".

use serde_json::Value;
use ullm_core::{Error, Result};

/// Compile a JSON Schema value into GBNF grammar text.
pub(crate) fn schema_to_gbnf(schema: &Value) -> Result<String> {
    let mut c = Compiler {
        defs: collect_defs(schema),
        ..Compiler::default()
    };
    let root = c.visit(schema)?;
    // No surrounding whitespace: structured output should start at the value, and
    // an unbounded leading `ws` lets a greedy model emit whitespace forever.
    let mut out = format!("root ::= {root}\n");
    // Inter-token whitespace is bounded for the same reason (32 is ample for
    // indentation); see `bounded_ws`.
    out.push_str(&format!("ws ::= {}\n", bounded_ws(32)));
    c.emit_primitives(&mut out);
    for rule in &c.rules {
        out.push_str(rule);
        out.push('\n');
    }
    Ok(out)
}

/// Gather the named subschemas from `$defs` / `definitions` (both conventions).
fn collect_defs(schema: &Value) -> serde_json::Map<String, Value> {
    let mut defs = serde_json::Map::new();
    for key in ["$defs", "definitions"] {
        if let Some(Value::Object(m)) = schema.get(key) {
            for (k, v) in m {
                defs.insert(k.clone(), v.clone());
            }
        }
    }
    defs
}

#[derive(Default)]
struct Compiler {
    rules: Vec<String>,
    counter: usize,
    /// Named subschemas (`$defs` / `definitions`) resolvable via `$ref`.
    defs: serde_json::Map<String, Value>,
    /// `$ref` target name -> the rule that realizes it (tie-the-knot recursion).
    ref_rules: std::collections::HashMap<String, String>,
    // Which shared primitives the grammar references.
    need_string: bool,
    need_integer: bool,
    need_number: bool,
    need_boolean: bool,
    need_null: bool,
    need_value: bool,
}

impl Compiler {
    fn fresh(&mut self) -> String {
        let n = self.counter;
        self.counter += 1;
        format!("r{n}")
    }

    /// Add a named rule `name ::= expr` and return `name`.
    fn add(&mut self, expr: String) -> String {
        let name = self.fresh();
        self.rules.push(format!("{name} ::= {expr}"));
        name
    }

    /// Compile `schema`, returning the name of a rule (or a primitive) matching it.
    fn visit(&mut self, schema: &Value) -> Result<String> {
        // `true` / `{}` accept anything; `false` accepts nothing (not useful).
        let obj = match schema {
            Value::Bool(true) => return Ok(self.value()),
            Value::Object(o) => o,
            _ => return Ok(self.value()),
        };

        if let Some(Value::String(pointer)) = obj.get("$ref") {
            return self.visit_ref(pointer);
        }
        if let Some(c) = obj.get("const") {
            return Ok(self.add(json_literal(c)));
        }
        if let Some(Value::Array(values)) = obj.get("enum") {
            let alts: Vec<String> = values.iter().map(json_literal).collect();
            return Ok(self.add(alts.join(" | ")));
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(Value::Array(subs)) = obj.get(key) {
                let mut alts = Vec::with_capacity(subs.len());
                for s in subs {
                    alts.push(self.visit(s)?);
                }
                return Ok(self.add(alts.join(" | ")));
            }
        }
        if let Some(Value::Array(subs)) = obj.get("allOf") {
            // Intersection is hard in general; honor the common single-element case.
            if subs.len() == 1 {
                return self.visit(&subs[0]);
            }
        }

        match obj.get("type") {
            Some(Value::String(t)) => self.visit_type(t, obj),
            Some(Value::Array(types)) => {
                let mut alts = Vec::new();
                for t in types {
                    if let Value::String(t) = t {
                        alts.push(self.visit_type(t, obj)?);
                    }
                }
                Ok(self.add(alts.join(" | ")))
            }
            // No explicit type: infer object from `properties`, else accept any value.
            _ if obj.contains_key("properties") => self.visit_object(obj),
            _ => Ok(self.value()),
        }
    }

    /// Resolve a local `$ref` (`#/$defs/Name`, `#/definitions/Name`) to a rule.
    /// The rule name is reserved *before* the target is compiled, so a schema
    /// that refers to itself (a recursive tree) ties the knot instead of looping.
    fn visit_ref(&mut self, pointer: &str) -> Result<String> {
        let name = pointer.rsplit('/').next().unwrap_or(pointer).to_string();
        if let Some(rule) = self.ref_rules.get(&name) {
            return Ok(rule.clone());
        }
        let rule = self.fresh();
        self.ref_rules.insert(name.clone(), rule.clone());
        let def =
            self.defs.get(&name).cloned().ok_or_else(|| {
                Error::Unsupported(format!("$ref to unknown definition {name:?}"))
            })?;
        let body = self.visit(&def)?;
        self.rules.push(format!("{rule} ::= {body}"));
        Ok(rule)
    }

    fn visit_type(&mut self, t: &str, obj: &serde_json::Map<String, Value>) -> Result<String> {
        Ok(match t {
            "object" => self.visit_object(obj)?,
            "array" => self.visit_array(obj)?,
            "string" => self.visit_string(obj)?,
            "integer" => {
                self.need_integer = true;
                "integer".into()
            }
            "number" => {
                self.need_number = true;
                "number".into()
            }
            "boolean" => {
                self.need_boolean = true;
                "boolean".into()
            }
            "null" => {
                self.need_null = true;
                "null".into()
            }
            other => return Err(Error::Unsupported(format!("JSON Schema type {other:?}"))),
        })
    }

    /// A string, optionally constrained by `pattern` (a regex) or `format`
    /// (a known regex). The regex matches the *content* between the quotes.
    fn visit_string(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String> {
        let pattern = obj
            .get("pattern")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| {
                obj.get("format")
                    .and_then(Value::as_str)
                    .and_then(format_pattern)
                    .map(str::to_string)
            });
        match pattern {
            Some(p) => {
                let body = crate::regex::regex_to_gbnf(&p)?;
                // A JSON string whose content matches the regex: `"` body `"`.
                Ok(self.add(format!("\"\\\"\" {body} \"\\\"\"")))
            }
            None => {
                self.need_string = true;
                Ok("string".into())
            }
        }
    }

    fn visit_array(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String> {
        let item = match obj.get("items") {
            Some(s) => self.visit(s)?,
            None => self.value(),
        };
        let min = obj.get("minItems").and_then(Value::as_u64).unwrap_or(0);
        let mut expr = String::from("\"[\" ws ");
        if min == 0 {
            // [] or a comma-separated list.
            expr.push_str(&format!("( {item} ( \",\" ws {item} )* )?"));
        } else {
            // At least `min` items: the first, then min-1 required, then any rest.
            expr.push_str(&item);
            for _ in 1..min {
                expr.push_str(&format!(" \",\" ws {item}"));
            }
            expr.push_str(&format!(" ( \",\" ws {item} )*"));
        }
        expr.push_str(" \"]\"");
        Ok(self.add(expr))
    }

    fn visit_object(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String> {
        let props = obj.get("properties").and_then(Value::as_object);
        let Some(props) = props else {
            // Object with no declared properties: any JSON object.
            self.need_string = true;
            let v = self.value();
            return Ok(self.add(format!(
                "\"{{\" ws ( string ws \":\" ws {v} ( \",\" ws string ws \":\" ws {v} )* )? \"}}\""
            )));
        };

        // Order required keys by the `required` array (the schema author's intent);
        // optional keys follow in `properties` order.
        let required_order: Vec<&str> = obj
            .get("required")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let is_required = |n: &str| required_order.contains(&n);

        // Build a `"key" ws ":" ws value` fragment for each property.
        let mut frag = std::collections::HashMap::new();
        for (name, sub) in props {
            let vrule = self.visit(sub)?;
            let key = json_literal(&Value::String(name.clone()));
            frag.insert(name.as_str(), format!("{key} ws \":\" ws {vrule}"));
        }

        let req: Vec<&String> = required_order.iter().filter_map(|n| frag.get(n)).collect();
        let opt: Vec<&String> = props
            .keys()
            .filter(|n| !is_required(n))
            .filter_map(|n| frag.get(n.as_str()))
            .collect();

        let mut expr = String::from("\"{\" ws ");
        if !req.is_empty() {
            // All required, in order; each optional independently with a leading comma.
            expr.push_str(req[0]);
            for f in &req[1..] {
                expr.push_str(&format!(" \",\" ws {f}"));
            }
            for f in &opt {
                expr.push_str(&format!(" ( \",\" ws {f} )?"));
            }
        } else if !opt.is_empty() {
            // No required: the object may be empty, or start at the first present
            // optional (no leading comma), with later ones comma-prefixed.
            let mut firsts = Vec::with_capacity(opt.len());
            for i in 0..opt.len() {
                let mut alt = opt[i].clone();
                for f in &opt[i + 1..] {
                    alt.push_str(&format!(" ( \",\" ws {f} )?"));
                }
                firsts.push(alt);
            }
            expr.push_str(&format!("( {} )?", firsts.join(" | ")));
        }
        expr.push_str(" \"}\"");
        Ok(self.add(expr))
    }

    /// A reference to the generic "any JSON value" rule (emitted once).
    fn value(&mut self) -> String {
        self.need_value = true;
        self.need_string = true;
        self.need_number = true;
        self.need_boolean = true;
        self.need_null = true;
        "value".into()
    }

    fn emit_primitives(&self, out: &mut String) {
        if self.need_value {
            out.push_str("value ::= object_any | array_any | string | number | boolean | null\n");
            out.push_str(
                "object_any ::= \"{\" ws ( string ws \":\" ws value ( \",\" ws string ws \":\" ws value )* )? \"}\"\n",
            );
            out.push_str("array_any ::= \"[\" ws ( value ( \",\" ws value )* )? \"]\"\n");
        }
        if self.need_string || self.need_value {
            out.push_str(
                "string ::= \"\\\"\" ( [^\"\\\\] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* \"\\\"\"\n",
            );
        }
        // Digit runs are BOUNDED (no real JSON number needs hundreds of digits):
        // under greedy constrained decoding an unbounded `[0-9]*` lets a confused
        // model spiral into endless digits. `d` = up to 18 further digits.
        let d = bounded_digits(18);
        let f = bounded_digits(18);
        if self.need_number || self.need_value {
            out.push_str(&format!(
                "number ::= \"-\"? (\"0\" | [1-9]{d}) (\".\" [0-9]{f})? ([eE] [-+]? [0-9] [0-9]? [0-9]?)?\n"
            ));
        }
        if self.need_integer {
            out.push_str(&format!("integer ::= \"-\"? (\"0\" | [1-9]{d})\n"));
        }
        if self.need_boolean || self.need_value {
            out.push_str("boolean ::= \"true\" | \"false\"\n");
        }
        if self.need_null || self.need_value {
            out.push_str("null ::= \"null\"\n");
        }
    }
}

/// `n` optional digits, e.g. `bounded_digits(2)` -> ` [0-9]? [0-9]?` — a
/// repetition `[0-9]{0,n}` expanded for our `*+?`-only GBNF.
fn bounded_digits(n: usize) -> String {
    " [0-9]?".repeat(n)
}

/// Up to `n` whitespace characters (`[ \t\n\r]{0,n}`), expanded for `*+?` GBNF.
fn bounded_ws(n: usize) -> String {
    let unit = "[ \\t\\n\\r]?";
    vec![unit; n].join(" ")
}

/// Map a well-known JSON Schema `format` to a regex (positive classes only, so
/// the string content can never contain a `"` that would break the JSON).
fn format_pattern(format: &str) -> Option<&'static str> {
    Some(match format {
        "date" => r"[0-9]{4}-[0-9]{2}-[0-9]{2}",
        "time" => r"[0-9]{2}:[0-9]{2}:[0-9]{2}",
        "date-time" => {
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})?"
        }
        "email" => r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "uuid" => r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        "ipv4" => r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}",
        _ => return None,
    })
}

/// Render a JSON value as a GBNF literal that matches its canonical serialization.
fn json_literal(v: &Value) -> String {
    gbnf_str_literal(&serde_json::to_string(v).unwrap_or_default())
}

/// Wrap `s` in a GBNF `"..."` literal, escaping what GBNF requires.
fn gbnf_str_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use crate::{Grammar, GrammarState};
    use serde_json::json;

    fn pieces(s: &[&str]) -> Vec<Vec<u8>> {
        s.iter().map(|t| t.as_bytes().to_vec()).collect()
    }

    /// Feed token pieces through a grammar, asserting each is allowed; return
    /// whether it ends in an acceptable state.
    fn run(g: &Grammar, steps: &[&str]) -> bool {
        let mut st = GrammarState::new(g);
        let ps = pieces(steps);
        for (i, p) in ps.iter().enumerate() {
            let mut mask = vec![false; ps.len()];
            st.allowed_mask(&ps, &mut mask);
            assert!(mask[i], "step {i} ({:?}) should be allowed", p);
            assert!(st.accept_token(p));
        }
        st.can_end()
    }

    #[test]
    fn object_with_required_and_typed_fields() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });
        let g = Grammar::from_json_schema(&schema).unwrap();
        assert!(run(
            &g,
            &[
                "{", "\"name\"", ":", "\"Jo\"", ",", "\"age\"", ":", "30", "}"
            ]
        ));

        // A non-integer age must be rejected.
        let mut st = GrammarState::new(&g);
        for p in [
            b"{".as_ref(),
            b"\"name\"",
            b":",
            b"\"Jo\"",
            b",",
            b"\"age\"",
            b":",
        ] {
            assert!(st.accept_token(p));
        }
        let ps = pieces(&["\"x\"", "5"]);
        let mut mask = vec![false; ps.len()];
        st.allowed_mask(&ps, &mut mask);
        assert!(!mask[0], "age value cannot be a string");
        assert!(mask[1], "age value can be a digit");
    }

    #[test]
    fn enum_restricts_to_fixed_values() {
        let schema = json!({ "enum": ["red", "green", "blue"] });
        let g = Grammar::from_json_schema(&schema).unwrap();
        assert!(run(&g, &["\"red\""]));
        let st = GrammarState::new(&g);
        let ps = pieces(&["\"red\"", "\"x\""]);
        let mut mask = vec![false; ps.len()];
        st.allowed_mask(&ps, &mut mask);
        assert!(mask[0] && !mask[1]);
    }

    #[test]
    fn array_of_integers_with_min_items() {
        let schema = json!({
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 1
        });
        let g = Grammar::from_json_schema(&schema).unwrap();
        assert!(run(&g, &["[", "1", ",", "2", "]"]));
        // Empty array violates minItems: after "[", "]" is not allowed.
        let mut st = GrammarState::new(&g);
        assert!(st.accept_token(b"["));
        let ps = pieces(&["]", "1"]);
        let mut mask = vec![false; ps.len()];
        st.allowed_mask(&ps, &mut mask);
        assert!(!mask[0], "empty array violates minItems:1");
        assert!(mask[1], "a digit is allowed");
    }

    #[test]
    fn ref_and_recursive_schema() {
        // A nested address via $ref...
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {"$ref": "#/$defs/addr"}
            },
            "required": ["name", "address"],
            "$defs": {
                "addr": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        });
        let g = Grammar::from_json_schema(&schema).unwrap();
        assert!(run(
            &g,
            &[
                "{",
                "\"name\"",
                ":",
                "\"Jo\"",
                ",",
                "\"address\"",
                ":",
                "{",
                "\"city\"",
                ":",
                "\"NY\"",
                "}",
                "}"
            ]
        ));

        // ...and a self-recursive tree (ties the knot, no infinite loop).
        let tree = json!({
            "$ref": "#/$defs/node",
            "$defs": {
                "node": {
                    "type": "object",
                    "properties": {
                        "v": {"type": "integer"},
                        "kids": {"type": "array", "items": {"$ref": "#/$defs/node"}}
                    },
                    "required": ["v"]
                }
            }
        });
        let g = Grammar::from_json_schema(&tree).unwrap();
        assert!(run(
            &g,
            &[
                "{", "\"v\"", ":", "1", ",", "\"kids\"", ":", "[", "{", "\"v\"", ":", "2", "}",
                "]", "}"
            ]
        ));
    }

    #[test]
    fn string_format_and_pattern() {
        // format: date -> the string content must be a date.
        let schema = json!({ "type": "string", "format": "date" });
        let g = Grammar::from_json_schema(&schema).unwrap();
        assert!(run(&g, &["\"", "2024", "-", "01", "-", "15", "\""]));

        // pattern: a fixed product code AB-1234.
        let schema = json!({ "type": "string", "pattern": r"[A-Z]{2}-[0-9]{4}" });
        let g = Grammar::from_json_schema(&schema).unwrap();
        assert!(run(&g, &["\"", "AB", "-", "1234", "\""]));
        let mut st = GrammarState::new(&g);
        st.accept_token(b"\"");
        let ps = pieces(&["A", "9"]);
        let mut mask = vec![false; ps.len()];
        st.allowed_mask(&ps, &mut mask);
        assert!(mask[0] && !mask[1], "code must start with a letter");
    }

    #[test]
    fn optional_properties_any_subset() {
        let schema = json!({
            "type": "object",
            "properties": { "a": {"type": "integer"}, "b": {"type": "integer"} }
        });
        let g = Grammar::from_json_schema(&schema).unwrap();
        assert!(run(&g, &["{", "}"]), "empty object ok (no required)");
        assert!(
            run(&g, &["{", "\"b\"", ":", "2", "}"]),
            "only the second optional"
        );
        assert!(run(
            &g,
            &["{", "\"a\"", ":", "1", ",", "\"b\"", ":", "2", "}"]
        ));
    }
}

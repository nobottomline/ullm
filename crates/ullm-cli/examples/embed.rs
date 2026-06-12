//! Embed uLLM as a library: load a GGUF model and generate JSON that is
//! *guaranteed* to match a schema — no server, no Python, ~30 lines.
//!
//!     cargo run --release -p ullm-cli --example embed -- path/to/model.gguf

use ullm_gguf::GgufModel;
use ullm_model::{
    Grammar, GrammarConstraint, LlamaModel, LogitConstraint, SampleParams, TokenTrie,
};

fn main() {
    let path = std::env::args().nth(1).expect("usage: embed <model.gguf>");

    let gguf = GgufModel::open(&path).expect("open gguf");
    let tk = gguf.tokenizer().expect("tokenizer");
    let mut model = LlamaModel::from_gguf(&gguf).expect("load model");

    // A JSON Schema compiled to a grammar the decoder physically cannot violate.
    let schema = r#"{
        "type": "object",
        "properties": { "name": {"type": "string"}, "age": {"type": "integer"} },
        "required": ["name", "age"]
    }"#;
    let grammar = Grammar::from_json_schema_str(schema).expect("compile schema");
    let trie = TokenTrie::new(tk.token_pieces());
    let mut constraint = GrammarConstraint::new(&grammar, &trie, tk.eos_id());

    let prompt = tk.encode("Extract a person. John is 30 years old. JSON:", true);
    let out = model.generate(
        &prompt,
        64,
        tk.eos_id(),
        &SampleParams::default(),
        Some(&mut constraint as &mut dyn LogitConstraint),
    );

    // Always parseable, always matching the schema.
    println!("{}", tk.decode(&out));
}

//! `ullm` — the universal LLM inference engine command-line interface.

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use ullm_core::device::Hardware;
use ullm_gguf::GgufModel;

#[derive(Parser)]
#[command(name = "ullm", version, about = "Universal LLM inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect the host hardware and report what uLLM can run.
    Doctor,
    /// Inspect a GGUF model file: architecture, hyperparameters, tensor count.
    Inspect {
        /// Path to a `.gguf` model file.
        path: PathBuf,
    },
    /// Tokenize text with a GGUF model's tokenizer, then round-trip it back.
    Tokenize {
        /// Path to a `.gguf` model file.
        path: PathBuf,
        /// Text to tokenize.
        text: String,
    },
    /// Run a model (not yet implemented — see docs/roadmap.md).
    Run,
    /// Start an OpenAI-compatible server (not yet implemented).
    Serve,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Doctor => doctor(),
        Command::Inspect { path } => inspect(&path),
        Command::Tokenize { path, text } => tokenize(&path, &text),
        Command::Run | Command::Serve => {
            eprintln!("not yet implemented — see docs/roadmap.md (Phase 0)");
            std::process::exit(1);
        }
    }
}

fn doctor() {
    let hw = Hardware::detect();
    println!("uLLM doctor");
    println!("  chip:           {}", hw.chip);
    println!("  unified memory: {:.1} GiB", hw.total_memory_gib());
    println!("  cpu cores:      {}", hw.cpu_cores);
    println!(
        "  apple silicon:  {}",
        if hw.apple_silicon { "yes" } else { "no" }
    );
    println!(
        "  metal backend:  {}",
        if hw.metal { "available" } else { "unavailable" }
    );
}

fn inspect(path: &Path) {
    let model = match GgufModel::open(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let s = model.model_spec();
    println!("gguf: {}", path.display());
    println!("  version:       {}", model.version);
    println!("  architecture:  {}", s.architecture);
    println!("  context len:   {}", s.context_length);
    println!("  hidden size:   {}", s.hidden_size);
    println!("  layers:        {}", s.num_layers);
    println!("  heads:         {} (kv {})", s.num_heads, s.num_kv_heads);
    println!("  vocab:         {}", s.vocab_size);
    println!("  tensors:       {}", model.tensors.len());

    if let Some(tk) = model
        .metadata_get("tokenizer.ggml.model")
        .and_then(|v| v.as_str())
    {
        println!("  tokenizer:     {tk}");
    }
    let arr_len = |k: &str| {
        model
            .metadata_get(k)
            .and_then(|v| v.as_array())
            .map(|a| a.len())
    };
    println!(
        "  tok arrays:    tokens={:?} scores={:?} merges={:?} types={:?}",
        arr_len("tokenizer.ggml.tokens"),
        arr_len("tokenizer.ggml.scores"),
        arr_len("tokenizer.ggml.merges"),
        arr_len("tokenizer.ggml.token_type"),
    );
    for key in [
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.unknown_token_id",
    ] {
        if let Some(id) = model.metadata_get(key).and_then(|v| v.to_u64()) {
            println!("  {key} = {id}");
        }
    }
    if let Some(toks) = model
        .metadata_get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
    {
        let sample: Vec<&str> = toks.iter().take(14).filter_map(|v| v.as_str()).collect();
        println!("  first tokens:  {sample:?}");
    }
}

fn tokenize(path: &Path, text: &str) {
    let model = match GgufModel::open(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let tk = match model.tokenizer() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let ids = tk.encode(text, true);
    println!("input:    {text:?}");
    println!("tokens:   {} -> {:?}", ids.len(), ids);
    println!("decoded:  {:?}", tk.decode(&ids));
}

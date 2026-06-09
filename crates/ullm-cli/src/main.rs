//! `ullm` — the universal LLM inference engine command-line interface.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use ullm_core::device::Hardware;
use ullm_gguf::GgufModel;
use ullm_metal::MetalContext;
use ullm_model::{LlamaModel, SampleParams};

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
    /// Generate text from a prompt (greedy decoding on CPU).
    Run {
        /// Path to a `.gguf` model file.
        path: PathBuf,
        /// The prompt text.
        prompt: String,
        /// Maximum number of new tokens to generate.
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        /// Sampling temperature (0 = greedy / deterministic).
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Keep only the top-k logits (0 = disabled).
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        /// Nucleus sampling threshold (1.0 = disabled).
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        /// RNG seed (0 = fixed default).
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Start an OpenAI-compatible HTTP server.
    Serve {
        /// Path to a `.gguf` model file.
        path: PathBuf,
        /// Host / interface to bind.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to listen on.
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Check the Metal GPU backend and validate a kernel against the CPU.
    MetalCheck,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Doctor => doctor(),
        Command::Inspect { path } => inspect(&path),
        Command::Tokenize { path, text } => tokenize(&path, &text),
        Command::Run {
            path,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
        } => run(
            &path,
            &prompt,
            max_tokens,
            SampleParams {
                temperature,
                top_k,
                top_p,
                seed,
            },
        ),
        Command::Serve { path, host, port } => {
            if let Err(e) = ullm_server::run(&path, &host, port) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Command::MetalCheck => metal_check(),
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

    let mut dtypes: BTreeMap<String, usize> = BTreeMap::new();
    for t in model.tensors.tensors.values() {
        *dtypes.entry(format!("{:?}", t.dtype)).or_default() += 1;
    }
    println!("  dtypes:        {dtypes:?}");

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

fn run(path: &Path, prompt: &str, max_tokens: usize, params: SampleParams) {
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
    let mut lm = match LlamaModel::from_gguf(&model) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let prompt_ids = tk.encode(prompt, true);
    let generated = lm.generate(&prompt_ids, max_tokens, tk.eos_id(), &params);

    let mut full = prompt_ids.clone();
    full.extend_from_slice(&generated);
    println!("{}", tk.decode(&full));
}

fn metal_check() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    println!("metal device:  {}", ctx.device_name());

    let (out_dim, in_dim) = (4096usize, 4096usize);
    let w: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

    let gpu = ctx.matvec(&w, &x, out_dim, in_dim);
    let cpu: Vec<f32> = (0..out_dim)
        .map(|o| {
            w[o * in_dim..o * in_dim + in_dim]
                .iter()
                .zip(&x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();
    let max_err = gpu
        .iter()
        .zip(&cpu)
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);

    println!("gemv {out_dim}x{in_dim}: max|gpu-cpu| = {max_err:.3e}");
    if max_err < 1e-2 {
        println!("validation:    OK (matches CPU reference)");
    } else {
        println!("validation:    MISMATCH");
        std::process::exit(1);
    }
}

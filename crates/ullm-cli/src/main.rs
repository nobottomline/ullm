//! `ullm` — the universal LLM inference engine command-line interface.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use ullm_core::device::Hardware;
use ullm_core::ir::WeightSource;
use ullm_gguf::GgufModel;
use ullm_metal::MetalContext;
use ullm_model::{LlamaModel, SampleParams};
use ullm_safetensors::SafeTensorsModel;

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

/// Whether `path` is a Hugging Face / SafeTensors model (a directory or a
/// `.safetensors` file) rather than a GGUF file.
fn is_safetensors(path: &Path) -> bool {
    path.is_dir() || path.extension().is_some_and(|e| e == "safetensors")
}

fn inspect(path: &Path) {
    if is_safetensors(path) {
        inspect_safetensors(path);
        return;
    }
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

fn inspect_safetensors(path: &Path) {
    let model = match SafeTensorsModel::open(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    println!("safetensors: {}", path.display());
    let arch = model
        .config()
        .get("architectures")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .or_else(|| model.config_str("model_type"))
        .unwrap_or("<unknown>");
    println!("  architecture:  {arch}");
    let field = |k: &str| model.config_usize(k);
    if let Some(v) = field("hidden_size") {
        println!("  hidden size:   {v}");
    }
    if let Some(v) = field("num_hidden_layers") {
        println!("  layers:        {v}");
    }
    let heads = field("num_attention_heads");
    let kv = field("num_key_value_heads").or(heads);
    if let (Some(h), Some(k)) = (heads, kv) {
        println!("  heads:         {h} (kv {k})");
    }
    if let Some(v) = field("head_dim") {
        println!("  head dim:      {v}");
    }
    if let Some(v) = field("intermediate_size") {
        println!("  ffn:           {v}");
    }
    if let Some(v) = field("vocab_size") {
        println!("  vocab:         {v}");
    }
    if let Some(v) = field("max_position_embeddings") {
        println!("  context len:   {v}");
    }
    if let Some(v) = model.config_f32("rope_theta") {
        println!("  rope theta:    {v}");
    }
    println!("  tensors:       {}", model.tensor_bag().len());

    let mut dtypes: BTreeMap<String, usize> = BTreeMap::new();
    for t in model.tensor_bag().tensors.values() {
        *dtypes.entry(format!("{:?}", t.dtype)).or_default() += 1;
    }
    println!("  dtypes:        {dtypes:?}");
    println!(
        "  tokenizer.json: {}",
        model
            .tokenizer_json_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<none>".into())
    );
    let mut names: Vec<&String> = model.tensor_bag().tensors.keys().collect();
    names.sort();
    let sample: Vec<&String> = names.into_iter().take(6).collect();
    println!("  first tensors: {sample:?}");
}

/// Load a tokenizer from a HF directory / `.safetensors` model via its
/// `tokenizer.json`, reading `bos`/`eos` ids from `config.json`.
fn load_hf_tokenizer(path: &Path) -> ullm_tokenizer::Tokenizer {
    let model = match SafeTensorsModel::open(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let tj = match model.tokenizer_json_path() {
        Some(p) => p,
        None => {
            eprintln!("error: no tokenizer.json next to the model");
            std::process::exit(1);
        }
    };
    let bytes = std::fs::read(&tj).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let bos = model.config_usize("bos_token_id").map(|v| v as u32);
    let eos = model.config_usize("eos_token_id").map(|v| v as u32);
    match ullm_tokenizer::Tokenizer::from_hf_json(&bytes, bos, eos, false) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn tokenize(path: &Path, text: &str) {
    let tk = if is_safetensors(path) {
        load_hf_tokenizer(path)
    } else {
        let model = match GgufModel::open(path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        };
        match model.tokenizer() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
    };
    let ids = tk.encode(text, true);
    println!("input:    {text:?}");
    println!("tokens:   {} -> {:?}", ids.len(), ids);
    println!("decoded:  {:?}", tk.decode(&ids));
}

fn run(path: &Path, prompt: &str, max_tokens: usize, params: SampleParams) {
    let exit = |e| -> ! {
        eprintln!("error: {e}");
        std::process::exit(1);
    };
    let (tk, mut lm) = if is_safetensors(path) {
        let st = SafeTensorsModel::open(path).unwrap_or_else(|e| exit(e));
        let tk = load_hf_tokenizer(path);
        let lm = LlamaModel::from_safetensors(&st).unwrap_or_else(|e| exit(e));
        (tk, lm) // `st`'s mmap drops here; `lm` owns copied weights
    } else {
        let model = GgufModel::open(path).unwrap_or_else(|e| exit(e));
        let tk = model.tokenizer().unwrap_or_else(|e| exit(e));
        let lm = LlamaModel::from_gguf(&model).unwrap_or_else(|e| exit(e));
        (tk, lm)
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

    let (o, i) = (1024usize, 4096usize);
    let x: Vec<f32> = (0..i).map(|k| ((k % 13) as f32 - 6.0) * 0.1).collect();

    let mut ok = true;
    {
        let w: Vec<f32> = (0..o * i).map(|k| ((k % 17) as f32 - 8.0) * 0.01).collect();
        ok &= report("f32 ", &ctx.matvec(&w, &x, o, i), &cpu_ref(&w, &x, o, i));
    }
    ok &= check_quant(&ctx, ullm_core::DType::Q4K, "Q4_K", &x, o, i);
    ok &= check_quant(&ctx, ullm_core::DType::Q6K, "Q6_K", &x, o, i);

    if !ok {
        std::process::exit(1);
    }
    bench_gemv(&ctx, 5632, 2048);
}

fn cpu_ref(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    (0..out_dim)
        .map(|o| {
            w[o * in_dim..o * in_dim + in_dim]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

fn report(label: &str, gpu: &[f32], cpu: &[f32]) -> bool {
    let scale = cpu.iter().map(|c| c.abs()).fold(0.0f32, f32::max).max(1e-6);
    let abs = gpu
        .iter()
        .zip(cpu)
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    let rel = abs / scale;
    let status = if rel < 1e-3 { "OK" } else { "MISMATCH" };
    println!("  {label} GEMV vs CPU: rel err {rel:.2e}  [{status}]");
    rel < 1e-3
}

fn check_quant(
    ctx: &MetalContext,
    dtype: ullm_core::DType,
    label: &str,
    x: &[f32],
    o: usize,
    i: usize,
) -> bool {
    let ts = dtype.type_size();
    let total = o * (i / 256) * ts;
    let half = 0x3000u16.to_le_bytes(); // 0.125 — keeps dequantized values finite
    let mut w: Vec<u8> = (0..total)
        .map(|k| (k.wrapping_mul(131).wrapping_add(7) % 251) as u8)
        .collect();
    let d_offsets: &[usize] = if dtype == ullm_core::DType::Q6K {
        &[208]
    } else {
        &[0, 2]
    };
    for blk in w.chunks_mut(ts) {
        for &off in d_offsets {
            blk[off] = half[0];
            blk[off + 1] = half[1];
        }
    }
    let cpu_w = ullm_core::dequant::dequantize(dtype, &w, o * i).expect("cpu dequant");
    let cpu = cpu_ref(&cpu_w, x, o, i);
    match ctx.matvec_quant(dtype, &w, x, o, i) {
        Ok(gpu) => report(label, &gpu, &cpu),
        Err(e) => {
            eprintln!("error: {e}");
            false
        }
    }
}

/// Compare GPU (quantized, resident) vs GPU/CPU f32 throughput on one GEMV.
fn bench_gemv(ctx: &MetalContext, o: usize, i: usize) {
    use rayon::prelude::*;

    let ts = 144usize;
    let total = o * (i / 256) * ts;
    let half = 0x3000u16.to_le_bytes();
    let mut wq: Vec<u8> = (0..total)
        .map(|k| (k.wrapping_mul(131).wrapping_add(7) % 251) as u8)
        .collect();
    for blk in wq.chunks_mut(ts) {
        blk[0] = half[0];
        blk[1] = half[1];
        blk[2] = half[0];
        blk[3] = half[1];
    }
    let wf = ullm_core::dequant::dequantize(ullm_core::DType::Q4K, &wq, o * i).expect("dequant");
    let x: Vec<f32> = (0..i).map(|k| ((k % 13) as f32 - 6.0) * 0.1).collect();
    let wbuf = ctx.upload(&wq);
    let n = 200;

    let _ = ctx.matvec_resident(ullm_core::DType::Q4K, &wbuf, &x, o, i); // warm up
    let gpu_q4k = time_ms(n, || {
        let _ = ctx.matvec_resident(ullm_core::DType::Q4K, &wbuf, &x, o, i);
    });
    let gpu_f32 = time_ms(n, || {
        let _ = ctx.matvec(&wf, &x, o, i);
    });
    let cpu_f32 = time_ms(n, || {
        let _: Vec<f32> = (0..o)
            .into_par_iter()
            .map(|r| {
                wf[r * i..r * i + i]
                    .iter()
                    .zip(&x)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();
    });

    println!();
    println!("throughput  {o}x{i} GEMV (avg of {n} ops):");
    println!("  GPU Q4_K (resident):  {gpu_q4k:.3} ms");
    println!("  GPU f32:              {gpu_f32:.3} ms");
    println!("  CPU f32 (rayon):      {cpu_f32:.3} ms");
    println!(
        "  GPU-Q4_K: {:.1}x vs CPU-f32, {:.1}x vs GPU-f32",
        cpu_f32 / gpu_q4k,
        gpu_f32 / gpu_q4k
    );
}

fn time_ms<F: FnMut()>(n: usize, mut f: F) -> f64 {
    let t = std::time::Instant::now();
    for _ in 0..n {
        f();
    }
    t.elapsed().as_secs_f64() * 1000.0 / n as f64
}

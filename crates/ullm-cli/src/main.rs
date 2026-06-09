//! `ullm` — the universal LLM inference engine command-line interface.

use clap::{Parser, Subcommand};
use ullm_core::device::Hardware;

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
    /// Run a model (not yet implemented — see docs/roadmap.md).
    Run,
    /// Start an OpenAI-compatible server (not yet implemented).
    Serve,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Doctor => doctor(),
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

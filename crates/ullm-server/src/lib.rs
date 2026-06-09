//! A minimal OpenAI-compatible HTTP server for uLLM.
//!
//! Phase 1: non-streaming `/v1/chat/completions` and `/v1/models`. One request
//! is served at a time (a mutex around the model); continuous batching is a
//! later phase.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

use ullm_core::{Error, Result};
use ullm_gguf::GgufModel;
use ullm_model::{LlamaModel, SampleParams};
use ullm_tokenizer::Tokenizer;

/// A loaded model + tokenizer, generation-ready.
struct Engine {
    model: LlamaModel,
    tokenizer: Tokenizer,
    model_id: String,
}

impl Engine {
    fn load(path: &Path) -> Result<Self> {
        let gguf = GgufModel::open(path)?;
        let tokenizer = gguf.tokenizer()?;
        let model = LlamaModel::from_gguf(&gguf)?;
        let model_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();
        Ok(Self {
            model,
            tokenizer,
            model_id,
        })
    }

    /// Returns (text, prompt_tokens, completion_tokens).
    fn complete(&mut self, prompt: &str, max_tokens: usize, params: &SampleParams) -> (String, usize, usize) {
        let prompt_ids = self.tokenizer.encode(prompt, true);
        let generated = self
            .model
            .generate(&prompt_ids, max_tokens, self.tokenizer.eos_id(), params);
        let text = self.tokenizer.decode(&generated);
        (text, prompt_ids.len(), generated.len())
    }
}

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<Engine>>,
    model_id: String,
}

/// Load the model and serve until the process is stopped (blocking).
pub fn run(model_path: &Path, host: &str, port: u16) -> Result<()> {
    let engine = Engine::load(model_path)?;
    let model_id = engine.model_id.clone();
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        model_id: model_id.clone(),
    };

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| Error::Format(format!("tokio runtime: {e}")))?;

    rt.block_on(async move {
        let addr = format!("{host}:{port}");
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .map_err(|e| Error::Format(format!("bind {addr}: {e}")))?;
        println!("uLLM server: model '{model_id}' ready on http://{addr}");
        axum::serve(listener, app)
            .await
            .map_err(|e| Error::Format(format!("serve: {e}")))?;
        Ok(())
    })
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

async fn list_models(State(s): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelInfo {
            id: s.model_id,
            object: "model",
            created: unix_now(),
            owned_by: "uLLM",
        }],
    })
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

async fn chat_completions(State(s): State<AppState>, Json(req): Json<ChatRequest>) -> Json<ChatResponse> {
    let prompt = build_prompt(&req.messages);
    let max_tokens = req.max_tokens.unwrap_or(128);
    let params = SampleParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_k: 0,
        top_p: req.top_p.unwrap_or(1.0),
        seed: req.seed.unwrap_or(0),
    };

    let engine = s.engine.clone();
    let (text, pt, ct) = tokio::task::spawn_blocking(move || {
        let mut e = engine.lock().expect("engine mutex poisoned");
        e.complete(&prompt, max_tokens, &params)
    })
    .await
    .unwrap_or_else(|_| (String::new(), 0, 0));

    Json(ChatResponse {
        id: "chatcmpl-ullm".to_string(),
        object: "chat.completion",
        created: unix_now(),
        model: s.model_id,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text.trim().to_string(),
            },
            finish_reason: "stop",
        }],
        usage: Usage {
            prompt_tokens: pt,
            completion_tokens: ct,
            total_tokens: pt + ct,
        },
    })
}

/// Render chat messages into a Zephyr/TinyLlama-style prompt.
fn build_prompt(messages: &[ChatMessage]) -> String {
    let mut p = String::new();
    for m in messages {
        let tag = match m.role.as_str() {
            "system" | "user" | "assistant" => m.role.as_str(),
            _ => "user",
        };
        p.push_str(&format!("<|{tag}|>\n{}\n", m.content));
    }
    p.push_str("<|assistant|>\n");
    p
}

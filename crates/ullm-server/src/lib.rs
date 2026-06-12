//! A minimal OpenAI-compatible HTTP server for uLLM.
//!
//! Phase 1: `/v1/chat/completions` (SSE streaming and non-streaming) and
//! `/v1/models`. One request is served at a time (a mutex around the model);
//! continuous batching is a later phase.

use std::convert::Infallible;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use ullm_core::{Error, Result};
use ullm_gguf::GgufModel;
use ullm_model::{Grammar, GrammarConstraint, LlamaModel, SampleParams};
use ullm_safetensors::SafeTensorsModel;
use ullm_tokenizer::Tokenizer;

/// A loaded model + tokenizer, generation-ready.
struct Engine {
    model: LlamaModel,
    tokenizer: Tokenizer,
    model_id: String,
    chat_format: ChatFormat,
}

impl Engine {
    fn load(path: &Path, gpu: bool) -> Result<Self> {
        let is_hf = path.is_dir() || path.extension().is_some_and(|e| e == "safetensors");
        let (tokenizer, mut model, template) = if is_hf {
            let st = SafeTensorsModel::open(path)?;
            let tj = st
                .tokenizer_json_path()
                .ok_or_else(|| Error::Format("no tokenizer.json next to the model".into()))?;
            let bytes = std::fs::read(&tj)?;
            let bos = st.config_usize("bos_token_id").map(|v| v as u32);
            let eos = st.config_usize("eos_token_id").map(|v| v as u32);
            let tk = Tokenizer::from_hf_json(&bytes, bos, eos, false)?;
            let m = LlamaModel::from_safetensors(&st)?;
            let template = hf_chat_template(path);
            (tk, m, template)
        } else {
            let gguf = GgufModel::open(path)?;
            let tk = gguf.tokenizer()?;
            let template = gguf
                .metadata_get("tokenizer.chat_template")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let m = LlamaModel::from_gguf(&gguf)?;
            (tk, m, template)
        };
        if gpu {
            if let Err(e) = model.enable_gpu() {
                eprintln!("uLLM server: GPU init failed ({e}); falling back to CPU");
            }
        }
        let chat_format = ChatFormat::detect(template.as_deref());
        let model_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();
        Ok(Self {
            model,
            tokenizer,
            model_id,
            chat_format,
        })
    }

    /// A grammar constraint bound to this engine's vocabulary, if `grammar` is set.
    fn constraint<'g>(&self, grammar: Option<&'g Grammar>) -> Option<GrammarConstraint<'g>> {
        grammar.map(|g| {
            GrammarConstraint::new(g, self.tokenizer.token_pieces(), self.tokenizer.eos_id())
        })
    }

    /// Returns (text, prompt_tokens, completion_tokens).
    fn complete(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        params: &SampleParams,
        grammar: Option<&Grammar>,
    ) -> (String, usize, usize) {
        let prompt_ids = self.tokenizer.encode(prompt, true);
        let mut constraint = self.constraint(grammar);
        let generated = self.model.generate(
            &prompt_ids,
            max_tokens,
            self.tokenizer.eos_id(),
            params,
            constraint
                .as_mut()
                .map(|c| c as &mut dyn ullm_model::LogitConstraint),
        );
        let text = self.tokenizer.decode(&generated);
        (text, prompt_ids.len(), generated.len())
    }

    /// Stream the completion, invoking `on_delta` with each new text piece.
    fn complete_stream<F: FnMut(&str)>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        params: &SampleParams,
        grammar: Option<&Grammar>,
        mut on_delta: F,
    ) {
        let prompt_ids = self.tokenizer.encode(prompt, true);
        let eos = self.tokenizer.eos_id();
        let mut all = prompt_ids.clone();
        let mut sent = self.tokenizer.decode(&prompt_ids).len();
        let mut constraint = self.constraint(grammar);
        let cons = constraint
            .as_mut()
            .map(|c| c as &mut dyn ullm_model::LogitConstraint);
        let tok = &self.tokenizer;
        self.model
            .generate_stream(&prompt_ids, max_tokens, eos, params, cons, |id| {
                all.push(id);
                let full = tok.decode(&all);
                if full.len() > sent {
                    if let Some(delta) = full.get(sent..) {
                        on_delta(delta);
                        sent = full.len();
                    }
                }
                true
            });
    }
}

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<Engine>>,
    model_id: String,
    chat_format: ChatFormat,
}

/// Load the model and serve until the process is stopped (blocking).
pub fn run(model_path: &Path, host: &str, port: u16, gpu: bool) -> Result<()> {
    let engine = Engine::load(model_path, gpu)?;
    let backend = if engine.model.gpu_enabled() {
        "gpu"
    } else {
        "cpu"
    };
    let model_id = engine.model_id.clone();
    let chat_format = engine.chat_format;
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        model_id: model_id.clone(),
        chat_format,
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
        println!("uLLM server: model '{model_id}' ({backend}) ready on http://{addr}");
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
    #[serde(default)]
    stream: Option<bool>,
    /// OpenAI Structured Outputs: `{"type": "json_object"}` or
    /// `{"type": "json_schema", "json_schema": {"schema": {...}}}`.
    #[serde(default)]
    response_format: Option<ResponseFormat>,
    /// uLLM extension: a raw GBNF grammar string (takes precedence).
    #[serde(default)]
    grammar: Option<String>,
}

#[derive(Deserialize)]
struct ResponseFormat {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    json_schema: Option<JsonSchemaSpec>,
}

#[derive(Deserialize)]
struct JsonSchemaSpec {
    #[serde(default)]
    schema: Option<serde_json::Value>,
}

/// Build the decoding constraint requested by a chat request (a raw GBNF
/// grammar, a JSON Schema, or plain JSON mode). `Ok(None)` means unconstrained.
fn request_grammar(req: &ChatRequest) -> Result<Option<Grammar>> {
    if let Some(gbnf) = &req.grammar {
        return Ok(Some(Grammar::from_gbnf(gbnf)?));
    }
    match &req.response_format {
        Some(rf) => match rf.kind.as_str() {
            "json_object" => Ok(Some(Grammar::json())),
            "json_schema" => {
                let schema = rf
                    .json_schema
                    .as_ref()
                    .and_then(|s| s.schema.as_ref())
                    .ok_or_else(|| {
                        Error::Format("response_format json_schema needs a `schema`".into())
                    })?;
                Ok(Some(Grammar::from_json_schema(schema)?))
            }
            _ => Ok(None), // "text" or unset
        },
        None => Ok(None),
    }
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

async fn chat_completions(State(s): State<AppState>, Json(req): Json<ChatRequest>) -> Response {
    let prompt = s.chat_format.build_prompt(&req.messages);
    let max_tokens = req.max_tokens.unwrap_or(128);
    let params = SampleParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_k: 0,
        top_p: req.top_p.unwrap_or(1.0),
        seed: req.seed.unwrap_or(0),
    };
    let model_id = s.model_id.clone();

    // Compile the requested structured-output constraint, if any (400 on error).
    let grammar = match request_grammar(&req) {
        Ok(g) => g,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                format!("invalid response_format / grammar: {e}"),
            )
                .into_response();
        }
    };

    if req.stream == Some(true) {
        let (tx, rx) = mpsc::channel::<std::result::Result<Event, Infallible>>(64);
        let engine = s.engine.clone();
        let mid = model_id;
        tokio::task::spawn_blocking(move || {
            let send = |data: String| {
                let _ = tx.blocking_send(Ok(Event::default().data(data)));
            };
            send(chunk_json(&mid, Some("assistant"), None, None));
            {
                let mut e = engine.lock().expect("engine mutex poisoned");
                e.complete_stream(&prompt, max_tokens, &params, grammar.as_ref(), |delta| {
                    send(chunk_json(&mid, None, Some(delta), None));
                });
            }
            send(chunk_json(&mid, None, None, Some("stop")));
            send("[DONE]".to_string());
        });
        return Sse::new(ReceiverStream::new(rx)).into_response();
    }

    let engine = s.engine.clone();
    let (text, pt, ct) = tokio::task::spawn_blocking(move || {
        let mut e = engine.lock().expect("engine mutex poisoned");
        e.complete(&prompt, max_tokens, &params, grammar.as_ref())
    })
    .await
    .unwrap_or_else(|_| (String::new(), 0, 0));

    Json(ChatResponse {
        id: "chatcmpl-ullm".to_string(),
        object: "chat.completion",
        created: unix_now(),
        model: model_id,
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
    .into_response()
}

#[derive(Serialize)]
struct ChatChunk {
    id: &'static str,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
}

#[derive(Serialize)]
struct ChunkChoice {
    index: usize,
    delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

/// Serialize one OpenAI `chat.completion.chunk`.
fn chunk_json(
    model: &str,
    role: Option<&str>,
    content: Option<&str>,
    finish: Option<&str>,
) -> String {
    let chunk = ChatChunk {
        id: "chatcmpl-ullm",
        object: "chat.completion.chunk",
        created: unix_now(),
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: role.map(str::to_string),
                content: content.map(str::to_string),
            },
            finish_reason: finish.map(str::to_string),
        }],
    };
    serde_json::to_string(&chunk).unwrap_or_default()
}

/// Read a Hugging Face model's chat template (`chat_template.jinja` or the
/// `chat_template` field of `tokenizer_config.json`).
fn hf_chat_template(dir: &Path) -> Option<String> {
    std::fs::read_to_string(dir.join("chat_template.jinja"))
        .ok()
        .or_else(|| {
            let cfg = std::fs::read_to_string(dir.join("tokenizer_config.json")).ok()?;
            let v: serde_json::Value = serde_json::from_str(&cfg).ok()?;
            v.get("chat_template")
                .and_then(|t| t.as_str())
                .map(str::to_string)
        })
}

/// The chat prompt format a model expects, detected from its chat template.
#[derive(Clone, Copy, Debug)]
enum ChatFormat {
    ChatML,
    Gemma,
    Llama3,
    Zephyr,
}

impl ChatFormat {
    /// Pick a format from the model's `chat_template` string (substring markers).
    fn detect(template: Option<&str>) -> Self {
        match template {
            Some(t) if t.contains("<|im_start|>") => ChatFormat::ChatML,
            Some(t) if t.contains("<start_of_turn>") => ChatFormat::Gemma,
            Some(t) if t.contains("<|start_header_id|>") => ChatFormat::Llama3,
            _ => ChatFormat::Zephyr,
        }
    }

    /// Render messages into the model's native chat prompt, ending with the
    /// open assistant turn. Special-token markers tokenize to single ids.
    fn build_prompt(&self, messages: &[ChatMessage]) -> String {
        let mut p = String::new();
        match self {
            ChatFormat::ChatML => {
                for m in messages {
                    let r = norm_role(&m.role, "user");
                    p.push_str(&format!("<|im_start|>{r}\n{}<|im_end|>\n", m.content));
                }
                p.push_str("<|im_start|>assistant\n");
            }
            ChatFormat::Gemma => {
                for m in messages {
                    // Gemma has no system role; fold it into a user turn.
                    let r = if m.role == "assistant" {
                        "model"
                    } else {
                        "user"
                    };
                    p.push_str(&format!("<start_of_turn>{r}\n{}<end_of_turn>\n", m.content));
                }
                p.push_str("<start_of_turn>model\n");
            }
            ChatFormat::Llama3 => {
                for m in messages {
                    let r = norm_role(&m.role, "user");
                    p.push_str(&format!(
                        "<|start_header_id|>{r}<|end_header_id|>\n\n{}<|eot_id|>",
                        m.content
                    ));
                }
                p.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
            ChatFormat::Zephyr => {
                for m in messages {
                    let r = norm_role(&m.role, "user");
                    p.push_str(&format!("<|{r}|>\n{}\n", m.content));
                }
                p.push_str("<|assistant|>\n");
            }
        }
        p
    }
}

fn norm_role<'a>(role: &'a str, default: &'a str) -> &'a str {
    match role {
        "system" | "user" | "assistant" => role,
        _ => default,
    }
}

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
use ullm_model::{Grammar, GrammarConstraint, LlamaModel, SampleParams, TokenTrie};
use ullm_safetensors::SafeTensorsModel;
use ullm_tokenizer::Tokenizer;
use ullm_tokenizer::chat::{ChatFormat, hf_chat_template};

/// A loaded model + tokenizer, generation-ready.
struct Engine {
    model: LlamaModel,
    tokenizer: Tokenizer,
    /// Vocabulary trie, built once at load, reused for fast grammar masking.
    trie: TokenTrie,
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
        let trie = TokenTrie::new(tokenizer.token_pieces());
        Ok(Self {
            model,
            tokenizer,
            trie,
            model_id,
            chat_format,
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
        let eos = self.tokenizer.eos_id();
        // Direct field borrows (self.trie / self.model) keep these disjoint.
        let mut constraint = grammar.map(|g| GrammarConstraint::new(g, &self.trie, eos));
        let generated = self.model.generate(
            &prompt_ids,
            max_tokens,
            eos,
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
        let mut constraint = grammar.map(|g| GrammarConstraint::new(g, &self.trie, eos));
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
    /// OpenAI function/tool definitions. When present (and `tool_choice` is not
    /// `"none"`), generation is constrained to a valid call of one of them.
    #[serde(default)]
    tools: Option<Vec<Tool>>,
    /// `"none"` | `"auto"` | `"required"` | `{"type":"function","function":{"name":..}}`.
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ResponseFormat {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    json_schema: Option<JsonSchemaSpec>,
}

#[derive(Deserialize)]
struct Tool {
    #[serde(default)]
    function: ToolFunction,
}

#[derive(Deserialize, Default)]
struct ToolFunction {
    #[serde(default)]
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<serde_json::Value>,
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

/// What a tool-calling request resolves to: a grammar that forces a valid call,
/// and a system message describing the tools for the model.
struct ToolPlan {
    grammar: Grammar,
    system: String,
}

/// If the request asks for tool calls, build the constraint + tool description.
/// `Ok(None)` means no tool calling (no tools, or `tool_choice: "none"`).
fn tool_plan(req: &ChatRequest) -> Result<Option<ToolPlan>> {
    let Some(tools) = req.tools.as_ref().filter(|t| !t.is_empty()) else {
        return Ok(None);
    };
    let choice = req.tool_choice.as_ref();
    if choice.and_then(|c| c.as_str()) == Some("none") {
        return Ok(None);
    }
    // A named `tool_choice` restricts to that one function.
    let forced = choice
        .and_then(|c| c.get("function"))
        .and_then(|f| f.get("name"))
        .and_then(|n| n.as_str());
    let selected: Vec<&Tool> = match forced {
        Some(name) => vec![
            tools
                .iter()
                .find(|t| t.function.name == name)
                .ok_or_else(|| {
                    Error::Format(format!("tool_choice names unknown function {name:?}"))
                })?,
        ],
        None => tools.iter().collect(),
    };

    // Each tool -> {"name": const, "arguments": <params schema>}; the call is an
    // anyOf over them. Reuses the JSON-Schema compiler (const/anyOf/object).
    let variant = |t: &Tool| {
        let params = t
            .function
            .parameters
            .clone()
            .unwrap_or_else(|| serde_json::json!({ "type": "object" }));
        serde_json::json!({
            "type": "object",
            "properties": { "name": { "const": t.function.name }, "arguments": params },
            "required": ["name", "arguments"]
        })
    };
    let variants: Vec<serde_json::Value> = selected.iter().map(|t| variant(t)).collect();
    let schema = if variants.len() == 1 {
        variants.into_iter().next().unwrap()
    } else {
        serde_json::json!({ "anyOf": variants })
    };
    let grammar = Grammar::from_json_schema(&schema)?;

    let mut system =
        String::from("You can call functions to answer the user.\nAvailable functions:\n");
    for t in &selected {
        let params = t
            .function
            .parameters
            .as_ref()
            .map(|p| p.to_string())
            .unwrap_or_else(|| "{}".into());
        let desc = t.function.description.as_deref().unwrap_or("");
        system.push_str(&format!(
            "- {} — {desc} (parameters: {params})\n",
            t.function.name
        ));
    }
    system.push_str(
        "Call exactly one function. Reply with ONLY a JSON object \
         {\"name\": <function>, \"arguments\": {<arguments>}} and nothing else.",
    );
    Ok(Some(ToolPlan { grammar, system }))
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
    message: OutMessage,
    finish_reason: &'static str,
}

/// An assistant response message: either text `content` or `tool_calls`.
#[derive(Serialize)]
struct OutMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallOut>>,
}

#[derive(Serialize)]
struct ToolCallOut {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: FnOut,
}

#[derive(Serialize)]
struct FnOut {
    name: String,
    /// OpenAI sends arguments as a JSON *string*.
    arguments: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

fn bad_request(msg: String) -> Response {
    (axum::http::StatusCode::BAD_REQUEST, msg).into_response()
}

async fn chat_completions(State(s): State<AppState>, Json(req): Json<ChatRequest>) -> Response {
    let max_tokens = req.max_tokens.unwrap_or(128);
    let params = SampleParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_k: 0,
        top_p: req.top_p.unwrap_or(1.0),
        seed: req.seed.unwrap_or(0),
        ..SampleParams::default()
    };
    let model_id = s.model_id.clone();

    // Tool calling takes precedence: constrain to a valid function call, render
    // the tools into a system message, and return `tool_calls` (streamed or not).
    match tool_plan(&req) {
        Err(e) => return bad_request(format!("invalid tools: {e}")),
        Ok(Some(plan)) => {
            let mut msgs = req.messages.clone();
            msgs.insert(
                0,
                ChatMessage {
                    role: "system".to_string(),
                    content: plan.system,
                },
            );
            let prompt = build_prompt(s.chat_format, &msgs);
            let engine = s.engine.clone();
            let grammar = plan.grammar;
            let p = params.clone();

            if req.stream == Some(true) {
                let (tx, rx) = mpsc::channel::<std::result::Result<Event, Infallible>>(64);
                let mid = model_id;
                tokio::task::spawn_blocking(move || {
                    let send = |data: String| {
                        let _ = tx.blocking_send(Ok(Event::default().data(data)));
                    };
                    send(chunk_json(&mid, Some("assistant"), None, None));
                    let text = {
                        let mut e = engine.lock().expect("engine mutex poisoned");
                        e.complete(&prompt, max_tokens, &p, Some(&grammar)).0
                    };
                    stream_tool_call(&mid, &text, &send);
                    send("[DONE]".to_string());
                });
                return Sse::new(ReceiverStream::new(rx)).into_response();
            }

            let (text, pt, ct) = tokio::task::spawn_blocking(move || {
                let mut e = engine.lock().expect("engine mutex poisoned");
                e.complete(&prompt, max_tokens, &p, Some(&grammar))
            })
            .await
            .unwrap_or_else(|_| (String::new(), 0, 0));
            return tool_call_response(model_id, &text, pt, ct);
        }
        Ok(None) => {}
    }

    let prompt = build_prompt(s.chat_format, &req.messages);

    // Compile the requested structured-output constraint, if any (400 on error).
    let grammar = match request_grammar(&req) {
        Ok(g) => g,
        Err(e) => return bad_request(format!("invalid response_format / grammar: {e}")),
    };

    if req.stream == Some(true) {
        let (tx, rx) = mpsc::channel::<std::result::Result<Event, Infallible>>(64);
        let engine = s.engine.clone();
        let mid = model_id;
        let p = params.clone();
        tokio::task::spawn_blocking(move || {
            let send = |data: String| {
                let _ = tx.blocking_send(Ok(Event::default().data(data)));
            };
            send(chunk_json(&mid, Some("assistant"), None, None));
            {
                let mut e = engine.lock().expect("engine mutex poisoned");
                e.complete_stream(&prompt, max_tokens, &p, grammar.as_ref(), |delta| {
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
            message: OutMessage {
                role: "assistant",
                content: Some(text.trim().to_string()),
                tool_calls: None,
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

/// Turn the (grammar-constrained) model output into an OpenAI `tool_calls`
/// response. Falls back to plain content if it somehow isn't a `{name,...}`.
fn tool_call_response(model_id: String, text: &str, pt: usize, ct: usize) -> Response {
    let (message, finish) = match parse_tool_call(text) {
        Some((name, arg_str)) => (
            OutMessage {
                role: "assistant",
                content: None,
                tool_calls: Some(vec![ToolCallOut {
                    id: format!("call_{name}"),
                    kind: "function",
                    function: FnOut {
                        name,
                        arguments: arg_str,
                    },
                }]),
            },
            "tool_calls",
        ),
        None => (
            OutMessage {
                role: "assistant",
                content: Some(text.trim().to_string()),
                tool_calls: None,
            },
            "stop",
        ),
    };
    Json(ChatResponse {
        id: "chatcmpl-ullm".to_string(),
        object: "chat.completion",
        created: unix_now(),
        model: model_id,
        choices: vec![ChatChoice {
            index: 0,
            message,
            finish_reason: finish,
        }],
        usage: Usage {
            prompt_tokens: pt,
            completion_tokens: ct,
            total_tokens: pt + ct,
        },
    })
    .into_response()
}

/// Parse the (grammar-constrained) `{"name":...,"arguments":...}` into the
/// function name and its arguments as a JSON string (OpenAI's wire format).
fn parse_tool_call(text: &str) -> Option<(String, String)> {
    let v: serde_json::Value = serde_json::from_str(text.trim()).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = v
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let arg_str = serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".into());
    Some((name, arg_str))
}

/// Emit a parsed tool call as OpenAI streaming `tool_calls` deltas: announce the
/// function (id + name), stream the arguments string in pieces, then finish.
fn stream_tool_call(model: &str, text: &str, send: &impl Fn(String)) {
    match parse_tool_call(text) {
        Some((name, args)) => {
            send(emit_chunk(
                model,
                Delta {
                    tool_calls: Some(vec![DeltaToolCall {
                        index: 0,
                        id: Some(format!("call_{name}")),
                        kind: Some("function"),
                        function: DeltaFunction {
                            name: Some(name),
                            arguments: Some(String::new()),
                        },
                    }]),
                    ..Delta::default()
                },
                None,
            ));
            for piece in chunk_chars(&args, 24) {
                send(emit_chunk(
                    model,
                    Delta {
                        tool_calls: Some(vec![DeltaToolCall {
                            index: 0,
                            id: None,
                            kind: None,
                            function: DeltaFunction {
                                name: None,
                                arguments: Some(piece),
                            },
                        }]),
                        ..Delta::default()
                    },
                    None,
                ));
            }
            send(chunk_json(model, None, None, Some("tool_calls")));
        }
        None => {
            send(chunk_json(model, None, Some(text.trim()), None));
            send(chunk_json(model, None, None, Some("stop")));
        }
    }
}

/// Split `s` into pieces of at most `n` characters (never mid-codepoint).
fn chunk_chars(s: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    chars.chunks(n).map(|c| c.iter().collect()).collect()
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

#[derive(Serialize, Default)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Serialize)]
struct DeltaToolCall {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<&'static str>,
    function: DeltaFunction,
}

#[derive(Serialize, Default)]
struct DeltaFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

/// Serialize a `chat.completion.chunk` with a ready-made delta.
fn emit_chunk(model: &str, delta: Delta, finish: Option<&str>) -> String {
    let chunk = ChatChunk {
        id: "chatcmpl-ullm",
        object: "chat.completion.chunk",
        created: unix_now(),
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason: finish.map(str::to_string),
        }],
    };
    serde_json::to_string(&chunk).unwrap_or_default()
}

/// Serialize one role/content/finish `chat.completion.chunk`.
fn chunk_json(
    model: &str,
    role: Option<&str>,
    content: Option<&str>,
    finish: Option<&str>,
) -> String {
    emit_chunk(
        model,
        Delta {
            role: role.map(str::to_string),
            content: content.map(str::to_string),
            tool_calls: None,
        },
        finish,
    )
}

/// Wrap chat messages in the model's prompt format (`ChatFormat` lives in
/// `ullm-tokenizer` so the CLI shares it).
fn build_prompt(fmt: ChatFormat, messages: &[ChatMessage]) -> String {
    let pairs: Vec<(&str, &str)> = messages
        .iter()
        .map(|m| (m.role.as_str(), m.content.as_str()))
        .collect();
    fmt.build_prompt(&pairs)
}

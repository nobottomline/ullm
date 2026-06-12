//! Chat-prompt formatting — wrap messages in the model's native turn markers so
//! an instruct model answers instead of free-completing the raw text.
//!
//! A full Jinja `chat_template` engine is out of scope; we detect the common
//! family from substring markers in the template and emit the matching prompt.
//! Shared by the CLI (`ullm run`) and the server.

use std::path::Path;

/// The chat prompt format a model expects, detected from its chat template.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatFormat {
    ChatML,
    Gemma,
    Llama3,
    Zephyr,
}

impl ChatFormat {
    /// Pick a format from a model's `chat_template` string (substring markers).
    pub fn detect(template: Option<&str>) -> Self {
        match template {
            Some(t) if t.contains("<|im_start|>") => ChatFormat::ChatML,
            Some(t) if t.contains("<start_of_turn>") => ChatFormat::Gemma,
            Some(t) if t.contains("<|start_header_id|>") => ChatFormat::Llama3,
            _ => ChatFormat::Zephyr,
        }
    }

    /// Render `(role, content)` messages into the model's native chat prompt,
    /// ending with the open assistant turn. Markers tokenize to single ids.
    pub fn build_prompt(&self, messages: &[(&str, &str)]) -> String {
        let mut p = String::new();
        match self {
            ChatFormat::ChatML => {
                for (role, content) in messages {
                    p.push_str(&format!(
                        "<|im_start|>{}\n{content}<|im_end|>\n",
                        norm_role(role, "user")
                    ));
                }
                p.push_str("<|im_start|>assistant\n");
            }
            ChatFormat::Gemma => {
                for (role, content) in messages {
                    // Gemma has no system role; fold it into a user turn.
                    let r = if *role == "assistant" {
                        "model"
                    } else {
                        "user"
                    };
                    p.push_str(&format!("<start_of_turn>{r}\n{content}<end_of_turn>\n"));
                }
                p.push_str("<start_of_turn>model\n");
            }
            ChatFormat::Llama3 => {
                for (role, content) in messages {
                    p.push_str(&format!(
                        "<|start_header_id|>{}<|end_header_id|>\n\n{content}<|eot_id|>",
                        norm_role(role, "user")
                    ));
                }
                p.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
            ChatFormat::Zephyr => {
                for (role, content) in messages {
                    p.push_str(&format!("<|{}|>\n{content}\n", norm_role(role, "user")));
                }
                p.push_str("<|assistant|>\n");
            }
        }
        p
    }

    /// Wrap a single user prompt as a one-turn chat.
    pub fn wrap(&self, user: &str) -> String {
        self.build_prompt(&[("user", user)])
    }
}

fn norm_role<'a>(role: &'a str, default: &'a str) -> &'a str {
    match role {
        "system" | "user" | "assistant" => role,
        _ => default,
    }
}

/// Read a Hugging Face model dir's chat template (`chat_template.jinja`, or the
/// `chat_template` field of `tokenizer_config.json`).
pub fn hf_chat_template(dir: &Path) -> Option<String> {
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

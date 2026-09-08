use super::{ApiError, Info};
use crate::{
    decode::{Options, SpecialTokens},
    tokenizer::TextCodec,
};
use serde_json::{Value, json};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Chat,
    Completion,
}

pub struct Prepared {
    pub kind: Kind,
    pub ids: Vec<u32>,
    pub options: Options,
    pub stream: bool,
    pub include_usage: bool,
    pub progress: bool,
    pub cache_prompt: bool,
    pub stops: Vec<String>,
    pub tools: Vec<Value>,
    pub required_tool: bool,
    pub parallel_tools: bool,
    pub assistant_prefix: String,
    pub json_object: bool,
}

pub fn boolean(v: &Value, key: &str, default: bool) -> Result<bool, ApiError> {
    match v.get(key).filter(|v| !v.is_null()) {
        None => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| ApiError::invalid(format!("{key} must be boolean"))),
    }
}

fn content(v: &Value) -> Result<String, ApiError> {
    if v.is_null() {
        return Ok(String::new());
    }
    if let Some(s) = v.as_str() {
        return Ok(s.to_owned());
    }
    let parts = v
        .as_array()
        .ok_or_else(|| ApiError::invalid("content must be text or text content parts"))?;
    let mut text = String::new();
    for part in parts {
        if !matches!(
            part["type"].as_str(),
            Some("text" | "input_text" | "output_text")
        ) {
            return Err(ApiError::invalid("this model supports text content only"));
        }
        text.push_str(
            part["text"]
                .as_str()
                .ok_or_else(|| ApiError::invalid("text part requires text"))?,
        );
    }
    Ok(text)
}

pub fn messages(v: &Value) -> Result<Vec<Value>, ApiError> {
    let messages = v
        .as_array()
        .filter(|m| !m.is_empty())
        .ok_or_else(|| ApiError::invalid("messages must be a nonempty array"))?;
    let mut result = Vec::new();
    let mut call_ids = std::collections::HashSet::new();
    for message in messages {
        let mut m = message
            .as_object()
            .cloned()
            .ok_or_else(|| ApiError::invalid("invalid message"))?;
        let role = match message["role"].as_str() {
            Some("developer") => "system",
            Some(role @ ("system" | "user" | "assistant" | "tool")) => role,
            _ => return Err(ApiError::invalid("unsupported message role")),
        };
        m.insert("role".into(), json!(role));
        m.insert("content".into(), json!(content(&message["content"])?));
        if let Some(calls) = message.get("tool_calls").filter(|x| !x.is_null()) {
            if role != "assistant" {
                return Err(ApiError::invalid("tool_calls require an assistant message"));
            }
            let calls = calls
                .as_array()
                .ok_or_else(|| ApiError::invalid("tool_calls must be an array"))?;
            for call in calls {
                let id = call["id"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| ApiError::invalid("tool call requires id"))?;
                if !call_ids.insert(id.to_owned()) {
                    return Err(ApiError::invalid("duplicate tool call id"));
                }
                if call["type"] != "function"
                    || call["function"]["name"].as_str().is_none()
                    || call["function"]["arguments"].as_str().is_none()
                {
                    return Err(ApiError::invalid("invalid function tool call"));
                }
            }
        }
        if role == "tool" {
            let id = message["tool_call_id"]
                .as_str()
                .ok_or_else(|| ApiError::invalid("tool message requires tool_call_id"))?;
            if !call_ids.remove(id) {
                return Err(ApiError::invalid(
                    "tool result has no matching preceding tool call",
                ));
            }
        }
        result.push(Value::Object(m));
    }
    Ok(result)
}

pub fn tools(v: &Value) -> Result<Vec<Value>, ApiError> {
    if v.is_null() {
        return Ok(vec![]);
    }
    let items = v
        .as_array()
        .filter(|a| a.len() <= 128)
        .ok_or_else(|| ApiError::invalid("tools must be an array of at most 128 functions"))?;
    let mut names = std::collections::HashSet::new();
    for t in items {
        let f = &t["function"];
        let name = f["name"].as_str().unwrap_or("");
        if t["type"] != "function"
            || name.is_empty()
            || name.len() > 128
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "_-".contains(c))
            || !names.insert(name)
        {
            return Err(ApiError::invalid(
                "tools require unique function names containing letters, digits, underscores or hyphens",
            ));
        }
        if !f["parameters"].is_null() && !f["parameters"].is_object() {
            return Err(ApiError::invalid(
                "function parameters must be a JSON Schema object",
            ));
        }
        if boolean(f, "strict", false)? {
            return Err(ApiError::invalid(
                "strict tool schemas are not supported; use strict=false",
            ));
        }
    }
    Ok(items.clone())
}

pub fn prepare(
    body: Value,
    kind: Kind,
    codec: &TextCodec,
    info: &Info,
) -> Result<Prepared, ApiError> {
    if !body.is_object() {
        return Err(ApiError::invalid("request must be an object"));
    }
    if let Some(model) = body.get("model").filter(|v| !v.is_null()) {
        let model = model
            .as_str()
            .ok_or_else(|| ApiError::invalid("model must be a string"))?;
        if !info.accepts_model(model) {
            return Err(ApiError::invalid("unknown model; see /v1/models"));
        }
    }
    if body
        .get("n")
        .is_some_and(|v| !v.is_null() && v.as_u64() != Some(1))
    {
        return Err(ApiError::invalid("only n=1 is supported"));
    }
    // Accept transport metadata and neutral llama.cpp sampler settings. Reject
    // active features we cannot implement instead of silently ignoring them.
    for (key, neutral) in [
        ("frequency_penalty", 0.0),
        ("presence_penalty", 0.0),
        ("repeat_penalty", 1.0),
        ("min_p", 0.0),
        ("typ_p", 1.0),
        ("dynatemp_range", 0.0),
        ("xtc_probability", 0.0),
        ("dry_multiplier", 0.0),
        ("mirostat", 0.0),
        ("top_n_sigma", 0.0),
    ] {
        if let Some(v) = body.get(key).filter(|v| !v.is_null())
            && v.as_f64() != Some(neutral)
        {
            return Err(ApiError::invalid(format!(
                "{key} is not supported at a non-default value"
            )));
        }
    }
    for key in ["logprobs", "echo", "ignore_eos", "continue_final_message"] {
        if boolean(&body, key, false)? {
            return Err(ApiError::invalid(format!("{key} is not supported")));
        }
    }
    if body.get("add_generation_prompt").is_some()
        && !boolean(&body, "add_generation_prompt", true)?
    {
        return Err(ApiError::invalid(
            "add_generation_prompt=false is not supported",
        ));
    }
    for key in ["grammar", "logit_bias"] {
        if let Some(v) = body.get(key).filter(|v| !v.is_null())
            && !matches!(v,Value::String(s) if s.is_empty())
            && !matches!(v,Value::Object(o) if o.is_empty())
            && !matches!(v,Value::Array(a) if a.is_empty())
        {
            return Err(ApiError::invalid(format!("{key} is not supported")));
        }
    }
    let allowed = [
        "model",
        "prompt",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "n_predict",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "stream",
        "stream_options",
        "n",
        "stop",
        "minnow",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "return_progress",
        "cache_prompt",
        "timings_per_token",
        "sse_ping_interval",
        "reasoning_format",
        "reasoning_control",
        "chat_template_kwargs",
        "reasoning_effort",
        "user",
        "metadata",
        "store",
        "service_tier",
        "safety_identifier",
        "prompt_cache_key",
        "frequency_penalty",
        "presence_penalty",
        "repeat_penalty",
        "repeat_last_n",
        "min_p",
        "typ_p",
        "dynatemp_range",
        "dynatemp_exponent",
        "xtc_probability",
        "xtc_threshold",
        "dry_multiplier",
        "dry_base",
        "dry_allowed_length",
        "dry_penalty_last_n",
        "dry_sequence_breakers",
        "mirostat",
        "top_n_sigma",
        "samplers",
        "backend_sampling",
        "logprobs",
        "top_logprobs",
        "echo",
        "ignore_eos",
        "continue_final_message",
        "add_generation_prompt",
        "grammar",
        "logit_bias",
    ];
    let diffusion = [
        "threshold",
        "editing_threshold",
        "max_post_steps",
        "steps",
        "max_steps_per_block",
    ];
    for key in body.as_object().unwrap().keys() {
        if !allowed.contains(&key.as_str()) && !diffusion.contains(&key.as_str()) {
            return Err(ApiError::invalid(format!(
                "unsupported request field: {key}"
            )));
        }
    }
    let mut defaults = serde_json::to_value(&info.defaults).unwrap();
    if let Some(overrides) = body.get("minnow").filter(|v| !v.is_null()) {
        let overrides = overrides
            .as_object()
            .ok_or_else(|| ApiError::invalid("minnow must be an options object"))?;
        defaults.as_object_mut().unwrap().extend(overrides.clone());
    }
    for key in diffusion {
        if let Some(value) = body.get(key).filter(|v| !v.is_null()) {
            defaults[key] = value.clone();
        }
    }
    let mut options: Options =
        serde_json::from_value(defaults).map_err(|e| ApiError::invalid(e.to_string()))?;
    for key in ["temperature", "top_p"] {
        if let Some(v) = body.get(key).filter(|v| !v.is_null()) {
            let v = v
                .as_f64()
                .ok_or_else(|| ApiError::invalid(format!("{key} must be a number")))?
                as f32;
            if key == "temperature" {
                options.temperature = v;
            } else {
                options.top_p = v;
            }
        }
    }
    if let Some(v) = body.get("top_k").filter(|v| !v.is_null()) {
        options.top_k = v
            .as_u64()
            .ok_or_else(|| ApiError::invalid("top_k must be nonnegative"))?
            as usize;
    }
    if let Some(v) = body.get("seed").filter(|v| !v.is_null()) {
        options.seed = if v.as_i64() == Some(-1) {
            rand::random()
        } else {
            v.as_u64()
                .ok_or_else(|| ApiError::invalid("seed must be nonnegative or -1"))?
        };
    }
    let mut tools = tools(&body["tools"])?;
    let mut assistant_prefix = String::new();
    let choice = &body["tool_choice"];
    let mut required_tool = false;
    if choice == "none" {
        tools.clear();
    } else if choice == "required" {
        required_tool = true;
        assistant_prefix = "<|tool_calls_section_begin|><|tool_call_begin|>functions.".into();
    } else if choice.is_object() && choice["type"] == "function" {
        let name = choice["function"]["name"]
            .as_str()
            .ok_or_else(|| ApiError::invalid("tool_choice requires function.name"))?;
        if !tools.iter().any(|t| t["function"]["name"] == name) {
            return Err(ApiError::invalid("tool_choice names an unknown function"));
        }
        tools.retain(|t| t["function"]["name"] == name);
        required_tool = true;
        assistant_prefix = format!(
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:0<|tool_call_argument_begin|>"
        );
    } else if !choice.is_null() && choice != "auto" {
        return Err(ApiError::invalid("unsupported tool_choice"));
    }
    if required_tool && tools.is_empty() {
        return Err(ApiError::invalid("tool_choice requires tools"));
    }
    let format = &body["response_format"];
    let json_object = !format.is_null() && format["type"] == "json_object";
    if !format.is_null() && !json_object && format["type"] != "text" {
        return Err(ApiError::invalid(
            "response_format supports text and json_object; JSON Schema constrained decoding is not implemented",
        ));
    }
    let ids = if kind == Kind::Completion {
        if !body["messages"].is_null() || !tools.is_empty() {
            return Err(ApiError::invalid(
                "use chat completions for messages and tools",
            ));
        }
        if let Some(text) = body["prompt"].as_str() {
            codec
                .encode(text)
                .map_err(|e| ApiError::invalid(e.to_string()))?
        } else {
            serde_json::from_value::<Vec<u32>>(body["prompt"].clone())
                .map_err(|_| ApiError::invalid("prompt must be a string or array of token IDs"))?
        }
    } else {
        if !body["prompt"].is_null() {
            return Err(ApiError::invalid("use messages for chat"));
        }
        let mut messages = messages(&body["messages"])?;
        if json_object {
            messages.insert(0,json!({"role":"system","content":"Respond with one valid JSON object, without markdown fences or other text."}));
        }
        let text = codec
            .chat_prompt_with_tools(&messages, &tools, true)
            .map_err(|e| ApiError::invalid(e.to_string()))?
            + &assistant_prefix;
        codec
            .encode(&text)
            .map_err(|e| ApiError::invalid(e.to_string()))?
    };
    if ids.iter().any(|&id| id as usize >= info.vocab_size) {
        return Err(ApiError::invalid(
            "prompt contains token IDs outside the vocabulary",
        ));
    }
    if ids.is_empty() {
        return Err(ApiError::invalid("prompt must not be empty"));
    }
    if ids.len() >= info.max_context {
        return Err(ApiError::context(ids.len(), info.max_context));
    }
    let limit = ["max_completion_tokens", "max_tokens", "n_predict"]
        .iter()
        .find_map(|key| body.get(*key).filter(|v| !v.is_null()));
    if let Some(limit) = limit {
        let n = limit
            .as_i64()
            .filter(|n| *n >= -1)
            .ok_or_else(|| ApiError::invalid("token limit must be nonnegative or -1"))?;
        options.max_tokens = if n == -1 {
            info.max_context - ids.len()
        } else {
            n as usize
        };
    } else {
        options.max_tokens = options.max_tokens.min(info.max_context - ids.len());
    }
    if options.max_tokens > info.max_context - ids.len() {
        return Err(ApiError::context(
            ids.len().saturating_add(options.max_tokens),
            info.max_context,
        ));
    }
    options
        .validate(info.vocab_size)
        .map_err(|e| ApiError::invalid(e.to_string()))?;
    let special = SpecialTokens::default();
    if ids
        .iter()
        .any(|t| [special.mask, special.delete, special.split].contains(t))
    {
        return Err(ApiError::invalid(
            "prompt contains reserved diffusion tokens",
        ));
    }
    let stops = match &body["stop"] {
        Value::Null => vec![],
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a
            .iter()
            .map(|s| {
                s.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| ApiError::invalid("stop entries must be strings"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err(ApiError::invalid("stop must be a string or array")),
    };
    if stops.len() > 16 || stops.iter().any(String::is_empty) {
        return Err(ApiError::invalid(
            "stop requires at most 16 nonempty strings",
        ));
    }
    Ok(Prepared {
        cache_prompt: boolean(&body, "cache_prompt", true)?,
        kind,
        ids,
        options,
        stream: boolean(&body, "stream", false)?,
        include_usage: boolean(&body["stream_options"], "include_usage", false)?,
        progress: boolean(&body, "return_progress", false)?
            || boolean(&body, "timings_per_token", false)?,
        stops,
        tools,
        required_tool,
        parallel_tools: boolean(&body, "parallel_tool_calls", true)?,
        assistant_prefix,
        json_object,
    })
}

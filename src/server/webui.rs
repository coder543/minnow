//! Optional external UI assets are mounted in server.rs; llama.cpp's metadata
//! and utility routes live here, independently of the inference protocol.
use super::{ApiError, App};
use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    routing::{get, post},
};
use serde_json::{Value, json};

async fn props(State(app): State<App>) -> Json<Value> {
    let o = &app.info.defaults;
    // Clients send these values back explicitly. Advertise remaining context as
    // -1 so the prompt is subtracted, instead of sending the entire context as output.
    let max_tokens = match o.max_tokens {
        Some(limit) if limit < app.info.max_context => json!(limit),
        _ => json!(-1),
    };
    Json(json!({
        "role":"model","model_alias":app.info.model_id,"model_path":app.info.model_path,
        "total_slots":app.info.parallel,"modalities":{"vision":false,"audio":false,"video":false},
        "chat_template":app.codec.chat_template,"chat_template_caps":{"supports_tools":true,"supports_parallel_tool_calls":true},
        "bos_token":"","eos_token":"<|endoftext|>","build_info":concat!("minnow ",env!("CARGO_PKG_VERSION")),
        "is_sleeping":false,"endpoint_slots":true,"endpoint_props":false,"endpoint_metrics":false,
        "webui":app.info.ui_dir.is_some(),"cors_proxy_enabled":false,
        "ui_settings":{"showMessageStats":true,"showToolCalls":true,"enableContinueGeneration":false},
        "default_generation_settings":{"id":0,"id_task":-1,"n_ctx":app.info.max_context,"speculative":false,"is_processing":false,"prompt":"",
            "params":{"n_predict":max_tokens,"max_tokens":max_tokens,"seed":o.seed.map_or(json!(-1), |seed| json!(seed)),"temperature":o.temperature,"top_k":o.top_k,"top_p":o.top_p,
                "dynatemp_range":0,"dynatemp_exponent":1,"min_p":0,"top_n_sigma":0,"xtc_probability":0,"xtc_threshold":0.1,"typ_p":1,
                "repeat_last_n":64,"repeat_penalty":1,"presence_penalty":0,"frequency_penalty":0,"dry_multiplier":0,"dry_base":1.75,"dry_allowed_length":2,"dry_penalty_last_n":-1,"dry_sequence_breakers":[],
                "mirostat":0,"mirostat_tau":5,"mirostat_eta":0.1,"stop":[],"n_keep":0,"n_discard":0,"ignore_eos":false,"stream":true,"logit_bias":[],"n_probs":0,"min_keep":0,
                "grammar":"","grammar_lazy":false,"grammar_triggers":[],"preserved_tokens":[],"chat_format":"llada22","reasoning_format":"none","reasoning_in_content":false,"generation_prompt":"<role>assistant</role>",
                "samplers":["top_k","top_p","temperature"],"backend_sampling":false,"speculative.n_max":0,"speculative.n_min":0,"speculative.p_min":0,"timings_per_token":true,"post_sampling_probs":false,"lora":[],
                "threshold":o.threshold,"editing_threshold":o.editing_threshold,"max_post_steps":o.max_post_steps},
            "next_token":{"has_next_token":false,"has_new_line":false,"n_remain":0,"n_decoded":0,"stopping_word":""}},
        "minnow":{"block_size":32,"attention_backend":app.info.attention_backend,"generation_defaults":o,"stream_resumption":false,"strict_tool_schemas":false,
            "cache":app.info.cache,"cache_budget_bytes":app.info.cache_budget_bytes,"inference_workers":1,"parallel":app.info.parallel,"batch_wait_us":app.info.batch_wait_us,
            "statistics":{"evaluated_tokens":"32 times refinement forwards, including repeated positions","processed_tokens":"all transformer input positions, including prefill and commit refreshes","timings":"prefill counts complete prompt blocks; predicted counts non-special completion tokens"}}
    }))
}
async fn slots(State(app): State<App>) -> Json<Value> {
    let active = app.active.lock().unwrap().clone();
    let cached = app.cache_slots.lock().unwrap();
    Json(Value::Array(
        (0..app.info.parallel)
            .map(|id| {
                let mut value = if let Some(value) = active.get(id).and_then(Option::as_ref) {
                    value.clone()
                } else {
                    json!({"id":id,"id_task":-1,"is_processing":false,"n_ctx":app.info.max_context})
                };
                let cache_id = if value["is_processing"] == true {
                    value["cache_slot"].as_u64().map(|id| id as usize)
                } else {
                    Some(id)
                };
                if let Some(slot) = cache_id.and_then(|id| cached.get(id)) {
                    value["cache"] = json!(slot);
                }
                value
            })
            .collect(),
    ))
}
async fn tokenize(
    State(app): State<App>,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let body = body.map_err(|e| ApiError::invalid(e.body_text()))?.0;
    let text = body["content"]
        .as_str()
        .ok_or_else(|| ApiError::invalid("content must be a string"))?;
    let ids = app
        .codec
        .encode(text)
        .map_err(|e| ApiError::invalid(e.to_string()))?;
    let tokens = if super::request::boolean(&body, "with_pieces", false)? {
        ids.into_iter()
            .map(|id| {
                app.codec
                    .decode_raw(&[id])
                    .map(|piece| json!({"id":id,"piece":piece}))
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .map_err(|e| ApiError::invalid(e.to_string()))?
    } else {
        ids.into_iter().map(|i| json!(i)).collect()
    };
    Ok(Json(json!({"tokens":tokens})))
}
async fn detokenize(
    State(app): State<App>,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let body = body.map_err(|e| ApiError::invalid(e.body_text()))?.0;
    let tokens: Vec<u32> = serde_json::from_value(body["tokens"].clone())
        .map_err(|e| ApiError::invalid(e.to_string()))?;
    if tokens.iter().any(|&t| t as usize >= app.info.vocab_size) {
        return Err(ApiError::invalid("token outside vocabulary"));
    }
    let content = app
        .codec
        .decode_raw(&tokens)
        .map_err(|e| ApiError::invalid(e.to_string()))?;
    Ok(Json(json!({"content":content})))
}
async fn apply_template(
    State(app): State<App>,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let body = body.map_err(|e| ApiError::invalid(e.body_text()))?.0;
    let messages = super::request::messages(&body["messages"])?;
    let tools = super::request::tools(&body["tools"])?;
    let prompt = app
        .codec
        .chat_prompt_with_tools(
            &messages,
            &tools,
            super::request::boolean(&body, "add_generation_prompt", true)?,
        )
        .map_err(|e| ApiError::invalid(e.to_string()))?;
    Ok(Json(json!({"prompt":prompt})))
}
async fn tools_disabled() -> ApiError {
    ApiError::new(
        axum::http::StatusCode::FORBIDDEN,
        "this feature is disabled: server-side tool execution",
    )
}
pub(super) fn routes() -> Router<App> {
    Router::new()
        .route("/props", get(props))
        .route("/slots", get(slots))
        .route("/tools", get(tools_disabled).post(tools_disabled))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/apply-template", post(apply_template))
        // The current UI probes this during startup even without saved streams.
        .route("/v1/streams/lookup", post(|| async { Json(json!([])) }))
}

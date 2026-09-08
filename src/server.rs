mod output;
mod request;
#[cfg(test)]
mod tests;
mod webui;

use crate::{
    decode::{BatchStats, Options, Progress, SpecialTokens, Stats, generate_observed, rate},
    model::Model,
    tokenizer::TextCodec,
};
use anyhow::{Result, ensure};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, State, rejection::JsonRejection},
    http::{StatusCode, header},
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{get, post},
};
use request::{Kind, Prepared};
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    net::SocketAddr,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub struct ServeConfig {
    pub listen: SocketAddr,
    pub max_context: usize,
    pub queue_capacity: usize,
    pub model_id: String,
    pub model_path: PathBuf,
    pub ui_dir: Option<PathBuf>,
    pub defaults: Options,
}
struct Info {
    model_id: String,
    model_path: PathBuf,
    max_context: usize,
    vocab_size: usize,
    trained_context: usize,
    defaults: Options,
    ui_dir: Option<PathBuf>,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
    details: Option<Value>,
}
impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            details: None,
        }
    }
    fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }
    fn unavailable(message: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, message)
    }
    fn context(n: usize, limit: usize) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: format!("request requires {n} tokens; context limit is {limit}"),
            details: Some(json!({"n_prompt_tokens":n,"n_ctx":limit})),
        }
    }
    fn value(&self) -> Value {
        let mut error = json!({"message":self.message,"type":if self.status.is_client_error(){"invalid_request_error"}else{"server_error"},"param":null,"code":self.status.as_u16()});
        if let Some(details) = &self.details {
            error
                .as_object_mut()
                .unwrap()
                .extend(details.as_object().unwrap().clone());
            error["type"] = json!("exceed_context_size_error");
        }
        json!({"error":error})
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.value())).into_response()
    }
}
type Reply = std::result::Result<Value, ApiError>;
type Wire = std::result::Result<Event, Infallible>;

enum Transport {
    Json(oneshot::Sender<Reply>),
    Stream(mpsc::Sender<Wire>),
}
impl Transport {
    fn closed(&self) -> bool {
        match self {
            Self::Json(s) => s.is_closed(),
            Self::Stream(s) => s.is_closed(),
        }
    }
    fn send(&self, value: Value, important: bool) -> Result<()> {
        if let Self::Stream(s) = self {
            // Progress is disposable. Reserve space for text and terminal events;
            // a client that stops reading must never block the inference worker.
            if !important && s.capacity() < 8 {
                return Ok(());
            }
            s.try_send(Ok(Event::default().data(value.to_string())))
                .map_err(|e| anyhow::anyhow!("stream receiver unavailable: {e}"))?;
        }
        Ok(())
    }
}
struct Job {
    request: Prepared,
    transport: Transport,
}
#[derive(Clone)]
struct App {
    tx: mpsc::Sender<Option<Job>>,
    healthy: Arc<AtomicBool>,
    stopping: Arc<AtomicBool>,
    codec: Arc<TextCodec>,
    info: Arc<Info>,
    active: Arc<Mutex<Option<Value>>>,
}

fn timings(stats: &Stats, text_tokens: usize) -> Value {
    json!({"prompt_n":stats.prefill_tokens,"prompt_ms":stats.prefill_seconds*1000.0,
        "prompt_per_token_ms":if stats.prefill_tokens>0 {stats.prefill_seconds*1000.0/stats.prefill_tokens as f64}else{0.0},
        "prompt_per_second":rate(stats.prefill_tokens,stats.prefill_seconds),
        "predicted_n":text_tokens,"predicted_ms":stats.decode_seconds*1000.0,
        "predicted_per_token_ms":if text_tokens>0 {stats.decode_seconds*1000.0/text_tokens as f64}else{0.0},
        "predicted_per_second":rate(text_tokens,stats.decode_seconds),"cache_n":0})
}
fn metadata(stats: &Stats, phase: &str, batch: Option<&BatchStats>, text_tokens: usize) -> Value {
    let mut value = serde_json::to_value(stats).unwrap();
    value["phase"] = json!(phase);
    value["text_tokens"] = json!(text_tokens);
    value["text_tokens_per_second"] = json!(rate(text_tokens, stats.decode_seconds));
    if let Some(batch) = batch {
        value["batch"] = json!(batch);
    }
    value
}

struct Emitter<'a> {
    job: &'a Job,
    app: &'a App,
    id: String,
    created: u64,
    output: output::Assistant,
    raw: String,
    stopped: bool,
    text_tokens: usize,
}
impl Emitter<'_> {
    fn chunk(&self, delta: Value, finish: Value) -> Value {
        let choice = if self.job.request.kind == Kind::Chat {
            json!({"index":0,"delta":delta,"finish_reason":finish})
        } else {
            json!({"index":0,"text":delta["content"].as_str().unwrap_or(""),"logprobs":null,"finish_reason":finish})
        };
        json!({"id":self.id,"object":if self.job.request.kind==Kind::Chat {"chat.completion.chunk"}else{"text_completion"},"created":self.created,"model":self.app.info.model_id,"choices":[choice]})
    }
    fn stats(
        &self,
        stats: &Stats,
        batch: Option<&BatchStats>,
        phase: &str,
        delta: Value,
        important: bool,
    ) -> Result<()> {
        let mut chunk = self.chunk(delta, Value::Null);
        chunk["timings"] = timings(stats, self.text_tokens);
        chunk["minnow"] = metadata(stats, phase, batch, self.text_tokens);
        if self.job.request.include_usage {
            chunk["usage"] = Value::Null;
        }
        *self.app.active.lock().unwrap() = Some(
            json!({"id":0,"id_task":self.id,"is_processing":true,"n_ctx":self.app.info.max_context,"n_prompt_tokens":stats.prompt_tokens,"n_decoded":stats.completion_tokens,"minnow":chunk["minnow"]}),
        );
        self.job.transport.send(chunk, important)
    }
    fn update(&mut self, ids: &[u32], final_output: bool, complete_tools: bool) -> Result<Value> {
        self.raw = self.job.request.assistant_prefix.clone() + &self.app.codec.decode_raw(ids)?;
        let mut current = if self.job.request.kind == Kind::Chat {
            let names = self
                .job
                .request
                .tools
                .iter()
                .filter_map(|t| t["function"]["name"].as_str().map(str::to_owned))
                .collect::<Vec<_>>();
            output::parse(&self.raw, &self.id, &names, final_output, complete_tools)?
        } else {
            output::Assistant {
                content: self.app.codec.decode(ids)?,
                ..output::Assistant::default()
            }
        };
        if self.job.request.kind == Kind::Completion && !final_output {
            current.content = current.content.trim_end_matches('\u{fffd}').to_owned();
        }
        self.stopped = current.stop(&self.job.request.stops, final_output);
        ensure!(
            self.job.request.parallel_tools || current.calls.len() <= 1,
            "model produced parallel tool calls with parallel_tool_calls=false"
        );
        // JSON mode is buffered until it can be validated. Ordinary text and tool
        // calls stream at each committed block boundary.
        if self.job.request.json_object && !final_output {
            return Ok(json!({}));
        }
        if self.job.request.json_object {
            let value: Value = serde_json::from_str(&current.content)?;
            ensure!(value.is_object(), "model did not produce a JSON object");
        }
        let delta = output::delta(&self.output, &current)?;
        self.output = current;
        self.text_tokens = self.app.codec.count_text_tokens(ids);
        Ok(delta)
    }
}

fn execute(model: &Model, app: &App, job: &Job) -> Reply {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let id = format!(
        "minnow-{:x}-{}",
        created.as_nanos(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    );
    let mut emitter = Emitter {
        job,
        app,
        id,
        created: created.as_secs(),
        output: output::Assistant::default(),
        raw: String::new(),
        stopped: false,
        text_tokens: 0,
    };
    let mut run = || -> Result<Value> {
        let initial = emitter.chunk(
            if job.request.kind == Kind::Chat {
                json!({"role":"assistant","content":""})
            } else {
                json!({"content":""})
            },
            Value::Null,
        );
        job.transport.send(initial, true)?;
        let result = generate_observed(
            model,
            &job.request.ids,
            &job.request.options,
            SpecialTokens::default(),
            || job.transport.closed() || app.stopping.load(Ordering::Acquire),
            |event| {
                match event {
                    Progress::Prefill {
                        processed,
                        total,
                        seconds,
                    } => {
                        if job.request.progress && total > 0 {
                            let mut chunk = emitter.chunk(json!({}), Value::Null);
                            chunk["prompt_progress"] = json!({"cache":0,"processed":processed,"total":total,"time_ms":seconds*1000.0});
                            chunk["minnow"] = json!({"phase":"prefill","prompt_tokens":job.request.ids.len(),"prefill_tokens":processed,"prefill_total":total});
                            job.transport.send(chunk, processed == total)?;
                            if processed == total {
                                // The current UI clears its preparing state on a timing
                                // event without prompt_progress. Do not invent a token.
                                let stats = Stats {
                                    prompt_tokens: job.request.ids.len(),
                                    prefill_tokens: total,
                                    prefill_seconds: seconds,
                                    ..Stats::default()
                                };
                                emitter.stats(&stats, None, "prefill_complete", json!({}), true)?;
                            }
                        }
                    }
                    Progress::Refinement { stats, batch } => {
                        if job.request.progress {
                            emitter.stats(stats, Some(batch), "refinement", json!({}), false)?;
                        }
                    }
                    Progress::Block {
                        token_ids,
                        stats,
                        batch,
                    } => {
                        let delta = emitter.update(token_ids, false, false)?;
                        emitter.stats(stats, Some(batch), "block", delta, true)?;
                        if emitter.stopped {
                            return Ok(false);
                        }
                    }
                }
                Ok(true)
            },
        )?;
        // A length cutoff can leave partial tool arguments. It must not be
        // reported as a successfully completed tool call.
        let delta = emitter.update(&result.token_ids, true, result.finish_reason != "length")?;
        if !delta.as_object().unwrap().is_empty() {
            emitter.stats(&result.stats, None, "complete", delta, true)?;
        }
        if job.request.required_tool && result.finish_reason != "length" {
            ensure!(
                !emitter.output.calls.is_empty(),
                "model did not produce the required tool call"
            );
        }
        let finish = if emitter.stopped {
            "stop"
        } else if result.finish_reason == "length" {
            "length"
        } else if !emitter.output.calls.is_empty() {
            "tool_calls"
        } else {
            "stop"
        };
        let usage = json!({"prompt_tokens":job.request.ids.len(),"completion_tokens":result.token_ids.len(),"total_tokens":job.request.ids.len()+result.token_ids.len(),"prompt_tokens_details":{"cached_tokens":0}});
        let timing = timings(&result.stats, emitter.text_tokens);
        let mut meta = metadata(
            &result.stats,
            "complete",
            result.batches.last(),
            emitter.text_tokens,
        );
        meta["batches"] = json!(result.batches);
        meta["generation_settings"] = json!(job.request.options);
        let mut terminal = emitter.chunk(json!({}), json!(finish));
        terminal["timings"] = timing.clone();
        terminal["minnow"] = meta.clone();
        if job.request.include_usage {
            terminal["usage"] = Value::Null;
        }
        job.transport.send(terminal, true)?;
        if job.request.include_usage {
            let mut chunk = emitter.chunk(json!({}), Value::Null);
            chunk["choices"] = json!([]);
            chunk["usage"] = usage.clone();
            job.transport.send(chunk, true)?;
        }
        let choice = if job.request.kind == Kind::Chat {
            json!({"index":0,"message":emitter.output.message(),"finish_reason":finish})
        } else {
            json!({"index":0,"text":emitter.output.content,"logprobs":null,"finish_reason":finish})
        };
        Ok(
            json!({"id":emitter.id,"object":if job.request.kind==Kind::Chat {"chat.completion"}else{"text_completion"},"created":emitter.created,"model":app.info.model_id,"choices":[choice],"usage":usage,"timings":timing,"minnow":meta}),
        )
    };
    run().map_err(|e| {
        tracing::error!(error=%e,"generation failed");
        ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })
}

async fn submit(app: App, body: Value, kind: Kind) -> std::result::Result<Response, ApiError> {
    if !app.healthy.load(Ordering::Acquire) || app.stopping.load(Ordering::Acquire) {
        return Err(ApiError::unavailable("inference worker unavailable"));
    }
    let info = app.info.clone();
    let codec = app.codec.clone();
    let request = tokio::task::spawn_blocking(move || request::prepare(body, kind, &codec, &info))
        .await
        .map_err(|_| ApiError::unavailable("request preparation failed"))??;
    let enqueue = |job| {
        app.tx.try_send(Some(job)).map_err(|e| match e {
            mpsc::error::TrySendError::Full(_) => {
                ApiError::new(StatusCode::TOO_MANY_REQUESTS, "inference queue is full")
            }
            mpsc::error::TrySendError::Closed(_) => {
                ApiError::unavailable("inference worker unavailable")
            }
        })
    };
    if request.stream {
        let (tx, rx) = mpsc::channel(128);
        enqueue(Job {
            request,
            transport: Transport::Stream(tx),
        })?;
        let mut response = Sse::new(ReceiverStream::new(rx))
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_secs(1))
                    .text("keep-alive"),
            )
            .into_response();
        response
            .headers_mut()
            .insert("x-accel-buffering", "no".parse().unwrap());
        response.headers_mut().insert(
            header::CACHE_CONTROL,
            "no-cache, no-transform".parse().unwrap(),
        );
        Ok(response)
    } else {
        let (tx, rx) = oneshot::channel();
        enqueue(Job {
            request,
            transport: Transport::Json(tx),
        })?;
        rx.await
            .map_err(|_| ApiError::unavailable("inference worker stopped"))?
            .map(|v| Json(v).into_response())
    }
}
async fn chat(
    State(app): State<App>,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> std::result::Result<Response, ApiError> {
    submit(
        app,
        body.map_err(|e| ApiError::invalid(e.body_text()))?.0,
        Kind::Chat,
    )
    .await
}
async fn completion(
    State(app): State<App>,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> std::result::Result<Response, ApiError> {
    submit(
        app,
        body.map_err(|e| ApiError::invalid(e.body_text()))?.0,
        Kind::Completion,
    )
    .await
}
async fn health(State(app): State<App>) -> std::result::Result<Json<Value>, ApiError> {
    if !app.healthy.load(Ordering::Acquire) || app.tx.is_closed() {
        return Err(ApiError::unavailable("inference worker unavailable"));
    }
    Ok(Json(
        json!({"status":"ok","model":app.info.model_id,"max_context":app.info.max_context,"queued_requests":app.tx.max_capacity()-app.tx.capacity()}),
    ))
}
fn model_info(info: &Info) -> Value {
    json!({"id":info.model_id,"object":"model","created":0,"owned_by":"local","context_length":info.max_context,"meta":{"n_ctx_train":info.trained_context,"n_ctx":info.max_context,"n_vocab":info.vocab_size},"capabilities":{"completion":true,"tool_calling":true,"vision":false,"audio":false}})
}
async fn models(State(app): State<App>) -> Json<Value> {
    Json(json!({"object":"list","data":[model_info(&app.info)]}))
}
async fn model_detail(
    State(app): State<App>,
    Path(id): Path<String>,
) -> std::result::Result<Json<Value>, ApiError> {
    if id != app.info.model_id {
        return Err(ApiError::new(StatusCode::NOT_FOUND, "unknown model"));
    }
    Ok(Json(model_info(&app.info)))
}
async fn not_found() -> ApiError {
    ApiError::new(StatusCode::NOT_FOUND, "endpoint not found")
}

fn router(app: App) -> Router {
    let mut router = Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/models/{id}", get(model_detail))
        .route("/v1/chat/completions", post(chat))
        .route("/chat/completions", post(chat))
        .route("/v1/completions", post(completion))
        .merge(webui::routes())
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024));
    if let Some(dir) = &app.info.ui_dir {
        router = router
            .nest("/v1", Router::new().fallback(not_found))
            .fallback_service(
                tower_http::services::ServeDir::new(dir)
                    .precompressed_gzip()
                    .precompressed_br(),
            )
            .layer(
                tower_http::set_header::SetResponseHeaderLayer::if_not_present(
                    header::CACHE_CONTROL,
                    header::HeaderValue::from_static("no-cache"),
                ),
            );
    } else {
        router = router.fallback(not_found);
    }
    router.with_state(app)
}

pub async fn serve(model: Model, codec: TextCodec, config: ServeConfig) -> Result<()> {
    ensure!(
        config.max_context > 0
            && config.max_context <= model.config.max_position_embeddings
            && config.max_context.is_multiple_of(model.config.block_size),
        "max-context must be a whole number of blocks within model context"
    );
    ensure!(
        (1..=1024).contains(&config.queue_capacity),
        "queue-capacity must be between 1 and 1024"
    );
    config.defaults.validate(model.config.vocab_size)?;
    if let Some(dir) = &config.ui_dir {
        ensure!(
            dir.join("index.html").is_file(),
            "ui-dir must point at built UI assets containing index.html"
        );
    }
    let listener = tokio::net::TcpListener::bind(config.listen).await?;
    let (tx, mut rx) = mpsc::channel::<Option<Job>>(config.queue_capacity);
    let app = App {
        tx,
        healthy: Arc::new(AtomicBool::new(true)),
        stopping: Arc::new(AtomicBool::new(false)),
        codec: Arc::new(codec),
        info: Arc::new(Info {
            model_id: config.model_id,
            model_path: config.model_path,
            max_context: config.max_context,
            vocab_size: model.config.vocab_size,
            trained_context: model.config.max_position_embeddings,
            defaults: config.defaults,
            ui_dir: config.ui_dir,
        }),
        active: Arc::new(Mutex::new(None)),
    };
    let worker = app.clone();
    let worker_tx = worker.tx.clone();
    let handle = std::thread::Builder::new()
        .name("minnow-inference".into())
        .spawn(move || {
            struct Guard(Arc<AtomicBool>);
            impl Drop for Guard {
                fn drop(&mut self) {
                    self.0.store(false, Ordering::Release);
                }
            }
            let _guard = Guard(worker.healthy.clone());
            while let Some(Some(job)) = rx.blocking_recv() {
                if worker.stopping.load(Ordering::Acquire) {
                    break;
                }
                if job.transport.closed() {
                    continue;
                }
                *worker.active.lock().unwrap() =
                    Some(json!({"id":0,"is_processing":true,"n_ctx":worker.info.max_context}));
                let result = execute(&model, &worker, &job);
                *worker.active.lock().unwrap() = None;
                match job.transport {
                    Transport::Json(reply) => {
                        let _ = reply.send(result);
                    }
                    Transport::Stream(reply) => {
                        if let Err(error) = result {
                            let _ = reply
                                .try_send(Ok(Event::default().data(error.value().to_string())));
                        }
                        let _ = reply.try_send(Ok(Event::default().data("[DONE]")));
                    }
                }
            }
        })?;
    let stopping = app.stopping.clone();
    let shutdown = stopping.clone();
    tracing::info!(listen=%config.listen,max_context=config.max_context,"serving minnow");
    let outcome = axum::serve(listener, router(app))
        .with_graceful_shutdown(async move {
            #[cfg(unix)]
            {
                let mut term =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("install SIGTERM handler");
                tokio::select! {_=tokio::signal::ctrl_c()=>{},_=term.recv()=>{}}
            }
            #[cfg(not(unix))]
            {
                let _ = tokio::signal::ctrl_c().await;
            }
            shutdown.store(true, Ordering::Release);
        })
        .await;
    stopping.store(true, Ordering::Release);
    // Wake an idle worker; a full queue is already sufficient to wake it.
    let _ = worker_tx.try_send(None);
    let _ = tokio::task::spawn_blocking(move || handle.join()).await;
    outcome?;
    Ok(())
}

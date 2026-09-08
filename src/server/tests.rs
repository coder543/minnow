use super::*;
use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use tower::ServiceExt;

fn app() -> (App, mpsc::Receiver<Option<Job>>) {
    let (tx, rx) = mpsc::channel(2);
    (
        App {
            tx,
            healthy: Arc::new(AtomicBool::new(true)),
            stopping: Arc::new(AtomicBool::new(false)),
            codec: Arc::new(TextCodec::fixture()),
            info: Arc::new(Info {
                model_id: "test".into(),
                model_path: "fixture".into(),
                max_context: 128,
                vocab_size: 320,
                trained_context: 131072,
                defaults: Options {
                    threshold: 0.7,
                    editing_threshold: 0.2,
                    max_post_steps: 5,
                    ..Options::default()
                },
                ui_dir: None,
            }),
            active: Arc::new(Mutex::new(None)),
        },
        rx,
    )
}

#[test]
fn defaults_and_partial_request_overrides_compose() {
    let (app, _rx) = app();
    let body = json!({"messages":[{"role":"developer","content":"hello"},{"role":"user","content":[{"type":"text","text":"hello"}]}],"minnow":{"threshold":0.6},"editing_threshold":0.3,"max_completion_tokens":16});
    let p = request::prepare(body, Kind::Chat, &app.codec, &app.info).unwrap();
    assert_eq!(p.options.threshold, 0.6);
    assert_eq!(p.options.editing_threshold, 0.3);
    assert_eq!(p.options.max_post_steps, 5);
    assert_eq!(p.options.max_tokens, 16);
    for (key, value) in [
        ("threshold", json!(-0.1)),
        ("editing_threshold", json!(1.1)),
        ("max_post_steps", json!(-1)),
    ] {
        let mut body = json!({"messages":[{"role":"user","content":"hello"}]});
        body[key] = value;
        assert!(request::prepare(body, Kind::Chat, &app.codec, &app.info).is_err());
    }
}

#[test]
fn tool_choice_and_history_are_validated() {
    let (app, _rx) = app();
    let body = json!({"messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}],"tool_choice":{"type":"function","function":{"name":"weather"}},"max_tokens":8});
    let p = request::prepare(body.clone(), Kind::Chat, &app.codec, &app.info).unwrap();
    assert!(
        p.assistant_prefix
            .ends_with("functions.weather:0<|tool_call_argument_begin|>")
    );
    let mut bad = body;
    bad["tool_choice"]["function"]["name"] = json!("other");
    assert!(request::prepare(bad, Kind::Chat, &app.codec, &app.info).is_err());
    let history = json!([{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_1","content":"sunny"}]);
    assert!(request::messages(&history).is_ok());
    let mut bad = history;
    bad[1]["tool_call_id"] = json!("missing");
    assert!(request::messages(&bad).is_err());
}

async fn body(response: Response) -> (StatusCode, Vec<u8>) {
    (
        response.status(),
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
}

#[tokio::test]
async fn api_errors_and_props_work_without_a_model() {
    let (app, mut rx) = app();
    let router = router(app);
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/props")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let (_, bytes) = body(response).await;
    let props: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        props["default_generation_settings"]["params"]["max_post_steps"],
        5
    );
    assert_eq!(props["default_generation_settings"]["n_ctx"], 128);
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/chat/completions")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"messages":[],"stream":true}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    let (status, bytes) = body(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(serde_json::from_slice::<Value>(&bytes).unwrap()["error"].is_object());
    assert!(rx.try_recv().is_err());
    let response = router
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn external_ui_is_read_live_and_does_not_replace_api_errors() {
    let (mut app, _rx) = app();
    let root = std::env::temp_dir().join(format!(
        "minnow-ui-{}-{}",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("index.html"), "first").unwrap();
    Arc::get_mut(&mut app.info).unwrap().ui_dir = Some(root.clone());
    let router = router(app);
    for text in ["first", "updated"] {
        std::fs::write(root.join("index.html"), text).unwrap();
        let response = router
            .clone()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-cache");
        let (status, bytes) = body(response).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(bytes, text.as_bytes());
    }
    for path in ["/v1/unsupported", "/%2e%2e/Cargo.toml", "/missing.js"] {
        let response = router
            .clone()
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn progress_drops_under_backpressure_and_closed_clients_cancel() {
    let (tx, rx) = mpsc::channel(8);
    let transport = Transport::Stream(tx);
    transport.send(json!({"text":"first"}), true).unwrap();
    transport.send(json!({"progress":1}), false).unwrap();
    if let Transport::Stream(tx) = &transport {
        assert_eq!(tx.capacity(), 7);
    }
    drop(rx);
    assert!(transport.closed());
    assert!(transport.send(json!({}), true).is_err());
}

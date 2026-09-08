# HTTP API and llama-server UI

Build with `cargo build --release --features cuda`. The native Rust process owns
one model, one inference worker, a bounded queue, HTTP handling, and optional static
assets. `server/request.rs`, `server/output.rs`, and `server/webui.rs` separate the
API contract, committed-output parsing, and UI compatibility.

## Serving

```sh
target/release/minnow serve --listen 127.0.0.1:8080 \
  --alias minnow-llada2.2-mini \
  --ui-dir ../llama.cpp/build/tools/ui/dist \
  --threshold 0.5 --editing-threshold 0 --max-post-steps 16
```

Omit `--ui-dir` for API-only operation. It must contain a built `index.html` and
its assets; the source `tools/ui` directory alone is insufficient. The directory
is read on demand, including precompressed assets, with cache revalidation, so a
new UI build is served without copying it into minnow. The tested UI is llama.cpp
b10819. No session resumption is implemented; the UI startup lookup returns an
empty list.

Context defaults to `config.json`'s `max_position_embeddings`: 131,072 for mini.
`--max-context` can reduce it to a whole number of 32-token blocks. This is a
per-request prompt-plus-output limit, not an eagerly allocated global cache.
K/V storage is allocated for the request's block-rounded budget. Full-context
BF16 K/V alone is 5 GiB (20 layers × K/V × 4 heads × 128 dimensions × 131,072 ×
2 bytes); working activations require additional memory. The full context is
advertised by `/health`, `/props`, and `/v1/models`. Advertising it does not imply
a completed full-length performance or soak test.

Defaults: queue 8, maximum output 256, greedy sampling, threshold 0.5,
editing_threshold 0, max_post_steps 16, steps 32, max_steps_per_block 1000,
top_k 0, top_p 1, seed 42. All decoding defaults have `serve --...` arguments.
The llama-swap entry overrides maximum output to 2,048.

## Routes and request options

- `POST /v1/chat/completions` (also `/chat/completions`): Chat Completions.
- `POST /v1/completions`: plain string prompts, streaming or non-streaming.
- `GET /v1/models`, `/v1/models/{id}`: model identity and context metadata.
- `GET /health`, `/v1/health`, `/props`, `/slots`: readiness and UI metadata.
- `POST /tokenize`, `/detokenize`, `/apply-template`: text utilities.

The primary interface follows the [Chat Completions contract](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
Messages support system/developer, user, assistant, and tool roles, text strings
and text-only content parts. Developer messages use the model's system role.

Supported controls include `max_completion_tokens` (preferred), `max_tokens`,
`temperature`, `top_p`, `top_k`, `seed`, `stop`, `n: 1`, `stream`, and
`stream_options.include_usage`. The llama-server `n_predict` alias is accepted;
`-1` uses remaining context. An omitted output limit uses the server default,
clamped to remaining context. Stops are held across block boundaries and stop
further block generation once matched. Disconnects cancel queued or active work
at the next forward boundary.

Diffusion controls `threshold`, `editing_threshold`, `max_post_steps`, `steps`,
and `max_steps_per_block` may be top-level request fields or members of `minnow`.
Precedence is server defaults, then partial `minnow` overrides, then top-level
fields. Final `minnow.generation_settings` shows the effective settings.
`steps` controls an unmasking schedule, not an exact number of forward passes.

```json
{
  "model": "minnow-llada2.2-mini",
  "messages": [{"role":"user","content":"What is the LHC?"}],
  "max_completion_tokens": 1024,
  "stream": true,
  "stream_options": {"include_usage":true},
  "return_progress": true,
  "threshold": 0.5,
  "editing_threshold": 0,
  "max_post_steps": 16
}
```

Unknown request fields and unsupported active features return JSON errors before
stream headers. Errors during streaming produce an error event and `[DONE]`.
Logprobs, multiple choices, non-neutral repetition penalties, multimodal inputs,
JSON Schema constraints, grammar/logit bias, and continuation are unsupported.
`response_format: {"type":"json_object"}` adds a JSON instruction and buffers
output for validation; it is not grammar-constrained decoding and can fail if the
model returns invalid JSON. Stochastic results need not match PyTorch's RNG.
The Responses API is deferred.

## Function tools

Use standard `tools: [{"type":"function","function":{...}}]` definitions.
`tool_choice` supports `auto`, `none`, `required`, and a named function. Required
and named calls prefill the model's native tool-call prefix. Tool definitions are
rendered with the checkpoint's chat template. Native tool markers are parsed
into `message.tool_calls` or incremental `delta.tool_calls`; they do not appear
as assistant prose. Complete arguments must be JSON objects and names must have
been declared. A completed call ends with `finish_reason: "tool_calls"`; output
cut off by the token budget ends with `"length"`, potentially with partial args.

Append the assistant message containing `tool_calls` to history, then one
`{"role":"tool","tool_call_id":"call_...","content":"..."}` result per call.
The next request renders results using the model's native return format.
Minnow does not execute tools. `/tools` returns the same disabled-feature response
as llama-server without server tools; browser-side and external MCP tools remain
client responsibilities. `parallel_tool_calls: false` rejects multiple
returned calls; strict schema-constrained generation (`strict: true`) is deferred.

## Streaming, progress and statistics

SSE uses `chat.completion.chunk`, a stable ID, assistant-role initialization,
content/tool deltas, a finish chunk, optional final usage chunk, then `[DONE]`.
Only finalized blocks emit text or tool arguments. Progress-only events are
requested with `return_progress: true` or `timings_per_token: true`.

Prefill events contain `prompt_progress` with `processed`, `total`, `cache: 0`,
and `time_ms`. A final timing event without `prompt_progress` clears the UI's
preparing state immediately when prefill ends. Refinement events carry metadata
without text or a prefill progress field. The UI displays ordinary prefill and
generation speeds; custom refinement statistics remain available to API clients.

| Field | Meaning |
| --- | --- |
| `usage.prompt_tokens` | Entire rendered prompt, including tool definitions/history. |
| `usage.completion_tokens` | Final generated model IDs, including special controls and positions generated past a stop within the final block. |
| `timings.prompt_n`, `prompt_per_second` | Complete prompt blocks processed by prefill and their measured throughput. A partial prompt block is processed during diffusion. |
| `timings.predicted_n`, `predicted_per_second` | Generated non-special token count and rate over decode time, including refinement/commit costs. Stop-boundary overgeneration remains counted. |
| `minnow.text_tokens`, `text_tokens_per_second` | Same non-special output count/rate. |
| `minnow.evaluated_tokens` | 32 × refinement forwards: all predicted positions, including repeated refinements, known prompt positions, and final padding. This is work, not useful output. |
| `minnow.total_tokens_per_second` | Evaluated positions divided by decode seconds. |
| `minnow.denoise_forwards` | Total refinement steps. |
| `minnow.refinement_steps_per_block` | Mean refinement steps per generated block. |
| `minnow.processed_tokens` | All transformer input positions: prefill, refinements, and necessary commit refreshes. |
| `minnow.commit_forwards`, `reused_commits` | Cache refreshes versus exact reuse after block finalization. |
| `minnow.tokens_per_second` | Generated model IDs / total elapsed inference time, including prefill (legacy metric). |
| `minnow.batch` | Current block: index, offset, finalized model-token count, refinement_steps, evaluated_tokens, elapsed_seconds, output and total-work rates. |
| `minnow.batches` | All finalized block records, included at response completion. |

Block output rates use final model IDs, including special controls. During a
refinement, the current block's completion count is zero until it is finalized.
Response averages use summed counts and elapsed time; they are not an unweighted
average of block rates. Request queuing is excluded from inference timings.

## Validation

`cargo test --release --features cuda -- --include-ignored` covers numerical
fixtures, progress and cancellation boundaries, incremental tool/stop parsing,
request validation, overrides, and external asset serving.
`scripts/check_compatibility.py` exercises the trained model over HTTP, including
stream/plain equivalence, multi-block prefill, tool/result round trips, progress
clearing, and disconnect cancellation. Run under `scripts/memory_guard.py`, with
only one model process. `artifacts/check-minnow-ui.mjs` exercises the actual UI
through llama-swap using the locally installed Playwright browser.

Validated on GB10 with the unquantized BF16 checkpoint: 33 Rust tests (including
CUDA checks), CPU/CUDA clippy, streamed/non-streamed tool equivalence, tool-result
history, an 8,827-token prompt with 8,800 prefilled positions, and the actual
llama-server UI through llama-swap. The browser also completed its built-in
`get_datetime` tool round trip. All three configured model names routed without
rewriting. Artifacts: `artifacts/minnow-swap-api.json`, `minnow-ui-check.json`, and
`minnow-aliases.json`. The guarded API run peaked at 33.72 GiB above its unloaded
system baseline, with no new swap-out; disconnect cancellation took about 54 ms.
This validates the configured full limit and prompts beyond the old 8K cap, not a
131,072-token soak run.

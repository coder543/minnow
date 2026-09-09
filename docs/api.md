# HTTP API and llama-server UI

Build with `cargo build --release --features cuda`. The native Rust process owns
one model, one inference worker, a bounded queue, HTTP handling, and optional static
assets. `server/request.rs`, `server/output.rs`, and `server/webui.rs` separate the
API contract, committed-output parsing, and UI compatibility.

## Serving

```sh
target/release/minnow --model models/mini-bf16.mnw serve --listen 127.0.0.1:8080 \
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

Context defaults to `config.json`'s `max_position_embeddings`: 131,072 for mini and flash.
`--max-context` can reduce it to a whole number of 32-token blocks. This is a
per-request prompt-plus-output limit, not an eagerly allocated global cache.
K/V initially covers the prompt and first output block, rounded up to 2,048 tokens.
It grows in 2,048-token chunks as generation advances, capped at the context limit.
An uncapped output does not reserve its entire possible context. Full-context
BF16 K/V alone is 5 GiB (20 layers × K/V × 4 heads × 128 dimensions × 131,072 ×
2 bytes); working activations require additional memory. The full context is
advertised by `/health`, `/props`, and `/v1/models`. Advertising it does not imply
a completed full-length performance or soak test.

Defaults: parallel 4, queue 8, no output cap within the remaining context,
greedy sampling, threshold 0.5, editing_threshold 0, max_post_steps 16,
steps 32, max_steps_per_block 1000,
top_k 0, top_p 1, and a fresh random seed per generation. Greedy temperature 0
matches upstream mini and flash defaults; set `temperature` above zero to sample.
All decoding defaults have `serve --...` arguments. `--seed N` fixes the seed for
CLI generation or supplies a server default. Requests can override it with
`seed: N`; `seed: -1` selects fresh randomness even when the server has a fixed
seed. An omitted or null request seed inherits the server default. `/props`
advertises `seed: -1` when seeds are random.
`serve --max-tokens N` sets a smaller default output budget; requests may override
it. Without that flag, responses continue until EOS, a stop string, cancellation,
or the context limit. The `generate` command uses the same output-limit default.

### Conversation prefix slots

`--cache-slots 4` retains up to four independent conversation prefixes, with one
inference worker. `--cache-slots 0` disables reuse. The default aggregate K/V
capacity budget is one full context (5 GiB for mini BF16); `--cache-max-mib N`
overrides it and must fit at least one configured `--max-context`. Idle slots are
evicted in LRU order before growing a buffer. Weights are shared by all slots.

Selection uses the longest exact token prefix, rounded down to a 32-token block
boundary. Committed blocks are bidirectional internally, so a changed token
invalidates its entire block and everything after it. A match covering at least
`--cache-reuse-threshold 0.8` of the shorter prefix updates that slot in place.
A smaller match copies only those committed blocks into an empty or LRU slot,
preserving the source conversation when the memory budget allows. If both buffers
cannot fit, the source is truncated and reused in place. Similarity is a heuristic
for slot retention; it never permits reuse of nonmatching tokens.
Changing prefill batch boundaries can change BF16 rounding, expert routing, and
generated wording even for identical prompts; warm/cold bitwise equivalence is
not promised. See [numerical checks and measurements](prefix-cache.md).

Prompt blocks and finalized generated blocks committed during decoding are
retained. The final output block may need recomputation on the next request;
uncommitted refinement K/V is never reused. Failed/cancelled generations discard
their active slot. `cache_prompt: false` performs a cold request without storing
its result; it can still evict idle slots if needed for the memory budget.
`/slots` exposes execution slots and the selected prefix's committed/reserved
capacity; `/props` exposes policy, budget, and parallelism. Active prefixes are
exclusively checked out and cannot be reused or evicted by another request. If
all other slots are busy, a divergent prefix reuses its source in place. A
zero-output request leaves cached conversations untouched. Stream resumption is
not implemented.

### Continuous batching

`--parallel N` (default 4, maximum 64) bounds concurrent requests. One GPU worker
combines ready token segments across sequences, sharing dense/MoE/head GEMMs.
Each sequence retains its own attention, positional offsets, K/V, block-capacity
routing, RNG, decoder state, and stream. Prefill and refinement may share a batch.
New requests are admitted as execution slots and the aggregate K/V budget allow.
Requests waiting for K/V count toward both `--queue-capacity` and
`/health.queued_requests`, including the scheduler's pending admission candidate.
Idle prefixes are evicted first; active reservations, including cache-bypass
requests, count toward the same budget. Long requests can consequently reduce
the achievable parallelism without reducing the advertised per-request context.
Uncapped requests can run concurrently: reservations track allocated chunks,
not their maximum possible output. Growth evicts idle caches first and waits
while other active requests can make progress. New admissions pause during that
wait. If every active request is blocked on growth, the youngest blocked request
fails with a capacity error (HTTP 503, or an error event for an existing stream),
releasing its cache so older requests can continue. This does not silently shorten
the requested output. Cancellation and failures release all grown chunks.

`--batch-wait-us` defaults to 200 and bounds the time spent collecting ready work.
A lone active request does not wait for another arrival. `--parallel 1` serializes
execution. `/health.batching` reports `forward_batches`, `sequence_forwards`, and
`max_batch_size`, allowing clients/tests to distinguish actual batched execution
from concurrently queued requests. Inference timings include scheduling between
forwards once admitted; they exclude time awaiting admission. Changing batch
composition may change floating-point reduction order and MoE selections, so
greedy output is not guaranteed bitwise invariant across concurrency levels.
`kv_growths` counts buffer growth operations; `kv_pressure_rejections` counts
requests failed to resolve a full-budget wait. `/slots` shows current capacities.

## Routes and request options

- `POST /v1/chat/completions` (also `/chat/completions`): Chat Completions.
- `POST /v1/completions`: plain string or token-ID prompts, streaming or non-streaming.
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
clamped to remaining context. `/props` advertises `max_tokens` and `n_predict` as
`-1` when the default uses remaining context, including for the web UI.
An explicitly saved client limit still applies. Stops are held across block
boundaries and stop further block generation once matched. Disconnects cancel queued or active work
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

Prefill events contain `prompt_progress` with `processed`, `total`, `cache`,
and `time_ms`. Processed and total include cached tokens; `cache` is the reused
prefix length. A fully cached prompt skips progress events. A final timing event
without `prompt_progress` clears the UI's
preparing state immediately when prefill ends. Refinement events carry metadata
without text or a prefill progress field. The UI displays ordinary prefill and
generation speeds; custom refinement statistics remain available to API clients.

| Field | Meaning |
| --- | --- |
| `usage.prompt_tokens` | Entire rendered prompt, including tool definitions/history. |
| `usage.completion_tokens` | Final generated model IDs, including special controls and positions generated past a stop within the final block. |
| `timings.prompt_n`, `prompt_per_second` | Newly computed complete prompt blocks and their measured throughput, excluding cached tokens and cache setup/copy time. A partial prompt block is processed during diffusion. |
| `usage.prompt_tokens_details.cached_tokens`, `timings.cache_n`, `minnow.cached_tokens` | Exact complete prompt tokens reused across requests. |
| `minnow.cache_slot`, `cache_copied_tokens`, `cache_seconds` | Selected slot (null for cold requests), prefix tokens copied when forking, and synchronized lookup/allocation/copy time. |
| `timings.predicted_n`, `predicted_per_second` | Generated non-special token count and rate over decode time, including refinement/commit costs. Stop-boundary overgeneration remains counted. |
| `minnow.text_tokens`, `text_tokens_per_second` | Same non-special output count/rate. |
| `minnow.evaluated_tokens` | 32 × refinement forwards: all predicted positions, including repeated refinements, known prompt positions, and final padding. This is work, not useful output. |
| `minnow.total_tokens_per_second` | Evaluated positions divided by decode seconds. |
| `minnow.denoise_forwards` | Total refinement steps. |
| `minnow.refinement_steps_per_block` | Mean refinement steps per generated block. |
| `minnow.processed_tokens` | All transformer input positions: prefill, refinements, and necessary commit refreshes. |
| `minnow.commit_forwards`, `reused_commits` | Cache refreshes versus exact reuse after block finalization. |
| `minnow.tokens_per_second` | Generated model IDs / total elapsed inference time, including prefill. |
| `minnow.batch` | Current block: index, offset, finalized model-token count, refinement_steps, evaluated_tokens, elapsed_seconds, output and total-work rates. |
| `minnow.batches` | All finalized block records, included at response completion. |

Block output rates use final model IDs, including special controls. During a
refinement, the current block's completion count is zero until it is finalized.
Response averages use summed counts and elapsed time; they are not an unweighted
average of block rates. Request queuing is excluded from inference timings.

## Validation

`scripts/check_compatibility.py --spawn --model MODEL` exercises streaming/plain
responses, multi-block prefill, tool/result round trips, progress clearing, and
disconnect cancellation. `scripts/check_batching.py --model MODEL` checks
concurrent requests, late admission, and cancellation recovery.
`scripts/check_prefix_cache.py --url URL` checks forks, growth, eviction, and
cache/progress accounting against a running server. See [validation](validation.md).

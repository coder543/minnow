# minnow

A model-specific Rust/Candle inference server for LLaDA2.2-mini and LLaDA2.2-flash.
It supports BF16 weights, mixed expert-only INT8/FP4 quantization, an FP32 diagnostic
path, and continuous batching. Validated on NVIDIA GB10 with CUDA 13.0; RTX 3090
has not yet been hardware-tested.

Minnow keeps committed-prefix K/V resident and evaluates only the current 32-token
block during refinement. It supports the reference's block-capacity MoE routing,
partial RoPE, Q/K normalization, and joint mask/token/delete/split decoder. CUDA
uses pointer-batched expert GEMMs for refinement and grouped expert GEMMs for
prefill. Fused SiLU/multiply, RMSNorm, Q/K normalization with RoPE, attention
softmax, and expert mixing retain FP32 arithmetic and BF16 rounding points.
Greedy selection reduces the vocabulary in parallel and transfers only token IDs
and confidences. RoPE tables are reused while refining the same block. A block
commit reuses the last forward's K/V when its input exactly matches the final
tokens; final edits trigger a refresh before committing.

Prefill batches up to 4,096 tokens through each transformer layer while attention
uses separate 1,024-query tiles. Configure these with `--prefill-chunk-tokens` and
`--attention-chunk-tokens`. Attention tiles shrink automatically at long contexts;
transformer batches are limited to 8,192 tokens to bound expert activations.

## Build and run

Requires Rust, Linux with direct I/O, and CUDA 13.0 or later with a GPU supporting BF16
(Ampere or later). The CUDA build uses `nvcc`; set `NVCC` if it is not on PATH.
The CPU build is useful for small FP32 fixtures.

`MINNOW_CUDA_ARCH` selects the PTX target for minnow's custom kernels, defaulting
to `compute_80`. For a GB10-specific build, use
`MINNOW_CUDA_ARCH=compute_121 cargo build --release --features cuda`.
This does not change Candle's independently built kernels or cuBLAS's GEMMs.
Changing the variable triggers a rebuild; newer targets require compatible GPUs.

```sh
cargo build --release --features cuda
target/release/minnow inspect
target/release/minnow generate 'What is the capital of France?' --max-tokens 64
target/release/minnow serve
```

The default checkpoint is `~/models/hf/inclusionAI/LLaDA2.2-mini`, on this machine's
local NVMe SSD. Override it with `--model PATH`. The original
`~/hf/inclusionAI/LLaDA2.2-mini` checkout is read-only. To create a fresh local copy:

```sh
python3 scripts/copy_checkpoint.py \
  ~/hf/inclusionAI/LLaDA2.2-mini ~/models/hf/inclusionAI/LLaDA2.2-mini
```

The copy command refuses an existing destination. It streams with an 8 MiB buffer
and verifies every file's SHA-256 before publishing the directory. Minnow reads
weights directly into their final allocations: no checkpoint mmap, complete host
model, or second packed expert copy. A process lock prevents concurrent full-model
runtime/reference loads. BF16 weights occupy 30.28 GiB; FP32 weights occupy
60.56 GiB. See [memory ownership and limits](docs/memory.md).

## HTTP interface

The default listener is `127.0.0.1:8080`, with up to four active requests, a queue
of eight, and the checkpoint's full context (131,072 tokens for mini and flash). Override
these with `serve --listen ADDRESS --parallel N --queue-capacity N --max-context N`. Four
conversation prefix slots share a K/V capacity budget of one full context (5 GiB
for mini BF16). `--cache-slots`, `--cache-reuse-threshold`, and `--cache-max-mib`
configure retention; `cache_prompt: false` requests a cold run. Exact prefixes
are reused at 32-token boundaries. See [cache policy and metrics](docs/api.md#conversation-prefix-slots).

One GPU worker combines ready prefill and refinement segments across requests.
Attention, K/V, block routing capacity, RNG, and output streams remain independent.
New requests enter as slots and the shared K/V budget become available. Set
`--parallel 1` for serialized execution; `--batch-wait-us` defaults to 200.
The CUDA allocator retains up to `--workspace-cache-mib 2048` of unused scratch
storage relative to live allocations; zero disables retention.

```sh
target/release/minnow serve \
  --ui-dir ../llama.cpp/build/tools/ui/dist \
  --threshold 0.5 --editing-threshold 0 --max-post-steps 16

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minnow-llada2.2-mini","messages":[{"role":"user","content":"What is the LHC?"}],"max_completion_tokens":1024,"stream":true,"stream_options":{"include_usage":true},"return_progress":true,"threshold":0.5,"editing_threshold":0,"max_post_steps":16}'
```

Chat Completions supports streaming, function tools and tool-result history,
common sampling controls, stop strings, and usage. Committed 32-token blocks
stream after refinement. Prefill progress clears when prefill completes; standard
llama-server timings report prefill and text-generation rates. Additional `minnow`
metadata reports refinement work and per-block/whole-response statistics.

`--ui-dir` serves the existing built llama-server UI directly from disk; assets
are neither embedded nor copied. UI compatibility and request handling are separate
Rust modules in the same process. No extra proxy or Docker is required.

See [API and UI details](docs/api.md) for supported routes, tool calling, metric
semantics, and limits. The local llama-swap entry is `llada-2.2-mini`; its UI is
[available through llama-swap](http://127.0.0.1:8083/upstream/llada-2.2-mini/).
Responses API and stream/session resumption are deferred.

## Self-contained models and quantization

`.mnw` bundles aligned weights, a checksummed MessagePack manifest, tokenizer,
configuration, and the exact chat template. Both the original directory and the
single-file container are supported by every model-loading command. Conversion
uses bounded direct reads/writes and never loads the whole checkpoint.

```sh
target/release/minnow convert ~/models/minnow/llada2.2-mini-bf16.mnw
target/release/minnow convert ~/models/minnow/llada2.2-mini-int8.mnw --experts int8
target/release/minnow --model ~/hf/inclusionAI/LLaDA2.2-flash \
  convert ~/models/minnow/llada2.2-flash-fp4.mnw --experts fp4
target/release/minnow --model ~/models/minnow/llada2.2-flash-fp4.mnw serve
```

INT8 uses symmetric groups of 128 weights with FP16 scales; FP4 uses E2M1 values
and groups of 32 with FP16 scales. This FP4 format is custom, not NVFP4's storage
format. Only routed experts are quantized; embeddings, attention, routers,
shared experts, and the head retain source precision. `--tensor-rules` selects
mixed precision by projection/layer. CUDA dequantizes into tensor-core registers
and keeps BF16 operands with FP32 accumulation. A lossless fragment layout avoids
strided small weight loads; `--quant-layout row` retains the comparison layout.

Weight payloads: mini BF16 30.28 GiB, mini INT8 16.25 GiB, mini FP4 9.79 GiB,
flash FP4 57.96 GiB. Mini INT8 leaves room on a 24 GiB card for K/V and scratch,
but full 131K K/V adds 5 GiB before runtime/workspace costs; context, parallelism,
and scratch budgets must fit the target card. See [model format](docs/model-format.md).

## Validation

Small checked-in fixtures use the checkpoint's actual Python classes and random
weights. Tests compare every layer and router, exercise edits and non-convergence,
check cache refinement/commit, and verify direct reads across buffer boundaries.
They do not require the trained checkpoint or Python.

```sh
cargo test --release
cargo test --release --features cuda
cargo test --release --features cuda --lib -- --ignored
cargo clippy --all-targets --features cuda -- -D warnings
```

The ignored CUDA tests cover expert pointer/index layouts, uneven grouped GEMMs,
fused block softmax, RMSNorm, Q/K normalization with RoPE, routing, and prediction
confidences. RMSNorm, QKV preparation, expert mixing, and SiLU are checked
bitwise; the SiLU check covers every finite BF16 input. Diagnostic switches
`--serial-experts`, `--unfused-activation`, `--unfused-attention`,
`--unfused-expert-mix`, `--unfused-norm`, and `--unfused-qkv` retain comparison paths.
`--device-routing` enables experimental GPU routing and fixed-capacity cuBLAS
batches. CPU routing remains the default because it measured faster on GB10.
`--compact-decode-experts` tests grouped GEMMs on assigned rows only; it measured
within about 1% of the default on the cached-forward sweep and remains optional.
Quantized experts use CPU routing and their own grouped CUDA kernels; the float
expert execution switches compare the original BF16 paths. Quantized CUDA
execution requires BF16 activations. The container and CPU fixture tests verify
mixed precision and lossless layout conversion; the GPU oracle compares all
encodings/group sizes against separately dequantized BF16 operands.

The Python oracle environment used here is Python 3.13, PyTorch 2.10.0+cu130,
Transformers 5.2.0, NumPy and safetensors. Python is not a runtime dependency.
`scripts/layerwise_reference.py` streams one layer at a time, including for FP32.
`scripts/reference_generate.py` uses meta initialization and direct assignment to
hold one BF16 copy when exercising the original complete decoder. Run these
sequentially with the Rust runtime stopped and use the memory guard:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/memory_guard.py \
  --report artifacts/reference-memory.json --max-growth-gib 12 --reserve-gib 32 -- \
  .venv/bin/python scripts/layerwise_reference.py \
  --dtype float32 --output artifacts/reference-f32

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/memory_guard.py \
  --report artifacts/native-memory.json --max-growth-gib 72 --reserve-gib 32 -- \
  target/release/minnow --dtype f32 validate --reference artifacts/reference-f32 \
  --max-abs-error 0.001 --max-relative-error 0.00002
```

The FP32 native diagnostic has a larger memory allowance because it holds one
60.56 GiB weight set. Routine serving uses BF16. Do not recreate a full CPU model
and then convert/copy it to CUDA. The guard observes whole-system memory and new
swap-out activity, rather than relying on process RSS on the GB10's shared RAM.

`forward --input IDS.json --output TRACE.safetensors [--cached]` exports logits and
intermediates. `bench --input IDS.json --prefix-blocks 0,8,32 --iterations 10`
compares repeated current-block evaluation against full-prefix recomputation.
Add `--cached-only` to measure refinement latency at longer prefixes without
full-prefix recomputation. `decode-bench --iterations 3` uses matched tutorial
and story prompts at 128, 512, and 1,024 output-token limits, with a warmup and
three measured repetitions. EOS remains enabled, and actual returned lengths
are recorded. It reports model-token and text-token rates separately, plus
refinement passes per block. Text-token counts exclude special end/role markers;
the historical `useful_tokens_per_second` field is an alias for model-token rate.
Use `--cases tests/generation_cases.json` for the short HTTP smoke cases.
Use `--cases tests/decode_natural_cases.json` for the LHC, React TypeScript, and
Fellowship prompts, each with a 2,048-token cap and normal EOS stopping.
`prefill-bench --input IDS.json --tokens 512,2048,4096 --iterations 5` times the
actual serving prompt-to-KV path with two warmups per length and no vocabulary
projection. Add `--chunk-sizes 128,512,2048,4096` to sweep transformer batches using
one resident model. `--profile` marks timed CUDA iterations for Nsight Systems
with `--capture-range=cudaProfilerApi`. The Python oracle supports the equivalent diagnostic with
`--prefill-input IDS.json --prefill-only --attention eager` or `--attention sdpa`.
`scripts/check_server.py` checks HTTP behavior and compares generation text to
saved `reference_generate.py` results. `scripts/check_compatibility.py` checks streaming, tools, live prefill progress,
settings, cancellation, and optional UI assets against the trained model. Use
`--spawn` to start one server, or `--url URL` for an existing server (including a
llama-swap `/upstream/llada-2.2-mini` URL). Large validations should use the guard.
`scripts/check_batching.py --model FILE.mnw` validates simultaneous generation,
late admission, and cancellation with one model process. `scripts/bench_models.py`
measures repeated cold prefill and long-response useful decode, records actual
lengths/refinement counts, and uses one resident model at a time.

See [prefill and useful generation throughput](docs/throughput-comparison.md),
[quantized mini/flash measurements](docs/quantization-performance.md),
[decode optimization measurements](docs/decode-performance.md),
[CUDA architecture comparison](docs/cuda-architecture.md),
[matched-prompt length measurements](docs/decode-length-sweep.md),
[measured validation and performance](docs/results.md), and
[execution invariants](docs/implementation.md) for evidence and current limits.

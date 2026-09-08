# minnow

Minnow is a Rust inference server for **LLaDA2.2-mini and LLaDA2.2-flash**. It
supports OpenAI Chat Completions, streaming, tool calling, conversation prefix
caching, continuous batching, and an optional llama-server web UI.

LLaDA generates by refining 32-token blocks. Minnow caches committed blocks and
recomputes only the current block during refinement. Completed blocks stream to
the client with prefill progress, useful token rates, and refinement statistics.

## Requirements

- Linux, Rust, and Git (for the pinned CUDA build dependency).
- CUDA 13 or later (`nvcc` on PATH, or set `NVCC`).
- An NVIDIA GPU with BF16 support (Ampere or newer). **NVFP4 requires SM120/121**
  hardware, such as RTX Blackwell or GB10.
- A local model checkpoint on a filesystem supporting direct I/O (`O_DIRECT`).

Only routed experts are quantized; attention, shared experts, embeddings, and the
output head retain their source precision. Approximate mini weight sizes are
30.3 GiB for BF16, 16.3 GiB for INT8, and 9.8 GiB for NVFP4. Allow additional
memory for K/V, activations, and CUDA workspaces. See [memory configuration](docs/memory.md).

## Build and run

Download [LLaDA2.2-mini](https://huggingface.co/inclusionAI/LLaDA2.2-mini) or
[LLaDA2.2-flash](https://huggingface.co/inclusionAI/LLaDA2.2-flash), including its
configuration, tokenizer, and chat template. Pass the checkpoint location
explicitly with `--model`; there is no assumed installation directory.

```sh
cargo build --release --features cuda

target/release/minnow --model models/LLaDA2.2-mini \
  generate 'What is the LHC?' --max-tokens 1024

target/release/minnow --model models/LLaDA2.2-mini serve --listen 127.0.0.1:8080
```

The first CUDA build fetches pinned CUTLASS headers through CudaForge; subsequent
builds reuse its cache. Third-party notices are in `vendor/flash-attention/`.

Serving and checkpoint conversion do not require Python. A CPU build
(`cargo build --release`) is available for small test fixtures and reference
diagnostics.

## Quantization

Conversion produces a self-contained `.mnw` file with weights, tokenizer,
configuration, and chat template. It streams the source and does not load a
complete model into memory. Existing destinations are never overwritten.

```sh
# Blackwell: native FP4 tensor cores for both prefill and decode.
target/release/minnow --model models/LLaDA2.2-mini \
  convert models/mini-nvfp4.mnw --experts nvfp4

# Ampere and newer: INT8 weights with BF16 activations.
target/release/minnow --model models/LLaDA2.2-mini \
  convert models/mini-int8.mnw --experts int8

target/release/minnow --model models/mini-nvfp4.mnw serve
```

NVFP4 uses E2M1 weights and activations, E4M3 scales per 16 values, and FP32
accumulation. INT8 uses groups of 128 weights with FP16 scales and dequantizes
into BF16 tensor-core registers. Both change model numerics; throughput results
are not evidence of equal answer quality. Convert from floating-point weights.
`--tensor-rules` permits mixed precision by layer or projection.
See [formats and conversion](docs/model-format.md).

NVFP4 uses fused gate/up/SiLU kernels and GPU routing for 32-token decode
blocks by default. `--host-routing` and `--unfused-activation` provide comparison
paths. Block FlashAttention improves long-prompt prefill by keeping attention scores
on chip and is the default on supported CUDA shapes. `--materialized-attention`
selects the previous attention path for numerical or performance comparisons.
Floating-point rounding differs between the two and can change generated responses.
See [optimization measurements](docs/nvfp4-optimizations.md).

## API and web UI

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the LHC?"}],"max_tokens":1024,"stream":true}'
```

The server supports Chat Completions and text Completions, function tools,
tool-result history, sampling controls, stop strings, usage, `/v1/models`,
`/health`, and llama-server-compatible `/props` and timing metadata. Responses
API and session resumption are not yet supported.

To use llama-server's web UI, build the UI from your llama.cpp checkout and pass
its output directory. Minnow serves those files directly; it does not bundle or
copy the UI. The UI is optional and no Docker or separate proxy is required.

```sh
target/release/minnow --model models/mini-nvfp4.mnw serve \
  --ui-dir ../llama.cpp/build/tools/ui/dist
```

`threshold`, `editing_threshold`, and `max_post_steps` can be set per request or
with `serve --threshold`, `--editing-threshold`, and `--max-post-steps`.
The full model context is exposed by default. `--max-context`, `--parallel`,
`--cache-slots`, and `--cache-max-mib` control request and cache capacity.
See [API, caching, and UI configuration](docs/api.md).

## Development and performance

```sh
cargo test --release
cargo test --release --features cuda
cargo test --release --features cuda --lib -- --ignored --test-threads=1
cargo clippy --all-targets --features cuda -- -D warnings
```

Ignored tests require compatible CUDA hardware; the NVFP4 tests require SM120/121.
Tests cover model/reference agreement, cache commits, decoding, container
integrity, and CUDA kernels against independent numerical oracles.

- [Implementation](docs/implementation.md)
- [Validation and benchmarking](docs/validation.md)
- [Measured quantization performance](docs/quantization-performance.md)

Pointwise, normalization, and routing kernels target `compute_80` by default;
`MINNOW_CUDA_ARCH` overrides their PTX target. Block FlashAttention uses
`compute_80`; Candle and cuBLAS build separately. NVFP4 builds as a
separate `compute_120f` module and always uses native Blackwell FP4 instructions.

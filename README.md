# minnow

Minnow is a Rust inference server for **LLaDA2.2-mini and LLaDA2.2-flash**. It
supports OpenAI Chat Completions, streaming, tool calling, conversation prefix
caching, continuous batching, and an optional llama-server web UI.

LLaDA generates by refining 32-token blocks. Minnow caches committed blocks and
recomputes only the current block during refinement. Completed blocks stream to
the client with prefill progress, useful token rates, and refinement statistics.

## Measured performance

### RTX 3090 (24 GiB)

LLaDA2.2-mini, one request at a time, CUDA 13, a 420 W GPU power limit, BF16 dense
layers, FlashAttention, and a 512 MiB workspace cache. Rates below are **tokens/second**:

| Expert execution | 4,096-token prefill | LHC decode | React TypeScript decode | Fellowship decode |
| --- | ---: | ---: | ---: | ---: |
| INT8 / BF16 activations | 13,094 | 210 | 667 | 151 |
| INT4 / INT8 activations | 15,667 | 238 | 697 | 194 |
| NVFP4 / BF16 tensor-core fallback | 11,578 | 133 | 300 | 108 |

Prefill averages ten samples and excludes loading and the vocabulary head.
Decode averages four warmed runs of each [prompt](tests/decode_natural_cases.json),
using default greedy diffusion settings and a 512-token output cap; it counts
generated text tokens and excludes prefill. Quantizations produce different
responses and refinement counts, so decode rates depend on both the prompt and
numerics. INT4 here uses `--int8-expert-activations`.
See [full RTX 3090 measurements and validation](docs/rtx3090.md).

The BF16 Transformers reference is not benchmarked on this GPU: its weights
alone require about 30.3 GiB, exceeding the 24 GiB VRAM before activations or K/V.
Layerwise reference checks fit, but do not measure end-to-end generation speed.

## Requirements

- Linux and Rust. CPU builds require no CUDA installation.
- For GPU acceleration: Git, CUDA 13 or later (`nvcc` on PATH, or set `NVCC`),
  and an NVIDIA GPU with BF16 support (Ampere or newer).
  NVFP4 uses native FP4 tensor cores on SM120/121 (RTX Blackwell or GB10),
  and a BF16 tensor-core fallback on other Ampere-or-newer GPUs.
- A local model checkpoint on a filesystem supporting direct I/O (`O_DIRECT`).

Only routed experts are quantized; attention, shared experts, embeddings, and the
output head retain their source precision. Approximate mini weight sizes are
30.3 GiB for BF16, 16.3 GiB for INT8, 9.1 GiB for INT4, and 9.8 GiB for NVFP4. Allow additional
memory for K/V, activations, and CUDA workspaces. See [memory configuration](docs/memory.md).

## Build and run

Download [LLaDA2.2-mini](https://huggingface.co/inclusionAI/LLaDA2.2-mini) or
[LLaDA2.2-flash](https://huggingface.co/inclusionAI/LLaDA2.2-flash), including its
configuration, tokenizer, and chat template. Pass the checkpoint location
explicitly with `--model`; there is no assumed installation directory.

Preconverted mini containers are available from
[coder543/LLaDA2.2-mini-minnow](https://huggingface.co/coder543/LLaDA2.2-mini-minnow):

```sh
hf download coder543/LLaDA2.2-mini-minnow \
  llada2.2-mini-bf16.mnw llada2.2-mini-int8.mnw llada2.2-mini-nvfp4.mnw \
  --local-dir models
```

```sh
cargo build --release --features cuda

target/release/minnow --model models/LLaDA2.2-mini \
  generate 'What is the LHC?' --max-tokens 1024

target/release/minnow --model models/LLaDA2.2-mini serve --listen 127.0.0.1:8080
```

The first CUDA build fetches pinned CUTLASS headers through CudaForge; subsequent
builds reuse its cache. Third-party notices are in `vendor/flash-attention/`.

Serving and checkpoint conversion do not require Python. For a CPU-only build:

```sh
cargo build --release

target/release/minnow --model models/mini-int8.mnw --device cpu serve
```

`--device auto` (the default) selects CUDA when available, otherwise CPU.
`--dtype auto` uses BF16 on CUDA and FP32 on CPU. A CUDA build still requires
its linked NVIDIA libraries to start; use the CPU build on systems without them.
CPU execution supports floating, INT4, INT8, and NVFP4 checkpoints with the same API,
caching, and streaming behavior, but is substantially slower than CUDA. Quantized
experts stay compressed; the CPU expands one selected expert at a time for GEMM.
Unquantized weights use FP32 on CPU, so budget twice their BF16 storage size,
plus expert scratch, K/V, and activations. CPU and CUDA results can differ through
floating-point rounding. Set `RAYON_NUM_THREADS` to limit CPU parallelism.

Candle's Apple Metal backend is not yet integrated into minnow. AMD/Intel GPU
acceleration is not supported by this build; those systems can use the CPU path.

## Quantization

Conversion produces a self-contained `.mnw` file with weights, tokenizer,
configuration, and chat template. It streams the source and does not load a
complete model into memory. Existing destinations are never overwritten.
The source may be a safetensors directory or a floating-point `.mnw` file.
Conversion overlaps direct reads, expert quantization, and ordered writes using
up to eight CPU workers by default. Use `convert --workers N` to set the worker
count (1–64). Memory is bounded by per-worker expert scratch and one queued
result per worker; large unquantized tensors are streamed. Worker count does
not change the checkpoint bytes.

```sh
# Blackwell: native FP4 tensor cores for both prefill and decode.
target/release/minnow --model models/LLaDA2.2-mini \
  convert models/mini-nvfp4.mnw --experts nvfp4

# Ampere and newer: INT8 weights with BF16 activations.
target/release/minnow --model models/LLaDA2.2-mini \
  convert models/mini-int8.mnw --experts int8

# Quantize directly from a self-contained BF16 checkpoint.
target/release/minnow --model models/llada2.2-mini-bf16.mnw \
  convert models/llada2.2-mini-int4.mnw --experts int4

target/release/minnow --model models/mini-nvfp4.mnw serve
```

NVFP4 uses E2M1 weights and activations, E4M3 scales per 16 values, and FP32
accumulation. INT4 and INT8 use groups of 128 weights with FP16 scales and dequantize
into BF16 tensor-core registers by default. All quantizations change model numerics; throughput results
are not evidence of equal answer quality. Convert from floating-point weights.
`--tensor-rules` permits mixed precision by layer or projection.
Use `minnow --model models/mini-int8.mnw validate` to verify checkpoint
checksums separately from loading. See [formats and conversion](docs/model-format.md).

`--int8-expert-activations` opts an instance with INT4/INT8 weights into W4A8/W8A8
integer tensor cores on SM80 and newer. It dynamically quantizes activations per
group, accumulates in INT32, then applies scales in FP32. This changes numerics
again; it is independent of checkpoint storage and leaves the default GB10 paths
unchanged. See [RTX 3090 measurements](docs/rtx3090.md).

INT4, INT8, and NVFP4 use fused gate/up/SiLU kernels and GPU routing for 32-token decode
blocks by default. `--host-routing` and `--unfused-activation` provide comparison
paths. Block FlashAttention improves long-prompt prefill by keeping attention scores
on chip and is the default on supported CUDA shapes. `--materialized-attention`
selects the previous attention path for numerical or performance comparisons.
Floating-point rounding differs between the two and can change generated responses.
See [NVFP4 measurements](docs/nvfp4-optimizations.md) and
[INT8 measurements and CPU fallback](docs/int8-optimizations.md).

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

Ignored tests require compatible CUDA hardware (SM80 or newer).
Tests cover model/reference agreement, cache commits, decoding, container
integrity, and CUDA kernels against independent numerical oracles.

- [Implementation](docs/implementation.md)
- [Validation and benchmarking](docs/validation.md)
- [Measured quantization performance](docs/quantization-performance.md)

Pointwise, normalization, and routing kernels target `compute_80` by default;
`MINNOW_CUDA_ARCH` overrides their PTX target. Block FlashAttention uses
`compute_80`; Candle and cuBLAS build separately. Native NVFP4 builds as a separate
`compute_120f` module. Its fallback and the opt-in integer kernels use separate
`compute_80` modules, selected independently of `MINNOW_CUDA_ARCH`.

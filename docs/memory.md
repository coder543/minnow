# Memory and checkpoint loading

Each minnow instance holds one resident weight set, shared across its requests.
Independent instances can coexist when memory permits. Checkpoint data is
read with Linux direct I/O into bounded staging buffers and final tensor
allocations. Loading does not use mmap or construct a complete CPU model before
copying it to CUDA. Reads and uploads overlap using two reusable batches of up
to 64 chunks, each at most 8 MiB. Buffers grow to fit the reads, with about 1 GiB
of maximum staging plus a separate 8 MiB conversion/validation buffer. CUDA
allocations start from metadata on a background stream and stay ahead of uploads;
these are the final weights, not extra copies. Matching-dtype CUDA uploads
write directly into final storage without a temporary device tensor. Checksums
are verified separately with `minnow --model CHECKPOINT.mnw validate`.
See [loader measurements](loading.md) for timings and profiling.
Use a filesystem that supports `O_DIRECT`; unsupported
filesystems produce an error rather than silently switching loading modes.

`--model` accepts a safetensors checkpoint directory or a self-contained `.mnw`
file. Conversion processes one expert matrix at a time and streams larger
unquantized tensors in 8 MiB chunks. `scripts/copy_checkpoint.py SOURCE DESTINATION`
can make a bounded-memory copy with SHA-256 verification when needed.

## Sizing a deployment

Approximate weight payloads, with only routed experts quantized:

| Model | BF16 | INT8 | NVFP4 |
| --- | ---: | ---: | ---: |
| Mini | 30.28 GiB | 16.25 GiB | 9.79 GiB |
| Flash | 191.65 GiB | 100.10 GiB | 57.96 GiB |

Weights are only part of the memory requirement. Reserve room for K/V,
activations, attention scores, the CUDA runtime, and allocator scratch. Mini's
BF16 K/V costs 40 KiB per token, or 5 GiB at 131,072 tokens:

```
20 layers × 2 (K,V) × 4 KV heads × 128 head dimension × 2 bytes
```

Flash uses 8 GiB of K/V at the same context length. Expert quantization does not
quantize the cache. Full-context memory estimates are not full-context throughput
measurements; current benchmark reports cover shorter contexts.

## Server controls

- `serve --max-context`: maximum prompt plus output budget per request; defaults
  to the checkpoint's full context.
- `serve --parallel`: active request limit, default four.
- `serve --cache-slots`: retained conversation prefix slots, default four.
- `serve --cache-max-mib`: aggregate K/V capacity across active and idle slots;
  defaults to one full context. Requests wait if their reservation cannot fit.
- `--workspace-cache-mib`: unused CUDA scratch retained for reuse, default 2048;
  zero disables retention. This is an allowance, not an eager allocation.
- `--prefill-chunk-tokens`: transformer prefill batch, default 4096, maximum 8192.
- `--attention-chunk-tokens`: attention query tile, default 1024; automatically
  reduced for long key sequences to bound score storage.

Cache allocation follows the prompt plus output reservation and grows in
2,048-token increments. Idle slots are evicted before allocation. Forks copy only
an exact committed prefix. During growth, one old layer's K/V may temporarily
remain allocated alongside its replacement. Attention reads committed K/V
through views. Four requests do not each reserve an entire maximum-context cache.

## Diagnostic resource controls

The loader checks available system memory before and during loading.
`--memory-reserve-mib` sets the host headroom it retains, default 16384 (16 GiB).
A deployment manager can choose a smaller reserve after budgeting for weights,
K/V, working allocations, and other resident processes. This is a load-time
check, not a runtime memory reservation or a replacement for GPU VRAM sizing
on discrete GPUs. There is no cross-process model lock.

For large benchmarks, `scripts/memory_guard.py` runs a command with explicit
system-memory growth and reserve budgets. It also monitors swap-out and Linux
memory-pressure stalls. Use `--help` for configurable limits. The report records
peak growth, pressure, and any abort reason. It measures system RAM, including
GPU allocations on unified-memory systems, but does not independently measure
discrete GPU VRAM.

Reference validation streams one layer at a time with
`scripts/layerwise_reference.py`. Complete generation comparisons use meta
initialization and direct parameter assignment in `scripts/reference_generate.py`.
Neither tool creates a second complete weight set. Run them sequentially with
serving stopped. See [validation](validation.md).

# Validation and benchmarking

Small checked-in fixtures exercise the checkpoint's model classes, block
attention, MoE routing, cache commits, and decoder edge cases. They do not need
trained weights or Python. CUDA tests compare fused kernels with independent
numerical oracles; NVFP4 tests require SM120/121 hardware.

```sh
cargo test --release
cargo test --release --features cuda
cargo test --release --features cuda --lib -- --ignored --test-threads=1
cargo clippy --all-targets --features cuda -- -D warnings
```

## Full-model measurements

Build the release binary before timing. Run one model at a time, with other
inference workloads stopped. `scripts/bench_models.py` launches a temporary
server, discards a warmup, and measures three repetitions at each prefill length
and for each long-response prompt. Requests disable prefix reuse. Decode reports
useful tokens, refinement counts, stopping reasons, and full response text.

```sh
python3 scripts/memory_guard.py \
  --report artifacts/benchmark-memory.json --max-growth-gib 30 --reserve-gib 8 -- \
  python3 scripts/bench_models.py --model models/mini-nvfp4.mnw \
  --report artifacts/benchmark.json
```

The budgets above are examples: size them for the checkpoint and host. The guard
measures system RAM; discrete GPU VRAM must also fit. `--prefill-only` and
`--decode-only` select parts of the suite. Avoid compiling or running another
benchmark during measurements. Loading and admission are excluded from reported
inference rates. Quantized outputs can require different numbers of refinements,
so useful decode rates are not identical-work kernel comparisons.

For focused measurements, `prefill-bench` accepts `--tokens` and `--chunk-sizes`;
`bench --cached-only` measures repeated current-block forwards against fixed K/V;
`decode-bench` accepts a JSON case file. Use each subcommand's `--help` for options.
`examples/bench_nvfp4.rs` measures native expert projections against BF16 at mini
and flash dimensions. `MINNOW_NVFP4_TILE_ROWS=16|32|64|128` selects a diagnostic
row tile; production selects measured defaults by matrix shape.
See the [NVFP4 profiling report](nvfp4-profile.md) for Nsight Systems and
Nsight Compute commands and the distinction between full-model and fixture
measurements.

## Independent reference

Python is an oracle, not a runtime dependency. The reference scripts require
PyTorch with CUDA, Transformers, NumPy, and safetensors, plus the checkpoint's
Python model files. The recorded reference environment uses PyTorch 2.10.0 and
Transformers 5.2.0. Run it sequentially with minnow stopped.

`scripts/layerwise_reference.py` loads one transformer layer at a time.
`scripts/reference_generate.py` uses meta initialization and direct parameter
assignment for one resident BF16 model. Both require an explicit `--model` path.
They never build a complete host model and then copy it to CUDA.

```sh
python3 scripts/layerwise_reference.py --model models/LLaDA2.2-mini \
  --dtype float32 --output artifacts/reference-f32

target/release/minnow --model models/LLaDA2.2-mini --dtype f32 \
  validate --reference artifacts/reference-f32 \
  --max-abs-error 0.001 --max-relative-error 0.00002
```

Use the memory guard with suitable budgets for these commands. FP32 mini weights
alone require about 60.6 GiB; the layerwise reference uses much less memory.
Numerical agreement verifies the implementation, not task accuracy. Evaluate
quantized model quality on representative tasks before selecting a deployment
precision.

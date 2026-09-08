# minnow contributor instructions

- Treat input checkpoints as read-only. Keep generated checkpoints in an explicit,
  user-selected model directory; do not assume a developer's home-directory layout.
- Keep one complete resident weight set across runtime and validation processes.
  Preserve the shared model lease. Small synthetic fixtures are exempt.
- Do not mmap checkpoint files. Use bounded direct reads into final allocations;
  do not construct a complete host model and then copy it to CUDA. Expert views
  must share their packed storage.
- Use `scripts/layerwise_reference.py` for numerical reference checks, holding one
  layer at a time. `scripts/reference_generate.py` uses meta initialization and
  direct assignment for one complete BF16 model. Do not introduce whole-model
  `from_pretrained(...).to('cuda')` validation paths that duplicate weights.
- Run large validations sequentially under `scripts/memory_guard.py` with explicit
  budgets appropriate for the model and host. Inspect peak-memory and pressure
  reports. Do not rely on swap or the OOM killer. Build before timing benchmarks.
- Establish numerical correctness before optimizing. FP32 register arithmetic is
  compatible with BF16 storage; prefer equivalent fused execution over separate
  intermediate tensor allocations.
- Keep the runtime in Rust. Python is an independent test oracle only.
- Keep user documentation and examples portable. Label benchmark hardware and
  settings explicitly; avoid embedding local deployment paths or incident history.

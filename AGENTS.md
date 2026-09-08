# minnow workspace instructions

- Treat `~/hf/inclusionAI/LLaDA2.2-mini` and its resolved checkpoint directory as
  read-only. Keep the working checkpoint under
  `~/models/hf/inclusionAI/LLaDA2.2-mini` on local NVMe, as requested by the user.
  Write other generated data under this workspace.
- The user requires only one complete resident copy of model weights across
  runtime and validation processes. Keep the shared model lock in place.
- Do not mmap checkpoint files. This machine handles file-backed mmap poorly.
  Use bounded direct reads into final allocations; do not pack a complete host
  model and then copy it to CUDA. Expert views must share their packed storage.
- Never restore complete-model `from_pretrained(...).to('cuda')` in validation.
  Use `scripts/layerwise_reference.py`; FP32 reference checks hold one layer at
  a time. Use native BF16 weights for routine full-model runs. A native FP32
  diagnostic may hold one 60.6 GiB weight copy, sequentially under the memory
  guard with a 72 GiB growth ceiling and at least 32 GiB system reserve.
  `scripts/reference_generate.py` uses meta initialization and direct assignment
  to hold one BF16 copy for comparison of complete generations.
- Run large validation commands sequentially under `scripts/memory_guard.py`.
  Inspect its peak-memory/swap report. Stop on memory pressure rather than
  relying on swap or the OOM killer. Small synthetic test fixtures are separate.
- Establish numerical correctness first, then optimize. FP32 arithmetic inside
  fused kernels is compatible with BF16 storage; avoid separate intermediate
  tensor allocations where equivalent fused execution is available.
- Continue building a Rust runtime. Python is only an independent test oracle.

# NVFP4 bottleneck profile

LLaDA2.2-mini on NVIDIA GB10, September 8, 2026. Nsight Systems 2025.3.2
measures the complete execution timeline; Nsight Compute 2025.3.1 measures
hardware counters. The model uses native NVFP4 for routed experts and source
precision elsewhere. [Counter data and source hashes](nvfp4-profile.json) and
[unprofiled throughput](quantization-performance.md) accompany this report.

Neither phase is explained by a single saturated resource. Prefill spends more
time on materialized attention than on NVFP4 expert GEMMs. Decode is strongly
affected by weight reads, but also has CPU routing and dispatch gaps. Native
GEMMs have low tensor-pipe activity and substantial load-dependency stalls.

## Whole-model timeline

Prefill uses cold prefixes, 4,096-token transformer batches and 1,024-query
attention tiles. Decode generates the complete LHC response: 352 text tokens,
222 refinement forwards, normal EOS, threshold 0.5, editing threshold 0, and
16 maximum post-steps. Loading and warmup are outside the captures.

| Component, percent of GPU kernel time | Prefill 4K | Prefill 8K | Decode LHC |
| --- | ---: | ---: | ---: |
| Native NVFP4 expert GEMMs | 24.5% | 18.4% | 49.3% |
| Attention QK + AV | 23.9% | 32.0% | Included in other |
| Attention softmax | 16.5% | 23.1% | 0.4% |
| Activation quantization | 1.5% | 1.1% | 0.8% |
| BF16 vocabulary projection | Omitted during prefill | Omitted during prefill | 18.1% |
| Other kernels | 33.6% | 25.4% | 31.7% |

At 4K, kernel time totals 322 ms over a 396 ms kernel span; at 8K, 874 ms over
1,021 ms. Decode totals 3,519 ms over 4,105 ms. Including memory copies, GPU
work occupies 81.8%, 86.0%, and 86.4% of those spans. The remainder includes
host work, synchronization and profiler overhead, rather than GPU arithmetic.
These are one capture each, not replacement throughput benchmarks.

The NVFP4 path currently bypasses `route_mini`, which serves BF16 experts.
It copies sigmoid router scores to the CPU, selects experts there, and uploads
assignments and GEMM descriptors. The decode trace contains 4,443 D2H calls,
approximately 20 per forward, and 25,986 H2D calls. Their GPU copy time is only
29.5 ms; synchronization and dispatch are the concern, not transfer bandwidth.
CPU API wait durations overlap GPU execution and must not be added to kernel
time as if they were independent costs.

## Hardware counters

The complete 4K prefill range averages 10.5% tensor-pipe active cycles and
18.6% SM throughput according to Nsight Compute. L2 fills from system memory
total 32.5 GB over 412 ms, or 78.8 GB/s. These heterogeneous phase averages
rule out sustained tensor-compute saturation across the whole phase; they do
not classify every individual attention kernel. Tensor-pipe activity is not
the fraction of peak FLOPs achieved.

For individual GEMMs, bounded synthetic fixtures call the production kernels.
There are 48 experts with either six or 128 assigned rows each. Gate/up has
N=512, K=2048; down has N=2048, K=512. The vocabulary fixture has M=32,
N=157184, K=2048 and selects the same cuBLAS kernel and grid as the decode trace.

| Fixture | Kernel time | System-memory fills | Fill rate | Tensor-pipe activity | Long-scoreboard stalls |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVFP4 gate/up, 6 rows/expert | 165 us | 28.68 MB | 174 GB/s | 2.0% | 98.2% |
| NVFP4 down, 6 rows/expert | 145 us | 28.45 MB | 196 GB/s | 2.2% | 95.0% |
| NVFP4 gate/up, 128 rows/expert | 344 us | 35.69 MB | 104 GB/s | 7.7% | 80.7% |
| NVFP4 down, 128 rows/expert | 400 us | 30.21 MB | 75.6 GB/s | 6.5% | 84.3% |
| BF16 vocabulary head | 2.88 ms | 644.57 MB | 224 GB/s | 5.8% | 65.7% |

Long-scoreboard stalls measure warps waiting for a dependency on a global/local
memory operation. They are not a percentage of wall time that can simply be
removed. Scheduler issue activity is only 5–6% in the NVFP4 decode fixtures and
16–17% in prefill fixtures, supporting load-latency/pipelining as an optimization
target. See NVIDIA's [metric interpretation guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

NVFP4 system-memory fills exceed the unique weights, quantized inputs, scales,
and descriptors by only 0.12–0.75% across these fixtures. There is no large
redundant system-memory weight-read multiplier to eliminate. Prefill does make
many repeated requests served by cache: gate/up requests 270 MB from L2 while
only 35.7 MB comes from system memory. Better reuse and overlapping loads with
MMA can still help. There is no expanded BF16/FP32 weight tensor; native FP4
operands feed block-scaled MMA directly.

The decode fixtures reach about 64–72% of the advertised
[273 GB/s memory bandwidth](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).
The vocabulary head reaches about 82%. These are read-fill rates, not total
memory-interface utilization; output writes are excluded, and advertised peak
is not a measured sustainable ceiling. Thus decode is memory-heavy, but this
does not establish that its entire duration is at the bandwidth limit.

## Optimization priorities

1. **Fuse prefill attention.** QK, softmax and AV together consume 40.4% of GPU
   kernel time at 4K and 55.1% at 8K. A FlashAttention-style path can avoid the
   score/probability buffers and separate passes. It must preserve the block
   mask and cache offsets, and validate numerical effects against the current
   BF16 rounding points. This is the largest measured prefill opportunity.
2. **Keep NVFP4 routing and descriptors on the GPU.** Extend the existing device
   routing machinery to quantized experts, including compact assignments and
   mixing. This removes per-layer CPU round trips and makes further launch
   reduction or CUDA graph capture more practical. It does not reduce the
   number of distinct expert weight matrices required.
3. **Pipeline native NVFP4 loads.** The current kernel loads fragments directly
   into registers in each K iteration, without an asynchronous shared-memory
   pipeline. Test prefetching/double buffering and larger reuse tiles against
   register pressure and occupancy. Merely increasing row tiles already had
   mixed results; there is no measured speedup for a new pipeline yet.
4. **Fuse gate/up and activation where useful.** This can share input loads,
   descriptors and launches, but both different weight matrices still have to
   be read. Quantization itself is below 1% of decode GPU time, so further
   quantizer fusion alone has a small ceiling.

The BF16 vocabulary head is a substantial remaining cost, but its measured
bandwidth is already higher than the native expert fixtures. Quantizing it is
a separate accuracy decision, not a prerequisite for the opportunities above.
All figures here concern mini; flash and long-context decode need separate
profiles before assigning them the same bottleneck proportions.

## Reproduction and limitations

Build release executables before profiling, stop other inference workloads,
and use the memory guard with budgets appropriate for the model and host.
Hardware-counter access may require administrator configuration.

```sh
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=repeat \
  -o artifacts/prefill \
  target/release/minnow --model models/mini-nvfp4.mnw prefill-bench \
  --input tests/prefill_benchmark_ids.json --tokens 4096,8192 \
  --iterations 1 --profile

nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o artifacts/decode \
  target/release/minnow --model models/mini-nvfp4.mnw decode-bench \
  --cases tests/decode_natural_cases.json --iterations 1 --profile

ncu --replay-mode app-range --cache-control none --clock-control none \
  --section SpeedOfLight \
  --metrics lts__d_sectors_fill_sysmem.sum,lts__t_sectors_op_read.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed \
  -o artifacts/prefill-counters \
  target/release/minnow --model models/mini-nvfp4.mnw prefill-bench \
  --input tests/prefill_benchmark_ids.json --tokens 4096 --iterations 1 --profile

cargo build --release --features cuda --example bench_nvfp4 --example profile_head
MINNOW_BENCH_INPUT=2048 MINNOW_BENCH_ROWS=6 MINNOW_BENCH_MODE=nvfp4 \
  ncu --replay-mode kernel --cache-control all --clock-control none \
  --kernel-name regex:minnow_nvfp4_gemm --launch-skip 5 --launch-count 1 \
  --metrics gpu__time_duration.sum,lts__d_sectors_fill_sysmem.sum,lts__t_sectors_op_read.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct \
  -o artifacts/expert-counters target/release/examples/bench_nvfp4
```

Use K=512 for down and 128 rows for the prefill fixture. `profile_head` runs
six projections; use the same kernel-replay command without the kernel-name
filter to capture the last one. Export reports with `ncu --import REPORT
--page raw --csv` or `nsys stats --report cuda_gpu_kern_sum,cuda_api_sum REPORT`.

Full-model counters use application-range replay, which reloads one model
serially without snapshotting its weights. Kernel replay is limited to bounded
synthetic fixtures because NVIDIA's replay can back up all accessible GPU
memory. If running the profiler under a different user, also hold the serving
user's model lease throughout the command. Full decode counter replay stalled
and was terminated; the whole-response findings use the valid Systems capture,
with Compute results explicitly restricted to fixtures.

Fixture counters flush caches between passes; the full prefill range does not.
The fixtures use uniform assignments and gathered inputs, whereas real prefill
uses 256 experts with irregular assignments and indexes gate/up inputs before
routing. Fixture counter times therefore differ from warmed microbenchmark or
serving times. Raw reports remain in `artifacts/nvfp4-profile-*` and
`artifacts/nvfp4-ncu-*`; the JSON preserves the selected measurements.

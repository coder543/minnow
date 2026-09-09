# Minnow model container v1

The `.mnw` file is self-contained. It preserves source config/tokenizer/template
text and places a bounded MessagePack manifest after aligned tensor regions.
MessagePack was chosen for a small, inspectable schema without a code generator;
weights stay outside the manifest and stream through bounded O_DIRECT reads
into final allocations. The runtime and converter do not mmap checkpoints.

## Header and manifest

All integers and tensor elements are little-endian. The first 64 bytes are:

| Offset | Bytes | Meaning |
| --- | --- | --- |
| 0 | 8 | ASCII `MINNOW01` |
| 8 | 8 | Manifest offset |
| 16 | 8 | Manifest byte length |
| 24 | 8 | Exact file length |
| 32 | 32 | BLAKE3 digest of serialized manifest |

Tensor data begins at offset 4096. Every code, scale, and manifest region begins
on a 4096-byte boundary. Padding is zero. The manifest ends exactly at EOF and
is limited to 128 MiB. The header, bounds, shape/encoding lengths, region overlap,
and tensor metadata are checked before loading. Normal loading does not compute
checksums. Use `minnow --model CHECKPOINT.mnw validate` to verify the manifest
and every weight/scale payload with bounded host memory and no GPU allocation.

The named-map MessagePack manifest has fields `version` (1), `architecture`
(`llada2_moe`), `model_id`, `assets`, and `tensors`. Required assets are
`config.json`, `tokenizer.json`, and `chat_template.jinja`. Optional source
`tokenizer_config.json`, `special_tokens_map.json`, and `generation_config.json`
are also retained. Each tensor has its original name, `shape`, `encoding`,
`data` region, optional `scales` region, and `group_size` (zero for floating data).
A region has `offset`, `bytes`, and a 64-character hexadecimal BLAKE3 digest.

Unknown encodings/versions are rejected. Original precision containers retain
each tensor's exact source bytes and mixed floating dtypes.

## Encodings

| Name | Values | Scale/layout |
| --- | --- | --- |
| `bf16`, `f16`, `f32` | Standard floating-point bits | Original row-major storage |
| `i8_sym` | Signed INT8, range -127…127 | FP16 scales, row-major |
| `i8_mma` | Same INT8 codes | Tensor-core fragment order |
| `i4_sym` | Signed INT4, range -7…7, two's complement, low nibble first | FP16 scales, row-major |
| `i4_mma` | Same INT4 codes | Tensor-core fragment order |
| `nvfp4` | E2M1 codes | E4M3/16 block scales, FP32 outer scale, native K64 fragment order |

Integer groups run along the input dimension and contain 16, 32, 64, or 128
weights (default 128). Each group stores an FP16 scale rounded from
`max_abs/127` for INT8 or `max_abs/7` for INT4; zero groups use one. Nonzero
scales are clamped to the smallest positive FP16 value before rounding. Codes
use the stored scale and nearest-even rounding, clamped to the symmetric range.
Decoders also accept the representable -128/-8 codes. Scales must be positive and finite.

Packed matrices require N divisible by 8 and K divisible by 16. For each N8/K16
tile, lane `l` contains four codes at row `l/4`, columns
`2*(l%4) + [0,1,8,9]`. They occupy four consecutive INT8 bytes or two INT4 bytes,
with consecutive pairs in the low/high nibbles.
Tile order is `[N/8][K/16][lane]`. FP16 scales use `[N/8][K/group][8]` order.
Packed experts concatenate directly without runtime repacking or expansion.
This permutation changes no quantized value or scale.

## Conversion and mixed precision

`convert OUTPUT --experts int4|int8|nvfp4` accepts a safetensors directory or a
floating-point `.mnw` input and defaults to the packed layout. Use
`--quant-layout row` for plain INT4/INT8 layout. `--experts original` preserves source
precision/layout. A quantized input can be copied or losslessly repacked to the
same codebook/group size; changing its precision requires the original source.
For example, an earlier row-layout container can be repacked on local SSD:

```sh
minnow --model mini-row.mnw convert mini-packed.mnw --experts int8
```

`--tensor-rules FILE.json` is an array of prefix rules. Later matches win; a null
encoding preserves the source. Rules must match at least one tensor. All experts
within one layer/projection must share the same encoding and group size.

```json
[
  {"prefix":"model.layers.1.mlp.experts.","encoding":null},
  {"prefix":"model.layers.2.mlp.experts.","encoding":"i8_mma","group_size":128}
]
```

Only routed expert matrices may currently be quantized. Other tensors retain
their original dtype. The converter handles one expert at a time (limited to
16M elements); large unquantized tensors stream in 8 MiB chunks. It synchronizes
the output and publishes atomically without overwriting existing files. Original
checkpoints are read-only inputs.

CUDA's default integer path dequantizes into registers and uses BF16 tensor-core
operands with FP32 accumulation. `--int8-expert-activations` selects signed INT8
tensor-core operands on SM80+, for either INT4 or INT8 weights. Each input row
is quantized independently per weight group: FP32 scale `max_abs/127` (one for
zero groups, at least the smallest normal FP32 value otherwise), nearest-even
signed codes clamped to -127…127. Each group's dot accumulates exactly in INT32,
then contributes `float(dot) * (activation_scale * weight_scale)` to the FP32
result, which rounds to BF16 at projection output. INT4 expands only in registers.
This opt-in execution choice is per model instance, not checkpoint metadata.

NVFP4 uses native FP4 tensor cores on SM120/121 and BF16 tensor cores elsewhere.
None of these paths stores an
expanded model in system/GPU RAM. CPU execution expands one selected expert at
a time into an FP32 GEMM buffer. Conversion retains source precision for shared
experts, attention, routers, embeddings, and the output head; the CPU runtime
loads those unquantized weights as FP32.

## Native NVFP4

`nvfp4` uses NVIDIA's E2M1/E4M3 block-scaled arithmetic, with a minnow-specific
fragment storage permutation. It is not a drop-in reader for other vendors'
checkpoint files. Each expert matrix has a positive finite `global_scale` in
the manifest. The stored value is `E2M1(code) * E4M3(block_scale) * global_scale`.
Block scales occupy one byte per 16 weights (nonnegative finite E4M3 codes 0–126).
The outer scale is `max_abs / (6 * 448)`, clamped to positive normal FP32;
all-zero matrices use one. Block scales round `(block_max / 6) / global_scale`
to nearest-even E4M3, with a minimum nonzero code for nonzero blocks. Codes round
using the stored scales. Zero blocks use scale one. No calibration is required.

Matrices require N divisible by 8 and K divisible by 64. Codes are packed as
`[N/8][K/64][register=2][lane=32]` of little-endian u32 values. Lane `l` reads
row `l/4`, columns `8*(l%4)+[0..7]`, then the same columns plus 32. Scales use
`[N/8][K/64][row=8][group=4]` byte order. Experts concatenate directly; the loader
streams into final U8 code/scale allocations and one small FP32 outer-scale
vector. Copying an NVFP4 container preserves every code and scale exactly.
Requantization and `--quant-layout row` are rejected for NVFP4 conversion.

On SM120/121, activations use the same block scheme with a dynamic FP32 outer
scale **per token**, so unrelated rows do not alter a token's quantization.
Gate/up quantize each original token once and use routing indices to share it
across expert assignments. Their input is never gathered into duplicated BF16
rows. Specialized quantizers retain inputs in registers across the scale
reduction. The down projection quantizes its own expert-specific input.
The native `mma.sync...m16n8k64...e2m1...ue4m3` instruction consumes FP4 operands
and E4M3 scales directly. FP32 accumulators are multiplied by both outer scales
and rounded to BF16. No dequantized weights are written to memory. This path
quantizes both weights and activations.

The separate `compute_120f` PTX module requires CUDA 13 and SM120/121. On other
SM80+ GPUs, a separate `compute_80` module stores quantized activation block
operands as exact BF16 values, decodes packed weights into registers, and uses
BF16 MMA with FP32 accumulation. It retains the E2M1/E4M3 quantization and outer
scales; only the tensor-core instruction and activation scratch layout differ.
Native GB10 dispatch and kernels are preserved. The slower CPU software path supports serving and
also provides an independent numerical oracle. See [CPU execution](int8-optimizations.md#cpu-execution).
GPU tests compare activation codes/scales exactly with
the scalar encoder, then compare native MMA against independently dequantized
operands, including uneven/repeated expert segments and all supported row tiles.

References: [NVIDIA NVFP4 format](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
and [PTX MMA/block scaling](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma).

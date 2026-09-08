# Minnow model container v1

The `.mnw` file is self-contained. It preserves source config/tokenizer/template
text and places a bounded MessagePack manifest after aligned tensor regions.
MessagePack was chosen for a small, inspectable schema without a code generator;
weights stay outside the manifest so readers can use direct I/O or mmap views.
The current runtime and converter use O_DIRECT, not mmap.

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
| `nvfp4` | E2M1 codes | E4M3/16 block scales, FP32 outer scale, native K64 fragment order |

INT8 groups run along the input dimension and contain 16, 32, 64, or 128
weights (default 128). Each group stores an FP16 scale rounded from
`max_abs/127`; zero groups use one. Codes use the stored scale and nearest-even
rounding. Scales must be positive and finite.

Packed matrices require N divisible by 8 and K divisible by 16. For each N8/K16
tile, lane `l` contains four codes at row `l/4`, columns
`2*(l%4) + [0,1,8,9]`. They occupy four consecutive INT8 bytes.
Tile order is `[N/8][K/16][lane]`. FP16 scales use `[N/8][K/group][8]` order.
Packed experts concatenate directly without runtime repacking or expansion.
This permutation changes no quantized value or scale.

## Conversion and mixed precision

`convert OUTPUT --experts int8|nvfp4` defaults to the packed layout. Use
`--quant-layout row` for plain INT8 layout. `--experts original` preserves source
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

CUDA's INT8 path dequantizes into registers and uses BF16 tensor-core operands
with FP32 accumulation. NVFP4 uses native FP4 tensor cores. Neither stores an
expanded model in system/GPU RAM. CPU execution expands one selected expert for
small-fixture tests. Shared experts, attention, routers, embeddings, and the
output head retain source precision.

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

The separate `compute_120f` PTX module requires CUDA 13 and SM120/121. There is
no CUDA emulation fallback. CPU execution is an independent small-fixture oracle,
not a production backend. GPU tests compare activation codes/scales exactly with
the scalar encoder, then compare native MMA against independently dequantized
operands, including uneven/repeated expert segments and all supported row tiles.

References: [NVIDIA NVFP4 format](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
and [PTX MMA/block scaling](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma).

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
and manifest digest are checked before loading; payload digests are checked
during direct reads into final allocations.

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
| `fp4_e2m1` | E2M1, low nibble first | FP16 scales, row-major |
| `i8_mma` | Same INT8 codes | Tensor-core fragment order |
| `fp4_mma` | Same E2M1 codes | Tensor-core fragment order |

Groups run along the input dimension and may contain 16, 32, 64, or 128 weights.
Defaults are INT8/128 and FP4/32. Each group uses an FP16 scale rounded from
`max_abs/127` or `max_abs/6`. Zero groups use scale one. Codes are selected using
the stored scale and nearest-even rounding. E2M1 magnitudes are
`[0, 0.5, 1, 1.5, 2, 3, 4, 6]`; bit 3 is sign. This is a custom FP4 format,
not NVFP4 or MXFP4 wire compatibility. Scales must be positive and finite.

Packed matrices require N divisible by 8 and K divisible by 16. For each N8/K16
tile, lane `l` contains four codes at row `l/4`, columns
`2*(l%4) + [0,1,8,9]`. They occupy four consecutive INT8 bytes or two FP4 bytes.
Tile order is `[N/8][K/16][lane]`. FP16 scales use `[N/8][K/group][8]` order.
Packed experts concatenate directly without runtime repacking or expansion.
This permutation changes no quantized value or scale.

## Conversion and mixed precision

`convert OUTPUT --experts int8|fp4` defaults to the packed layout. Use
`--quant-layout row` for the plain layout. `--experts original` preserves source
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

CUDA's normal quantized path dequantizes directly into registers, converts
operands to BF16, and uses FP32 tensor-core accumulators. No expanded model is
stored in system/GPU RAM. A CPU fallback expands one selected expert for testing.
Tests cover exact original bytes/metadata, missing external assets, checksum and
truncation rejection, mixed-precision execution, lossless repacking, and GPU
results against separately decoded BF16 operands.

INT8 was selected for the 3090 target rather than depending on native FP8
instructions: Ampere provides BF16 and INT8 tensor cores. This W8A16 path uses
the BF16 form after register dequantization, preserving BF16 activations.
See NVIDIA's [Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).

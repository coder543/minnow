//! Expert quantization: INT8 weight-only and native NVFP4 weight/activation formats.
use crate::container::Encoding;
use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Tensor};
use half::{bf16, f16};
pub mod nvfp4;

pub struct Weights {
    pub codes: Tensor,
    pub scales: Tensor,
    pub encoding: Encoding,
    pub group_size: usize,
    pub shape: (usize, usize, usize),
    pub global_scales: Option<Tensor>,
}
impl Weights {
    pub fn grouped(&self, x: &Tensor, segments: &[(usize, usize)]) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if x.device().is_cuda() {
            if self.encoding == Encoding::Nvfp4 {
                return Ok(crate::cuda::nvfp4::grouped(x, self, segments)?);
            }
            return Ok(crate::cuda::quantized::grouped(x, self, segments)?);
        }
        if self.encoding == Encoding::Nvfp4 {
            return nvfp4::cpu_grouped(x, self, segments);
        }
        // CPU fallback expands one selected expert, never a whole model. This
        // path is an oracle/small-fixture fallback, not the CUDA serving path.
        let mut outputs = Vec::new();
        let (_, out, input) = self.shape;
        let elements = out * input;
        let bytes = elements;
        let scale_count = elements / self.group_size;
        let mut row = 0;
        for &(e, n) in segments {
            ensure!(
                e < self.shape.0 && n > 0 && row + n <= x.dim(0)?,
                "invalid expert segment"
            );
            let codes = self.codes.narrow(0, e * bytes, bytes)?.to_vec1::<u8>()?;
            let scales = self
                .scales
                .narrow(0, e * scale_count, scale_count)?
                .to_vec1::<f16>()?;
            let scales: Vec<u8> = scales
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect();
            let values = decode_matrix(&codes, &scales, self.encoding, self.group_size, input)?;
            // Match the BF16 tensor-core operand conversion, including for F32
            // fixture activations, so this is an independent CUDA oracle.
            let weight = Tensor::from_vec(values, (out, input), x.device())?
                .to_dtype(DType::BF16)?
                .to_dtype(x.dtype())?;
            outputs.push(x.narrow(0, row, n)?.matmul(&weight.t()?)?);
            row += n;
        }
        ensure!(row == x.dim(0)?, "expert segments do not cover input");
        Ok(Tensor::cat(&outputs, 0)?)
    }
}

pub fn encode_parallel(
    values: &[f32],
    encoding: Encoding,
    group: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    use rayon::prelude::*;
    if values.len() <= 32768 {
        return encode(values, encoding, group);
    }
    let chunks: Result<Vec<_>> = values
        .par_chunks(32768)
        .map(|v| encode(v, encoding, group))
        .collect();
    let chunks = chunks?;
    let mut codes = Vec::with_capacity(chunks.iter().map(|c| c.0.len()).sum());
    let mut scales = Vec::with_capacity(chunks.iter().map(|c| c.1.len()).sum());
    for (c, s) in chunks {
        codes.extend(c);
        scales.extend(s);
    }
    Ok((codes, scales))
}

pub fn decode_float_bytes(bytes: &[u8], encoding: Encoding) -> Result<Vec<f32>> {
    Ok(match encoding {
        Encoding::Bf16 => {
            ensure!(bytes.len().is_multiple_of(2), "truncated BF16 data");
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_bits(u16::from_le_bytes(*b)).to_f32())
                .collect()
        }
        Encoding::F16 => {
            ensure!(bytes.len().is_multiple_of(2), "truncated F16 data");
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f16::from_bits(u16::from_le_bytes(*b)).to_f32())
                .collect()
        }
        Encoding::F32 => {
            ensure!(bytes.len().is_multiple_of(4), "truncated F32 data");
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        }
        _ => bail!("expected unquantized floating data"),
    })
}
pub fn encode(values: &[f32], encoding: Encoding, group: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    ensure!(
        encoding == Encoding::I8Sym
            && [16, 32, 64, 128].contains(&group)
            && values.len().is_multiple_of(group),
        "invalid quantization shape/encoding"
    );
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "cannot quantize nonfinite weights"
    );
    let mut codes = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(values.len() / group * 2);
    for values in values.chunks_exact(group) {
        let maximum = values.iter().map(|v| v.abs()).fold(0., f32::max);
        let scale = if maximum == 0. {
            f16::ONE
        } else {
            f16::from_f32((maximum / 127.).max(f16::from_bits(1).to_f32()))
        };
        ensure!(
            scale.is_finite() && scale.to_f32() > 0.,
            "weight scale cannot be represented in FP16"
        );
        scales.extend_from_slice(&scale.to_bits().to_le_bytes());
        let scale = scale.to_f32();
        codes.extend(
            values
                .iter()
                .map(|v| (v / scale).round_ties_even().clamp(-127., 127.) as i8 as u8),
        );
    }
    Ok((codes, scales))
}

pub fn decode(codes: &[u8], scales: &[u8], encoding: Encoding, group: usize) -> Result<Vec<f32>> {
    ensure!(
        encoding == Encoding::I8Sym && [16, 32, 64, 128].contains(&group),
        "invalid quantized encoding/group"
    );
    let count = codes.len();
    ensure!(
        count.is_multiple_of(group) && scales.len() == count / group * 2,
        "invalid quantized payload length"
    );
    (0..count)
        .map(|i| {
            let s = i / group * 2;
            let scale = f16::from_bits(u16::from_le_bytes([scales[s], scales[s + 1]])).to_f32();
            ensure!(scale.is_finite() && scale > 0., "invalid quantized scale");
            let value = codes[i] as i8 as f32;
            Ok(value * scale)
        })
        .collect()
}

/// Lossless layout conversion. Each N8/K16 tile holds four weights per lane:
/// row=lane/4, columns=2*(lane%4)+[0,1,8,9]. Scales use [N8][K/group][8].
/// Multiple experts can be flattened along N because every expert is N8 aligned.
pub fn repack(
    codes: &[u8],
    scales: &[u8],
    from: Encoding,
    to: Encoding,
    group: usize,
    input: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    ensure!(
        from.int8() && to.int8(),
        "repacking must preserve the quantized codebook"
    );
    let count = codes.len();
    ensure!(
        [16, 32, 64, 128].contains(&group)
            && input > 0
            && input.is_multiple_of(group)
            && count.is_multiple_of(input)
            && scales.len() == count / group * 2,
        "invalid repacking dimensions"
    );
    if from == to {
        return Ok((codes.to_vec(), scales.to_vec()));
    }
    let out = count / input;
    ensure!(
        out.is_multiple_of(8),
        "packed quantization requires eight output rows per tile"
    );
    let mut next_codes = vec![0; codes.len()];
    let mut next_scales = vec![0; scales.len()];
    let bytes = 4;
    for n in (0..out).step_by(8) {
        for k in (0..input).step_by(16) {
            for lane in 0..32 {
                let packed = ((n / 8 * (input / 16) + k / 16) * 32 + lane) * bytes;
                let row = (n + lane / 4) * input + k + (lane % 4) * 2;
                for pair in 0..2 {
                    let row_byte = row + pair * 8;
                    let width = bytes / 2;
                    let packed_byte = packed + pair * width;
                    let (source, dest) = if to.packed() {
                        (row_byte, packed_byte)
                    } else {
                        (packed_byte, row_byte)
                    };
                    next_codes[dest..dest + width].copy_from_slice(&codes[source..source + width]);
                }
            }
        }
        for k in 0..input / group {
            for row in 0..8 {
                let plain = ((n + row) * (input / group) + k) * 2;
                let packed = ((n / 8 * (input / group) + k) * 8 + row) * 2;
                let (source, dest) = if to.packed() {
                    (plain, packed)
                } else {
                    (packed, plain)
                };
                next_scales[dest..dest + 2].copy_from_slice(&scales[source..source + 2]);
            }
        }
    }
    Ok((next_codes, next_scales))
}

pub fn encode_matrix(
    values: &[f32],
    encoding: Encoding,
    group: usize,
    input: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let (codes, scales) = encode_parallel(values, encoding.row_major(), group)?;
    if encoding.packed() {
        repack(
            &codes,
            &scales,
            encoding.row_major(),
            encoding,
            group,
            input,
        )
    } else {
        Ok((codes, scales))
    }
}

pub fn decode_matrix(
    codes: &[u8],
    scales: &[u8],
    encoding: Encoding,
    group: usize,
    input: usize,
) -> Result<Vec<f32>> {
    if encoding.packed() {
        let (codes, scales) = repack(codes, scales, encoding, encoding.row_major(), group, input)?;
        decode(&codes, &scales, encoding.row_major(), group)
    } else {
        decode(codes, scales, encoding, group)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fragment_repacking_preserves_every_code_and_scale() -> Result<()> {
        for packed in [Encoding::I8Mma] {
            for group in [16, 32, 64, 128] {
                let input = 256;
                let values: Vec<f32> = (0..24 * input)
                    .map(|i| ((i * 13 % 157) as f32 - 78.) / 97.)
                    .collect();
                let (codes, scales) = encode(&values, packed.row_major(), group)?;
                let (p, s) = repack(&codes, &scales, packed.row_major(), packed, group, input)?;
                let (round_codes, round_scales) =
                    repack(&p, &s, packed, packed.row_major(), group, input)?;
                assert_eq!(codes, round_codes);
                assert_eq!(scales, round_scales);
            }
        }
        Ok(())
    }
    #[test]
    fn int8_quantization_bounds_error() -> Result<()> {
        let values: Vec<f32> = (0..512)
            .map(|i| ((i * 31 % 257) as f32 - 128.) / 100.)
            .collect();
        let (codes, scales) = encode(&values, Encoding::I8Sym, 128)?;
        let decoded = decode(&codes, &scales, Encoding::I8Sym, 128)?;
        assert!(
            values
                .iter()
                .zip(decoded)
                .all(|(a, b)| (a - b).abs() < 0.0051)
        );
        assert!(encode(&[f32::NAN; 32], Encoding::I8Sym, 32).is_err());
        assert!(decode(&[0; 32], &[0, 0], Encoding::I8Sym, 32).is_err());
        Ok(())
    }
}

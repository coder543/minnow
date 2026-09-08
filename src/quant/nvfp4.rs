//! NVFP4 reference conversion and lossless native-mma packing.
use super::*;
const FP4: [f32; 8] = [0., 0.5, 1., 1.5, 2., 3., 4., 6.];

pub fn e4m3(code: u8) -> f32 {
    let e = (code >> 3) & 15;
    if code >= 127 {
        return f32::NAN;
    }
    if e == 0 {
        (code & 7) as f32 / 512.
    } else {
        f32::from_bits(((e as u32 + 120) << 23) | ((code as u32 & 7) << 20))
    }
}
fn scale_code(value: f32) -> u8 {
    // Independent scalar nearest-even conversion, including E4M3 subnormals.
    let mut lo = 0u8;
    let mut hi = 126u8;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if e4m3(mid) < value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == 0 {
        return 0;
    }
    let lower = lo - 1;
    let a = (value - e4m3(lower)).abs();
    let b = (e4m3(lo) - value).abs();
    if a < b || (a == b && lower.is_multiple_of(2)) {
        lower
    } else {
        lo
    }
}
fn fp4_code(value: f32) -> u8 {
    let a = value.abs();
    let code = if a <= 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5. {
        6
    } else {
        7
    };
    code | if value.is_sign_negative() { 8 } else { 0 }
}
/// Plain E2M1 nibbles/E4M3 scales, and a multiplicative FP32 tensor scale.
pub fn encode(values: &[f32]) -> Result<(Vec<u8>, Vec<u8>, f32)> {
    ensure!(
        !values.is_empty()
            && values.len().is_multiple_of(16)
            && values.iter().all(|v| v.is_finite()),
        "invalid NVFP4 input"
    );
    let max = values.iter().map(|v| v.abs()).fold(0., f32::max);
    let global = if max == 0. {
        1.
    } else {
        (max / 2688.).max(f32::MIN_POSITIVE)
    };
    let mut codes = vec![0; values.len() / 2];
    let mut scales = vec![0; values.len() / 16];
    let encode_chunk = |((codes, scales), values): ((&mut [u8], &mut [u8]), &[f32])| {
        for (group, values) in values.as_chunks::<16>().0.iter().enumerate() {
            let max = values.iter().map(|v| v.abs()).fold(0., f32::max);
            let scale = if max == 0. {
                56
            } else {
                scale_code((max / 6.) / global).max(1)
            };
            scales[group] = scale;
            let divisor = e4m3(scale) * global;
            for (i, pair) in values.as_chunks::<2>().0.iter().enumerate() {
                codes[group * 8 + i] =
                    fp4_code(pair[0] / divisor) | (fp4_code(pair[1] / divisor) << 4);
            }
        }
    };
    if values.len() > 65536 {
        use rayon::prelude::*;
        codes
            .par_chunks_mut(8192)
            .zip(scales.par_chunks_mut(1024))
            .zip(values.par_chunks(16384))
            .for_each(encode_chunk);
    } else {
        encode_chunk(((&mut codes, &mut scales), values));
    }
    Ok((codes, scales, global))
}
pub fn decode(codes: &[u8], scales: &[u8], global: f32) -> Result<Vec<f32>> {
    ensure!(
        codes.len().is_multiple_of(8)
            && scales.len() == codes.len() / 8
            && global.is_finite()
            && global > 0.
            && scales.iter().all(|s| *s < 127),
        "invalid NVFP4 payload/scales"
    );
    Ok((0..codes.len() * 2)
        .map(|i| {
            let code = (codes[i / 2] >> (i % 2 * 4)) & 15;
            let v = FP4[(code & 7) as usize] * if code & 8 == 0 { 1. } else { -1. };
            (v * e4m3(scales[i / 16])) * global
        })
        .collect())
}
/// Weights: [N/8][K/64][register 2][lane 32] u32 fragments.
/// Scales: [N/8][K/64][row 8][group 4] bytes. No numerical conversion.
pub fn pack(
    codes: &[u8],
    scales: &[u8],
    input: usize,
    inverse: bool,
) -> Result<(Vec<u8>, Vec<u8>)> {
    ensure!(
        input > 0
            && input.is_multiple_of(64)
            && (codes.len() * 2).is_multiple_of(input)
            && (codes.len() * 2 / input).is_multiple_of(8)
            && scales.len() == codes.len() / 8,
        "NVFP4 packing requires N8/K64 matrices"
    );
    let out = codes.len() * 2 / input;
    let mut c = vec![0; codes.len()];
    let mut s = vec![0; scales.len()];
    for n in (0..out).step_by(8) {
        for k in (0..input).step_by(64) {
            let tile = (n / 8 * (input / 64) + k / 64) * 256;
            for lane in 0..32 {
                for r in 0..2 {
                    let plain = ((n + lane / 4) * input + k + lane % 4 * 8 + r * 32) / 2;
                    let packed = tile + (r * 32 + lane) * 4;
                    let (src, dst) = if inverse {
                        (packed, plain)
                    } else {
                        (plain, packed)
                    };
                    c[dst..dst + 4].copy_from_slice(&codes[src..src + 4]);
                }
            }
            for row in 0..8 {
                let plain = ((n + row) * input + k) / 16;
                let packed = (n / 8 * (input / 64) + k / 64) * 32 + row * 4;
                let (src, dst) = if inverse {
                    (packed, plain)
                } else {
                    (plain, packed)
                };
                s[dst..dst + 4].copy_from_slice(&scales[src..src + 4]);
            }
        }
    }
    Ok((c, s))
}
pub fn quantize_rows(x: &Tensor) -> Result<Tensor> {
    let (rows, k) = x.dims2()?;
    let values = x.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let mut result = Vec::with_capacity(rows * k);
    for row in values {
        let (c, s, g) = encode(&row)?;
        result.extend(decode(&c, &s, g)?);
    }
    Ok(Tensor::from_vec(result, (rows, k), x.device())?)
}
pub(super) fn cpu_grouped(x: &Tensor, w: &Weights, segments: &[(usize, usize)]) -> Result<Tensor> {
    use candle_core::Storage;
    use rayon::prelude::*;
    let (_, out, input) = w.shape;
    ensure!(
        input.is_multiple_of(64)
            && input > 0
            && out.is_multiple_of(8)
            && x.dim(1)? == input
            && w.codes.is_contiguous()
            && w.scales.is_contiguous(),
        "invalid CPU NVFP4 shape/layout"
    );
    let globals = w
        .global_scales
        .as_ref()
        .context("missing NVFP4 tensor scales")?
        .to_vec1::<f32>()?;
    ensure!(
        globals.len() == w.shape.0 && globals.iter().all(|g| g.is_finite() && *g > 0.),
        "invalid NVFP4 tensor scales"
    );
    let (codes, cl) = w.codes.storage_and_layout();
    let (scales, sl) = w.scales.storage_and_layout();
    let (Storage::Cpu(codes), Storage::Cpu(scales)) = (&*codes, &*scales) else {
        anyhow::bail!("CPU quantized weights must reside on CPU");
    };
    let codes =
        &codes.as_slice::<u8>()?[cl.start_offset()..cl.start_offset() + cl.shape().elem_count()];
    let scales =
        &scales.as_slice::<u8>()?[sl.start_offset()..sl.start_offset() + sl.shape().elem_count()];
    ensure!(
        codes.len() == w.shape.0 * out * input / 2 && scales.len() == codes.len() / 8,
        "invalid CPU NVFP4 payload"
    );
    let selected = quantize_rows(x)?;
    let mut result = Vec::new();
    let mut row = 0;
    for &(e, n) in segments {
        ensure!(
            e < w.shape.0 && n > 0 && row + n <= x.dim(0)?,
            "invalid expert segment"
        );
        let c = &codes[e * out * input / 2..(e + 1) * out * input / 2];
        let s = &scales[e * out * input / 16..(e + 1) * out * input / 16];
        ensure!(s.iter().all(|s| *s < 127), "invalid NVFP4 scales");
        let mut values = vec![0f32; out * input];
        values
            .par_chunks_mut(input)
            .enumerate()
            .for_each(|(r, values)| {
                for (k, value) in values.iter_mut().enumerate() {
                    let tile = r / 8 * (input / 64) + k / 64;
                    let ci =
                        tile * 256 + (k % 64 / 32 * 32 + r % 8 * 4 + k % 32 / 8) * 4 + k % 8 / 2;
                    let code = (c[ci] >> (k % 2 * 4)) & 15;
                    let v = FP4[(code & 7) as usize] * if code & 8 == 0 { 1. } else { -1. };
                    *value = (v * e4m3(s[tile * 32 + r % 8 * 4 + k % 64 / 16])) * globals[e];
                }
            });
        let weight = Tensor::from_vec(values, (out, input), x.device())?;
        result.push(
            selected
                .narrow(0, row, n)?
                .matmul(&weight.t()?)?
                .to_dtype(x.dtype())?,
        );
        row += n;
    }
    ensure!(row == x.dim(0)?, "expert segments do not cover input");
    Ok(Tensor::cat(&result, 0)?)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nvfp4_scale_rounding_packing_and_error() -> Result<()> {
        for code in 0..127 {
            assert_eq!(scale_code(e4m3(code)), code);
        }
        for code in 0..126 {
            let mid = (e4m3(code) + e4m3(code + 1)) / 2.;
            assert_eq!(
                scale_code(mid),
                if code.is_multiple_of(2) {
                    code
                } else {
                    code + 1
                }
            );
        }
        let values: Vec<_> = (0..24 * 192)
            .map(|i| ((i * 31 % 257) as f32 - 128.) / 113.)
            .collect();
        let (c, s, g) = encode(&values)?;
        let (p, t) = pack(&c, &s, 192, false)?;
        assert_eq!(pack(&p, &t, 192, true)?, (c.clone(), s.clone()));
        let d = decode(&c, &s, g)?;
        assert!(
            values
                .iter()
                .zip(d)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                / (values.len() as f32)
                < 0.01
        );
        let (c, s, g) = encode(&[0.; 64])?;
        assert_eq!(decode(&c, &s, g)?, vec![0.; 64]);
        assert!(encode(&[f32::NAN; 16]).is_err());
        assert!(decode(&[0; 8], &[127], 1.).is_err());
        Ok(())
    }
}

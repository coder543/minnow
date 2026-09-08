//! Grouped W8A16/W4A16 tensor-core GEMM. Expanded weights live only in a
//! register file (or shared tiles in the diagnostic WMMA path), never global storage.
use crate::quant::Weights;
use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp3, Layout, Result, Shape, Tensor};
use half::{bf16, f16};

struct Gemm<'a> {
    weight: &'a Weights,
    segments: &'a [(usize, usize)],
}
impl CustomOp3 for Gemm<'_> {
    fn name(&self) -> &'static str {
        "minnow-quantized-expert-gemm"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("quantized tensor-core GEMM requires CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        w: &CudaStorage,
        wl: &Layout,
        s: &CudaStorage,
        sl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (experts, out, input) = self.weight.shape;
        let (rows, k) = xl.shape().dims2()?;
        let divisor = if self.weight.encoding.int8() { 1 } else { 2 };
        let group = self.weight.group_size;
        if !self.weight.encoding.quantized()
            || !xl.is_contiguous()
            || !wl.is_contiguous()
            || !sl.is_contiguous()
            || !xl.start_offset().is_multiple_of(2)
            || !wl
                .start_offset()
                .is_multiple_of(if self.weight.encoding.packed() {
                    4 / divisor
                } else {
                    2 / divisor
                })
            || ![16, 32, 64, 128].contains(&group)
            || !input.is_multiple_of(group)
            || k != input
            || rows == 0
            || out == 0
            || (self.weight.encoding.packed() && !out.is_multiple_of(8))
            || input == 0
            || self.segments.iter().any(|&(e, n)| e >= experts || n == 0)
            || self
                .segments
                .iter()
                .try_fold(0usize, |sum, (_, n)| sum.checked_add(*n))
                != Some(rows)
            || wl.shape().elem_count() != experts * out * input / divisor
            || sl.shape().elem_count() != experts * out * input / group
            || [rows, out, input, experts]
                .iter()
                .any(|n| *n > i32::MAX as usize)
        {
            candle_core::bail!("invalid quantized GEMM shape, layout or segments");
        }
        let dev = &x.device;
        let stream = dev.cuda_stream();
        let xs = x
            .as_cuda_slice::<bf16>()?
            .slice(xl.start_offset()..xl.start_offset() + rows * input);
        let ws = w
            .as_cuda_slice::<u8>()?
            .slice(wl.start_offset()..wl.start_offset() + wl.shape().elem_count());
        let ss = s
            .as_cuda_slice::<f16>()?
            .slice(sl.start_offset()..sl.start_offset() + sl.shape().elem_count());
        let wmma = !self.weight.encoding.packed()
            && std::env::var("MINNOW_QUANT_KERNEL").as_deref() == Ok("wmma");
        let tile_rows = if wmma {
            32
        } else {
            std::env::var("MINNOW_QUANT_TILE_ROWS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|n| [32, 64, 128].contains(n))
                .unwrap_or(32)
        };
        let mut descriptors = Vec::<i32>::new();
        let mut start = 0;
        for &(expert, n) in self.segments {
            for row in (0..n).step_by(tile_rows) {
                descriptors.extend_from_slice(&[
                    expert as i32,
                    (start + row) as i32,
                    (n - row).min(tile_rows) as i32,
                ]);
            }
            start += n;
        }
        let tiles = descriptors.len() / 3;
        let descriptors = stream.clone_htod(&descriptors).w()?;
        // SAFETY: descriptors cover all rows exactly, and the kernel bounds N.
        let mut output = unsafe { dev.alloc::<bf16>(rows * out)? };
        let bits = if self.weight.encoding.int8() { 8 } else { 4 };
        let name = if wmma {
            format!("minnow_quant_gemm_{bits}")
        } else if self.weight.encoding.packed() {
            format!("minnow_quant_packed_{bits}_{tile_rows}")
        } else {
            format!("minnow_quant_mma_{bits}_{tile_rows}")
        };
        let f = dev.get_or_load_custom_func(
            &name,
            "minnow-v1",
            include_str!(concat!(env!("OUT_DIR"), "/minnow.ptx")),
        )?;
        let (input, out_dim, group) = (input as i32, out as i32, group as i32);
        let mut call = f.builder();
        call.arg(&xs)
            .arg(&ws)
            .arg(&ss)
            .arg(&descriptors)
            .arg(&mut output)
            .arg(&input)
            .arg(&out_dim)
            .arg(&group);
        // SAFETY: validated shapes/indices, live buffers, same Candle stream.
        unsafe {
            call.launch(LaunchConfig {
                grid_dim: (out.div_ceil(64) as u32, tiles as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, dev.clone()),
            Shape::from((rows, out)),
        ))
    }
}
pub fn grouped(x: &Tensor, w: &Weights, segments: &[(usize, usize)]) -> Result<Tensor> {
    if !x.device().same_device(w.codes.device()) || !x.device().same_device(w.scales.device()) {
        candle_core::bail!("quantized GEMM buffers must share a device");
    }
    x.apply_op3_no_bwd(
        &w.codes,
        &w.scales,
        &Gemm {
            weight: w,
            segments,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::Encoding;
    use candle_core::{DType, Device};
    #[test]
    #[ignore = "requires CUDA"]
    fn tensor_core_quantized_experts_match_independent_dequantization() -> anyhow::Result<()> {
        let dev = Device::new_cuda(0)?;
        // Uneven expert rows, repeated expert IDs, N tail, K tile tail, and
        // all scale-group sizes exercise layout and padding independently.
        for encoding in [
            Encoding::I8Sym,
            Encoding::Fp4E2m1,
            Encoding::I8Mma,
            Encoding::Fp4Mma,
        ] {
            for group_size in [16, 32, 64, 128] {
                let (experts, out, input) = (3, 80, if group_size == 16 { 48 } else { 128 });
                let values: Vec<f32> = (0..experts * out * input)
                    .map(|i| ((i * 17 % 257) as f32 - 128.) / 97.)
                    .collect();
                let (plain_codes, plain_scales) =
                    crate::quant::encode(&values, encoding.row_major(), group_size)?;
                let (codes, scale_bytes) = crate::quant::repack(
                    &plain_codes,
                    &plain_scales,
                    encoding.row_major(),
                    encoding,
                    group_size,
                    input,
                )?;
                let scales: Vec<f16> = scale_bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f16::from_bits(u16::from_le_bytes(*b)))
                    .collect();
                let rows = 75;
                let x = Tensor::from_vec(
                    (0..rows * input)
                        .map(|i| ((i * 13 % 101) as f32 - 50.) / 59.)
                        .collect::<Vec<_>>(),
                    (rows, input),
                    &dev,
                )?
                .to_dtype(DType::BF16)?;
                let w = Weights {
                    codes: Tensor::from_vec(codes.clone(), codes.len(), &dev)?,
                    scales: Tensor::from_vec(scales.clone(), scales.len(), &dev)?,
                    encoding,
                    group_size,
                    shape: (experts, out, input),
                };
                let segments = [(2, 1), (0, 33), (2, 9), (1, 32)];
                let actual = grouped(&x, &w, &segments)?
                    .to_dtype(DType::F32)?
                    .to_vec2::<f32>()?;
                let decoded = crate::quant::decode(
                    &plain_codes,
                    &plain_scales,
                    encoding.row_major(),
                    group_size,
                )?;
                let full = Tensor::from_vec(decoded, (experts, out, input), &dev)?
                    .to_dtype(DType::BF16)?;
                let mut row = 0;
                for (e, n) in segments {
                    let expected = x
                        .narrow(0, row, n)?
                        .matmul(&full.get(e)?.t()?)?
                        .to_dtype(DType::F32)?
                        .to_vec2::<f32>()?;
                    for (a, b) in actual[row..row + n]
                        .iter()
                        .flatten()
                        .zip(expected.iter().flatten())
                    {
                        assert!(
                            (a - b).abs() <= 0.001 + 0.008 * b.abs(),
                            "{encoding:?} group{group_size}: {a} vs {b}"
                        );
                    }
                    row += n;
                }
                assert!(grouped(&x, &w, &[(3, 75)]).is_err());
                // Register loads read BF16 pairs; reject a contiguous but
                // misaligned view instead of launching an invalid CUDA load.
                let odd_x = Tensor::zeros(rows * input + 1, DType::BF16, &dev)?
                    .narrow(0, 1, rows * input)?
                    .reshape((rows, input))?;
                assert!(grouped(&odd_x, &w, &segments).is_err());
                if encoding.int8() || encoding.packed() {
                    let codes = Tensor::zeros(codes.len() + 1, DType::U8, &dev)?.narrow(
                        0,
                        1,
                        codes.len(),
                    )?;
                    let odd_w = Weights {
                        codes,
                        scales: w.scales.clone(),
                        encoding,
                        group_size,
                        shape: w.shape,
                    };
                    assert!(grouped(&x, &odd_w, &segments).is_err());
                }
            }
        }
        Ok(())
    }
}

//! Native block-scaled FP4 MMA on SM120/121. Activations are quantized once per
//! projection input, and the gate/up pair shares that temporary allocation.
use crate::{container::Encoding, quant::Weights};
use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{
    CpuStorage, CudaStorage, CustomOp1, CustomOp3, DType, Device, Layout, Result, Shape, Storage,
    Tensor,
};
use half::bf16;
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/minnow-nvfp4.ptx"));
pub fn check_device(device: &Device) -> Result<()> {
    let Device::Cuda(device) = device else {
        candle_core::bail!("native NVFP4 requires CUDA");
    };
    let capability = device.cuda_stream().context().compute_capability().w()?;
    if capability.0 != 12 || ![0, 1].contains(&capability.1) {
        candle_core::bail!("native NVFP4 kernels require SM120/121 (RTX Blackwell or GB10)");
    }
    Ok(())
}
struct Quantize;
impl CustomOp1 for Quantize {
    fn name(&self) -> &'static str {
        "minnow-nvfp4-quantize"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("requires CUDA")
    }
    fn cuda_fwd(&self, x: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)> {
        let (rows, input) = l.shape().dims2()?;
        if rows == 0
            || rows > 65536
            || input == 0
            || input > 16384
            || !input.is_multiple_of(64)
            || !l.is_contiguous()
            || !l.start_offset().is_multiple_of(2)
        {
            candle_core::bail!("invalid NVFP4 activation shape/layout");
        }
        let dev = &x.device;
        let xs = x
            .as_cuda_slice::<bf16>()?
            .slice(l.start_offset()..l.start_offset() + rows * input);
        let bytes = rows * input * 9 / 16 + rows * 4;
        // SAFETY: the kernel writes every code, scale and FP32 outer scale.
        let mut output = unsafe { dev.alloc::<u8>(bytes)? };
        let name = if [512, 1024, 2048, 4096].contains(&input) {
            format!("minnow_nvfp4_quantize_{input}")
        } else {
            "minnow_nvfp4_quantize".into()
        };
        let f = dev.get_or_load_custom_func(&name, "minnow-nvfp4-v1", PTX)?;
        let (k, m) = (input as i32, rows as i32);
        let mut call = f.builder();
        call.arg(&xs).arg(&mut output).arg(&k).arg(&m);
        // SAFETY: a CTA owns one checked input row and its disjoint output regions.
        unsafe {
            call.launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, dev.clone()),
            Shape::from(bytes),
        ))
    }
}
struct Gemm<'a> {
    w: &'a Weights,
    segments: &'a [(usize, usize)],
    rows: usize,
    source_rows: usize,
    indices: Option<&'a [u32]>,
    tile: usize,
}
impl CustomOp3 for Gemm<'_> {
    fn name(&self) -> &'static str {
        "minnow-native-nvfp4-gemm"
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
        candle_core::bail!("requires CUDA")
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
        let (experts, out, input) = self.w.shape;
        let rows = self.rows;
        let source_rows = self.source_rows;
        if self.w.encoding != Encoding::Nvfp4
            || self.w.group_size != 16
            || experts == 0
            || experts > 1024
            || out == 0
            || out > 65536
            || !out.is_multiple_of(8)
            || input == 0
            || input > 16384
            || !input.is_multiple_of(64)
            || rows == 0
            || rows > 65536
            || source_rows == 0
            || source_rows > 65536
            || self.indices.is_some_and(|ids| {
                ids.len() != rows || ids.iter().any(|&i| i as usize >= source_rows)
            })
            || ![xl, wl, sl]
                .iter()
                .all(|l| l.is_contiguous() && l.start_offset().is_multiple_of(4))
            || xl.shape().elem_count() != source_rows * input * 9 / 16 + source_rows * 4
            || wl.shape().elem_count() != experts * out * input / 2
            || sl.shape().elem_count() != experts * out * input / 16
            || self.segments.iter().any(|&(e, n)| e >= experts || n == 0)
            || self
                .segments
                .iter()
                .try_fold(0usize, |sum, (_, n)| sum.checked_add(*n))
                != Some(rows)
        {
            candle_core::bail!("invalid NVFP4 weights, shape, layout or expert segments");
        }
        let globals = self
            .w
            .global_scales
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("missing NVFP4 global scales".into()))?;
        if globals.dims() != [experts] || globals.dtype() != DType::F32 || !globals.is_contiguous()
        {
            candle_core::bail!("invalid NVFP4 global scales");
        }
        let (gs, gl) = globals.storage_and_layout();
        let Storage::Cuda(gs) = &*gs else {
            candle_core::bail!("NVFP4 scales must be on CUDA");
        };
        let dev = &x.device;
        let stream = dev.cuda_stream();
        let xs = x
            .as_cuda_slice::<u8>()?
            .slice(xl.start_offset()..xl.start_offset() + xl.shape().elem_count());
        let ws = w
            .as_cuda_slice::<u8>()?
            .slice(wl.start_offset()..wl.start_offset() + wl.shape().elem_count());
        let ss = s
            .as_cuda_slice::<u8>()?
            .slice(sl.start_offset()..sl.start_offset() + sl.shape().elem_count());
        let gs = gs
            .as_cuda_slice::<f32>()?
            .slice(gl.start_offset()..gl.start_offset() + experts);
        let mut descriptors = Vec::<i32>::new();
        let mut start = 0;
        for &(e, n) in self.segments {
            for row in (0..n).step_by(self.tile) {
                descriptors.extend_from_slice(&[
                    e as i32,
                    (start + row) as i32,
                    (n - row).min(self.tile) as i32,
                ]);
            }
            start += n;
        }
        let tiles = descriptors.len() / 3;
        if let Some(indices) = self.indices {
            descriptors.extend(indices.iter().map(|&i| i as i32));
        }
        let descriptors = stream.clone_htod(&descriptors).w()?;
        // SAFETY: disjoint descriptors cover all rows; kernel bounds output columns.
        let mut output = unsafe { dev.alloc::<bf16>(rows * out)? };
        let f = dev.get_or_load_custom_func(
            &format!("minnow_nvfp4_gemm_{}", self.tile),
            "minnow-nvfp4-v1",
            PTX,
        )?;
        let (k, n, m) = (input as i32, out as i32, source_rows as i32);
        let indexed = i32::from(self.indices.is_some());
        let mut call = f.builder();
        call.arg(&xs)
            .arg(&ws)
            .arg(&ss)
            .arg(&gs)
            .arg(&descriptors)
            .arg(&mut output)
            .arg(&k)
            .arg(&n)
            .arg(&m)
            .arg(&indexed);
        // SAFETY: validated buffers, alignment, descriptor indices, and N8/K64 shapes.
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
fn run(
    a: &Tensor,
    source_rows: usize,
    w: &Weights,
    segments: &[(usize, usize)],
    indices: Option<&[u32]>,
) -> Result<Tensor> {
    let rows = indices.map_or(source_rows, |v| v.len());
    for t in [
        &w.codes,
        &w.scales,
        w.global_scales.as_ref().unwrap_or(&w.scales),
    ] {
        if !a.device().same_device(t.device()) {
            candle_core::bail!("NVFP4 buffers must share a device");
        }
    }
    let tile = std::env::var("MINNOW_NVFP4_TILE_ROWS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| [16, 32, 64, 128].contains(v))
        .unwrap_or(if w.shape.2 >= 2048 || rows / segments.len().max(1) <= 16 {
            16
        } else {
            32
        });
    a.apply_op3_no_bwd(
        &w.codes,
        &w.scales,
        &Gemm {
            w,
            segments,
            rows,
            source_rows,
            indices,
            tile,
        },
    )
}
pub fn grouped(x: &Tensor, w: &Weights, segments: &[(usize, usize)]) -> Result<Tensor> {
    check_device(x.device())?;
    let a = x.apply_op1_no_bwd(&Quantize)?;
    run(&a, x.dim(0)?, w, segments, None)
}
pub fn grouped_pair_indexed(
    x: &Tensor,
    a: &Weights,
    b: &Weights,
    segments: &[(usize, usize)],
    indices: &[u32],
) -> Result<(Tensor, Tensor)> {
    check_device(x.device())?;
    if a.shape != b.shape {
        candle_core::bail!("NVFP4 paired projections require equal shapes");
    }
    let q = x.apply_op1_no_bwd(&Quantize)?;
    Ok((
        run(&q, x.dim(0)?, a, segments, Some(indices))?,
        run(&q, x.dim(0)?, b, segments, Some(indices))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::nvfp4 as reference;
    #[test]
    #[ignore = "requires SM120/121"]
    fn native_mma_matches_independent_nvfp4_operands() -> anyhow::Result<()> {
        let dev = Device::new_cuda(0)?;
        check_device(&dev)?;
        for input in [192, 512, 1024, 2048, 4096] {
            let (experts, out, rows) = (3, 80, 299);
            let mut codes = Vec::new();
            let mut scales = Vec::new();
            let mut globals = Vec::new();
            let mut weights = Vec::new();
            for e in 0..experts {
                let v: Vec<_> = (0..out * input)
                    .map(|i| ((i * 17 % 257) as f32 - 128.) / 113. * 2f32.powi(e as i32 - 1))
                    .collect();
                let (c, s, g) = reference::encode(&v)?;
                weights.extend(reference::decode(&c, &s, g)?);
                let (c, s) = reference::pack(&c, &s, input, false)?;
                codes.extend(c);
                scales.extend(s);
                globals.push(g);
            }
            let w = Weights {
                codes: Tensor::from_vec(codes, experts * out * input / 2, &dev)?,
                scales: Tensor::from_vec(scales, experts * out * input / 16, &dev)?,
                global_scales: Some(Tensor::from_vec(globals, experts, &dev)?),
                encoding: Encoding::Nvfp4,
                group_size: 16,
                shape: (experts, out, input),
            };
            let x = Tensor::from_vec(
                (0..rows * input)
                    .map(|i| {
                        if i / input % 11 == 0 {
                            0.
                        } else {
                            ((i * 13 % 101) as f32 - 50.) / 59.
                                * 2f32.powi((i / input % 9) as i32 - 4)
                        }
                    })
                    .collect::<Vec<_>>(),
                (rows, input),
                &dev,
            )?
            .to_dtype(DType::BF16)?;
            let quantized = x.apply_op1_no_bwd(&Quantize)?;
            let actual = quantized.to_vec1::<u8>()?;
            let original = x.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            let mut decoded = Vec::new();
            for (row, values) in original.iter().enumerate() {
                let (c, s, g) = reference::encode(values)?;
                assert_eq!(
                    &actual[row * input / 2..(row + 1) * input / 2],
                    &c,
                    "activation codes row {row}"
                );
                assert_eq!(
                    &actual[rows * input / 2 + row * input / 16
                        ..rows * input / 2 + (row + 1) * input / 16],
                    &s,
                    "activation scales row {row}"
                );
                let offset = rows * input * 9 / 16 + row * 4;
                assert_eq!(&actual[offset..offset + 4], &g.to_le_bytes());
                decoded.extend(reference::decode(&c, &s, g)?);
            }
            let decoded = Tensor::from_vec(decoded, (rows, input), &dev)?;
            let weights = Tensor::from_vec(weights, (experts, out, input), &dev)?;
            let segments = [(2, 1), (0, 129), (2, 9), (1, 160)];
            // Routing can reuse one token across several experts. Quantizing
            // before this gather must preserve every resulting projection.
            let indices: Vec<u32> = (0..rows).map(|i| ((i * 17) % 71) as u32).collect();
            let selected = x.index_select(&Tensor::from_slice(&indices, rows, &dev)?, 0)?;
            let expected = grouped(&selected, &w, &segments)?.to_vec2::<bf16>()?;
            let (gate, up) = grouped_pair_indexed(&x, &w, &w, &segments, &indices)?;
            assert_eq!(gate.to_vec2::<bf16>()?, expected);
            assert_eq!(up.to_vec2::<bf16>()?, expected);
            assert!(grouped_pair_indexed(&x, &w, &w, &segments, &vec![rows as u32; rows]).is_err());
            for tile in [16, 32, 64, 128] {
                let actual = quantized
                    .apply_op3_no_bwd(
                        &w.codes,
                        &w.scales,
                        &Gemm {
                            w: &w,
                            segments: &segments,
                            rows,
                            source_rows: rows,
                            indices: None,
                            tile,
                        },
                    )?
                    .to_dtype(DType::F32)?
                    .to_vec2::<f32>()?;
                let mut row = 0;
                for (e, n) in segments {
                    let expected = decoded
                        .narrow(0, row, n)?
                        .matmul(&weights.get(e)?.t()?)?
                        .to_vec2::<f32>()?;
                    for (a, b) in actual[row..row + n]
                        .iter()
                        .flatten()
                        .zip(expected.iter().flatten())
                    {
                        assert!(
                            (a - b).abs() <= 0.002 + 0.008 * b.abs(),
                            "tile{tile}: {a} vs {b}"
                        );
                    }
                    row += n;
                }
            }
            assert!(grouped(&x, &w, &[(3, rows)]).is_err());
            let odd = Tensor::zeros(rows * input + 1, DType::BF16, &dev)?
                .narrow(0, 1, rows * input)?
                .reshape((rows, input))?;
            assert!(grouped(&odd, &w, &segments).is_err());
        }

        Ok(())
    }
}

//! Grouped W8A16 tensor-core GEMM. Expanded weights live only in a
//! register file (or shared tiles in the diagnostic WMMA path), never global storage.
use crate::quant::Weights;
use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp3, Layout, Result, Shape, Storage, Tensor};
use half::{bf16, f16};

struct Gemm<'a> {
    weight: &'a Weights,
    segments: &'a [(usize, usize)],
    indices: Option<&'a [u32]>,
    pair: Option<&'a Weights>,
    device_plan: Option<&'a super::CompactRoutingPlan>,
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
        let (source_rows, k) = xl.shape().dims2()?;
        let rows = if self.device_plan.is_some() {
            256
        } else {
            self.indices.map_or(source_rows, |v| v.len())
        };
        let group = self.weight.group_size;
        if !self.weight.encoding.int8()
            || !xl.is_contiguous()
            || !wl.is_contiguous()
            || !sl.is_contiguous()
            || !xl.start_offset().is_multiple_of(2)
            || !wl
                .start_offset()
                .is_multiple_of(if self.weight.encoding.packed() { 4 } else { 2 })
            || ![16, 32, 64, 128].contains(&group)
            || !input.is_multiple_of(group)
            || k != input
            || rows == 0
            || out == 0
            || (self.weight.encoding.packed() && !out.is_multiple_of(8))
            || input == 0
            || self.segments.iter().any(|&(e, n)| e >= experts || n == 0)
            || (self.device_plan.is_none()
                && self
                    .segments
                    .iter()
                    .try_fold(0usize, |sum, (_, n)| sum.checked_add(*n))
                    != Some(rows))
            || self.indices.is_some_and(|ids| {
                ids.len() != rows || ids.iter().any(|&i| i as usize >= source_rows)
            })
            || (self.device_plan.is_some() && (experts != 256 || ![32, 256].contains(&source_rows)))
            || wl.shape().elem_count() != experts * out * input
            || sl.shape().elem_count() != experts * out * input / group
            || [rows, source_rows, out, input, experts]
                .iter()
                .any(|n| *n > i32::MAX as usize)
        {
            candle_core::bail!("invalid quantized GEMM shape, layout or segments");
        }
        let dev = &x.device;
        let stream = dev.cuda_stream();
        let xs = x
            .as_cuda_slice::<bf16>()?
            .slice(xl.start_offset()..xl.start_offset() + source_rows * input);
        let ws = w
            .as_cuda_slice::<u8>()?
            .slice(wl.start_offset()..wl.start_offset() + wl.shape().elem_count());
        let ss = s
            .as_cuda_slice::<f16>()?
            .slice(sl.start_offset()..sl.start_offset() + sl.shape().elem_count());
        let wmma = self.pair.is_none()
            && self.indices.is_none()
            && self.device_plan.is_none()
            && !self.weight.encoding.packed()
            && std::env::var("MINNOW_QUANT_KERNEL").as_deref() == Ok("wmma");
        let tile_rows = if self.device_plan.is_some() {
            16
        } else if wmma {
            32
        } else {
            std::env::var("MINNOW_QUANT_TILE_ROWS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|n| [16, 32, 64, 128].contains(n))
                .unwrap_or(if self.weight.encoding.packed() {
                    16
                } else {
                    32
                })
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
        if let Some(indices) = self.indices {
            descriptors.extend(indices.iter().map(|&i| i as i32));
        }
        let host_descriptors = if self.device_plan.is_none() {
            Some(stream.clone_htod(&descriptors).w()?)
        } else {
            None
        };
        let plan_storage = self.device_plan.map(|p| p.descriptors.storage_and_layout());
        let device_descriptors = if let Some((storage, layout)) = &plan_storage {
            let Storage::Cuda(storage) = &**storage else {
                candle_core::bail!("plan must be on CUDA");
            };
            Some(
                storage
                    .as_cuda_slice::<f32>()?
                    .slice(layout.start_offset()..layout.start_offset() + 544),
            )
        } else {
            None
        };
        let tiles = if self.device_plan.is_some() {
            96
        } else {
            tiles
        };
        if let Some(b) = self.pair
            && (b.shape != self.weight.shape
                || b.encoding != self.weight.encoding
                || b.group_size != group
                || b.codes.elem_count() != wl.shape().elem_count()
                || b.scales.elem_count() != sl.shape().elem_count()
                || !b.codes.is_contiguous()
                || !b.scales.is_contiguous()
                || !b
                    .codes
                    .layout()
                    .start_offset()
                    .is_multiple_of(if b.encoding.packed() { 4 } else { 2 }))
        {
            candle_core::bail!("invalid INT8 paired weights");
        }
        let paired_storage = self
            .pair
            .map(|b| (b.codes.storage_and_layout(), b.scales.storage_and_layout()));
        let paired_views = if let Some(((bc, cl), (bs, sl))) = &paired_storage {
            let (Storage::Cuda(bc), Storage::Cuda(bs)) = (&**bc, &**bs) else {
                candle_core::bail!("paired weights must be on CUDA");
            };
            Some((
                bc.as_cuda_slice::<u8>()?
                    .slice(cl.start_offset()..cl.start_offset() + cl.shape().elem_count()),
                bs.as_cuda_slice::<f16>()?
                    .slice(sl.start_offset()..sl.start_offset() + sl.shape().elem_count()),
            ))
        } else {
            None
        };
        // SAFETY: descriptors cover all rows exactly, and the kernel bounds N.
        let mut output = unsafe { dev.alloc::<bf16>(rows * out)? };
        let bits = 8;
        let extended = self.pair.is_some() || self.indices.is_some() || self.device_plan.is_some();
        let name = if extended {
            format!(
                "minnow_quant_{}_{}_{tile_rows}",
                if self.pair.is_some() {
                    "pair"
                } else {
                    "indexed"
                },
                if self.weight.encoding.packed() {
                    "packed"
                } else {
                    "mma"
                }
            )
        } else if wmma {
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
        call.arg(&xs).arg(&ws).arg(&ss);
        if let Some(p) = &device_descriptors {
            call.arg(p);
        } else {
            call.arg(host_descriptors.as_ref().unwrap());
        }
        call.arg(&mut output).arg(&input).arg(&out_dim).arg(&group);
        let indexed =
            i32::from(self.indices.is_some() || (self.device_plan.is_some() && source_rows == 32));
        if extended {
            call.arg(&indexed);
        }
        if let Some((bc, bs)) = &paired_views {
            call.arg(bc).arg(bs);
        }
        // SAFETY: validated shapes/indices, live buffers, same Candle stream.
        unsafe {
            call.launch(LaunchConfig {
                grid_dim: (out.div_ceil(64) as u32, tiles as u32, 1),
                block_dim: (if self.pair.is_some() { 256 } else { 128 }, 1, 1),
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
    x: &Tensor,
    w: &Weights,
    segments: &[(usize, usize)],
    indices: Option<&[u32]>,
    pair: Option<&Weights>,
    device_plan: Option<&super::CompactRoutingPlan>,
) -> Result<Tensor> {
    for weight in std::iter::once(w).chain(pair) {
        if !x.device().same_device(weight.codes.device())
            || !x.device().same_device(weight.scales.device())
        {
            candle_core::bail!("quantized GEMM buffers must share a device");
        }
    }
    if device_plan.is_some_and(|p| !x.device().same_device(p.descriptors.device())) {
        candle_core::bail!("routing plan must share the input device");
    }
    x.apply_op3_no_bwd(
        &w.codes,
        &w.scales,
        &Gemm {
            weight: w,
            segments,
            indices,
            pair,
            device_plan,
        },
    )
}
pub fn grouped(x: &Tensor, w: &Weights, segments: &[(usize, usize)]) -> Result<Tensor> {
    run(x, w, segments, None, None, None)
}
pub fn grouped_silu_indexed(
    x: &Tensor,
    gate: &Weights,
    up: &Weights,
    segments: &[(usize, usize)],
    indices: &[u32],
) -> Result<Tensor> {
    run(x, gate, segments, Some(indices), Some(up), None)
}
pub fn routed(x: &Tensor, w: &Weights, plan: &super::CompactRoutingPlan) -> Result<Tensor> {
    run(x, w, &[], None, None, Some(plan))
}
pub fn routed_silu(
    x: &Tensor,
    gate: &Weights,
    up: &Weights,
    plan: &super::CompactRoutingPlan,
) -> Result<Tensor> {
    run(x, gate, &[], None, Some(up), Some(plan))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::Encoding;
    use candle_core::{DType, Device};
    #[test]
    #[ignore = "requires CUDA"]
    fn compact_device_route_matches_host_projections_and_mix() -> anyhow::Result<()> {
        let dev = Device::new_cuda(0)?;
        let make_weights = |out, input, salt| -> anyhow::Result<Weights> {
            let mut codes = Vec::new();
            let mut scales = Vec::new();

            for e in 0..256 {
                let v: Vec<_> = (0..out * input)
                    .map(|i| ((i * 17 + e * 31 + salt) % 257) as f32 / 257. - 0.5)
                    .collect();
                let (c, s) = crate::quant::encode_matrix(&v, Encoding::I8Mma, 64, input)?;
                codes.extend(c);
                scales.extend(
                    s.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|v| f16::from_bits(u16::from_le_bytes(*v))),
                );
            }
            Ok(Weights {
                codes: Tensor::from_vec(codes, 256 * out * input, &dev)?,
                scales: Tensor::from_vec(scales, 256 * out * input / 64, &dev)?,
                global_scales: None,
                encoding: Encoding::I8Mma,
                group_size: 64,
                shape: (256, out, input),
            })
        };
        let w = make_weights(128, 192, 0)?;
        let up_weights = make_weights(128, 192, 79)?;
        let down = make_weights(192, 128, 17)?;
        let x = Tensor::from_vec(
            (0..32 * 192)
                .map(|i| ((i * 13 % 101) as f32 - 50.) / 59.)
                .collect::<Vec<_>>(),
            (32, 192),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        for tied in [true, false] {
            let logits = Tensor::from_vec(
                (0..8192)
                    .map(|i| {
                        if tied {
                            0.
                        } else {
                            (i as f32 * 0.019).sin() * 11.
                        }
                    })
                    .collect::<Vec<_>>(),
                (32, 256),
                &dev,
            )?;
            let route =
                super::super::route_mini(&logits, &Tensor::zeros(256, DType::F32, &dev)?, 2.5)?;
            let ids = route.expert_ids()?.flatten_all()?.to_vec1::<u32>()?;
            let plan = super::super::route_mini_compact(
                &logits,
                &Tensor::zeros(256, DType::F32, &dev)?,
                2.5,
            )?;
            assert_eq!(plan.expert_ids()?.flatten_all()?.to_vec1::<u32>()?, ids);
            let mut segments = Vec::new();
            let mut indices = Vec::new();
            let mut inverse = vec![0u32; 256];
            for e in 0..256 {
                let n = ids.iter().filter(|&&id| id == e).count();
                if n == 0 {
                    continue;
                }
                segments.push((e as usize, n));
                for (slot, &id) in ids.iter().enumerate() {
                    if id == e {
                        inverse[slot] = indices.len() as u32;
                        indices.push((slot / 8) as u32);
                    }
                }
            }
            let selected =
                x.index_select(&Tensor::from_slice(&indices, indices.len(), &dev)?, 0)?;
            let expected = grouped(&selected, &w, &segments)?;
            let expected_up = grouped(&selected, &up_weights, &segments)?;
            let actual = routed(&x, &w, &plan)?;
            let up = routed(&x, &up_weights, &plan)?;
            assert_eq!(actual.to_vec2::<bf16>()?, expected.to_vec2::<bf16>()?);
            assert_eq!(up.to_vec2::<bf16>()?, expected_up.to_vec2::<bf16>()?);
            let expected_hidden = super::super::silu_mul(&expected, &expected_up)?;
            assert_eq!(
                routed_silu(&x, &w, &up_weights, &plan)?.to_vec2::<bf16>()?,
                expected_hidden.to_vec2::<bf16>()?
            );
            assert_eq!(
                grouped_silu_indexed(&x, &w, &up_weights, &segments, &indices)?
                    .to_vec2::<bf16>()?,
                expected_hidden.to_vec2::<bf16>()?
            );
            let expected_down = grouped(&expected, &down, &segments)?;
            let actual_down = routed(&actual, &down, &plan)?;
            assert_eq!(
                actual_down.to_vec2::<bf16>()?,
                expected_down.to_vec2::<bf16>()?
            );
            let weights = route.0.narrow(0, 561, 256)?;
            let expected_mix =
                super::super::mix_experts(&expected_down, &inverse, &weights.to_vec1::<f32>()?, 8)?;
            let actual_mix = super::super::mix_compact_experts(&actual_down, &plan)?;
            assert_eq!(
                actual_mix.to_vec2::<bf16>()?,
                expected_mix.to_vec2::<bf16>()?
            );
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires CUDA"]
    fn tensor_core_quantized_experts_match_independent_dequantization() -> anyhow::Result<()> {
        let dev = Device::new_cuda(0)?;
        // Uneven expert rows, repeated expert IDs, N tail, K tile tail, and
        // all scale-group sizes exercise layout and padding independently.
        for encoding in [Encoding::I8Sym, Encoding::I8Mma] {
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
                    global_scales: None,
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
                        global_scales: None,
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

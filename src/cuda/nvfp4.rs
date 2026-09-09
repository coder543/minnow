//! Native block-scaled FP4 MMA on SM120/121, with a BF16 MMA fallback on SM80+.
//! Activations are quantized once per projection input; both paths keep weights
//! packed and share that temporary activation allocation across gate/up.
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
const BF16_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/minnow-nvfp4-bf16.ptx"));
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Native,
    Bf16,
}
impl Backend {
    fn for_capability(capability: (i32, i32)) -> Result<Self> {
        if capability.0 == 12 && [0, 1].contains(&capability.1) {
            Ok(Self::Native)
        } else if capability.0 >= 8 {
            Ok(Self::Bf16)
        } else {
            candle_core::bail!("NVFP4 CUDA execution requires SM80 or newer")
        }
    }
    fn module(self) -> (&'static str, &'static str) {
        match self {
            Self::Native => ("minnow-nvfp4-v1", PTX),
            Self::Bf16 => ("minnow-nvfp4-bf16-v1", BF16_PTX),
        }
    }
}
fn activation_bytes(native: bool, rows: usize, input: usize) -> usize {
    let operands = if native {
        rows * input * 9 / 16
    } else {
        rows * input * 2
    };
    operands + rows * 4
}
fn backend(device: &candle_core::CudaDevice) -> Result<Backend> {
    Backend::for_capability(device.cuda_stream().context().compute_capability().w()?)
}
fn default_tile(capability: (i32, i32), input: usize, rows: usize, segments: usize) -> usize {
    if capability == (8, 6) && rows > 256 {
        32
    } else if input >= 2048 || rows / segments.max(1) <= 16 {
        16
    } else {
        32
    }
}
pub fn check_device(device: &Device) -> Result<()> {
    let Device::Cuda(device) = device else {
        candle_core::bail!("NVFP4 CUDA execution requires CUDA");
    };
    backend(device)?;
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
        let backend = backend(dev)?;
        let (module, ptx) = backend.module();
        let xs = x
            .as_cuda_slice::<bf16>()?
            .slice(l.start_offset()..l.start_offset() + rows * input);
        let bytes = activation_bytes(backend == Backend::Native, rows, input);
        // SAFETY: the kernel writes all quantized operands and FP32 outer scales.
        let mut output = unsafe { dev.alloc::<u8>(bytes)? };
        let name = if [512, 1024, 2048, 4096].contains(&input) {
            format!("minnow_nvfp4_quantize_{input}")
        } else {
            "minnow_nvfp4_quantize".into()
        };
        let f = dev.get_or_load_custom_func(&name, module, ptx)?;
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
    pair: Option<&'a Weights>,
    device_plan: Option<&'a super::CompactRoutingPlan>,
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
        let backend = backend(&x.device)?;
        let (module, ptx) = backend.module();
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
            || xl.shape().elem_count()
                != activation_bytes(backend == Backend::Native, source_rows, input)
            || wl.shape().elem_count() != experts * out * input / 2
            || sl.shape().elem_count() != experts * out * input / 16
            || self.segments.iter().any(|&(e, n)| e >= experts || n == 0)
            || (self.device_plan.is_none()
                && self
                    .segments
                    .iter()
                    .try_fold(0usize, |sum, (_, n)| sum.checked_add(*n))
                    != Some(rows))
            || (self.device_plan.is_some()
                && (rows != 256
                    || experts != 256
                    || self.tile != 16
                    || ![32, 256].contains(&source_rows)))
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
        if let Some(b) = self.pair
            && (b.shape != self.w.shape
                || b.encoding != Encoding::Nvfp4
                || b.group_size != 16
                || b.codes.elem_count() != wl.shape().elem_count()
                || b.scales.elem_count() != sl.shape().elem_count()
                || ![&b.codes, &b.scales]
                    .iter()
                    .all(|t| t.is_contiguous() && t.layout().start_offset().is_multiple_of(4))
                || b.global_scales
                    .as_ref()
                    .is_none_or(|g| g.dims() != [experts] || !g.is_contiguous()))
        {
            candle_core::bail!("invalid NVFP4 paired weights");
        }
        let paired_storage = self.pair.map(|b| {
            (
                b.codes.storage_and_layout(),
                b.scales.storage_and_layout(),
                b.global_scales.as_ref().unwrap().storage_and_layout(),
            )
        });
        let paired_views = if let Some(((bc, cl), (bs, sl), (bg, gl))) = &paired_storage {
            let (Storage::Cuda(bc), Storage::Cuda(bs), Storage::Cuda(bg)) = (&**bc, &**bs, &**bg)
            else {
                candle_core::bail!("paired weights must be on CUDA");
            };
            Some((
                bc.as_cuda_slice::<u8>()?
                    .slice(cl.start_offset()..cl.start_offset() + cl.shape().elem_count()),
                bs.as_cuda_slice::<u8>()?
                    .slice(sl.start_offset()..sl.start_offset() + sl.shape().elem_count()),
                bg.as_cuda_slice::<f32>()?
                    .slice(gl.start_offset()..gl.start_offset() + experts),
            ))
        } else {
            None
        };
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
        // SAFETY: disjoint descriptors cover all rows; kernel bounds output columns.
        let mut output = unsafe { dev.alloc::<bf16>(rows * out)? };
        let f = dev.get_or_load_custom_func(
            &format!(
                "minnow_nvfp4_{}_{}",
                if self.pair.is_some() { "pair" } else { "gemm" },
                self.tile
            ),
            module,
            ptx,
        )?;
        let (k, n, m) = (input as i32, out as i32, source_rows as i32);
        let indexed =
            i32::from(self.indices.is_some() || (self.device_plan.is_some() && source_rows == 32));
        let mut call = f.builder();
        call.arg(&xs).arg(&ws).arg(&ss).arg(&gs);
        // Device descriptors store integer bits in a private FP32 allocation.
        if let Some(p) = &device_descriptors {
            call.arg(p);
        } else {
            call.arg(host_descriptors.as_ref().unwrap());
        }
        call.arg(&mut output).arg(&k).arg(&n).arg(&m).arg(&indexed);
        if let Some((bc, bs, bg)) = &paired_views {
            call.arg(bc).arg(bs).arg(bg);
        }
        // SAFETY: validated buffers, alignment, descriptor indices, and N8/K64 shapes.
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
    a: &Tensor,
    source_rows: usize,
    w: &Weights,
    segments: &[(usize, usize)],
    indices: Option<&[u32]>,
    pair: Option<&Weights>,
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
    if let Some(b) = pair {
        for t in [
            &b.codes,
            &b.scales,
            b.global_scales.as_ref().unwrap_or(&b.scales),
        ] {
            if !a.device().same_device(t.device()) {
                candle_core::bail!("paired weights must share a device");
            }
        }
    }
    let Device::Cuda(device) = a.device() else {
        candle_core::bail!("NVFP4 CUDA execution requires CUDA");
    };
    let capability = device.cuda_stream().context().compute_capability().w()?;
    let tile = std::env::var("MINNOW_NVFP4_TILE_ROWS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| [16, 32, 64, 128].contains(v))
        .unwrap_or_else(|| default_tile(capability, w.shape.2, rows, segments.len()));
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
            pair,
            device_plan: None,
        },
    )
}
pub fn grouped(x: &Tensor, w: &Weights, segments: &[(usize, usize)]) -> Result<Tensor> {
    check_device(x.device())?;
    let a = x.apply_op1_no_bwd(&Quantize)?;
    run(&a, x.dim(0)?, w, segments, None, None)
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
        run(&q, x.dim(0)?, a, segments, Some(indices), None)?,
        run(&q, x.dim(0)?, b, segments, Some(indices), None)?,
    ))
}

pub fn routed(x: &Tensor, w: &Weights, plan: &super::CompactRoutingPlan) -> Result<Tensor> {
    let q = x.apply_op1_no_bwd(&Quantize)?;
    routed_quantized(&q, x.dim(0)?, w, plan, None)
}
fn routed_quantized(
    q: &Tensor,
    source_rows: usize,
    w: &Weights,
    plan: &super::CompactRoutingPlan,
    pair: Option<&Weights>,
) -> Result<Tensor> {
    for t in [
        &w.codes,
        &w.scales,
        w.global_scales.as_ref().unwrap_or(&w.scales),
        &plan.descriptors,
    ] {
        if !q.device().same_device(t.device()) {
            candle_core::bail!("NVFP4 buffers must share a device");
        }
    }
    if let Some(b) = pair {
        for t in [
            &b.codes,
            &b.scales,
            b.global_scales.as_ref().unwrap_or(&b.scales),
        ] {
            if !q.device().same_device(t.device()) {
                candle_core::bail!("paired weights must share a device");
            }
        }
    }
    q.apply_op3_no_bwd(
        &w.codes,
        &w.scales,
        &Gemm {
            w,
            segments: &[],
            rows: 256,
            source_rows,
            indices: None,
            tile: 16,
            pair,
            device_plan: Some(plan),
        },
    )
}
pub fn routed_pair(
    x: &Tensor,
    a: &Weights,
    b: &Weights,
    plan: &super::CompactRoutingPlan,
) -> Result<(Tensor, Tensor)> {
    if x.dim(0)? != 32 || a.shape != b.shape {
        candle_core::bail!("invalid routed pair");
    }
    let q = x.apply_op1_no_bwd(&Quantize)?;
    Ok((
        routed_quantized(&q, 32, a, plan, None)?,
        routed_quantized(&q, 32, b, plan, None)?,
    ))
}
pub fn routed_silu(
    x: &Tensor,
    a: &Weights,
    b: &Weights,
    plan: &super::CompactRoutingPlan,
) -> Result<Tensor> {
    if x.dim(0)? != 32 {
        candle_core::bail!("invalid routed input");
    }
    let q = x.apply_op1_no_bwd(&Quantize)?;
    routed_quantized(&q, 32, a, plan, Some(b))
}
pub fn grouped_silu_indexed(
    x: &Tensor,
    a: &Weights,
    b: &Weights,
    segments: &[(usize, usize)],
    indices: &[u32],
) -> Result<Tensor> {
    let q = x.apply_op1_no_bwd(&Quantize)?;
    run(&q, x.dim(0)?, a, segments, Some(indices), Some(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::nvfp4 as reference;
    #[test]
    fn ampere_dispatch_preserves_blackwell_defaults() -> Result<()> {
        assert!(Backend::for_capability((7, 5)).is_err());
        for capability in [(8, 0), (8, 6), (8, 9), (9, 0)] {
            assert_eq!(Backend::for_capability(capability)?, Backend::Bf16);
        }
        for capability in [(12, 0), (12, 1)] {
            assert_eq!(Backend::for_capability(capability)?, Backend::Native);
            assert_eq!(default_tile(capability, 2048, 32768, 256), 16);
            assert_eq!(default_tile(capability, 512, 32768, 256), 32);
        }
        assert_eq!(default_tile((8, 6), 2048, 32768, 256), 32);
        assert_eq!(default_tile((8, 6), 2048, 256, 48), 16);
        assert_eq!(activation_bytes(true, 32, 2048), 36992);
        assert_eq!(activation_bytes(false, 32, 2048), 131200);
        Ok(())
    }
    #[test]
    #[ignore = "requires SM80+ CUDA"]
    fn compact_device_route_matches_host_projections_and_mix() -> anyhow::Result<()> {
        let dev = Device::new_cuda(0)?;
        let make_weights = |out, input, salt| -> anyhow::Result<Weights> {
            let mut codes = Vec::new();
            let mut scales = Vec::new();
            let mut globals = Vec::new();
            for e in 0..256 {
                let v: Vec<_> = (0..out * input)
                    .map(|i| ((i * 17 + e * 31 + salt) % 257) as f32 / 257. - 0.5)
                    .collect();
                let (c, s, g) = reference::encode(&v)?;
                let (c, s) = reference::pack(&c, &s, input, false)?;
                codes.extend(c);
                scales.extend(s);
                globals.push(g);
            }
            Ok(Weights {
                int8_activations: false,
                codes: Tensor::from_vec(codes, 256 * out * input / 2, &dev)?,
                scales: Tensor::from_vec(scales, 256 * out * input / 16, &dev)?,
                global_scales: Some(Tensor::from_vec(globals, 256, &dev)?),
                encoding: Encoding::Nvfp4,
                group_size: 16,
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
            let (expected, expected_up) =
                grouped_pair_indexed(&x, &w, &up_weights, &segments, &indices)?;
            let (actual, up) = routed_pair(&x, &w, &up_weights, &plan)?;
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
    #[ignore = "requires SM80+ CUDA"]
    fn mma_matches_independent_nvfp4_operands() -> anyhow::Result<()> {
        let dev = Device::new_cuda(0)?;
        check_device(&dev)?;
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let native = backend(cuda)? == Backend::Native;
        for input in [192, 512, 1024, 2048, 4096] {
            let (experts, out, rows) = (3, 80, 299);
            let mut codes = vec![0; 4];
            let mut scales = vec![0; 4];
            let mut globals = vec![1.];
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
            let w =
                Weights {
                    int8_activations: false,
                    codes: Tensor::from_vec(codes, experts * out * input / 2 + 4, &dev)?.narrow(
                        0,
                        4,
                        experts * out * input / 2,
                    )?,
                    scales: Tensor::from_vec(scales, experts * out * input / 16 + 4, &dev)?
                        .narrow(0, 4, experts * out * input / 16)?,
                    global_scales: Some(
                        Tensor::from_vec(globals, experts + 1, &dev)?.narrow(0, 1, experts)?,
                    ),
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
                if native {
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
                } else {
                    let operands = reference::decode(&c, &s, 1.)?;
                    for (k, &v) in operands.iter().enumerate() {
                        let offset = (row * input + k) * 2;
                        let bits = u16::from_le_bytes(actual[offset..offset + 2].try_into()?);
                        assert_eq!(bits, bf16::from_f32(v).to_bits(), "row{row} col{k}");
                    }
                }
                let offset = activation_bytes(native, rows, input) - rows * 4 + row * 4;
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
                let mut op = Gemm {
                    w: &w,
                    segments: &segments,
                    rows,
                    source_rows: rows,
                    indices: None,
                    tile,
                    pair: None,
                    device_plan: None,
                };
                let base = quantized.apply_op3_no_bwd(&w.codes, &w.scales, &op)?;
                op.pair = Some(&w);
                let fused = quantized.apply_op3_no_bwd(&w.codes, &w.scales, &op)?;
                assert_eq!(
                    fused.to_vec2::<bf16>()?,
                    super::super::silu_mul(&base, &base)?.to_vec2::<bf16>()?,
                    "fused tile{tile} input{input}"
                );
                let actual = base.to_dtype(DType::F32)?.to_vec2::<f32>()?;
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

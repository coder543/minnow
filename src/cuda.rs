//! Pointer-batched expert GEMMs. Weights stay resident; only small pointer arrays
//! cross the host/device boundary. All operations use Candle's own CUDA stream.
mod norm;
pub use norm::rms_norm;
mod predict;
use candle_core::cuda_backend::{
    WrapErr,
    cudarc::{
        cublas::sys,
        driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg},
    },
};
use candle_core::{
    CpuStorage, CudaStorage, CustomOp1, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor,
};
use half::bf16;
pub use predict::greedy_confidence;
mod routing;
pub use routing::{RoutingPlan, route_mini};
mod routed_experts;
pub use routed_experts::{mix_routed_experts, routed_expert_gemm};
mod qkv;
pub mod quantized;
pub mod workspace;
pub use qkv::prepare_qkv;
use std::ffi::c_void;

struct BlockSoftmax {
    queries: usize,
    offset: usize,
    block_size: usize,
    scale: f32,
}
impl CustomOp1 for BlockSoftmax {
    fn name(&self) -> &'static str {
        "minnow-block-softmax"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused attention softmax requires CUDA")
    }
    fn cuda_fwd(&self, x: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let (heads, rows, cols) = layout.shape().dims3()?;
        if !layout.is_contiguous()
            || self.queries == 0
            || self.block_size == 0
            || !rows.is_multiple_of(self.queries)
            || self.offset.checked_add(self.queries) != Some(cols)
            || cols == 0
            || cols > 131072
            || heads * rows > u32::MAX as usize
            || !self.queries.is_multiple_of(self.block_size)
            || !self.offset.is_multiple_of(self.block_size)
            || !self.scale.is_finite()
        {
            candle_core::bail!("invalid block softmax shape or parameters");
        }
        let n = layout.shape().elem_count();
        let dev = &x.device;
        let input = x
            .as_cuda_slice::<bf16>()?
            .slice(layout.start_offset()..layout.start_offset() + n);
        // SAFETY: every attention probability is written by the kernel.
        let mut out = unsafe { dev.alloc::<bf16>(n)? };
        let items = cols.div_ceil(128).next_power_of_two();
        let long = cols > 8192;
        let name = if long {
            "minnow_block_softmax_long".to_owned()
        } else {
            format!("minnow_block_softmax_{items}")
        };
        let func = dev.get_or_load_custom_func(
            &name,
            "minnow-v1",
            include_str!(concat!(env!("OUT_DIR"), "/minnow.ptx")),
        )?;
        let mut builder = func.builder();
        let (cols, queries, offset, block_size) = (
            cols as i32,
            self.queries as i32,
            self.offset as i32,
            self.block_size as i32,
        );
        builder
            .arg(&input)
            .arg(&mut out)
            .arg(&cols)
            .arg(&queries)
            .arg(&offset)
            .arg(&block_size)
            .arg(&self.scale);
        // SAFETY: contiguous inputs, matching output size, and checked row/column
        // bounds. Each CTA writes one complete row.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: ((heads * rows) as u32, 1, 1),
                block_dim: (if long { 256 } else { 128 }, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, dev.clone()),
            layout.shape().clone(),
        ))
    }
}

pub fn block_softmax(
    scores: &Tensor,
    queries: usize,
    offset: usize,
    block_size: usize,
    scale: f32,
) -> Result<Tensor> {
    scores.apply_op1_no_bwd(&BlockSoftmax {
        queries,
        offset,
        block_size,
        scale,
    })
}

struct MixExperts {
    top_k: usize,
}
impl CustomOp3 for MixExperts {
    fn name(&self) -> &'static str {
        "minnow-mix-experts"
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
        candle_core::bail!("fused expert mixing requires CUDA")
    }
    fn cuda_fwd(
        &self,
        experts: &CudaStorage,
        el: &Layout,
        rows: &CudaStorage,
        rl: &Layout,
        weights: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (_, hidden) = el.shape().dims2()?;
        let count = rl.shape().dims1()?;
        if !el.is_contiguous()
            || !rl.is_contiguous()
            || !wl.is_contiguous()
            || wl.shape().dims1()? != count
            || self.top_k == 0
            || self.top_k > 32
            || count == 0
            || !count.is_multiple_of(self.top_k)
            || hidden == 0
            || hidden > i32::MAX as usize
        {
            candle_core::bail!("invalid expert mixing layout or dimensions");
        }
        let tokens = count / self.top_k;
        let n = tokens
            .checked_mul(hidden)
            .ok_or_else(|| candle_core::Error::Msg("expert mix size overflow".into()))?;
        if n > u32::MAX as usize {
            candle_core::bail!("expert mix output too large");
        }
        let dev = &experts.device;
        let experts = experts
            .as_cuda_slice::<bf16>()?
            .slice(el.start_offset()..el.start_offset() + el.shape().elem_count());
        let rows = rows
            .as_cuda_slice::<u32>()?
            .slice(rl.start_offset()..rl.start_offset() + count);
        let weights = weights
            .as_cuda_slice::<f32>()?
            .slice(wl.start_offset()..wl.start_offset() + count);
        // SAFETY: every output element is written.
        let mut out = unsafe { dev.alloc::<bf16>(n)? };
        let func = dev.get_or_load_custom_func(
            &format!("minnow_mix_experts_{}", self.top_k.next_power_of_two()),
            "minnow-v1",
            include_str!(concat!(env!("OUT_DIR"), "/minnow.ptx")),
        )?;
        let hidden_i32 = hidden as i32;
        let top_k = self.top_k as i32;
        let mut builder = func.builder();
        builder
            .arg(&n)
            .arg(&hidden_i32)
            .arg(&top_k)
            .arg(&experts)
            .arg(&rows)
            .arg(&weights)
            .arg(&mut out);
        // SAFETY: the sole caller validates every row index before upload. These
        // contiguous slices and the launch cover exactly tokens * hidden outputs.
        unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, dev.clone()),
            Shape::from((tokens, hidden)),
        ))
    }
}

pub fn mix_experts(
    experts: &Tensor,
    rows: &[u32],
    weights: &[f32],
    top_k: usize,
) -> Result<Tensor> {
    let (available, _) = experts.dims2()?;
    if rows.len() != weights.len() || rows.iter().any(|&r| r as usize >= available) {
        candle_core::bail!("invalid expert mixing assignments");
    }
    let indices = Tensor::from_slice(rows, rows.len(), experts.device())?;
    let weights = Tensor::from_slice(weights, weights.len(), experts.device())?;
    experts.apply_op3_no_bwd(&indices, &weights, &MixExperts { top_k })
}

struct SiluMul;
impl CustomOp2 for SiluMul {
    fn name(&self) -> &'static str {
        "minnow-silu-mul"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused SiLU requires CUDA")
    }
    fn cuda_fwd(
        &self,
        gate: &CudaStorage,
        gl: &Layout,
        up: &CudaStorage,
        ul: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let n = gl.shape().elem_count();
        if gl.shape() != ul.shape()
            || !gl.is_contiguous()
            || !ul.is_contiguous()
            || n == 0
            || n > u32::MAX as usize
        {
            candle_core::bail!("invalid fused SiLU layout or size");
        }
        let dev = &gate.device;
        let gate = gate
            .as_cuda_slice::<bf16>()?
            .slice(gl.start_offset()..gl.start_offset() + n);
        let up = up
            .as_cuda_slice::<bf16>()?
            .slice(ul.start_offset()..ul.start_offset() + n);
        // SAFETY: the kernel writes all n elements.
        let mut out = unsafe { dev.alloc::<bf16>(n)? };
        let func = dev.get_or_load_custom_func(
            "minnow_silu_mul_bf16",
            "minnow-v1",
            include_str!(concat!(env!("OUT_DIR"), "/minnow.ptx")),
        )?;
        let mut builder = func.builder();
        builder.arg(&n).arg(&gate).arg(&up).arg(&mut out);
        // SAFETY: validated equal contiguous BF16 inputs and n-element output;
        // access guards keep all allocations alive on Candle's CUDA stream.
        unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, dev.clone()),
            gl.shape().clone(),
        ))
    }
}

pub fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if !gate.device().same_device(up.device()) {
        candle_core::bail!("SiLU inputs must be on the same device");
    }
    gate.apply_op2_no_bwd(up, &SiluMul)
}

struct ExpertGemm<'a> {
    experts: &'a [usize],
    input_batched: bool,
}

impl CustomOp2 for ExpertGemm<'_> {
    fn name(&self) -> &'static str {
        "minnow-expert-gemm"
    }
    fn cpu_fwd(
        &self,
        _x: &CpuStorage,
        _xl: &Layout,
        _w: &CpuStorage,
        _wl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("expert GEMM requires CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        w: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (expert_count, o, k) = wl.shape().dims3()?;
        let batch = self.experts.len();
        let n = if self.input_batched {
            let (b, n, ik) = xl.shape().dims3()?;
            if b != batch || ik != k {
                candle_core::bail!("batched expert input shape mismatch");
            }
            n
        } else {
            let (n, ik) = xl.shape().dims2()?;
            if ik != k {
                candle_core::bail!("expert input shape mismatch");
            }
            n
        };
        if !xl.is_contiguous()
            || !wl.is_contiguous()
            || batch == 0
            || self.experts.iter().any(|&e| e >= expert_count)
            || [n, o, k, batch]
                .iter()
                .any(|&v| v == 0 || v > i32::MAX as usize)
        {
            candle_core::bail!("invalid expert GEMM layout, dimensions, or indices");
        }
        let dev = &x.device;
        let stream = dev.cuda_stream();
        let xs = x.as_cuda_slice::<bf16>()?;
        let ws = w.as_cuda_slice::<bf16>()?;
        // SAFETY: every output element is written by GEMM with beta=0 below.
        let mut out = unsafe { dev.alloc::<bf16>(batch * n * o)? };
        {
            // Keep cudarc's access guards alive until the BLAS work is enqueued.
            let (xp, _x_access) = xs.device_ptr(&stream);
            let (wp, _w_access) = ws.device_ptr(&stream);
            let (yp, _y_access) = out.device_ptr_mut(&stream);
            let ap: Vec<u64> = self
                .experts
                .iter()
                .map(|&e| wp + ((wl.start_offset() + e * o * k) * 2) as u64)
                .collect();
            let bp: Vec<u64> = (0..batch)
                .map(|b| {
                    xp + ((xl.start_offset() + if self.input_batched { b * n * k } else { 0 }) * 2)
                        as u64
                })
                .collect();
            let cp: Vec<u64> = (0..batch).map(|b| yp + (b * n * o * 2) as u64).collect();
            let ap = stream.clone_htod(&ap).w()?;
            let bp = stream.clone_htod(&bp).w()?;
            let cp = stream.clone_htod(&cp).w()?;
            let (ap, _a_access) = ap.device_ptr(&stream);
            let (bp, _b_access) = bp.device_ptr(&stream);
            let (cp, _c_access) = cp.device_ptr(&stream);
            let alpha = 1f32;
            let beta = 0f32;
            let blas = dev.cublas_handle();
            // SAFETY: validated contiguous [expert,out,in] weights and input
            // shapes. Pointer arrays reference live buffers on the same stream.
            // Column-major BLAS computes W^T-layout * X^T-layout => Y^T-layout.
            unsafe {
                sys::cublasGemmBatchedEx(
                    *blas.handle(),
                    sys::cublasOperation_t::CUBLAS_OP_T,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    o as i32,
                    n as i32,
                    k as i32,
                    (&alpha as *const f32).cast::<c_void>(),
                    ap as *const *const c_void,
                    sys::cudaDataType::CUDA_R_16BF,
                    k as i32,
                    bp as *const *const c_void,
                    sys::cudaDataType::CUDA_R_16BF,
                    k as i32,
                    (&beta as *const f32).cast::<c_void>(),
                    cp as *const *mut c_void,
                    sys::cudaDataType::CUDA_R_16BF,
                    o as i32,
                    batch as i32,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
                .result()
                .w()?;
            }
        }
        Ok((
            CudaStorage::wrap_cuda_slice(out, dev.clone()),
            Shape::from((batch, n, o)),
        ))
    }
}

pub fn expert_gemm(
    x: &Tensor,
    weights: &Tensor,
    experts: &[usize],
    input_batched: bool,
) -> Result<Tensor> {
    if !x.device().same_device(weights.device()) {
        candle_core::bail!("expert inputs and weights must be on the same device");
    }
    x.apply_op2_no_bwd(
        weights,
        &ExpertGemm {
            experts,
            input_batched,
        },
    )
}

/// Variable-size expert batches. Each segment consumes its own consecutive input
/// rows and writes the same rows in the output; resident weights are only viewed.
struct GroupedExpertGemm<'a> {
    segments: &'a [(usize, usize)], // (expert index, number of assigned tokens)
}

impl CustomOp2 for GroupedExpertGemm<'_> {
    fn name(&self) -> &'static str {
        "minnow-grouped-expert-gemm"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("grouped expert GEMM requires CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        w: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (experts, o, k) = wl.shape().dims3()?;
        let (rows, ik) = xl.shape().dims2()?;
        let groups = self.segments.len();
        let assigned = self
            .segments
            .iter()
            .try_fold(0usize, |n, &(_, r)| n.checked_add(r));
        if !xl.is_contiguous() || !wl.is_contiguous() || ik != k
            || assigned != Some(rows)
            || [rows, o, k, groups].iter().any(|&n| n == 0 || n > i32::MAX as usize)
            || self.segments.iter().any(|&(e, n)| e >= experts || n == 0)
            // All matrix addresses must meet cuBLAS's 16-byte alignment rule.
            || !k.is_multiple_of(8) || !o.is_multiple_of(8)
            || !xl.start_offset().is_multiple_of(8) || !wl.start_offset().is_multiple_of(8)
        {
            candle_core::bail!("invalid grouped expert GEMM layout, dimensions, or segments");
        }
        let dev = &x.device;
        let stream = dev.cuda_stream();
        let xs = x.as_cuda_slice::<bf16>()?;
        let ws = w.as_cuda_slice::<bf16>()?;
        let count = rows
            .checked_mul(o)
            .ok_or_else(|| candle_core::Error::Msg("expert output size overflow".into()))?;
        // SAFETY: disjoint segments cover all rows, and GEMM uses beta=0.
        let mut out = unsafe { dev.alloc::<bf16>(count)? };
        {
            let (xp, _x_access) = xs.device_ptr(&stream);
            let (wp, _w_access) = ws.device_ptr(&stream);
            let (yp, _y_access) = out.device_ptr_mut(&stream);
            let mut ap = Vec::with_capacity(groups);
            let mut bp = Vec::with_capacity(groups);
            let mut cp = Vec::with_capacity(groups);
            let mut ns = Vec::with_capacity(groups);
            let mut offset = 0;
            for &(expert, n) in self.segments {
                ap.push(wp + ((wl.start_offset() + expert * o * k) * 2) as u64);
                bp.push(xp + ((xl.start_offset() + offset * k) * 2) as u64);
                cp.push(yp + (offset * o * 2) as u64);
                ns.push(n as i32);
                offset += n;
            }
            // Matrix pointer arrays are device-resident; size/scalar arrays are
            // host-resident, as specified by cublasGemmGroupedBatchedEx.
            let ap = stream.clone_htod(&ap).w()?;
            let bp = stream.clone_htod(&bp).w()?;
            let cp = stream.clone_htod(&cp).w()?;
            let (ap, _a_access) = ap.device_ptr(&stream);
            let (bp, _b_access) = bp.device_ptr(&stream);
            let (cp, _c_access) = cp.device_ptr(&stream);
            let transa = vec![sys::cublasOperation_t::CUBLAS_OP_T; groups];
            let transb = vec![sys::cublasOperation_t::CUBLAS_OP_N; groups];
            let ms = vec![o as i32; groups];
            let ks = vec![k as i32; groups];
            let alpha = vec![1f32; groups];
            let beta = vec![0f32; groups];
            let sizes = vec![1i32; groups];
            let blas = dev.cublas_handle();
            // SAFETY: checked BF16 shapes, alignment, expert indices, and row
            // coverage; access guards retain all device buffers on this stream.
            unsafe {
                sys::cublasGemmGroupedBatchedEx(
                    *blas.handle(),
                    transa.as_ptr(),
                    transb.as_ptr(),
                    ms.as_ptr(),
                    ns.as_ptr(),
                    ks.as_ptr(),
                    alpha.as_ptr().cast(),
                    ap as *const *const c_void,
                    sys::cudaDataType::CUDA_R_16BF,
                    ks.as_ptr(),
                    bp as *const *const c_void,
                    sys::cudaDataType::CUDA_R_16BF,
                    ks.as_ptr(),
                    beta.as_ptr().cast(),
                    cp as *const *mut c_void,
                    sys::cudaDataType::CUDA_R_16BF,
                    ms.as_ptr(),
                    groups as i32,
                    sizes.as_ptr(),
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                )
                .result()
                .w()?;
            }
        }
        Ok((
            CudaStorage::wrap_cuda_slice(out, dev.clone()),
            Shape::from((rows, o)),
        ))
    }
}

pub fn grouped_expert_gemm(
    x: &Tensor,
    weights: &Tensor,
    segments: &[(usize, usize)],
) -> Result<Tensor> {
    if !x.device().same_device(weights.device()) {
        candle_core::bail!("expert inputs and weights must be on the same device");
    }
    x.apply_op2_no_bwd(weights, &GroupedExpertGemm { segments })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Module};
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn fused_expert_mix_matches_explicit_gather_multiply_and_sum_bitwise() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let experts = Tensor::arange(0f32, 130. * 256., &dev)?
            .reshape((130, 256))?
            .affine(0.01, -1.)?
            .sin()?
            .to_dtype(DType::BF16)?
            .narrow(0, 2, 128)?;
        for top_k in [1, 2, 3, 8, 16, 32] {
            let rows: Vec<u32> = (0..37 * top_k).map(|i| (i * 17 % 128) as u32).collect();
            let weights: Vec<f32> = (0..rows.len())
                .map(|i| ((i * 3 % 17) as f32 - 8.) * 0.073)
                .collect();
            let expected = experts
                .index_select(&Tensor::from_slice(&rows, rows.len(), &dev)?, 0)?
                .reshape((37, top_k, 256))?
                .to_dtype(DType::F32)?
                .broadcast_mul(&Tensor::from_slice(&weights, (37, top_k, 1), &dev)?)?
                .sum(1)?
                .to_dtype(DType::BF16)?
                .flatten_all()?
                .to_vec1::<bf16>()?;
            let actual = mix_experts(&experts, &rows, &weights, top_k)?
                .flatten_all()?
                .to_vec1::<bf16>()?;
            for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "top_k={top_k}, output={i}: {a} vs {b}"
                );
            }
        }
        assert!(mix_experts(&experts, &[128], &[1.], 1).is_err());
        Ok(())
    }
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn fused_block_softmax_matches_explicit_scaling_mask_and_fp32_softmax() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for (queries, offset) in [
            (32, 0),
            (96, 64),
            (512, 0),
            (2048, 2048),
            (32, 8160),
            (32, 8192),
            (96, 8192),
            (128, 18432),
            (32, 131040),
        ] {
            let cols = queries + offset;
            let n = 2 * queries * cols;
            let raw = Tensor::arange(0f32, n as f32, &dev)?
                .affine(0.017, 0.)?
                .sin()?
                .affine(20., -5.)?
                .to_dtype(DType::BF16)?
                .reshape((2, queries, cols))?;
            let scale = 128f64.powf(-0.5);
            let mask: Vec<f32> = (0..queries)
                .flat_map(|q| {
                    (0..cols).map(move |k| {
                        if k / 32 <= (offset + q) / 32 {
                            0.
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                })
                .collect();
            let mask = Tensor::from_vec(mask, (1, queries, cols), &dev)?.to_dtype(DType::BF16)?;
            let scaled = (raw.to_dtype(DType::F32)? * scale)?.to_dtype(DType::BF16)?;
            let expected = candle_nn::ops::softmax_last_dim(
                &scaled.broadcast_add(&mask)?.to_dtype(DType::F32)?,
            )?
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?;
            let actual =
                block_softmax(&raw, queries, offset, 32, scale as f32)?.to_dtype(DType::F32)?;
            let error = (&actual - &expected)?.abs()?;
            // At most one BF16 rounding step; FP32 reductions use a different
            // tree, with the same BF16 score and probability rounding points.
            let ratio = error
                .broadcast_div(&expected.affine(0.008, 1e-8)?)?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(
                ratio <= 1.,
                "queries={queries}, offset={offset}, error/allowance={ratio}"
            );
            let sums = actual.sum(candle_core::D::Minus1)?;
            let error = sums.affine(1., -1.)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(error < 0.004, "probability mass error {error}");
            let actual = actual.flatten_all()?.to_vec1::<f32>()?;
            for (i, &p) in actual.iter().enumerate() {
                if i % cols / 32 > (offset + i / cols % queries) / 32 {
                    assert_eq!(p, 0., "future attention must be exactly zero");
                }
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn grouped_experts_match_individual_gemms_with_uneven_assignments() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let x = Tensor::arange(0f32, 80. * 64., &dev)?
            .reshape((80, 64))?
            .affine(0.001, -2.5)?
            .sin()?
            .to_dtype(DType::BF16)?;
        let w = Tensor::arange(0f32, 5. * 32. * 64., &dev)?
            .reshape((5, 32, 64))?
            .affine(0.01, -0.3)?
            .cos()?
            .to_dtype(DType::BF16)?;
        // Offset input storage, reordered/repeated experts, odd row counts, and
        // a group larger than a diffusion block exercise pointer arithmetic.
        let x = x.narrow(0, 3, 74)?;
        let segments = [(4, 1), (0, 37), (2, 7), (4, 29)];
        let y = grouped_expert_gemm(&x, &w, &segments)?;
        let down = w.transpose(1, 2)?.contiguous()?;
        let z = grouped_expert_gemm(&y, &down, &segments)?;
        let mut offset = 0;
        for (e, n) in segments {
            for (actual, input, weights) in [(&y, &x, &w), (&z, &y, &down)] {
                let expected = candle_nn::Linear::new(weights.get(e)?, None)
                    .forward(&input.narrow(0, offset, n)?)?
                    .to_dtype(DType::F32)?;
                let error =
                    (actual.narrow(0, offset, n)?.to_dtype(DType::F32)? - &expected)?.abs()?;
                let relative = error
                    .broadcast_div(&expected.abs()?.affine(0.01, 0.01)?)?
                    .max_all()?
                    .to_scalar::<f32>()?;
                assert!(relative <= 1., "grouped GEMM error/allowance = {relative}");
            }
            offset += n;
        }
        assert!(grouped_expert_gemm(&x, &w, &[(0, 73)]).is_err());
        assert!(grouped_expert_gemm(&x, &w, &[(5, 74)]).is_err());
        Ok(())
    }
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn fused_silu_matches_explicit_fp32_for_all_finite_bf16_inputs() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let values: Vec<bf16> = (0..=u16::MAX)
            .map(bf16::from_bits)
            .filter(|v| v.is_finite())
            .collect();
        let ups: Vec<bf16> = (0..values.len())
            .map(|i| bf16::from_f32([-2., -0.25, 0., 0.5, 1., 3.][i % 6]))
            .collect();
        let n = values.len();
        let gate = Tensor::from_vec(values, n, &dev)?;
        let up = Tensor::from_vec(ups, n, &dev)?;
        let expected =
            (gate.to_dtype(DType::F32)?.silu()?.to_dtype(DType::BF16)? * &up)?.to_vec1::<bf16>()?;
        let actual = silu_mul(&gate, &up)?.to_vec1::<bf16>()?;
        for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "BF16 input at {i}: {a} vs {b}");
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn pointer_batches_select_correct_experts_and_input_rows() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let x = Tensor::arange(0f32, 256f32, &dev)?
            .reshape((4, 64))?
            .affine(0.001, -0.1)?
            .to_dtype(DType::BF16)?;
        let w = Tensor::arange(0f32, 3. * 32. * 64., &dev)?
            .reshape((3, 32, 64))?
            .affine(0.0001, -0.3)?
            .to_dtype(DType::BF16)?;
        let ids = [2, 0];
        let y = expert_gemm(&x, &w, &ids, false)?;
        for (b, e) in ids.into_iter().enumerate() {
            let expected = candle_nn::Linear::new(w.get(e)?, None).forward(&x)?;
            let diff = (y.get(b)?.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(diff <= 0.001, "{diff}");
        }
        let down = w.transpose(1, 2)?.contiguous()?;
        let z = expert_gemm(&y, &down, &ids, true)?;
        for (b, e) in ids.into_iter().enumerate() {
            let expected = candle_nn::Linear::new(down.get(e)?, None).forward(&y.get(b)?)?;
            let diff = (z.get(b)?.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(diff <= 0.01, "{diff}");
        }
        Ok(())
    }
}

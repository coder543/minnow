//! Device-selected expert batches: no routing scores or pointer arrays cross
//! the host boundary. The checkpoint's packed BF16 allocations are only viewed.
use super::routing::{PLAN_SIZE, RoutingPlan};
use candle_core::cuda_backend::{
    WrapErr,
    cudarc::{
        cublas::sys,
        driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg},
    },
};
use candle_core::{CpuStorage, CudaStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};
use half::bf16;
use std::ffi::c_void;

struct ExpertGemm {
    batched: bool,
}
impl CustomOp3 for ExpertGemm {
    fn name(&self) -> &'static str {
        "minnow-device-expert-gemm"
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
        candle_core::bail!("device expert GEMMs require CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        w: &CudaStorage,
        wl: &Layout,
        p: &CudaStorage,
        pl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (experts, out_dim, in_dim) = wl.shape().dims3()?;
        let valid_input = if self.batched {
            xl.dims() == [48, 32, in_dim]
        } else {
            xl.dims() == [32, in_dim]
        };
        if !valid_input
            || experts != 256
            || !xl.is_contiguous()
            || !wl.is_contiguous()
            || !xl.start_offset().is_multiple_of(8)
            || !wl.start_offset().is_multiple_of(8)
            || !pl.is_contiguous()
            || pl.dims() != [PLAN_SIZE]
            || [in_dim, out_dim]
                .iter()
                .any(|&n| n == 0 || n > i32::MAX as usize || !n.is_multiple_of(8))
        {
            candle_core::bail!("invalid device expert GEMM shape");
        }
        let dev = &x.device;
        let stream = dev.cuda_stream();
        let xs = x.as_cuda_slice::<bf16>()?;
        let ws = w.as_cuda_slice::<bf16>()?;
        let plan = p
            .as_cuda_slice::<f32>()?
            .slice(pl.start_offset()..pl.start_offset() + PLAN_SIZE);
        // SAFETY: pointer setup initializes all addresses; cuBLAS writes all
        // output batches, including ignored padding, with beta=0.
        let mut output = unsafe { dev.alloc::<bf16>(48 * 32 * out_dim)? };
        let mut pointers = unsafe { dev.alloc::<u64>(3 * 48)? };
        {
            let (xp, _xa) = xs.device_ptr(&stream);
            let (wp, _wa) = ws.device_ptr(&stream);
            let (yp, _ya) = output.device_ptr_mut(&stream);
            let xp = xp + (xl.start_offset() * 2) as u64;
            let wp = wp + (wl.start_offset() * 2) as u64;
            let f = dev.get_or_load_custom_func(
                "minnow_expert_pointers",
                "minnow-v1",
                include_str!(concat!(env!("OUT_DIR"), "/minnow.ptx")),
            )?;
            let m = out_dim as i32;
            let k = in_dim as i32;
            let n = 32i32;
            let batched = i32::from(self.batched);
            let mut call = f.builder();
            call.arg(&plan)
                .arg(&mut pointers)
                .arg(&xp)
                .arg(&wp)
                .arg(&yp)
                .arg(&m)
                .arg(&k)
                .arg(&batched);
            // SAFETY: addresses refer to live BF16 buffers, with dimensions
            // checked above. RoutingPlan construction bounds every expert ID.
            unsafe {
                call.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (64, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .w()?;
            let (ptr, _pa) = pointers.device_ptr(&stream);
            let alpha = 1f32;
            let beta = 0f32;
            let blas = dev.cublas_handle();
            // SAFETY: pointer setup and BLAS execute in order on Candle's stream.
            unsafe {
                sys::cublasGemmBatchedEx(
                    *blas.handle(),
                    sys::cublasOperation_t::CUBLAS_OP_T,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    m,
                    n,
                    k,
                    (&alpha as *const f32).cast::<c_void>(),
                    ptr as *const *const c_void,
                    sys::cudaDataType::CUDA_R_16BF,
                    k,
                    (ptr + 48 * 8) as *const *const c_void,
                    sys::cudaDataType::CUDA_R_16BF,
                    k,
                    (&beta as *const f32).cast::<c_void>(),
                    (ptr + 96 * 8) as *const *mut c_void,
                    sys::cudaDataType::CUDA_R_16BF,
                    m,
                    48,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
                .result()
            }
            .w()?;
        }
        Ok((
            CudaStorage::wrap_cuda_slice(output, dev.clone()),
            Shape::from((48, 32, out_dim)),
        ))
    }
}

pub fn routed_expert_gemm(
    x: &Tensor,
    weights: &Tensor,
    plan: &RoutingPlan,
    batched: bool,
) -> Result<Tensor> {
    if !x.device().same_device(weights.device()) || !x.device().same_device(plan.0.device()) {
        candle_core::bail!("device expert inputs must share a device");
    }
    x.apply_op3_no_bwd(weights, &plan.0, &ExpertGemm { batched })
}

struct Mix {
    compact: bool,
}
impl CustomOp2 for Mix {
    fn name(&self) -> &'static str {
        "minnow-device-expert-mix"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("device expert mixing requires CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        p: &CudaStorage,
        pl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (batch, rows, hidden) = xl.shape().dims3()?;
        if batch != if self.compact { 8 } else { 48 }
            || rows != 32
            || hidden == 0
            || hidden > i32::MAX as usize
            || !xl.is_contiguous()
            || !pl.is_contiguous()
            || pl.dims() != [PLAN_SIZE]
            || rows * hidden > u32::MAX as usize
        {
            candle_core::bail!("invalid device expert mix shape");
        }
        let dev = &x.device;
        let experts = x
            .as_cuda_slice::<bf16>()?
            .slice(xl.start_offset()..xl.start_offset() + batch * 32 * hidden);
        let plan = p
            .as_cuda_slice::<f32>()?
            .slice(pl.start_offset()..pl.start_offset() + PLAN_SIZE);
        // SAFETY: one thread writes each of the 32*hidden outputs.
        let mut output = unsafe { dev.alloc::<bf16>(32 * hidden)? };
        let f = dev.get_or_load_custom_func(
            "minnow_mix_routed",
            "minnow-v1",
            include_str!(concat!(env!("OUT_DIR"), "/minnow.ptx")),
        )?;
        let h = hidden as i32;
        let mut call = f.builder();
        call.arg(&experts).arg(&plan).arg(&mut output).arg(&h);
        // SAFETY: RoutingPlan construction bounds gathered rows to [0,48*32).
        unsafe { call.launch(LaunchConfig::for_num_elems((32 * hidden) as u32)) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, dev.clone()),
            Shape::from((32, hidden)),
        ))
    }
}
pub fn mix_routed_experts(experts: &Tensor, plan: &RoutingPlan) -> Result<Tensor> {
    if !experts.device().same_device(plan.0.device()) {
        candle_core::bail!("expert mix inputs must share a device");
    }
    experts.apply_op2_no_bwd(&plan.0, &Mix { compact: false })
}
pub fn mix_compact_experts(
    experts: &Tensor,
    plan: &super::routing::CompactRoutingPlan,
) -> Result<Tensor> {
    let (rows, hidden) = experts.dims2()?;
    if rows != 256 || !experts.device().same_device(plan.mix.0.device()) {
        candle_core::bail!("invalid compact expert outputs");
    }
    experts
        .reshape((8, 32, hidden))?
        .apply_op2_no_bwd(&plan.mix.0, &Mix { compact: true })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn device_expert_batches_and_mixing_match_host_assignment_pipeline() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let x = Tensor::arange(0f32, 32. * 128., &dev)?
            .affine(1. / 8192., -0.2)?
            .to_dtype(DType::BF16)?
            .reshape((32, 128))?;
        let w = Tensor::arange(0f32, 256. * 64. * 128., &dev)?
            .affine(0.0001, -1.)?
            .sin()?
            .affine(0.03, 0.)?
            .to_dtype(DType::BF16)?
            .reshape((256, 64, 128))?;
        let down = w.transpose(1, 2)?.contiguous()?;
        for uniform in [true, false] {
            let logits = Tensor::arange(0f32, 8192., &dev)?
                .affine(if uniform { 0. } else { 0.31 }, 0.)?
                .sin()?
                .reshape((32, 256))?;
            let bias = Tensor::zeros(256, DType::F32, &dev)?;
            let plan = super::super::route_mini(&logits, &bias, 2.5)?;
            let values = plan.0.to_vec1::<f32>()?;
            let count = values[0] as usize;
            let selected: Vec<usize> = values[1..1 + count].iter().map(|&v| v as usize).collect();
            let rows: Vec<u32> = values[305..561].iter().map(|&v| v as u32).collect();
            let host_gate = super::super::expert_gemm(&x, &w, &selected, false)?;
            let host_hidden = super::super::silu_mul(&host_gate, &host_gate)?;
            let host_out =
                super::super::expert_gemm(&host_hidden, &down, &selected, true)?.flatten_to(1)?;
            let expected = super::super::mix_experts(&host_out, &rows, &values[561..817], 8)?;
            let gate = routed_expert_gemm(&x, &w, &plan, false)?;
            let hidden = super::super::silu_mul(&gate, &gate)?;
            let out = routed_expert_gemm(&hidden, &down, &plan, true)?;
            let actual = mix_routed_experts(&out, &plan)?;
            let expected = expected.flatten_all()?.to_vec1::<bf16>()?;
            let actual = actual.flatten_all()?.to_vec1::<bf16>()?;
            for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "uniform={uniform}, active={count}, index={i}"
                );
            }
        }
        Ok(())
    }
}

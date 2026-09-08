use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp2, Layout, Result, Shape, Tensor};
use half::bf16;

struct RmsNorm {
    eps: f32,
}
impl CustomOp2 for RmsNorm {
    fn name(&self) -> &'static str {
        "minnow-rms-norm"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused RMSNorm requires CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        w: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (outer, rows, hidden, stride_outer, stride_row) = match *xl.dims() {
            [rows, hidden] => (1, rows, hidden, 0, xl.stride()[0]),
            [outer, rows, hidden] => (outer, rows, hidden, xl.stride()[0], xl.stride()[1]),
            _ => candle_core::bail!("RMSNorm requires a rank-2 or rank-3 input"),
        };
        if xl.stride().last() != Some(&1)
            || !wl.is_contiguous()
            || wl.shape().dims1()? != hidden
            || hidden == 0
            || hidden > i32::MAX as usize
            || rows == 0
            || rows > i32::MAX as usize
            || outer * rows > u32::MAX as usize
            || !self.eps.is_finite()
            || self.eps <= 0.
        {
            candle_core::bail!("invalid RMSNorm layout or parameters");
        }
        let dev = &x.device;
        let input = x.as_cuda_slice::<bf16>()?.slice(xl.start_offset()..);
        let weights = w
            .as_cuda_slice::<bf16>()?
            .slice(wl.start_offset()..wl.start_offset() + hidden);
        // SAFETY: the kernel writes every output element.
        let mut out = unsafe { dev.alloc::<bf16>(outer * rows * hidden)? };
        let func = dev.get_or_load_custom_func(
            "minnow_rms_norm_bf16",
            "minnow-v1",
            include_str!(concat!(env!("OUT_DIR"), "/minnow.ptx")),
        )?;
        let threads = hidden.min(1024).next_power_of_two() as u32;
        let (hidden, rows) = (hidden as i32, rows as i32);
        let mut builder = func.builder();
        builder
            .arg(&input)
            .arg(&weights)
            .arg(&mut out)
            .arg(&hidden)
            .arg(&rows)
            .arg(&stride_outer)
            .arg(&stride_row)
            .arg(&self.eps);
        // SAFETY: strides come from a valid Tensor layout, the hidden dimension
        // is contiguous, and the launch/output cover every logical input row.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: ((outer * rows as usize) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, dev.clone()),
            xl.shape().clone(),
        ))
    }
}

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    if !x.device().same_device(weight.device()) {
        candle_core::bail!("RMSNorm inputs must share a device");
    }
    x.apply_op2_no_bwd(weight, &RmsNorm { eps: eps as f32 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{D, DType, Device};
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn rms_norm_matches_explicit_fp32_bitwise_including_strided_qkv() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for hidden in [8, 128, 2048, 3072] {
            let source = Tensor::arange(0f32, (32 * 24 * hidden) as f32, &dev)?
                .affine(0.0123, -2.)?
                .sin()?
                .affine(12., 0.2)?
                .to_dtype(DType::BF16)?
                .reshape((32, 24, hidden))?;
            let weight = Tensor::arange(0f32, hidden as f32, &dev)?
                .affine(0.001, -0.5)?
                .to_dtype(DType::BF16)?;
            let inputs = [
                source.narrow(1, 16, 4)?.transpose(0, 1)?,
                source.flatten_to(1)?,
            ];
            for input in inputs {
                let x = input.to_dtype(DType::F32)?;
                let inv = (x.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?
                    .sqrt()?
                    .recip()?;
                let expected = x
                    .broadcast_mul(&inv)?
                    .to_dtype(DType::BF16)?
                    .broadcast_mul(&weight)?
                    .flatten_all()?
                    .to_vec1::<bf16>()?;
                let actual = rms_norm(&input, &weight, 1e-6)?
                    .flatten_all()?
                    .to_vec1::<bf16>()?;
                for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "hidden={hidden}, index={i}: {a} vs {b}"
                    );
                }
            }
        }
        Ok(())
    }
}

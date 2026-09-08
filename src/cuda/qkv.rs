//! Fuse head normalization, partial RoPE, and the token-to-head transpose.
use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp3, Layout, Result, Shape, Storage, Tensor};
use half::bf16;

struct Prepare<'a> {
    cos: &'a Tensor,
    sin: &'a Tensor,
    eps: f32,
}
impl CustomOp3 for Prepare<'_> {
    fn name(&self) -> &'static str {
        "minnow-qkv-norm-rope"
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
        candle_core::bail!("fused QKV preparation requires CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (tokens, heads, dim) = xl.shape().dims3()?;
        let (cs, cl) = self.cos.storage_and_layout();
        let (ss, sl) = self.sin.storage_and_layout();
        if ![24, 40].contains(&heads)
            || dim != 128
            || tokens == 0
            || tokens > 8192
            || ql.dims() != [128]
            || kl.dims() != [128]
            || cl.dims() != [1, tokens, 64]
            || sl.dims() != [1, tokens, 64]
            || [xl, ql, kl, cl, sl].iter().any(|l| !l.is_contiguous())
            || !self.eps.is_finite()
            || self.eps <= 0.
        {
            candle_core::bail!("invalid LLaDA QKV layout or parameters");
        }
        let (Storage::Cuda(cos), Storage::Cuda(sin)) = (&*cs, &*ss) else {
            candle_core::bail!("RoPE tables must be on CUDA");
        };
        let dev = &x.device;
        let input = x
            .as_cuda_slice::<bf16>()?
            .slice(xl.start_offset()..xl.start_offset() + tokens * heads * 128);
        let qw = q
            .as_cuda_slice::<bf16>()?
            .slice(ql.start_offset()..ql.start_offset() + 128);
        let kw = k
            .as_cuda_slice::<bf16>()?
            .slice(kl.start_offset()..kl.start_offset() + 128);
        let cos = cos
            .as_cuda_slice::<bf16>()?
            .slice(cl.start_offset()..cl.start_offset() + tokens * 64);
        let sin = sin
            .as_cuda_slice::<bf16>()?
            .slice(sl.start_offset()..sl.start_offset() + tokens * 64);
        // SAFETY: every Q, K, and V element is written, in head-major order.
        let mut output = unsafe { dev.alloc::<bf16>(tokens * heads * 128)? };
        let f = dev.get_or_load_custom_func(
            "minnow_prepare_qkv",
            "minnow-v1",
            include_str!(concat!(env!("OUT_DIR"), "/minnow.ptx")),
        )?;
        let n = tokens as i32;
        let query_heads = (heads - 8) as i32;
        let mut call = f.builder();
        call.arg(&input)
            .arg(&qw)
            .arg(&kw)
            .arg(&cos)
            .arg(&sin)
            .arg(&mut output)
            .arg(&n)
            .arg(&query_heads)
            .arg(&self.eps);
        // SAFETY: one 128-thread block owns one head/token pair; all shapes and
        // storage offsets are checked. Kernel barriers are uniform within a CTA.
        unsafe {
            call.launch(LaunchConfig {
                grid_dim: ((tokens * heads) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, dev.clone()),
            Shape::from((heads, tokens, 128)),
        ))
    }
}

pub fn prepare_qkv(
    x: &Tensor,
    qweight: &Tensor,
    kweight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    if [qweight, kweight, cos, sin]
        .iter()
        .any(|t| !x.device().same_device(t.device()))
    {
        candle_core::bail!("QKV inputs must share a device");
    }
    x.apply_op3_no_bwd(
        qweight,
        kweight,
        &Prepare {
            cos,
            sin,
            eps: eps as f32,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{D, DType, Device};
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn qkv_fusion_matches_explicit_fp32_norm_and_bf16_rope_bitwise() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for (query_heads, n) in [16, 32]
            .into_iter()
            .flat_map(|q| [32, 96, 512].map(|n| (q, n)))
        {
            let heads = query_heads + 8;
            let x = Tensor::arange(0f32, ((n + 32) * heads * 128) as f32, &dev)?
                .affine(0.000123, -3.)?
                .sin()?
                .affine(8., -0.2)?
                .to_dtype(DType::BF16)?
                .reshape((n + 32, heads, 128))?
                .narrow(0, 32, n)?;
            let qweight = Tensor::arange(0f32, 128., &dev)?
                .affine(0.011, -0.6)?
                .to_dtype(DType::BF16)?;
            let kweight = qweight.affine(-0.71, 0.18)?;
            let angles = Tensor::arange(0f32, (n * 64) as f32, &dev)?
                .affine(0.13, 10.)?
                .reshape((1, n, 64))?;
            let cos = angles.cos()?.to_dtype(DType::BF16)?;
            let sin = angles.sin()?.to_dtype(DType::BF16)?;
            let normalize = |head: Tensor, weight: &Tensor| -> Result<Tensor> {
                let head = head.to_dtype(DType::F32)?;
                let inv = (head.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?
                    .sqrt()?
                    .recip()?;
                head.broadcast_mul(&inv)?
                    .to_dtype(DType::BF16)?
                    .broadcast_mul(weight)
            };
            let rotate = |head: Tensor| -> Result<Tensor> {
                let rot = head.narrow(2, 0, 64)?;
                let half = Tensor::cat(&[rot.narrow(2, 32, 32)?.neg()?, rot.narrow(2, 0, 32)?], 2)?;
                Tensor::cat(
                    &[
                        (rot.broadcast_mul(&cos)? + half.broadcast_mul(&sin)?)?,
                        head.narrow(2, 64, 64)?,
                    ],
                    2,
                )
            };
            let q = rotate(normalize(
                x.narrow(1, 0, query_heads)?.transpose(0, 1)?,
                &qweight,
            )?)?;
            let k = rotate(normalize(
                x.narrow(1, query_heads, 4)?.transpose(0, 1)?,
                &kweight,
            )?)?;
            let v = x.narrow(1, query_heads + 4, 4)?.transpose(0, 1)?;
            let expected = Tensor::cat(&[q, k, v], 0)?
                .flatten_all()?
                .to_vec1::<bf16>()?;
            let actual = prepare_qkv(&x, &qweight, &kweight, &cos, &sin, 1e-6)?
                .flatten_all()?
                .to_vec1::<bf16>()?;
            for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "tokens={n}, index={i}");
            }
        }
        Ok(())
    }
}

//! Block-causal BF16 attention with bounded on-chip score/probability storage.
use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp3, Layout, Result, Shape, Tensor};
use half::bf16;
struct Flash {
    offset: usize,
}
impl CustomOp3 for Flash {
    fn name(&self) -> &'static str {
        "minnow-flash-block32"
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
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
        v: &CudaStorage,
        vl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (heads, queries, dim) = ql.shape().dims3()?;
        let (kv, total, kdim) = kl.shape().dims3()?;
        if heads == 0
            || dim != 128
            || kdim != 128
            || kv == 0
            || !heads.is_multiple_of(kv)
            || heads > 64
            || queries == 0
            || !queries.is_multiple_of(32)
            || !self.offset.is_multiple_of(32)
            || self
                .offset
                .checked_add(queries)
                .is_none_or(|n| n > total || n > 131072)
            || kl.dims() != vl.dims()
            || [ql, kl, vl].iter().any(|l| {
                l.stride()[1] != 128 || l.stride()[2] != 1 || l.stride()[0] < l.dims()[1] * 128
            })
        {
            candle_core::bail!("invalid block attention shape or layout");
        }
        let extent = |l: &Layout| (l.dims()[0] - 1) * l.stride()[0] + l.dims()[1] * 128;
        let qs = q
            .as_cuda_slice::<bf16>()?
            .slice(ql.start_offset()..ql.start_offset() + extent(ql));
        let ks = k
            .as_cuda_slice::<bf16>()?
            .slice(kl.start_offset()..kl.start_offset() + extent(kl));
        let vs = v
            .as_cuda_slice::<bf16>()?
            .slice(vl.start_offset()..vl.start_offset() + extent(vl));
        let dev = &q.device;
        // SAFETY: one CTA owns each disjoint 64-token/head output block.
        let mut output = unsafe { dev.alloc::<bf16>(queries * heads * 128)? };
        let f = dev.get_or_load_custom_func(
            "minnow_flash_block32",
            "minnow-flash-v1",
            include_str!(concat!(env!("OUT_DIR"), "/minnow-flash.ptx")),
        )?;
        let (n, h, g, o) = (
            queries as i32,
            heads as i32,
            (heads / kv) as i32,
            self.offset as i32,
        );
        let (qh, kh, vh) = (ql.stride()[0], kl.stride()[0], vl.stride()[0]);
        // SAFETY: the forward kernel writes one log-sum-exp value per query/head.
        let mut lse = unsafe { dev.alloc::<f32>(queries * heads)? };
        {
            let mut call = f.builder();
            call.arg(&qs)
                .arg(&ks)
                .arg(&vs)
                .arg(&mut output)
                .arg(&n)
                .arg(&h)
                .arg(&g)
                .arg(&o)
                .arg(&qh)
                .arg(&kh)
                .arg(&vh)
                .arg(&mut lse);
            // SAFETY: checked dimensions, whole blocks and strided source extents.
            unsafe {
                call.launch(LaunchConfig {
                    grid_dim: (queries.div_ceil(64) as u32, 1, heads as u32),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 49152,
                })
            }
            .w()?;
        }
        Ok((
            CudaStorage::wrap_cuda_slice(output, dev.clone()),
            Shape::from((queries, heads * 128)),
        ))
    }
}
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, offset: usize) -> Result<Tensor> {
    if !q.device().same_device(k.device()) || !q.device().same_device(v.device()) {
        candle_core::bail!("attention inputs must share a device");
    }
    q.apply_op3_no_bwd(k, v, &Flash { offset })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    #[test]
    #[ignore = "requires CUDA"]
    fn online_attention_matches_block_mask_and_cache_offsets() -> anyhow::Result<()> {
        let dev = Device::new_cuda(0)?;
        let dim = 128;
        for (heads, kv) in [(16, 4), (32, 4)] {
            for (offset, n) in [(0, 32), (0, 160), (96, 64), (992, 32), (131040, 32)] {
                let total = offset + n;
                let make = |h, n, seed| -> Result<Tensor> {
                    Tensor::from_vec(
                        (0..h * n * dim)
                            .map(|i| ((i * seed % 1009) as f32 - 504.) / 800.)
                            .collect::<Vec<_>>(),
                        (h, n, dim),
                        &dev,
                    )?
                    .to_dtype(DType::BF16)
                };
                let q = make(heads, n, 17)?;
                let k = make(kv, total + 64, 31)?.narrow(1, 0, total)?;
                let v = make(kv, total + 64, 47)?.narrow(1, 0, total)?;
                let actual = attention(&q, &k, &v, offset)?;
                let qg = q.reshape((kv, heads / kv * n, dim))?;
                let scores = qg.matmul(&k.t()?)?;
                let p = super::super::block_softmax(
                    &scores,
                    n,
                    offset,
                    32,
                    (dim as f32).sqrt().recip(),
                )?;
                let expected = p
                    .matmul(&v)?
                    .reshape((heads, n, dim))?
                    .transpose(0, 1)?
                    .reshape((n, heads * dim))?;
                let a = actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let b = expected
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let max = a
                    .iter()
                    .zip(&b)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let rms = (a.iter().zip(&b).map(|(a, b)| (a - b).powi(2)).sum::<f32>()
                    / a.len() as f32)
                    .sqrt();
                eprintln!("heads={heads} offset={offset} n={n}: max={max} rms={rms}");
                assert!(max < 0.002 && rms < 0.0003);
                for start in (0..n).step_by(32) {
                    let single = attention(&q.narrow(1, start, 32)?, &k, &v, offset + start)?;
                    assert_eq!(
                        single.to_vec2::<bf16>()?,
                        actual.narrow(0, start, 32)?.to_vec2::<bf16>()?
                    );
                }
            }
        }
        Ok(())
    }
}

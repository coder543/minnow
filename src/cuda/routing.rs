use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp2, Layout, Result, Shape, Tensor};

pub(super) const PLAN_SIZE: usize = 817;
struct Route {
    scale: f32,
}
impl CustomOp2 for Route {
    fn name(&self) -> &'static str {
        "minnow-route-mini"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("device routing requires CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        b: &CudaStorage,
        bl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        if xl.dims() != [32, 256]
            || bl.dims() != [256]
            || !xl.is_contiguous()
            || !bl.is_contiguous()
            || !self.scale.is_finite()
            || self.scale <= 0.
        {
            candle_core::bail!("invalid mini routing shape or scale");
        }
        let dev = &x.device;
        let logits = x
            .as_cuda_slice::<f32>()?
            .slice(xl.start_offset()..xl.start_offset() + 8192);
        let bias = b
            .as_cuda_slice::<f32>()?
            .slice(bl.start_offset()..bl.start_offset() + 256);
        // SAFETY: the routing kernel initializes every field, including padding.
        let mut plan = unsafe { dev.alloc::<f32>(PLAN_SIZE)? };
        let f = dev.get_or_load_custom_func(
            "minnow_route_mini",
            "minnow-v1",
            include_str!(concat!(env!("OUT_DIR"), "/minnow.ptx")),
        )?;
        let mut call = f.builder();
        call.arg(&logits).arg(&bias).arg(&mut plan).arg(&self.scale);
        // SAFETY: this specialized kernel has exactly 256 threads and validated
        // 32x256 inputs. Internal ranks are bounded by the selected 48 experts.
        unsafe {
            call.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(plan, dev.clone()),
            Shape::from(PLAN_SIZE),
        ))
    }
}

/// Opaque, GPU-resident routing result. Construction validates its source shape.
pub struct RoutingPlan(pub(super) Tensor);
impl RoutingPlan {
    pub fn expert_ids(&self) -> Result<Tensor> {
        self.0
            .narrow(0, 49, 256)?
            .reshape((32, 8))?
            .to_dtype(candle_core::DType::U32)
    }
}

pub fn route_mini(logits: &Tensor, bias: &Tensor, scale: f32) -> Result<RoutingPlan> {
    if !logits.device().same_device(bias.device()) {
        candle_core::bail!("routing inputs must share a device");
    }
    Ok(RoutingPlan(
        logits.apply_op2_no_bwd(bias, &Route { scale })?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn device_routing_matches_host_ids_weights_and_active_rows_bitwise() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let mut config = crate::config::Config::load(std::path::Path::new("tests/fixtures/tiny"))
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        config.block_size = 32;
        config.num_experts = 256;
        config.num_experts_per_tok = 8;
        config.expert_capacity = 48;
        config.routed_scaling_factor = 2.5;
        for mode in 0..3 {
            let logits: Vec<f32> = (0..8192)
                .map(|i| match mode {
                    0 => (i as f32 * 0.019).sin() * 11.,
                    1 => 0.,
                    _ => ((i / 256 % 3 * 71 + i % 256) as f32 * 0.31).cos() * 40.,
                })
                .collect();
            let bias: Vec<f32> = (0..256)
                .map(|i| {
                    if mode == 1 {
                        0.
                    } else {
                        (i as f32 * 0.073).sin() * 0.25
                    }
                })
                .collect();
            let logits = Tensor::from_vec(logits, (32, 256), &dev)?;
            let scores = candle_nn::ops::sigmoid(&logits)?.to_vec2::<f32>()?;
            let (ids, weights) = crate::model::route(&scores, &bias, &config)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let plan = route_mini(&logits, &Tensor::from_vec(bias, 256, &dev)?, 2.5)?
                .0
                .to_vec1::<f32>()?;
            let mut active = ids.clone();
            active.sort_unstable();
            active.dedup();
            assert_eq!(plan[0] as usize, active.len());
            for (i, &id) in active.iter().enumerate() {
                assert_eq!(plan[1 + i] as u32, id);
            }
            for i in 0..256 {
                assert_eq!(plan[49 + i] as u32, ids[i], "mode={mode}, slot={i}");
                assert_eq!(
                    plan[561 + i].to_bits(),
                    weights[i].to_bits(),
                    "mode={mode}, slot={i}"
                );
                let rank = active.binary_search(&ids[i]).unwrap();
                assert_eq!(plan[305 + i] as usize, rank * 32 + i / 8);
            }
        }
        Ok(())
    }
}

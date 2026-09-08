//! Greedy token selection and confidence without a vocabulary-sized softmax.
use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape, Tensor};

struct Greedy;
impl CustomOp1 for Greedy {
    fn name(&self) -> &'static str {
        "minnow-greedy-confidence"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused predictions require CUDA")
    }
    fn cuda_fwd(&self, x: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let (rows, cols) = layout.shape().dims2()?;
        // IDs are represented exactly as floats in this small transfer buffer.
        if !layout.is_contiguous()
            || rows == 0
            || rows > u32::MAX as usize
            || cols == 0
            || cols > (1 << 24)
        {
            candle_core::bail!("invalid greedy prediction shape");
        }
        let tiles = cols.div_ceil(4096);
        let dev = &x.device;
        let input = x
            .as_cuda_slice::<f32>()?
            .slice(layout.start_offset()..layout.start_offset() + rows * cols);
        // SAFETY: both kernels write every element of their respective outputs.
        let mut partials = unsafe { dev.alloc::<f32>(rows * tiles * 3)? };
        let mut output = unsafe { dev.alloc::<f32>(rows * 2)? };
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/minnow.ptx"));
        let first = dev.get_or_load_custom_func("minnow_greedy_partials", "minnow-v1", ptx)?;
        let second = dev.get_or_load_custom_func("minnow_greedy_finish", "minnow-v1", ptx)?;
        let (cols, tiles) = (cols as i32, tiles as i32);
        let mut launch = first.builder();
        launch.arg(&input).arg(&mut partials).arg(&cols).arg(&tiles);
        // SAFETY: each block owns one row/tile; bounds are checked inside the kernel.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (tiles as u32, rows as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        let mut launch = second.builder();
        launch.arg(&partials).arg(&mut output).arg(&tiles);
        // SAFETY: each block reduces exactly the initialized partials for its row.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, dev.clone()),
            Shape::from((rows, 2)),
        ))
    }
}

/// Returns [token ID, probability of that token] for each row. Only this tiny
/// tensor needs a host transfer. Ties select the smallest vocabulary index.
pub fn greedy_confidence(logits: &Tensor) -> Result<Tensor> {
    logits.apply_op1_no_bwd(&Greedy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn greedy_matches_argmax_and_softmax_with_ties_and_extreme_logits() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for cols in [1, 127, 4096, 4103, 157184] {
            let mut values: Vec<f32> = (0..4 * cols)
                .map(|i| ((i * 71 % 1031) as f32 * 0.023).sin() * 12.)
                .collect();
            values[..cols].fill(0.); // Uniform row with many equal maxima.
            values[cols] = 10000.; // Extreme values must stay numerically stable.
            values[2 * cols - 1] = 10000.; // Tie, including across partial tiles.
            values[2 * cols..3 * cols].fill(f32::NEG_INFINITY);
            values[2 * cols] = -10000.;
            let exact: Vec<f64> = values
                .chunks_exact(cols)
                .map(|row| {
                    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                    1. / row.iter().map(|&v| (v as f64 - maximum).exp()).sum::<f64>()
                })
                .collect();
            // PyTorch argmax chooses the lowest vocabulary index on a tie.
            // Candle's CUDA reduction instead breaks ties by thread order.
            let ids: Vec<u32> = values
                .chunks_exact(cols)
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then(ib.cmp(ia)))
                        .unwrap()
                        .0 as u32
                })
                .collect();
            let ids = Tensor::from_vec(ids, 4, &dev)?;
            let x = Tensor::from_vec(values, (4, cols), &dev)?;
            let expected = candle_nn::ops::softmax_last_dim(&x)?
                .gather(&ids.unsqueeze(1)?, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let actual = greedy_confidence(&x)?.to_vec2::<f32>()?;
            for (row, (&id, &p)) in ids.to_vec1::<u32>()?.iter().zip(&expected).enumerate() {
                assert_eq!(actual[row][0] as u32, id, "cols={cols}, row={row}");
                assert!(
                    (actual[row][1] - p).abs() <= 1e-5 * p.max(1e-5),
                    "cols={cols}, row={row}: {} vs {p}",
                    actual[row][1]
                );
                assert!(
                    (actual[row][1] as f64 - exact[row]).abs() <= 5e-7 * exact[row],
                    "fused confidence differs from FP64 oracle: cols={cols}, row={row}"
                );
            }
        }
        Ok(())
    }
}

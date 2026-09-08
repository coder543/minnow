//! Fill final weight storage before exposing it as a tensor. CUDA uploads do
//! not allocate a temporary device tensor or zero the destination first.
use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};

pub(super) struct WeightBuffer {
    storage: Storage,
    dtype: DType,
    device: Device,
    count: usize,
    written: usize,
}
enum Storage {
    Tensor(Tensor),
    #[cfg(feature = "cuda")]
    Cuda(candle_core::CudaStorage),
}
impl WeightBuffer {
    pub(super) fn new(count: usize, dtype: DType, device: &Device) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(dev) = device {
            macro_rules! allocate {
                ($ty:ty) => {{
                    // SAFETY: this storage stays private until every element is
                    // initialized by write; failures only free the allocation.
                    let data = unsafe { dev.alloc::<$ty>(count) }?;
                    Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(data, dev.clone()))
                }};
            }
            let storage = match dtype {
                DType::U8 => allocate!(u8),
                DType::BF16 => allocate!(half::bf16),
                DType::F16 => allocate!(half::f16),
                DType::F32 => allocate!(f32),
                _ => Storage::Tensor(Tensor::zeros(count, dtype, device)?),
            };
            return Ok(Self {
                storage,
                dtype,
                device: device.clone(),
                count,
                written: 0,
            });
        }
        Ok(Self {
            storage: Storage::Tensor(Tensor::zeros(count, dtype, device)?),
            dtype,
            device: device.clone(),
            count,
            written: 0,
        })
    }
    pub(super) fn write(&mut self, bytes: &[u8], source: DType) -> Result<()> {
        ensure!(
            bytes.len().is_multiple_of(source.size_in_bytes()),
            "partial weight element"
        );
        let n = bytes.len() / source.size_in_bytes();
        ensure!(
            n <= self.count - self.written,
            "weight upload exceeds allocation"
        );
        match &mut self.storage {
            Storage::Tensor(out) => {
                let chunk = Tensor::from_raw_buffer(bytes, source, &[n], &self.device)?
                    .to_dtype(self.dtype)?;
                out.slice_set(&chunk, 0, self.written)?;
                self.device.synchronize()?;
            }
            #[cfg(feature = "cuda")]
            Storage::Cuda(out) => {
                if source == self.dtype {
                    match self.dtype {
                        DType::U8 => upload::<u8>(out, self.written, bytes)?,
                        DType::BF16 => upload::<half::bf16>(out, self.written, bytes)?,
                        DType::F16 => upload::<half::f16>(out, self.written, bytes)?,
                        DType::F32 => upload::<f32>(out, self.written, bytes)?,
                        _ => unreachable!(),
                    }
                } else {
                    // Dtype conversion is needed for small router/norm tensors.
                    // Preserve Candle's conversion semantics with bounded scratch.
                    let chunk = Tensor::from_raw_buffer(bytes, source, &[n], &self.device)?
                        .to_dtype(self.dtype)?;
                    let (storage, _) = chunk.storage_and_layout();
                    let candle_core::Storage::Cuda(input) = &*storage else {
                        unreachable!()
                    };
                    match self.dtype {
                        DType::U8 => copy::<u8>(out, self.written, input, n)?,
                        DType::BF16 => copy::<half::bf16>(out, self.written, input, n)?,
                        DType::F16 => copy::<half::f16>(out, self.written, input, n)?,
                        DType::F32 => copy::<f32>(out, self.written, input, n)?,
                        _ => unreachable!(),
                    }
                }
            }
        }
        self.written += n;
        Ok(())
    }
    pub(super) fn finish(self) -> Result<Tensor> {
        ensure!(self.written == self.count, "incomplete weight upload");
        Ok(match self.storage {
            Storage::Tensor(t) => t,
            #[cfg(feature = "cuda")]
            Storage::Cuda(s) => Tensor::from_storage(
                candle_core::Storage::Cuda(s),
                self.count,
                candle_core::op::BackpropOp::none(),
                false,
            ),
        })
    }
}

#[cfg(feature = "cuda")]
fn upload<
    T: candle_core::cuda_backend::CudaDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
>(
    out: &mut candle_core::CudaStorage,
    offset: usize,
    bytes: &[u8],
) -> Result<()> {
    let stream = out.device.cuda_stream();
    // SAFETY: callers specialize T to u8/f16/bf16/f32, which accept all bit
    // patterns. Read buffers may have unaligned safetensors offsets.
    let (prefix, aligned, suffix) = unsafe { bytes.align_to::<T>() };
    let owned;
    let values = if prefix.is_empty() && suffix.is_empty() {
        aligned
    } else {
        owned = bytes
            .chunks(std::mem::size_of::<T>())
            .map(|b| {
                // SAFETY: each chunk contains one complete T; alignment is not required.
                unsafe { std::ptr::read_unaligned(b.as_ptr().cast::<T>()) }
            })
            .collect::<Vec<_>>();
        &owned
    };
    let mut destination = out
        .as_cuda_slice_mut::<T>()?
        .slice_mut(offset..offset + values.len());
    stream.memcpy_htod(values, &mut destination)?;
    // The read pool may reuse the host bytes as soon as this call returns.
    stream.synchronize()?;
    Ok(())
}
#[cfg(feature = "cuda")]
fn copy<
    T: candle_core::cuda_backend::CudaDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
>(
    out: &mut candle_core::CudaStorage,
    offset: usize,
    input: &candle_core::CudaStorage,
    n: usize,
) -> Result<()> {
    let stream = out.device.cuda_stream();
    let mut destination = out.as_cuda_slice_mut::<T>()?.slice_mut(offset..offset + n);
    stream.memcpy_dtod(input.as_cuda_slice::<T>()?, &mut destination)?;
    stream.synchronize()?;
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires a CUDA device"]
    fn direct_upload_handles_unaligned_chunks_and_dtype_conversion() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let values = [0.0f32, 1.25, -2.5, 4.0, 0.125, 64.0, -0.5, 8.0, 16.0];
        for source in [DType::F32, DType::F16, DType::BF16] {
            let cpu = Tensor::from_slice(&values, values.len(), &Device::Cpu)?.to_dtype(source)?;
            let raw: Vec<u8> = match source {
                DType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                DType::F16 => cpu
                    .to_vec1::<half::f16>()?
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect(),
                DType::BF16 => cpu
                    .to_vec1::<half::bf16>()?
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect(),
                _ => unreachable!(),
            };
            let mut unaligned = vec![0u8];
            unaligned.extend_from_slice(&raw);
            let cut = 5 * source.size_in_bytes();
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let mut out = WeightBuffer::new(values.len(), dtype, &device)?;
                out.write(&unaligned[1..1 + cut], source)?;
                out.write(&unaligned[1 + cut..], source)?;
                assert_eq!(
                    out.finish()?.to_dtype(DType::F32)?.to_vec1::<f32>()?,
                    cpu.to_dtype(dtype)?
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?
                );
            }
        }
        let bytes: Vec<u8> = (0..37).map(|i| i * 7).collect();
        let mut codes = WeightBuffer::new(bytes.len(), DType::U8, &device)?;
        codes.write(&bytes[..13], DType::U8)?;
        codes.write(&bytes[13..], DType::U8)?;
        assert_eq!(codes.finish()?.to_vec1::<u8>()?, bytes);
        let mut incomplete = WeightBuffer::new(2, DType::U8, &device)?;
        assert!(incomplete.write(&[1, 2, 3], DType::U8).is_err());
        incomplete.write(&[1], DType::U8)?;
        assert!(incomplete.finish().is_err());
        Ok(())
    }
}

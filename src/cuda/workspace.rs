//! Bound reusable allocation storage across synchronizations. The limit follows
//! live allocations; retained pages are scratch, not another model weight set.
use anyhow::Result;
use candle_core::{Device, cuda_backend::cudarc::driver::sys};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct WorkspaceCache {
    device: Device,
    extra: AtomicU64,
}
impl WorkspaceCache {
    pub fn new(device: &Device) -> Option<Self> {
        let supported =
            matches!(device,Device::Cuda(cuda) if cuda.cuda_stream().context().has_async_alloc());
        supported.then(|| Self {
            device: device.clone(),
            extra: AtomicU64::new(2 * 1024 * 1024 * 1024),
        })
    }
    fn pool(&self) -> Result<sys::CUmemoryPool> {
        let Device::Cuda(device) = &self.device else {
            unreachable!()
        };
        let stream = device.cuda_stream();
        let context = stream.context();
        context.bind_to_thread()?;
        let mut pool = std::ptr::null_mut();
        // SAFETY: valid current context/device and a correctly typed output.
        unsafe {
            sys::cuDeviceGetMemPool(&mut pool, context.cu_device()).result()?;
        }
        Ok(pool)
    }
    fn threshold(&self, mut bytes: u64) -> Result<()> {
        // SAFETY: this attribute takes a pointer to u64.
        unsafe {
            sys::cuMemPoolSetAttribute(
                self.pool()?,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut bytes as *mut u64).cast(),
            )
            .result()?;
        }
        Ok(())
    }
    pub fn set(&self, bytes: u64) -> Result<()> {
        self.extra.store(bytes, Ordering::Relaxed);
        if bytes == 0 {
            self.threshold(0)
        } else {
            self.refresh()
        }
    }
    pub fn refresh(&self) -> Result<()> {
        let extra = self.extra.load(Ordering::Relaxed);
        if extra == 0 {
            return Ok(());
        }
        let mut live = 0u64;
        // SAFETY: this attribute writes a u64 to the supplied valid pointer.
        unsafe {
            sys::cuMemPoolGetAttribute(
                self.pool()?,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                (&mut live as *mut u64).cast(),
            )
            .result()?;
        }
        self.threshold(live.saturating_add(extra))
    }
}
impl Drop for WorkspaceCache {
    fn drop(&mut self) {
        let release = || -> Result<()> {
            self.threshold(0)?;
            self.device.synchronize()?;
            // SAFETY: weight fields drop before this field; trimming affects
            // unused pages and cannot invalidate remaining live tensors.
            unsafe {
                sys::cuMemPoolTrimTo(self.pool()?, 0).result()?;
            }
            Ok(())
        };
        if let Err(error) = release() {
            tracing::warn!(%error,"releasing CUDA workspace cache");
        }
    }
}

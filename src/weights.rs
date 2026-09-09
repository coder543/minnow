//! Bounded checkpoint loading: direct reads into an 8 MiB staging buffer and
//! copies into the final allocation. No checkpoint mmap or complete host copy.
use crate::container::{Container, Encoding, Region};
use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::{Init, var_builder::SimpleBackend};
use serde::Deserialize;
use std::{
    collections::{BTreeSet, HashMap},
    fs::{File, OpenOptions},
    io::Read,
    os::{
        fd::AsRawFd,
        unix::fs::{FileExt, OpenOptionsExt},
    },
    path::Path,
    sync::Mutex,
    time::{Duration, Instant},
};

#[cfg(feature = "cuda")]
mod allocation;
mod buffer;
mod reader;
use buffer::WeightBuffer;

const ALIGN: usize = 4096;
pub const STAGING_BYTES: usize = 8 * 1024 * 1024;
const LARGE_MODEL: u64 = 256 * 1024 * 1024;
pub const DEFAULT_MEMORY_RESERVE_MIB: u64 = 16 * 1024;

pub fn available_memory() -> Result<u64> {
    let mem = std::fs::read_to_string("/proc/meminfo")?;
    let value = mem
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .context("MemAvailable is missing")?;
    Ok(value
        .split_whitespace()
        .next()
        .context("invalid MemAvailable")?
        .parse::<u64>()?
        * 1024)
}

#[derive(Clone, Debug)]
struct Location {
    file: usize,
    offset: u64,
    bytes: usize,
    shape: Vec<usize>,
    dtype: DType,
    encoding: Encoding,
    checksum: Option<String>,
    scales: Option<Region>,
    group_size: usize,
    global_scale: Option<f32>,
}

// O_DIRECT needs alignment that Vec<u8> does not guarantee. This is an ordinary
// aligned staging allocation; the loader never maps checkpoint files.
pub(crate) struct Staging {
    pointer: std::ptr::NonNull<u8>,
    len: usize,
}
impl Staging {
    pub(crate) fn new() -> Result<Self> {
        Self::with_capacity(STAGING_BYTES + 2 * ALIGN)
    }
    fn with_capacity(len: usize) -> Result<Self> {
        let len = len.max(ALIGN).div_ceil(ALIGN) * ALIGN;
        let layout = std::alloc::Layout::from_size_align(len, ALIGN)?;
        // SAFETY: valid nonempty layout, released with this layout in Drop.
        let p = unsafe { std::alloc::alloc_zeroed(layout) };
        Ok(Self {
            pointer: std::ptr::NonNull::new(p).context("allocating direct-I/O staging")?,
            len,
        })
    }
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: exclusive access to the live allocation of self.len bytes.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.len) }
    }
}
// SAFETY: staging is accessed under the loader's mutex and owns its allocation.
unsafe impl Send for Staging {}
impl Drop for Staging {
    fn drop(&mut self) {
        let layout =
            std::alloc::Layout::from_size_align(self.len, ALIGN).expect("valid staging layout");
        // SAFETY: the pointer was allocated with this layout and has no live users.
        unsafe {
            std::alloc::dealloc(self.pointer.as_ptr(), layout);
        }
    }
}

pub struct WeightLoader {
    files: Vec<File>,
    tensors: HashMap<String, Location>,
    staging: Mutex<Staging>,
    read_buffers: Mutex<Vec<Staging>>,
    read_pool: std::sync::OnceLock<Result<rayon::ThreadPool, String>>,
    large: bool,
    reserve_bytes: u64,
    pub elements: u64,
    timings: Mutex<ReadTimings>,
    verify_checksums: bool,
    #[cfg(feature = "cuda")]
    allocations: Option<allocation::Allocations>,
}

#[derive(Default)]
struct ReadTimings {
    bytes: u64,
    reads: u64,
    read: Duration,
    read_wall: Duration,
    read_wait: Duration,
    allocation_wait: Duration,
    staging: Duration,
    checksum: Duration,
    consume: Duration,
}
impl Drop for WeightLoader {
    fn drop(&mut self) {
        if let Ok(t) = self.timings.get_mut()
            && t.bytes > 0
        {
            tracing::info!(
                bytes = t.bytes,
                reads = t.reads,
                read_seconds = t.read.as_secs_f64(),
                read_wall_seconds = t.read_wall.as_secs_f64(),
                read_wait_seconds = t.read_wait.as_secs_f64(),
                allocation_wait_seconds = t.allocation_wait.as_secs_f64(),
                staging_seconds = t.staging.as_secs_f64(),
                checksum_seconds = t.checksum.as_secs_f64(),
                consume_seconds = t.consume.as_secs_f64(),
                "checkpoint I/O timings"
            );
        }
    }
}

pub(crate) struct SharedLoader(pub std::sync::Arc<WeightLoader>);
impl SimpleBackend for SharedLoader {
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        self.0.get(s, name, h, dtype, dev)
    }
    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.0.get_unchecked(name, dtype, dev)
    }
    fn contains_tensor(&self, name: &str) -> bool {
        self.0.contains_tensor(name)
    }
}
impl WeightLoader {
    pub fn open(path: &Path) -> Result<Self> {
        if path.is_file() {
            return Self::open_container(path);
        }
        #[derive(Deserialize)]
        struct Index {
            weight_map: HashMap<String, String>,
        }
        let names = if path.join("model.safetensors.index.json").exists() {
            let index: Index =
                serde_json::from_slice(&std::fs::read(path.join("model.safetensors.index.json"))?)?;
            index.weight_map.into_values().collect::<BTreeSet<_>>()
        } else {
            BTreeSet::from(["model.safetensors".to_owned()])
        };
        let mut files = Vec::new();
        let mut tensors = HashMap::new();
        let mut elements = 0u64;
        let mut total_bytes = 0u64;
        for name in names {
            ensure!(
                Path::new(&name).file_name() == Some(std::ffi::OsStr::new(&name)),
                "invalid shard filename"
            );
            let filename = path.join(name);
            let mut header_file = File::open(&filename)?;
            let size = header_file.metadata()?.len();
            let mut length = [0u8; 8];
            header_file.read_exact(&mut length)?;
            let header_len = u64::from_le_bytes(length);
            ensure!(
                header_len <= 16 * 1024 * 1024 && header_len + 8 <= size,
                "invalid safetensors header length"
            );
            let mut bytes = vec![0; header_len as usize];
            header_file.read_exact(&mut bytes)?;
            let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&bytes)?;
            // Discard header read-ahead; do not accumulate a file-cache weight copy.
            unsafe {
                libc::posix_fadvise(header_file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
            }
            drop(header_file);
            let file = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(&filename)
                .with_context(|| {
                    format!(
                        "{} must support O_DIRECT for bounded loading",
                        filename.display()
                    )
                })?;
            for (name, info) in header {
                if name == "__metadata__" {
                    continue;
                }
                #[derive(Deserialize)]
                struct Info {
                    dtype: String,
                    shape: Vec<usize>,
                    data_offsets: [usize; 2],
                }
                let info: Info = serde_json::from_value(info)?;
                let dtype = match info.dtype.as_str() {
                    "BF16" => DType::BF16,
                    "F32" => DType::F32,
                    "F16" => DType::F16,
                    other => bail!("unsupported weight dtype {other}"),
                };
                let n = info
                    .shape
                    .iter()
                    .try_fold(1usize, |a, &b| a.checked_mul(b))
                    .context("weight shape overflow")?;
                let bytes = n
                    .checked_mul(dtype.size_in_bytes())
                    .context("weight byte size overflow")?;
                let [begin, end] = info.data_offsets;
                ensure!(
                    end >= begin
                        && end - begin == bytes
                        && (end as u64)
                            .checked_add(header_len + 8)
                            .is_some_and(|v| v <= size),
                    "invalid offsets for {name}"
                );
                elements = elements
                    .checked_add(n as u64)
                    .context("model size overflow")?;
                total_bytes += bytes as u64;
                ensure!(
                    tensors
                        .insert(
                            name.clone(),
                            Location {
                                file: files.len(),
                                offset: 8 + header_len + begin as u64,
                                bytes,
                                shape: info.shape,
                                dtype,
                                encoding: Encoding::from_dtype(dtype)?,
                                checksum: None,
                                scales: None,
                                group_size: 0,
                                global_scale: None,
                            }
                        )
                        .is_none(),
                    "duplicate tensor {name}"
                );
            }
            files.push(file);
        }
        Ok(Self {
            files,
            tensors,
            staging: Mutex::new(Staging::new()?),
            read_buffers: Mutex::new(Vec::new()),
            read_pool: std::sync::OnceLock::new(),
            large: total_bytes > LARGE_MODEL,
            reserve_bytes: DEFAULT_MEMORY_RESERVE_MIB * 1024 * 1024,
            elements,
            timings: Mutex::new(ReadTimings::default()),
            verify_checksums: false,
            #[cfg(feature = "cuda")]
            allocations: None,
        })
    }
    fn open_container(path: &Path) -> Result<Self> {
        let container = Container::open(path)?;
        let mut tensors = HashMap::new();
        let mut elements = 0;
        for (name, info) in &container.manifest.tensors {
            elements += info.elements()? as u64;
            tensors.insert(
                name.clone(),
                Location {
                    file: 0,
                    offset: info.data.offset,
                    bytes: info.data.bytes as usize,
                    shape: info.shape.clone(),
                    dtype: if info.encoding.quantized() {
                        DType::U8
                    } else {
                        info.encoding.float_dtype()?
                    },
                    encoding: info.encoding,
                    checksum: Some(info.data.blake3.clone()),
                    scales: info.scales.clone(),
                    group_size: info.group_size,
                    global_scale: info.global_scale,
                },
            );
        }
        Ok(Self {
            files: vec![
                OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(path)?,
            ],
            large: container.manifest.weight_bytes() > LARGE_MODEL,
            reserve_bytes: DEFAULT_MEMORY_RESERVE_MIB * 1024 * 1024,
            tensors,
            staging: Mutex::new(Staging::new()?),
            read_buffers: Mutex::new(Vec::new()),
            read_pool: std::sync::OnceLock::new(),
            elements,
            timings: Mutex::new(ReadTimings::default()),
            verify_checksums: false,
            #[cfg(feature = "cuda")]
            allocations: None,
        })
    }
    /// Metadata only; no tensor bytes are loaded by an inventory operation.
    pub fn inventory(&self) -> Vec<(String, Vec<usize>, Encoding)> {
        let mut entries: Vec<_> = self
            .tensors
            .iter()
            .map(|(name, t)| (name.clone(), t.shape.clone(), t.encoding))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
    }
    fn visit(
        &self,
        name: &str,
        part: &Location,
        mut consume: impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        let mut staging = self
            .staging
            .lock()
            .map_err(|_| anyhow::anyhow!("weight staging lock poisoned"))?;
        let mut copied = 0;
        let mut timings = ReadTimings::default();
        let mut hash = blake3::Hasher::new();
        while copied < part.bytes {
            let bytes = (part.bytes - copied).min(STAGING_BYTES);
            let offset = part.offset + copied as u64;
            let aligned = offset / ALIGN as u64 * ALIGN as u64;
            let skip = (offset - aligned) as usize;
            let requested = (skip + bytes).div_ceil(ALIGN) * ALIGN;
            let buffer = &mut staging.bytes()[..requested];
            let started = Instant::now();
            let got = loop {
                match self.files[part.file].read_at(buffer, aligned) {
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    result => break result.with_context(|| format!("direct read of {name}"))?,
                }
            };
            ensure!(got >= skip + bytes, "short direct read for {name}");
            timings.read += started.elapsed();
            timings.read_wall = timings.read;
            timings.reads += 1;
            timings.bytes += bytes as u64;
            let bytes = &buffer[skip..skip + bytes];
            let started = Instant::now();
            if self.verify_checksums && part.checksum.is_some() {
                hash.update(bytes);
            }
            timings.checksum += started.elapsed();
            let started = Instant::now();
            consume(bytes)?;
            timings.consume += started.elapsed();
            copied += bytes.len();
        }
        if self.verify_checksums
            && let Some(expected) = &part.checksum
        {
            ensure!(
                hash.finalize()
                    .to_hex()
                    .as_str()
                    .eq_ignore_ascii_case(expected),
                "tensor checksum mismatch: {name}"
            );
        }
        let mut total = self.timings.lock().unwrap();
        total.bytes += timings.bytes;
        total.reads += timings.reads;
        total.read += timings.read;
        total.read_wall += timings.read_wall;
        total.checksum += timings.checksum;
        total.consume += timings.consume;
        Ok(())
    }
    pub(crate) fn validate_payloads(mut self) -> Result<()> {
        self.verify_checksums = true;
        let mut parts: Vec<_> = self.tensors.iter().collect();
        parts.sort_unstable_by_key(|(_, p)| (p.file, p.offset));
        for (name, part) in parts {
            ensure!(
                part.checksum.is_some(),
                "validation requires a .mnw container"
            );
            self.visit_tensor(name, |_| Ok(()))?;
            if part.scales.is_some() {
                self.visit_scales(name, |_| Ok(()))?;
            }
        }
        Ok(())
    }
    /// Pipeline bounded direct reads with consumption into final storage.
    fn visit_parts(
        &self,
        name: &str,
        parts: &[Location],
        consume: impl FnMut(usize, bool, &[u8]) -> Result<()>,
    ) -> Result<()> {
        self.read_parts(name, parts, consume)
    }
    pub(crate) fn start_allocations(&mut self, dtype: DType, device: &Device) -> Result<()> {
        #[cfg(feature = "cuda")]
        if device.is_cuda() {
            self.allocations = Some(allocation::Allocations::start(
                &self.tensors,
                dtype,
                device,
            )?);
        }
        #[cfg(not(feature = "cuda"))]
        let _ = (dtype, device);
        Ok(())
    }
    fn buffer(
        &self,
        name: &str,
        scales: bool,
        count: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<WeightBuffer> {
        #[cfg(feature = "cuda")]
        if device.is_cuda()
            && let Some(allocations) = &self.allocations
        {
            let start = Instant::now();
            let result = allocations.take(name, scales, count, dtype, device);
            self.timings.lock().unwrap().allocation_wait += start.elapsed();
            return result;
        }
        let _ = (name, scales);
        WeightBuffer::new(count, dtype, device)
    }
    /// Bounded raw traversal for conversion; validation is a separate operation.
    pub fn visit_tensor(&self, name: &str, consume: impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
        self.visit(
            name,
            self.tensors.get(name).context("missing tensor")?,
            consume,
        )
    }
    pub fn quantization_group(&self, name: &str) -> Result<usize> {
        Ok(self.tensors.get(name).context("missing tensor")?.group_size)
    }
    pub fn global_scale(&self, name: &str) -> Result<Option<f32>> {
        Ok(self
            .tensors
            .get(name)
            .context("missing tensor")?
            .global_scale)
    }
    /// Bounded traversal of the separate scale region.
    pub fn visit_scales(&self, name: &str, consume: impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
        let part = self.tensors.get(name).context("missing tensor")?;
        let region = part
            .scales
            .as_ref()
            .context("missing quantization scales")?;
        let location = Location {
            offset: region.offset,
            bytes: region.bytes as usize,
            checksum: Some(region.blake3.clone()),
            dtype: DType::F16,
            ..part.clone()
        };
        self.visit(name, &location, consume)
    }
    pub fn set_memory_reserve_mib(&mut self, mib: u64) -> Result<()> {
        self.reserve_bytes = mib
            .checked_mul(1024 * 1024)
            .context("memory reserve overflow")?;
        Ok(())
    }
    pub fn check_memory(&self, dtype: DType) -> Result<()> {
        if !self.large {
            return Ok(());
        }
        let required = self.tensors.iter().try_fold(0u64, |sum, (name, t)| {
            let bytes = if t.encoding.quantized() {
                t.bytes as u64 + t.scales.as_ref().map_or(0, |s| s.bytes)
            } else {
                let size = if name.ends_with(".gate.weight") {
                    4
                } else {
                    dtype.size_in_bytes()
                };
                t.bytes as u64 / t.dtype.size_in_bytes() as u64 * size as u64
            };
            sum.checked_add(bytes).context("model size overflow")
        })?;
        let available = available_memory()?;
        ensure!(
            available
                >= required
                    .checked_add(self.reserve_bytes)
                    .context("memory requirement overflow")?,
            "loading needs {:.2} GiB for weights plus {:.2} GiB system headroom; only {:.2} GiB available",
            required as f64 / 2f64.powi(30),
            self.reserve_bytes as f64 / 2f64.powi(30),
            available as f64 / 2f64.powi(30)
        );
        tracing::info!(
            weight_gib = required as f64 / 2f64.powi(30),
            staging_mib = STAGING_BYTES / 1024 / 1024,
            "loading one resident weight copy with direct I/O"
        );
        Ok(())
    }
    fn parts(&self, name: &str, shape: &Shape) -> Result<Vec<Location>> {
        if let Some(loc) = self.tensors.get(name) {
            ensure!(loc.shape == shape.dims(), "shape mismatch for {name}");
            return Ok(vec![loc.clone()]);
        }
        // Packed expert projections are assembled directly in their final storage.
        let (prefix, projection) = name
            .rsplit_once(".experts.")
            .with_context(|| format!("missing weight {name}"))?;
        ensure!(
            ["gate_proj.weight", "up_proj.weight", "down_proj.weight"].contains(&projection),
            "missing weight {name}"
        );
        let (count, out, input) = shape.dims3()?;
        (0..count)
            .map(|e| {
                let key = format!("{prefix}.experts.{e}.{projection}");
                let loc = self
                    .tensors
                    .get(&key)
                    .with_context(|| format!("missing weight {key}"))?;
                ensure!(loc.shape == [out, input], "shape mismatch for {key}");
                Ok(loc.clone())
            })
            .collect()
    }
    pub(crate) fn quantized_experts(
        &self,
        name: &str,
        shape: (usize, usize, usize),
        device: &Device,
    ) -> Result<Option<crate::quant::Weights>> {
        let parts = self.parts(name, &shape.into())?;
        if parts.iter().all(|t| !t.encoding.quantized()) {
            return Ok(None);
        }
        let first = &parts[0];
        ensure!(
            parts
                .iter()
                .all(|t| t.encoding == first.encoding && t.group_size == first.group_size),
            "all experts within one projection must use the same encoding and group size: {name}"
        );
        let encoding = first.encoding;
        let group_size = first.group_size;
        let count = shape.0 * shape.1 * shape.2;
        let mut codes = self.buffer(
            name,
            false,
            if encoding.int8() { count } else { count / 2 },
            DType::U8,
            device,
        )?;
        let native = encoding == Encoding::Nvfp4;
        let scale_dtype = if native { DType::U8 } else { DType::F16 };
        let globals = if native {
            Some(Tensor::from_vec(
                parts
                    .iter()
                    .map(|p| p.global_scale.context("missing NVFP4 global scale"))
                    .collect::<Result<Vec<_>>>()?,
                shape.0,
                device,
            )?)
        } else {
            None
        };
        let mut scales = self.buffer(name, true, count / group_size, scale_dtype, device)?;
        self.visit_parts(name, &parts, |_, is_scale, bytes| {
            if !is_scale {
                codes.write(bytes, DType::U8)?;
            } else {
                if native {
                    ensure!(
                        bytes.iter().all(|s| *s < 127),
                        "invalid NVFP4 scale in {name}"
                    );
                } else {
                    for value in bytes.as_chunks::<2>().0.iter() {
                        let value = half::f16::from_bits(u16::from_le_bytes(*value));
                        ensure!(
                            value.is_finite() && value.to_f32() > 0.,
                            "invalid scale in {name}"
                        );
                    }
                }
                scales.write(bytes, scale_dtype)?;
            }
            Ok(())
        })?;
        let codes = codes.finish()?;
        let scales = scales.finish()?;
        Ok(Some(crate::quant::Weights {
            int8_activations: false,
            codes,
            scales,
            encoding,
            group_size,
            shape,
            global_scales: globals,
        }))
    }
    fn load(&self, name: &str, shape: Shape, dtype: DType, device: &Device) -> Result<Tensor> {
        let parts = self.parts(name, &shape)?;
        ensure!(
            parts.iter().all(|t| !t.encoding.quantized()),
            "quantized tensor {name} requires a quantized execution path"
        );
        if self.large {
            ensure!(
                available_memory()? > self.reserve_bytes,
                "memory headroom fell below {} MiB; stopping weight load",
                self.reserve_bytes / (1024 * 1024)
            );
        }
        let mut out = self.buffer(name, false, shape.elem_count(), dtype, device)?;
        self.visit_parts(name, &parts, |index, is_scale, buffer| {
            ensure!(!is_scale, "unexpected scale for {name}");
            let part = &parts[index];
            out.write(buffer, part.dtype)?;
            Ok(())
        })?;
        Ok(out.finish()?.reshape(shape)?)
    }
}
impl SimpleBackend for WeightLoader {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _h: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        self.load(name, s, dtype, dev)
            .map_err(|e| candle_core::Error::Msg(format!("{e:#}")))
    }
    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let shape = self
            .tensors
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing weight {name}")))?
            .shape
            .clone();
        self.get(shape.into(), name, Init::Const(0.0), dtype, dev)
    }
    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    pub(super) struct Directory(pub(super) PathBuf);
    impl Directory {
        pub(super) fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "minnow-weights-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn direct_reads_cross_chunks_and_pack_experts_without_reordering() -> Result<()> {
        let dir = Directory::new();
        // A non-page-aligned tensor spanning more than one staging chunk, plus
        // small adjacent tensors whose aligned reads overlap on disk.
        let n = STAGING_BYTES / 4 + 133;
        let values: Vec<f32> = (0..n).map(|i| (i % 1009) as f32 / 16.0).collect();
        let mut weights = HashMap::from([(
            "large".to_owned(),
            Tensor::from_vec(values.clone(), n, &Device::Cpu)?,
        )]);
        for e in 0..137 {
            weights.insert(
                format!("mlp.experts.{e}.gate_proj.weight"),
                Tensor::from_vec(
                    (0..15).map(|i| (e * 100 + i) as f32).collect::<Vec<_>>(),
                    (3, 5),
                    &Device::Cpu,
                )?,
            );
        }
        candle_core::safetensors::save(&weights, dir.0.join("model.safetensors"))?;
        drop(weights);
        let loader = WeightLoader::open(&dir.0)?;
        loader.check_memory(DType::F32)?;
        let actual = loader
            .load("large", n.into(), DType::F32, &Device::Cpu)?
            .to_vec1::<f32>()?;
        assert_eq!(actual, values);
        let packed = loader.load(
            "mlp.experts.gate_proj.weight",
            (137, 3, 5).into(),
            DType::F32,
            &Device::Cpu,
        )?;
        let expected: Vec<f32> = (0..137)
            .flat_map(|e| (0..15).map(move |i| (e * 100 + i) as f32))
            .collect();
        assert_eq!(packed.flatten_all()?.to_vec1::<f32>()?, expected);
        assert!(
            loader
                .load("large", (n - 1).into(), DType::F32, &Device::Cpu)
                .is_err()
        );
        assert!(
            loader
                .load(
                    "mlp.experts.gate_proj.weight",
                    (138, 3, 5).into(),
                    DType::F32,
                    &Device::Cpu
                )
                .is_err()
        );
        Ok(())
    }
}

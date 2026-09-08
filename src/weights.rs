//! Bounded checkpoint loading: direct reads into an 8 MiB staging buffer and
//! copies into the final allocation. No checkpoint mmap or complete host copy.
use crate::container::{Container, Encoding, Region};
use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::{Init, var_builder::SimpleBackend};
use fs2::FileExt as LockExt;
use serde::Deserialize;
use std::{
    collections::{BTreeSet, HashMap},
    fs::{File, OpenOptions},
    io::Read,
    os::{
        fd::AsRawFd,
        unix::fs::{FileExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
    sync::Mutex,
};

const ALIGN: usize = 4096;
pub const STAGING_BYTES: usize = 8 * 1024 * 1024;
const LARGE_MODEL: u64 = 256 * 1024 * 1024;
const RESERVE: u64 = 16 * 1024 * 1024 * 1024;

pub struct ModelLease {
    _file: File,
}
impl ModelLease {
    pub fn acquire() -> Result<Self> {
        // Same path and flock protocol as scripts/weight_io.py.
        let path = PathBuf::from(format!("/tmp/minnow-model-{}.lock", unsafe {
            libc::geteuid()
        }));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(&path)?;
        file.try_lock_exclusive().with_context(|| {
            format!(
                "another minnow/reference model is resident; lock {}",
                path.display()
            )
        })?;
        Ok(Self { _file: file })
    }
}

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
}

// O_DIRECT needs alignment that Vec<u8> does not guarantee. This is an ordinary
// aligned staging allocation; the loader never maps checkpoint files.
pub(crate) struct Staging {
    pointer: std::ptr::NonNull<u8>,
    len: usize,
}
impl Staging {
    pub(crate) fn new() -> Result<Self> {
        let len = STAGING_BYTES + 2 * ALIGN;
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
    large: bool,
    pub elements: u64,
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
            large: total_bytes > LARGE_MODEL,
            elements,
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
            tensors,
            staging: Mutex::new(Staging::new()?),
            elements,
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
        let mut hash = blake3::Hasher::new();
        while copied < part.bytes {
            let bytes = (part.bytes - copied).min(STAGING_BYTES);
            let offset = part.offset + copied as u64;
            let aligned = offset / ALIGN as u64 * ALIGN as u64;
            let skip = (offset - aligned) as usize;
            let requested = (skip + bytes).div_ceil(ALIGN) * ALIGN;
            let buffer = &mut staging.bytes()[..requested];
            let got = loop {
                match self.files[part.file].read_at(buffer, aligned) {
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    result => break result.with_context(|| format!("direct read of {name}"))?,
                }
            };
            ensure!(got >= skip + bytes, "short direct read for {name}");
            let bytes = &buffer[skip..skip + bytes];
            if part.checksum.is_some() {
                hash.update(bytes);
            }
            consume(bytes)?;
            copied += bytes.len();
        }
        if let Some(expected) = &part.checksum {
            ensure!(
                hash.finalize()
                    .to_hex()
                    .as_str()
                    .eq_ignore_ascii_case(expected),
                "tensor checksum mismatch: {name}"
            );
        }
        Ok(())
    }
    /// Bounded raw traversal for conversion, with payload verification for .mnw.
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
    /// Bounded traversal of the separate scale region, including its checksum.
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
    pub fn lease_and_check(&self, dtype: DType) -> Result<Option<ModelLease>> {
        if !self.large {
            return Ok(None);
        }
        let lease = ModelLease::acquire()?;
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
            available >= required + RESERVE,
            "loading needs {:.2} GiB for weights plus 16 GiB system headroom; only {:.2} GiB available",
            required as f64 / 2f64.powi(30),
            available as f64 / 2f64.powi(30)
        );
        tracing::info!(
            weight_gib = required as f64 / 2f64.powi(30),
            staging_mib = STAGING_BYTES / 1024 / 1024,
            "loading one resident weight copy with direct I/O"
        );
        Ok(Some(lease))
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
        let codes = Tensor::zeros(
            if encoding.int8() { count } else { count / 2 },
            DType::U8,
            device,
        )?;
        let scales = Tensor::zeros(count / group_size, DType::F16, device)?;
        let mut code_offset = 0;
        let mut scale_offset = 0;
        for part in parts {
            self.visit(name, &part, |bytes| {
                let chunk = Tensor::from_raw_buffer(bytes, DType::U8, &[bytes.len()], device)?;
                codes.slice_set(&chunk, 0, code_offset)?;
                device.synchronize()?;
                code_offset += bytes.len();
                Ok(())
            })?;
            let region = part
                .scales
                .as_ref()
                .context("missing quantization scales")?;
            let scale_location = Location {
                offset: region.offset,
                bytes: region.bytes as usize,
                checksum: Some(region.blake3.clone()),
                dtype: DType::F16,
                ..part.clone()
            };
            self.visit(name, &scale_location, |bytes| {
                // Reject corrupt but checksummed scales before a kernel can use them.
                for value in bytes.as_chunks::<2>().0.iter() {
                    let value = half::f16::from_bits(u16::from_le_bytes(*value));
                    ensure!(
                        value.is_finite() && value.to_f32() > 0.,
                        "invalid scale in {name}"
                    );
                }
                let chunk = Tensor::from_raw_buffer(bytes, DType::F16, &[bytes.len() / 2], device)?;
                scales.slice_set(&chunk, 0, scale_offset)?;
                device.synchronize()?;
                scale_offset += bytes.len() / 2;
                Ok(())
            })?;
        }
        ensure!(
            code_offset == codes.elem_count() && scale_offset == scales.elem_count(),
            "incomplete quantized weight {name}"
        );
        Ok(Some(crate::quant::Weights {
            codes,
            scales,
            encoding,
            group_size,
            shape,
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
                available_memory()? > RESERVE,
                "memory headroom fell below 16 GiB; stopping weight load"
            );
        }
        let out = Tensor::zeros(shape.elem_count(), dtype, device)?;
        let mut dst = 0;
        for part in parts {
            let item_size = part.dtype.size_in_bytes();
            self.visit(name, &part, |buffer| {
                let chunk = Tensor::from_raw_buffer(
                    buffer,
                    part.dtype,
                    &[buffer.len() / item_size],
                    device,
                )?
                .to_dtype(dtype)?;
                out.slice_set(&chunk, 0, dst)?;
                // Bound the lifetime of async copies before reusing host staging.
                device.synchronize()?;
                dst += buffer.len() / item_size;
                Ok(())
            })?;
        }
        ensure!(dst == shape.elem_count(), "incomplete weight {name}");
        Ok(out.reshape(shape)?)
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
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
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
        for e in 0..3 {
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
        assert!(loader.lease_and_check(DType::F32)?.is_none());
        let actual = loader
            .load("large", n.into(), DType::F32, &Device::Cpu)?
            .to_vec1::<f32>()?;
        assert_eq!(actual, values);
        let packed = loader.load(
            "mlp.experts.gate_proj.weight",
            (3, 3, 5).into(),
            DType::F32,
            &Device::Cpu,
        )?;
        let expected: Vec<f32> = (0..3)
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
                    (4, 3, 5).into(),
                    DType::F32,
                    &Device::Cpu
                )
                .is_err()
        );
        Ok(())
    }
}

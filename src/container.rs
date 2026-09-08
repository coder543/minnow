//! Minnow model container: bounded MessagePack manifest + aligned raw regions.
//! Weight regions can be mapped directly; normal loading uses O_DIRECT instead.
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

pub const MAGIC: &[u8; 8] = b"MINNOW01";
pub const ALIGNMENT: u64 = 4096;
pub const HEADER_BYTES: usize = 64;
pub const MAX_MANIFEST_BYTES: u64 = 128 * 1024 * 1024;

mod writer;
pub use writer::{Conversion, TensorRule, convert};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Encoding {
    Bf16,
    F16,
    F32,
    /// Signed symmetric INT8, FP16 scales along the innermost dimension.
    I8Sym,
    /// INT8 in m16n8k16 tensor-core fragment order, with tiled FP16 scales.
    I8Mma,
    /// Native NVFP4: E2M1, E4M3 scales per 16, FP32 scale per expert matrix.
    Nvfp4,
}
impl Encoding {
    pub fn quantized(self) -> bool {
        matches!(self, Self::I8Sym | Self::I8Mma | Self::Nvfp4)
    }
    pub fn int8(self) -> bool {
        matches!(self, Self::I8Sym | Self::I8Mma)
    }
    pub fn packed(self) -> bool {
        matches!(self, Self::I8Mma)
    }
    pub fn row_major(self) -> Self {
        match self {
            Self::I8Mma => Self::I8Sym,
            other => other,
        }
    }
    pub fn float_dtype(self) -> Result<candle_core::DType> {
        Ok(match self {
            Self::Bf16 => candle_core::DType::BF16,
            Self::F16 => candle_core::DType::F16,
            Self::F32 => candle_core::DType::F32,
            _ => bail!("quantized weights require a quantized execution path"),
        })
    }
    pub fn from_dtype(dtype: candle_core::DType) -> Result<Self> {
        Ok(match dtype {
            candle_core::DType::BF16 => Self::Bf16,
            candle_core::DType::F16 => Self::F16,
            candle_core::DType::F32 => Self::F32,
            _ => bail!("unsupported floating weight dtype {dtype:?}"),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Region {
    pub offset: u64,
    pub bytes: u64,
    pub blake3: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub encoding: Encoding,
    pub data: Region,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scales: Option<Region>,
    #[serde(default)]
    pub group_size: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_scale: Option<f32>,
}
impl TensorInfo {
    pub fn elements(&self) -> Result<usize> {
        ensure!(
            !self.shape.is_empty() && self.shape.len() <= 8 && self.shape.iter().all(|n| *n > 0),
            "invalid tensor shape"
        );
        self.shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .context("tensor size overflow")
    }
    pub fn validate(&self) -> Result<()> {
        let n = self.elements()?;
        let expected = match self.encoding {
            Encoding::Bf16 | Encoding::F16 => n.checked_mul(2),
            Encoding::F32 => n.checked_mul(4),
            Encoding::I8Sym | Encoding::I8Mma => Some(n),
            Encoding::Nvfp4 => Some(n.div_ceil(2)),
        }
        .context("tensor byte size overflow")?;
        ensure!(
            self.data.bytes == expected as u64,
            "tensor payload length disagrees with shape/encoding"
        );
        if self.encoding.quantized() {
            let native = self.encoding == Encoding::Nvfp4;
            ensure!(
                if native {
                    self.shape.len() == 2
                        && self.shape[0].is_multiple_of(8)
                        && self.shape[1].is_multiple_of(64)
                        && self.group_size == 16
                        && self.global_scale.is_some_and(|v| v.is_finite() && v > 0.)
                } else {
                    self.global_scale.is_none()
                },
                "invalid NVFP4 shape or global scale"
            );
            if self.encoding.packed() {
                ensure!(
                    self.shape.len() == 2
                        && self.shape[0].is_multiple_of(8)
                        && self.shape[1].is_multiple_of(16),
                    "invalid packed quantized matrix shape"
                );
            }
            ensure!(
                [16, 32, 64, 128].contains(&self.group_size)
                    && self.shape.last().unwrap().is_multiple_of(self.group_size),
                "invalid quantization group size"
            );
            ensure!(
                self.scales.as_ref().is_some_and(
                    |s| s.bytes == (n / self.group_size * if native { 1 } else { 2 }) as u64
                ),
                "invalid quantization scale payload"
            );
        } else {
            ensure!(
                self.scales.is_none() && self.group_size == 0 && self.global_scale.is_none(),
                "floating tensor has quantization metadata"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Manifest {
    pub version: u32,
    pub architecture: String,
    pub model_id: String,
    /// Exact source JSON/Jinja text; no external tokenizer/config files needed.
    pub assets: BTreeMap<String, String>,
    pub tensors: BTreeMap<String, TensorInfo>,
}
impl Manifest {
    pub fn asset(&self, name: &str) -> Result<&str> {
        self.assets
            .get(name)
            .map(String::as_str)
            .with_context(|| format!("missing model asset {name}"))
    }
    pub fn config(&self) -> Result<crate::config::Config> {
        crate::config::Config::from_json(self.asset("config.json")?.as_bytes())
    }
    pub fn weight_bytes(&self) -> u64 {
        self.tensors
            .values()
            .map(|t| t.data.bytes + t.scales.as_ref().map_or(0, |s| s.bytes))
            .sum()
    }
}

pub struct Container {
    pub manifest: Manifest,
    pub file_bytes: u64,
}
impl Container {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_impl(path, false)
    }
    /// Verify the manifest and every payload checksum without loading a model.
    pub fn validate(path: &Path) -> Result<Self> {
        let container = Self::open_impl(path, true)?;
        crate::weights::WeightLoader::open(path)?.validate_payloads()?;
        Ok(container)
    }
    fn open_impl(path: &Path, verify_checksum: bool) -> Result<Self> {
        let mut file = File::open(path)?;
        let size = file.metadata()?.len();
        let mut header = [0u8; HEADER_BYTES];
        file.read_exact(&mut header)
            .context("reading minnow container header")?;
        ensure!(
            &header[..8] == MAGIC,
            "not a supported minnow model container"
        );
        let u64_at = |offset| u64::from_le_bytes(header[offset..offset + 8].try_into().unwrap());
        let (offset, length, declared_size) = (u64_at(8), u64_at(16), u64_at(24));
        ensure!(
            declared_size == size
                && offset >= ALIGNMENT
                && offset.is_multiple_of(ALIGNMENT)
                && length > 0
                && length <= MAX_MANIFEST_BYTES
                && offset.checked_add(length) == Some(size),
            "invalid minnow manifest bounds"
        );
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0; length as usize];
        file.read_exact(&mut bytes)?;
        if verify_checksum {
            ensure!(
                blake3::hash(&bytes).as_bytes() == &header[32..64],
                "minnow manifest checksum mismatch"
            );
        }
        let manifest: Manifest =
            rmp_serde::from_slice(&bytes).context("decoding MessagePack model manifest")?;
        ensure!(
            manifest.version == 1 && manifest.architecture == "llada2_moe",
            "unsupported minnow model version/architecture"
        );
        let config = manifest.config()?;
        ensure!(
            config.model_type == manifest.architecture,
            "model architecture mismatch"
        );
        manifest.asset("tokenizer.json")?;
        manifest.asset("chat_template.jinja")?;
        ensure!(!manifest.tensors.is_empty(), "model has no tensors");
        let mut regions = Vec::new();
        for (name, tensor) in &manifest.tensors {
            tensor
                .validate()
                .with_context(|| format!("invalid tensor {name}"))?;
            for region in std::iter::once(&tensor.data).chain(tensor.scales.iter()) {
                ensure!(
                    region.offset >= ALIGNMENT
                        && region.offset.is_multiple_of(ALIGNMENT)
                        && region.bytes > 0
                        && region
                            .offset
                            .checked_add(region.bytes)
                            .is_some_and(|end| end <= offset),
                    "invalid payload bounds for {name}"
                );
                ensure!(
                    region.blake3.len() == 64
                        && region.blake3.bytes().all(|b| b.is_ascii_hexdigit()),
                    "invalid payload checksum for {name}"
                );
                regions.push((region.offset, region.offset + region.bytes));
            }
        }
        regions.sort_unstable();
        ensure!(
            regions.windows(2).all(|p| p[0].1 <= p[1].0),
            "overlapping tensor regions"
        );
        Ok(Self {
            manifest,
            file_bytes: size,
        })
    }
}

pub fn asset(path: &Path, name: &str) -> Result<String> {
    if path.is_file() {
        Ok(Container::open(path)?.manifest.asset(name)?.to_owned())
    } else {
        Ok(std::fs::read_to_string(path.join(name))?)
    }
}

use super::*;
use crate::weights::{STAGING_BYTES, Staging, WeightLoader};
use std::{
    fs::OpenOptions,
    os::unix::fs::{FileExt, OpenOptionsExt},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TensorRule {
    pub prefix: String,
    /// Null preserves the source encoding.
    pub encoding: Option<Encoding>,
    #[serde(default)]
    pub group_size: usize,
}
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Conversion {
    pub expert_encoding: Option<Encoding>,
    #[serde(default)]
    pub group_size: usize,
    /// Later matching rules override earlier ones and the expert default.
    #[serde(default)]
    pub rules: Vec<TensorRule>,
}
impl Conversion {
    fn for_tensor(
        &self,
        name: &str,
        original: Encoding,
        original_group: usize,
    ) -> Result<(Encoding, usize)> {
        let expert = name.contains(".experts.") && name.ends_with(".weight");
        let mut encoding = if expert {
            self.expert_encoding.unwrap_or(original)
        } else {
            original
        };
        let mut group = self.group_size;
        for rule in &self.rules {
            if name.starts_with(&rule.prefix) {
                encoding = rule.encoding.unwrap_or(original);
                group = rule.group_size;
            }
        }
        if encoding.quantized() {
            ensure!(
                expert,
                "only routed expert projections can currently be quantized: {name}"
            );
            if group == 0 {
                group = if original_group > 0 {
                    original_group
                } else if encoding == Encoding::Nvfp4 {
                    16
                } else {
                    128
                };
            }
            ensure!(
                [16, 32, 64, 128].contains(&group),
                "unsupported group size {group}"
            );
            ensure!(
                encoding != Encoding::Nvfp4 || group == 16,
                "NVFP4 requires groups of 16"
            );
        } else {
            ensure!(
                encoding == original,
                "floating dtype conversion is not supported; preserve the source dtype"
            );
            group = 0;
        }
        Ok((encoding, group))
    }
}

// Direct writes avoid turning the filesystem cache into a second weight store.
struct Writer {
    file: File,
    staging: Staging,
    used: usize,
    position: u64,
    temporary: PathBuf,
}
impl Writer {
    fn new(destination: &Path) -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        ensure!(
            !destination.exists(),
            "destination already exists: {}",
            destination.display()
        );
        let filename = destination
            .file_name()
            .context("destination needs a filename")?
            .to_string_lossy();
        let temporary = destination.with_file_name(format!(
            ".{filename}.{}-{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let file = OpenOptions::new()
            .write(true)
            .read(true)
            .create_new(true)
            .mode(0o644)
            .custom_flags(libc::O_DIRECT)
            .open(&temporary)?;
        Ok(Self {
            file,
            staging: Staging::new()?,
            used: 0,
            position: 0,
            temporary,
        })
    }
    fn write(&mut self, mut bytes: &[u8]) -> Result<()> {
        while !bytes.is_empty() {
            let take = bytes.len().min(STAGING_BYTES - self.used);
            self.staging.bytes()[self.used..self.used + take].copy_from_slice(&bytes[..take]);
            self.used += take;
            self.position += take as u64;
            bytes = &bytes[take..];
            if self.used == STAGING_BYTES {
                self.flush()?;
            }
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<()> {
        if self.used == 0 {
            return Ok(());
        }
        let padded = self.used.div_ceil(ALIGNMENT as usize) * ALIGNMENT as usize;
        self.staging.bytes()[self.used..padded].fill(0);
        self.file.write_all_at(
            &self.staging.bytes()[..padded],
            self.position - self.used as u64,
        )?;
        self.used = 0;
        Ok(())
    }
    fn align(&mut self) -> Result<u64> {
        let padding = (ALIGNMENT - self.position % ALIGNMENT) % ALIGNMENT;
        self.write(&vec![0; padding as usize])?;
        Ok(self.position)
    }
    fn finish(mut self, manifest: &Manifest, destination: &Path) -> Result<Container> {
        let bytes = rmp_serde::to_vec_named(manifest)?;
        ensure!(
            bytes.len() as u64 <= MAX_MANIFEST_BYTES,
            "model manifest exceeds limit"
        );
        let offset = self.align()?;
        self.write(&bytes)?;
        self.flush()?;
        // All tensor and manifest writes have completed before publishing a header.
        let header = &mut self.staging.bytes()[..ALIGNMENT as usize];
        header.fill(0);
        header[..8].copy_from_slice(MAGIC);
        header[8..16].copy_from_slice(&offset.to_le_bytes());
        header[16..24].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
        header[24..32].copy_from_slice(&self.position.to_le_bytes());
        header[32..64].copy_from_slice(blake3::hash(&bytes).as_bytes());
        self.file.write_all_at(header, 0)?;
        self.file.set_len(self.position)?;
        self.file.sync_all()?;
        let container = Container::open(&self.temporary)?;
        // Same-filesystem hard linking publishes atomically and never overwrites.
        std::fs::hard_link(&self.temporary, destination)?;
        std::fs::remove_file(&self.temporary)?;
        File::open(
            destination
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new(".")),
        )?
        .sync_all()?;
        Ok(container)
    }
}
impl Drop for Writer {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.temporary);
    }
}

pub fn convert(source: &Path, destination: &Path, options: &Conversion) -> Result<Container> {
    let loader = WeightLoader::open(source)?;
    let mut assets = BTreeMap::new();
    for name in [
        "config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ] {
        match asset(source, name) {
            Ok(value) => {
                assets.insert(name.to_owned(), value);
            }
            Err(e) if ["config.json", "tokenizer.json", "chat_template.jinja"].contains(&name) => {
                return Err(e);
            }
            Err(_) => {}
        }
    }
    let config = crate::config::Config::from_json(assets["config.json"].as_bytes())?;
    let inventory = loader.inventory();
    for rule in &options.rules {
        ensure!(
            inventory
                .iter()
                .any(|(n, _, _)| n.starts_with(&rule.prefix)),
            "tensor rule matches no weights: {}",
            rule.prefix
        );
    }
    // Validate every conversion choice before creating any output.
    let mut families = BTreeMap::new();
    for (name, shape, encoding) in &inventory {
        let original_group = loader.quantization_group(name)?;
        let (selected, group) = options.for_tensor(name, *encoding, original_group)?;
        if encoding.quantized() {
            ensure!(
                selected.row_major() == encoding.row_major() && group == original_group,
                "re-quantization is not supported; only lossless repacking of quantized input is allowed"
            );
        }
        if selected.quantized() {
            ensure!(
                shape.len() == 2 && shape.iter().product::<usize>() <= 16 * 1024 * 1024,
                "quantized conversion supports individual expert matrices up to 16M elements"
            );
        }
        if selected.packed() {
            ensure!(
                shape[0].is_multiple_of(8) && shape[1].is_multiple_of(16),
                "packed matrix {name} requires N8/K16 alignment"
            );
        }
        if selected == Encoding::Nvfp4 {
            ensure!(
                shape[0].is_multiple_of(8) && shape[1].is_multiple_of(64),
                "NVFP4 requires N8/K64 alignment: {name}"
            );
        }
        ensure!(
            group == 0 || shape.last().is_some_and(|n| n.is_multiple_of(group)),
            "group size does not divide {name}"
        );
        if let Some((prefix, expert)) = name.rsplit_once(".experts.") {
            let (_, projection) = expert
                .split_once('.')
                .context("invalid expert tensor name")?;
            let family = format!("{prefix}.experts.{projection}");
            if let Some(previous) = families.insert(family.clone(), (selected, group)) {
                ensure!(
                    previous == (selected, group),
                    "all experts within projection {family} must share an encoding/group size"
                );
            }
        }
    }
    let mut manifest = Manifest {
        version: 1,
        architecture: config.model_type,
        model_id: source
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        assets,
        tensors: BTreeMap::new(),
    };
    let mut writer = Writer::new(destination)?;
    writer.write(&[0; ALIGNMENT as usize])?;
    for (index, (name, shape, original)) in inventory.iter().enumerate() {
        let (encoding, group_size) =
            options.for_tensor(name, *original, loader.quantization_group(name)?)?;
        let offset = writer.align()?;
        let mut hash = blake3::Hasher::new();
        let mut scales = Vec::new();
        let mut global_scale = None;
        if encoding.quantized() {
            // One expert at a time, bounded above before opening the output.
            let mut bytes = Vec::new();
            loader.visit_tensor(name, |input| {
                bytes.extend_from_slice(input);
                Ok(())
            })?;
            let (codes, converted_scales) = if encoding == Encoding::Nvfp4 {
                if *original == Encoding::Nvfp4 {
                    global_scale = loader.global_scale(name)?;
                    loader.visit_scales(name, |input| {
                        scales.extend_from_slice(input);
                        Ok(())
                    })?;
                    (bytes, scales)
                } else {
                    let values = crate::quant::decode_float_bytes(&bytes, *original)?;
                    let (codes, scales, global) = crate::quant::nvfp4::encode(&values)?;
                    global_scale = Some(global);
                    crate::quant::nvfp4::pack(&codes, &scales, shape[1], false)?
                }
            } else if original.quantized() {
                loader.visit_scales(name, |input| {
                    scales.extend_from_slice(input);
                    Ok(())
                })?;
                crate::quant::repack(&bytes, &scales, *original, encoding, group_size, shape[1])?
            } else {
                let values = crate::quant::decode_float_bytes(&bytes, *original)?;
                crate::quant::encode_matrix(&values, encoding, group_size, shape[1])?
            };
            hash.update(&codes);
            writer.write(&codes)?;
            scales = converted_scales;
        } else {
            loader.visit_tensor(name, |input| {
                hash.update(input);
                writer.write(input)?;
                Ok(())
            })?;
        }
        let data = Region {
            offset,
            bytes: writer.position - offset,
            blake3: hash.finalize().to_hex().to_string(),
        };
        let scales = if scales.is_empty() {
            None
        } else {
            let offset = writer.align()?;
            writer.write(&scales)?;
            Some(Region {
                offset,
                bytes: scales.len() as u64,
                blake3: blake3::hash(&scales).to_hex().to_string(),
            })
        };
        let tensor = TensorInfo {
            shape: shape.clone(),
            encoding,
            data,
            scales,
            group_size,
            global_scale,
        };
        tensor.validate()?;
        manifest.tensors.insert(name.clone(), tensor);
        if index % 256 == 0 || index + 1 == inventory.len() {
            tracing::info!(
                tensors = index + 1,
                total = inventory.len(),
                gib = writer.position as f64 / 1073741824.,
                "converting checkpoint"
            );
        }
    }
    writer.finish(&manifest, destination)
}

use anyhow::Result;
use candle_core::{DType, Device};
use minnow::{
    container::{Container, Conversion, convert},
    model::{Cache, Model},
    tokenizer::TextCodec,
};
use std::{
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "minnow-container-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root)?;
        let source = root.join("source");
        std::fs::create_dir(&source)?;
        for name in ["config.json", "model.safetensors"] {
            std::fs::copy(
                Path::new("tests/fixtures/tiny").join(name),
                source.join(name),
            )?;
        }
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [("[UNK]".to_string(), 0), ("hello".to_string(), 1)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        tokenizers::Tokenizer::new(model)
            .save(source.join("tokenizer.json"), false)
            .unwrap();
        std::fs::write(
            source.join("chat_template.jinja"),
            "{% for m in messages %}{{m.content}}{% endfor %}",
        )?;
        Ok(Self(root))
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn original_container_is_self_contained_and_preserves_model_execution() -> Result<()> {
    let fixture = Fixture::new()?;
    let source = fixture.0.join("source");
    let dest = fixture.0.join("original.mnw");
    let manifest = convert(&source, &dest, &Conversion::default())?;
    assert!(
        manifest
            .manifest
            .tensors
            .values()
            .all(|t| t.data.offset % 4096 == 0)
    );
    assert!(convert(&source, &dest, &Conversion::default()).is_err());
    let model = Model::load(&source, DType::F32, &Device::Cpu)?;
    let mut cache = Cache::new(&model.config, 32)?;
    let expected = model
        .forward(&[1; 32], &mut cache, true, None)?
        .unwrap()
        .to_vec2::<f32>()?;
    drop((model, cache));
    std::fs::remove_dir_all(source)?;
    let codec = TextCodec::load(&dest)?;
    assert_eq!(codec.encode("hello")?, vec![1]);
    assert_eq!(
        codec.chat_prompt(&[serde_json::json!({"role":"user","content":"hello"})])?,
        "hello"
    );
    let model = Model::load(&dest, DType::F32, &Device::Cpu)?;
    let mut cache = Cache::new(&model.config, 32)?;
    assert_eq!(
        model
            .forward(&[1; 32], &mut cache, true, None)?
            .unwrap()
            .to_vec2::<f32>()?,
        expected
    );
    Ok(())
}

#[test]
fn corrupt_payload_and_manifest_are_rejected() -> Result<()> {
    let fixture = Fixture::new()?;
    let source = fixture.0.join("source");
    let dest = fixture.0.join("corrupt.mnw");
    let container = convert(&source, &dest, &Conversion::default())?;
    let tensor = &container.manifest.tensors["model.word_embeddings.weight"];
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&dest)?;
    file.seek(SeekFrom::Start(tensor.data.offset))?;
    file.write_all(&[0x7f; 4])?;
    file.sync_all()?;
    let error = Model::load(&dest, DType::F32, &Device::Cpu)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("checksum"), "{error}");
    file.seek(SeekFrom::Start(8))?;
    let mut bytes = [0u8; 8];
    file.read_exact(&mut bytes)?;
    let offset = u64::from_le_bytes(bytes);
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(&[0xc1])?;
    file.sync_all()?;
    assert!(
        Container::open(&dest)
            .err()
            .unwrap()
            .to_string()
            .contains("checksum")
    );
    file.set_len(10)?;
    assert!(Container::open(&dest).is_err());
    Ok(())
}

#[test]
fn mixed_quantization_matches_a_separately_decoded_checkpoint() -> Result<()> {
    use minnow::container::{Encoding, TensorRule};
    let fixture = Fixture::new()?;
    let source = fixture.0.join("source");
    let dest = fixture.0.join("mixed.mnw");
    let container = convert(
        &source,
        &dest,
        &Conversion {
            expert_encoding: Some(Encoding::Fp4E2m1),
            group_size: 16,
            rules: vec![TensorRule {
                prefix: "model.layers.2.mlp.experts.".into(),
                encoding: Some(Encoding::I8Sym),
                group_size: 16,
            }],
        },
    )?;
    let mut weights =
        candle_core::safetensors::load(source.join("model.safetensors"), &Device::Cpu)?;
    let mut file = std::fs::File::open(&dest)?;
    let mut encodings = std::collections::HashSet::new();
    for (name, t) in &container.manifest.tensors {
        if !t.encoding.quantized() {
            continue;
        }
        encodings.insert(format!("{:?}", t.encoding));
        let mut codes = vec![0; t.data.bytes as usize];
        file.seek(SeekFrom::Start(t.data.offset))?;
        file.read_exact(&mut codes)?;
        let scale = t.scales.as_ref().unwrap();
        let mut scales = vec![0; scale.bytes as usize];
        file.seek(SeekFrom::Start(scale.offset))?;
        file.read_exact(&mut scales)?;
        let values = minnow::quant::decode(&codes, &scales, t.encoding, t.group_size)?;
        let weight = candle_core::Tensor::from_vec(values, t.shape.as_slice(), &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?;
        weights.insert(name.clone(), weight);
    }
    assert_eq!(encodings.len(), 2);
    candle_core::safetensors::save(&weights, source.join("model.safetensors"))?;
    drop(weights);
    let evaluate = |path: &Path| -> Result<Vec<f32>> {
        let model = Model::load(path, DType::F32, &Device::Cpu)?;
        let mut cache = Cache::new(&model.config, 64)?;
        Ok(model
            .forward(&[1; 64], &mut cache, true, None)?
            .unwrap()
            .flatten_all()?
            .to_vec1::<f32>()?)
    };
    let expected = evaluate(&source)?;
    let actual = evaluate(&dest)?;
    assert!(
        expected
            .iter()
            .zip(actual)
            .all(|(a, b)| (a - b).abs() < 1e-5)
    );
    // Repack the already quantized container without returning to source weights
    // or changing any scale/code. Both CPU execution and the on-disk reader use it.
    let packed = fixture.0.join("packed.mnw");
    convert(
        &dest,
        &packed,
        &Conversion {
            expert_encoding: Some(Encoding::Fp4Mma),
            group_size: 16,
            rules: vec![TensorRule {
                prefix: "model.layers.2.mlp.experts.".into(),
                encoding: Some(Encoding::I8Mma),
                group_size: 16,
            }],
        },
    )?;
    assert_eq!(evaluate(&dest)?, evaluate(&packed)?);
    assert!(
        convert(
            &dest,
            &fixture.0.join("invalid.mnw"),
            &Conversion {
                expert_encoding: Some(Encoding::I8Mma),
                group_size: 16,
                rules: vec![]
            }
        )
        .is_err(),
        "repacking must not silently change precision"
    );
    Ok(())
}

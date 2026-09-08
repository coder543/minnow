use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use clap::{Parser, Subcommand, ValueEnum};
use minnow::{
    config::Config,
    decode::{Options, SpecialTokens, generate, prefill},
    model::{Cache, Model, Trace},
    tokenizer::TextCodec,
};
use serde_json::json;
use std::{fs, path::PathBuf, time::Instant};

#[derive(Clone, Copy, ValueEnum)]
enum Precision {
    Auto,
    Bf16,
    F32,
}
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Checkpoint directory or self-contained .mnw file (required).
    #[arg(long, global = true)]
    model: Option<PathBuf>,
    /// auto selects CUDA when available, otherwise CPU. Also cpu, cuda, cuda:N.
    #[arg(long, global = true, default_value = "auto")]
    device: String,
    /// auto uses BF16 on CUDA and FP32 on CPU.
    #[arg(long, global = true, value_enum, default_value = "auto")]
    dtype: Precision,
    /// Use individual expert GEMMs for comparison with the batched CUDA path.
    #[arg(long, global = true)]
    serial_experts: bool,
    /// Experimental grouped decode GEMMs using only each expert's assigned rows.
    #[arg(long, global = true, conflicts_with = "device_routing")]
    compact_decode_experts: bool,
    /// Materialize FP32 SiLU tensors instead of using the equivalent fused kernel.
    #[arg(long, global = true)]
    unfused_activation: bool,
    /// Materialize attention scaling, masking, and FP32 softmax intermediates.
    #[arg(long, global = true)]
    unfused_attention: bool,
    /// Use materialized attention instead of the default block FlashAttention.
    #[arg(long, global = true)]
    materialized_attention: bool,
    /// Use host routing for comparisons (INT8/NVFP4 decode defaults to GPU).
    #[arg(long, global = true, conflicts_with = "device_routing")]
    host_routing: bool,
    /// Materialize expert gathering, FP32 weighting, and reduction intermediates.
    #[arg(long, global = true)]
    unfused_expert_mix: bool,
    /// Materialize the FP32 RMSNorm intermediates for numerical comparison.
    #[arg(long, global = true)]
    unfused_norm: bool,
    /// Experimental GPU routing with fixed-capacity expert batches.
    #[arg(long, global = true)]
    device_routing: bool,
    /// Evaluate head normalization, RoPE, and layout conversion separately.
    #[arg(long, global = true)]
    unfused_qkv: bool,
    /// Maximum transformer prefill batch (up to 8192 tokens).
    #[arg(long, global = true, default_value_t = 4096)]
    prefill_chunk_tokens: usize,
    /// Maximum query tile; reduced automatically to bound attention workspace.
    #[arg(long, global = true, default_value_t = 1024)]
    attention_chunk_tokens: usize,
    /// Reuse up to this much unused CUDA allocation storage between forwards.
    #[arg(long, global = true, default_value_t = 2048)]
    workspace_cache_mib: usize,
    /// System-memory headroom retained while loading (independent of GPU VRAM).
    #[arg(long, global = true, default_value_t = minnow::weights::DEFAULT_MEMORY_RESERVE_MIB)]
    memory_reserve_mib: u64,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Convert a checkpoint to a self-contained .mnw file using bounded direct I/O.
    Convert {
        output: PathBuf,
        /// Quantize routed expert projections; other weights retain their dtype.
        #[arg(long, value_parser = ["original", "int8", "nvfp4"], default_value = "original")]
        experts: String,
        /// Quantization group size (default: 128 for INT8, 16 for NVFP4).
        #[arg(long, default_value_t = 0)]
        group_size: usize,
        /// CUDA fragment packing is lossless; row layout is available for comparisons.
        #[arg(long, value_parser = ["mma", "row"], default_value = "mma")]
        quant_layout: String,
        /// JSON array of tensor-prefix rules for mixed precision; later rules win.
        #[arg(long)]
        tensor_rules: Option<PathBuf>,
    },
    /// Read configuration without loading weights.
    Inspect,
    /// Encode a raw or chat prompt without loading weights.
    Tokenize {
        prompt: String,
        #[arg(long)]
        raw: bool,
    },
    /// Generate with the checkpoint chat template (or --raw).
    Generate {
        prompt: String,
        #[arg(long)]
        raw: bool,
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        options: Options,
    },
    /// Export forward logits and intermediate tensors for numerical comparison.
    Forward {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        cached: bool,
    },
    /// Verify .mnw checksums, or compare model execution when --reference is supplied.
    Validate {
        /// Compare tensors with a reference fixture instead of checking the container.
        #[arg(long)]
        reference: Option<PathBuf>,
        #[arg(long, default_value_t = 0.0001)]
        max_abs_error: f64,
        /// Per-element allowance: abs(actual-reference) <= atol + rtol*abs(reference).
        #[arg(long, default_value_t = 0.0)]
        max_relative_error: f64,
    },
    /// Serve text completions and chat through HTTP.
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: std::net::SocketAddr,
        /// Context limit; defaults to the checkpoint's full supported context.
        #[arg(long)]
        max_context: Option<usize>,
        #[arg(long, default_value_t = 8)]
        queue_capacity: usize,
        /// Maximum concurrent requests sharing transformer batches.
        #[arg(long, default_value_t = 4)]
        parallel: usize,
        /// Maximum wait to combine ready requests into one GPU invocation.
        #[arg(long, default_value_t = 200)]
        batch_wait_us: u64,
        /// Model ID advertised to clients; defaults to the mini/flash architecture.
        #[arg(long)]
        alias: Option<String>,
        /// Serve a built UI directory directly, without embedding or copying it.
        #[arg(long)]
        ui_dir: Option<PathBuf>,
        #[command(flatten)]
        cache: minnow::prefix::CacheOptions,
        /// Default decoding settings; requests can override these.
        #[command(flatten)]
        options: Options,
    },
    /// Compare current-block evaluation with literal full-prefix recomputation.
    Bench {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "0,8,32")]
        prefix_blocks: Vec<usize>,
        #[arg(long, default_value_t = 3)]
        iterations: usize,
        /// Measure refinement only, including longer committed prefixes.
        #[arg(long)]
        cached_only: bool,
        #[cfg(feature = "cuda")]
        #[arg(long)]
        profile: bool,
    },
    /// Repeated generations: model/text token rates and refinement work.
    DecodeBench {
        #[arg(long, default_value = "tests/decode_throughput_cases.json")]
        cases: PathBuf,
        #[arg(long, default_value_t = 5)]
        iterations: usize,
        #[cfg(feature = "cuda")]
        #[arg(long)]
        profile: bool,
    },
    /// Measure the serving prompt-to-KV path, without vocabulary projection or decoding.
    PrefillBench {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "512,2048,4096")]
        tokens: Vec<usize>,
        #[arg(long, default_value_t = 5)]
        iterations: usize,
        /// Sweep batch sizes using one resident model (defaults to the global setting).
        #[arg(long, value_delimiter = ',')]
        chunk_sizes: Vec<usize>,
        /// Mark timed CUDA iterations for Nsight's --capture-range=cudaProfilerApi.
        #[cfg(feature = "cuda")]
        #[arg(long)]
        profile: bool,
    },
}

fn device(name: &str) -> Result<Device> {
    if name == "cpu" {
        return Ok(Device::Cpu);
    }
    if name == "auto" {
        #[cfg(feature = "cuda")]
        {
            match Device::new_cuda(0) {
                Ok(device) => return Ok(device),
                Err(error) => tracing::info!(%error, "CUDA unavailable; selecting CPU"),
            }
        }
        return Ok(Device::Cpu);
    }
    let ordinal = if name == "cuda" {
        0
    } else {
        name.strip_prefix("cuda:")
            .context("device must be auto, cpu, cuda, or cuda:N")?
            .parse()?
    };
    Device::new_cuda(ordinal).context("initializing CUDA (build with --features cuda)")
}

fn error(a: &Tensor, b: &Tensor, atol: f64, rtol: f64) -> Result<(f64, f64, usize)> {
    ensure!(
        a.dims() == b.dims(),
        "shape mismatch: {:?} vs {:?}",
        a.dims(),
        b.dims()
    );
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut max = 0.0f64;
    let mut sq = 0.0;
    let mut outside = 0;
    for (&a, &b) in a.iter().zip(&b) {
        ensure!(a.is_finite() && b.is_finite(), "nonfinite comparison");
        let d = (a as f64 - b as f64).abs();
        max = max.max(d);
        sq += d * d;
        outside += usize::from(d > atol + rtol * (b as f64).abs());
    }
    Ok((max, (sq / a.len() as f64).sqrt(), outside))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "minnow=info".into()),
        )
        .init();
    let cli = Cli::parse();
    let model_path = cli
        .model
        .context("pass --model with a checkpoint directory or .mnw file")?;
    match &cli.command {
        Command::Convert {
            output,
            experts,
            group_size,
            quant_layout,
            tensor_rules,
        } => {
            use minnow::container::{Conversion, Encoding, convert};
            anyhow::ensure!(
                experts != "nvfp4" || quant_layout == "mma",
                "NVFP4 requires its native MMA layout; --quant-layout row is unsupported"
            );
            let options = Conversion {
                expert_encoding: match experts.as_str() {
                    "nvfp4" => Some(Encoding::Nvfp4),
                    "int8" => Some(if quant_layout == "mma" {
                        Encoding::I8Mma
                    } else {
                        Encoding::I8Sym
                    }),
                    _ => None,
                },
                group_size: *group_size,
                rules: match tensor_rules {
                    Some(path) => serde_json::from_slice(&fs::read(path)?)?,
                    None => vec![],
                },
            };
            let start = Instant::now();
            let container = convert(&model_path, output, &options)?;
            println!(
                "{}",
                json!({"path":output,"file_bytes":container.file_bytes,"weight_bytes":container.manifest.weight_bytes(),"tensors":container.manifest.tensors.len(),"seconds":start.elapsed().as_secs_f64()})
            );
            return Ok(());
        }
        Command::Validate {
            reference: None, ..
        } => {
            let start = Instant::now();
            let container = minnow::container::Container::validate(&model_path)?;
            println!(
                "{}",
                json!({"path":model_path,"valid":true,"tensors":container.manifest.tensors.len(),"weight_bytes":container.manifest.weight_bytes(),"seconds":start.elapsed().as_secs_f64()})
            );
            return Ok(());
        }
        Command::Inspect => {
            println!(
                "{}",
                serde_json::to_string_pretty(&Config::load(&model_path)?)?
            );
            return Ok(());
        }
        Command::Tokenize { prompt, raw } => {
            let codec = TextCodec::load(&model_path)?;
            let text = if *raw {
                prompt.clone()
            } else {
                codec.chat_prompt(&[json!({"role":"user","content":prompt})])?
            };
            println!("{}", json!({"input_ids":codec.encode(&text)?,"text":text}));
            return Ok(());
        }
        _ => {}
    }
    let device = device(&cli.device)?;
    let dtype = match cli.dtype {
        Precision::Auto => {
            if device.is_cuda() {
                DType::BF16
            } else {
                DType::F32
            }
        }
        Precision::Bf16 => DType::BF16,
        Precision::F32 => DType::F32,
    };
    tracing::info!(?device, ?dtype, "execution backend");
    let start = Instant::now();
    let mut model =
        Model::load_with_memory_reserve(&model_path, dtype, &device, cli.memory_reserve_mib)?;
    model.set_workspace_cache_mib(cli.workspace_cache_mib)?;
    model.set_batched_experts(!cli.serial_experts);
    model.set_compact_decode_experts(cli.compact_decode_experts);
    model.set_fused_activation(!cli.unfused_activation);
    model.set_fused_attention(!cli.unfused_attention);
    model.set_flash_attention(!cli.materialized_attention);
    model.set_host_routing(cli.host_routing);
    model.set_fused_expert_mix(!cli.unfused_expert_mix);
    model.set_fused_norm(!cli.unfused_norm);
    model.set_device_routing(cli.device_routing);
    model.set_fused_qkv(!cli.unfused_qkv);
    model.set_prefill_chunk_tokens(cli.prefill_chunk_tokens)?;
    model.set_attention_chunk_tokens(cli.attention_chunk_tokens)?;
    device.synchronize()?;
    tracing::info!(backend = model.attention_backend(), "attention backend");
    tracing::info!(seconds = start.elapsed().as_secs_f64(), "model loaded");
    match cli.command {
        Command::Generate {
            prompt,
            raw,
            json: as_json,
            options,
        } => {
            let codec = TextCodec::load(&model_path)?;
            let text = if raw {
                prompt
            } else {
                codec.chat_prompt(&[json!({"role":"user","content":prompt})])?
            };
            let result = generate(
                &model,
                &codec.encode(&text)?,
                &options,
                SpecialTokens::default(),
                || false,
            )?;
            let text = codec.decode(&result.token_ids)?;
            if as_json {
                println!("{}", json!({"text":text,"generation":result}));
            } else {
                println!("{text}");
                eprintln!("{}", serde_json::to_string(&result.stats)?);
            }
        }
        Command::Forward {
            input,
            output,
            cached,
        } => {
            let tokens: Vec<u32> = serde_json::from_slice(&fs::read(input)?)?;
            ensure!(
                tokens.len() >= model.config.block_size,
                "input must contain at least one complete block"
            );
            let mut cache = Cache::new(&model.config, tokens.len())?;
            let mut trace = Trace::new();
            if cached {
                let b = model.config.block_size;
                prefill(&model, &tokens[..tokens.len() - b], &mut cache, || false)?;
                let logits = model
                    .forward(
                        &tokens[tokens.len() - b..],
                        &mut cache,
                        true,
                        Some(&mut trace),
                    )?
                    .unwrap();
                trace.insert("logits".into(), logits);
            } else {
                let logits = model
                    .forward(&tokens, &mut cache, true, Some(&mut trace))?
                    .unwrap();
                trace.insert("logits".into(), logits);
            }
            candle_core::safetensors::save(&trace, output)?;
        }
        Command::Validate {
            reference,
            max_abs_error,
            max_relative_error,
        } => {
            let reference =
                reference.context("--reference is required for numerical validation")?;
            ensure!(
                max_abs_error.is_finite()
                    && max_abs_error >= 0.0
                    && max_relative_error.is_finite()
                    && max_relative_error >= 0.0,
                "invalid tolerance"
            );
            let tokens: Vec<u32> =
                serde_json::from_slice(&fs::read(reference.join("input.json"))?)?;
            let expected = candle_core::safetensors::load(
                reference.join("reference.safetensors"),
                &Device::Cpu,
            )?;
            let mut cache = Cache::new(&model.config, tokens.len())?;
            let mut trace = Trace::new();
            let full = model
                .forward(&tokens, &mut cache, true, Some(&mut trace))?
                .unwrap();
            trace.insert("logits".into(), full.clone());
            let mut failures = Vec::new();
            let mut report = serde_json::Map::new();
            for (name, target) in &expected {
                let actual = trace
                    .get(name)
                    .with_context(|| format!("missing trace {name}"))?;
                let (atol, rtol) = if name.ends_with("router_ids") {
                    (0.0, 0.0)
                } else {
                    (max_abs_error, max_relative_error)
                };
                let (max, rms, outside) = error(actual, target, atol, rtol)?;
                report.insert(name.clone(), json!({"max_abs_error":max,"rms_error":rms,"elements_outside_tolerance":outside}));
                if outside > 0 {
                    failures.push(name.clone());
                }
            }
            let b = model.config.block_size;
            let mut cached = Cache::new(&model.config, tokens.len())?;
            let mut cached_logits = Vec::new();
            for chunk in tokens.chunks(b) {
                let logits = model.forward(chunk, &mut cached, true, None)?.unwrap();
                // Staging is replaceable, and committing different tokens must fail.
                let mut wrong = chunk.to_vec();
                wrong[0] = (wrong[0] + 1) % model.config.vocab_size as u32;
                ensure!(
                    cached.commit(&wrong).is_err(),
                    "accepted stale cache commit"
                );
                cached.commit(chunk)?;
                cached_logits.push(logits);
            }
            let (max, rms, outside) = error(
                &Tensor::cat(&cached_logits, 0)?,
                &full,
                max_abs_error,
                max_relative_error,
            )?;
            report.insert(
                "cache_vs_full".into(),
                json!({"max_abs_error":max,"rms_error":rms,"elements_outside_tolerance":outside}),
            );
            if outside > 0 {
                failures.push("cache_vs_full".into());
            }
            let actual_top = full.argmax(1)?.to_vec1::<u32>()?;
            let target_top = expected["logits"].argmax(1)?.to_vec1::<u32>()?;
            report.insert(
                "top1_agreement".into(),
                json!(
                    actual_top
                        .iter()
                        .zip(&target_top)
                        .filter(|(a, b)| a == b)
                        .count() as f64
                        / tokens.len() as f64
                ),
            );
            println!("{}", serde_json::to_string_pretty(&report)?);
            ensure!(
                failures.is_empty(),
                "validation exceeded tolerance for: {}",
                failures.join(", ")
            );
        }
        Command::Serve {
            listen,
            max_context,
            queue_capacity,
            parallel,
            batch_wait_us,
            alias,
            ui_dir,
            cache,
            options,
        } => {
            let codec = TextCodec::load(&model_path)?;
            let max_context = max_context.unwrap_or(model.config.max_position_embeddings);
            let alias = alias.unwrap_or_else(|| {
                format!(
                    "minnow-{}",
                    model.config.model_family().to_ascii_lowercase()
                )
            });
            minnow::server::serve(
                model,
                codec,
                minnow::server::ServeConfig {
                    listen,
                    max_context,
                    queue_capacity,
                    parallel,
                    batch_wait_us,
                    model_id: alias,
                    model_path,
                    ui_dir,
                    defaults: options,
                    cache,
                },
            )
            .await?;
        }
        Command::PrefillBench {
            input,
            tokens,
            iterations,
            chunk_sizes,
            #[cfg(feature = "cuda")]
            profile,
        } => {
            ensure!(
                (1..=100).contains(&iterations),
                "iterations must be between 1 and 100"
            );
            let seed: Vec<u32> = serde_json::from_slice(&fs::read(input)?)?;
            ensure!(!seed.is_empty(), "prefill input must not be empty");
            let mut reports = Vec::new();
            let chunk_sizes = if chunk_sizes.is_empty() {
                vec![model.prefill_chunk_tokens()]
            } else {
                chunk_sizes
            };
            for chunk_size in chunk_sizes {
                model.set_prefill_chunk_tokens(chunk_size)?;
                for &n in &tokens {
                    ensure!(
                        n > 0
                            && n <= model.config.max_position_embeddings
                            && n.is_multiple_of(model.config.block_size),
                        "prefill benchmark requires whole blocks within model context"
                    );
                    let prompt: Vec<u32> = seed.iter().copied().cycle().take(n).collect();
                    let mut times = Vec::new();
                    let mut forwards = 0;
                    for iteration in 0..iterations + 2 {
                        device.synchronize()?;
                        #[cfg(feature = "cuda")]
                        let capture = if profile && iteration >= 2 {
                            Some(candle_core::cuda_backend::cudarc::driver::Profiler::new()?)
                        } else {
                            None
                        };
                        let start = Instant::now();
                        let mut cache = Cache::new(&model.config, n)?;
                        prefill(&model, &prompt, &mut cache, || false)?;
                        let elapsed = start.elapsed().as_secs_f64();
                        #[cfg(feature = "cuda")]
                        drop(capture);
                        ensure!(cache.len() == n, "incomplete prompt cache");
                        forwards = cache.forwards;
                        if iteration >= 2 {
                            times.push(elapsed);
                        }
                    }
                    let mean = times.iter().sum::<f64>() / times.len() as f64;
                    let report = json!({"prompt_tokens":n,"iterations":iterations,"warmups":2,"seconds":times,"mean_seconds":mean,"tokens_per_second":n as f64 / mean,"max_chunk_tokens":chunk_size,"attention_chunk_tokens":model.attention_chunk_tokens(),"forwards":forwards,"batched_experts":!cli.serial_experts,"fused_attention":!cli.unfused_attention,"flash_attention":model.uses_flash_attention(),"fused_expert_mix":!cli.unfused_expert_mix,"dtype":format!("{dtype:?}")});
                    tracing::info!(%report,"prefill benchmark");
                    reports.push(report);
                }
            }
            println!("{}", serde_json::to_string_pretty(&reports)?);
        }
        Command::Bench {
            input,
            prefix_blocks,
            iterations,
            cached_only,
            #[cfg(feature = "cuda")]
            profile,
        } => {
            ensure!(
                (1..=100).contains(&iterations),
                "iterations must be between 1 and 100"
            );
            let seed: Vec<u32> = serde_json::from_slice(&fs::read(input)?)?;
            let b = model.config.block_size;
            ensure!(
                seed.len() >= b,
                "benchmark input must contain at least one block"
            );
            let current = &seed[seed.len() - b..];
            let mut reports = Vec::new();
            for blocks in prefix_blocks {
                ensure!(
                    blocks <= if cached_only { 255 } else { 64 },
                    "benchmark prefix exceeds the diagnostic limit"
                );
                let prefix: Vec<u32> = seed
                    .iter()
                    .cycle()
                    .take(blocks * b)
                    .map(|&t| if t == 156895 { 220 } else { t })
                    .collect();
                let mut cached = Cache::new(&model.config, (blocks + 1) * b)?;
                prefill(&model, &prefix, &mut cached, || false)?;
                model.forward(current, &mut cached, true, None)?;
                device.synchronize()?;
                #[cfg(feature = "cuda")]
                let capture = if profile {
                    Some(candle_core::cuda_backend::cudarc::driver::Profiler::new()?)
                } else {
                    None
                };
                let start = Instant::now();
                for _ in 0..iterations {
                    model.forward(current, &mut cached, true, None)?;
                }
                device.synchronize()?;
                let cached_seconds = start.elapsed().as_secs_f64() / iterations as f64;
                #[cfg(feature = "cuda")]
                drop(capture);
                if cached_only {
                    let report = json!({"prefix_tokens":prefix.len(),"current_tokens":b,"iterations":iterations,"cached_forward_seconds":cached_seconds,"dtype":format!("{dtype:?}"),"batched_experts":!cli.serial_experts});
                    tracing::info!(%report,"benchmark");
                    reports.push(report);
                    continue;
                }
                let all = [prefix.as_slice(), current].concat();
                let mut full = Cache::new(&model.config, all.len())?;
                model.forward(&all, &mut full, true, None)?;
                device.synchronize()?;
                let start = Instant::now();
                for _ in 0..iterations {
                    model.forward(&all, &mut full, true, None)?;
                }
                device.synchronize()?;
                let full_seconds = start.elapsed().as_secs_f64() / iterations as f64;
                let report = json!({"prefix_tokens":prefix.len(),"current_tokens":b,"iterations":iterations,"cached_forward_seconds":cached_seconds,"full_forward_seconds":full_seconds,"speedup":full_seconds/cached_seconds,"dtype":format!("{dtype:?}"),"batched_experts":!cli.serial_experts,"fused_activation":!cli.unfused_activation});
                tracing::info!(%report,"benchmark");
                reports.push(report);
            }
            println!("{}", serde_json::to_string_pretty(&reports)?);
        }
        Command::DecodeBench {
            cases,
            iterations,
            #[cfg(feature = "cuda")]
            profile,
        } => {
            ensure!(
                (1..=100).contains(&iterations),
                "iterations must be between 1 and 100"
            );
            let cases: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(cases)?)?;
            ensure!(
                !cases.is_empty(),
                "decode benchmark requires at least one case"
            );
            let codec = TextCodec::load(&model_path)?;
            let mut reports = Vec::new();
            for case in cases {
                let messages = case["messages"]
                    .as_array()
                    .context("case requires messages")?;
                let prompt = codec.encode(&codec.chat_prompt(messages)?)?;
                let options = Options {
                    max_tokens: case["max_tokens"]
                        .as_u64()
                        .context("case requires max_tokens")?
                        as usize,
                    ..Options::default()
                };
                ensure!(
                    (1..=8192).contains(&options.max_tokens),
                    "decode benchmark max_tokens must be between 1 and 8192"
                );
                ensure!(
                    prompt
                        .len()
                        .checked_add(options.max_tokens)
                        .is_some_and(|n| n <= 8192),
                    "decode benchmark context exceeds 8192 tokens"
                );
                let warm = generate(&model, &prompt, &options, SpecialTokens::default(), || {
                    false
                })?;
                let mut runs = Vec::new();
                let (mut seconds, mut decode_seconds) = (0., 0.);
                let (mut count, mut text_count, mut denoise_forwards, mut blocks) = (0, 0, 0, 0);
                for _ in 0..iterations {
                    #[cfg(feature = "cuda")]
                    let capture = if profile {
                        Some(candle_core::cuda_backend::cudarc::driver::Profiler::new()?)
                    } else {
                        None
                    };
                    let result =
                        generate(&model, &prompt, &options, SpecialTokens::default(), || {
                            false
                        })?;
                    #[cfg(feature = "cuda")]
                    drop(capture);
                    let text_tokens = codec.count_text_tokens(&result.token_ids);
                    seconds += result.stats.elapsed_seconds;
                    decode_seconds += result.stats.elapsed_seconds - result.stats.prefill_seconds;
                    count += result.token_ids.len();
                    text_count += text_tokens;
                    denoise_forwards += result.stats.denoise_forwards;
                    blocks += result.stats.blocks;
                    runs.push(json!({"matches_warmup":result.token_ids == warm.token_ids, "text_tokens":text_tokens, "generation":result}));
                }
                let report = json!({
                    "name":case["name"], "prompt_tokens":prompt.len(), "max_tokens":options.max_tokens,
                    "iterations":iterations, "warmups":1, "mean_seconds":seconds/iterations as f64,
                    "compact_decode_experts":cli.compact_decode_experts,
                    "mean_decode_seconds":decode_seconds/iterations as f64,
                    "mean_model_tokens":count as f64/iterations as f64,
                    "mean_text_tokens":text_count as f64/iterations as f64,
                    "model_tokens_per_second":count as f64/seconds,
                    // Retain the historical field as an explicit alias for model-token throughput.
                    "useful_tokens_per_second":count as f64/seconds,
                    "text_tokens_per_second":text_count as f64/seconds,
                    "decode_text_tokens_per_second":text_count as f64/decode_seconds,
                    "denoise_forwards_per_block":denoise_forwards as f64/blocks as f64,
                    "text_tokens_per_denoise_forward":text_count as f64/denoise_forwards as f64,
                    "token_counting":"model tokens include special tokens; text tokens exclude them",
                    "text":codec.decode(&warm.token_ids)?, "runs":runs
                });
                tracing::info!(%report,"decode benchmark");
                reports.push(report);
            }
            println!("{}", serde_json::to_string_pretty(&reports)?);
        }
        _ => unreachable!(),
    }
    Ok(())
}

#[cfg(test)]
mod backend_tests {
    use super::*;
    #[test]
    fn cpu_and_invalid_device_selection() {
        assert!(device("cpu").unwrap().is_cpu());
        assert!(device("metal").is_err());
        assert!(device("cuda:abc").is_err());
        #[cfg(not(feature = "cuda"))]
        assert!(device("auto").unwrap().is_cpu());
    }
}

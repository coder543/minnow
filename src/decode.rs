// Joint decoder ported from inclusionAI's Apache-2.0 LLaDA2.2 reference.
use crate::model::{Cache, Model};
use crate::prefix::{CacheOptions, CachePool};
use anyhow::{Context, Result, bail, ensure};
use candle_core::{D, Tensor};
use clap::Args;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, time::Instant};

#[derive(Clone, Debug, Args, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    /// Maximum new tokens; omitted means no output cap within the available context.
    #[arg(long)]
    pub max_tokens: Option<usize>,
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,
    #[arg(long, default_value_t = 0.5)]
    pub threshold: f32,
    #[arg(long, default_value_t = 0.0)]
    pub editing_threshold: f32,
    #[arg(long, default_value_t = 32)]
    pub steps: usize,
    #[arg(long, default_value_t = 16)]
    pub max_post_steps: usize,
    #[arg(long, default_value_t = 1000)]
    pub max_steps_per_block: usize,
    #[arg(long, default_value_t = 0)]
    pub top_k: usize,
    #[arg(long, default_value_t = 1.0)]
    pub top_p: f32,
    /// Random seed; omitted chooses fresh randomness for each generation.
    #[arg(long)]
    pub seed: Option<u64>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_tokens: None,
            temperature: 0.0,
            threshold: 0.5,
            editing_threshold: 0.0,
            steps: 32,
            max_post_steps: 16,
            max_steps_per_block: 1000,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        }
    }
}
impl Options {
    pub fn output_tokens(&self, prompt_tokens: usize, context_tokens: usize) -> Result<usize> {
        let remaining = context_tokens
            .checked_sub(prompt_tokens)
            .context("prompt exceeds context")?;
        let limit = self.max_tokens.unwrap_or(remaining);
        ensure!(limit <= remaining, "request exceeds context");
        Ok(limit)
    }

    pub fn validate(&self, vocab: usize) -> Result<()> {
        ensure!(
            self.temperature.is_finite() && self.temperature >= 0.0,
            "temperature must be finite and nonnegative"
        );
        ensure!(
            self.threshold.is_finite()
                && (0.0..=1.0).contains(&self.threshold)
                && self.editing_threshold.is_finite()
                && (0.0..=1.0).contains(&self.editing_threshold),
            "thresholds must be between zero and one"
        );
        ensure!(
            self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0 && self.top_k <= vocab,
            "invalid top_p or top_k"
        );
        ensure!(
            self.max_steps_per_block > 0
                && self.max_steps_per_block <= 1000
                && self.steps <= 1000
                && self.max_post_steps <= 1000,
            "decode step limits must be <= 1000 and max_steps_per_block must be positive"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SpecialTokens {
    pub mask: u32,
    pub delete: u32,
    pub split: u32,
    pub eos: u32,
}
impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            mask: 156895,
            delete: 156930,
            split: 156931,
            eos: 156892,
        }
    }
}

#[derive(Clone, Default, Debug, Serialize)]
pub struct Stats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub blocks: usize,
    pub denoise_forwards: usize,
    /// Extra evaluations needed to refresh K/V after final token edits.
    pub commit_forwards: usize,
    /// Block commits that reuse the last successful refinement's K/V.
    pub reused_commits: usize,
    pub total_forwards: usize,
    pub processed_tokens: usize,
    pub prefill_tokens: usize,
    /// Exact complete prompt blocks reused from a previous request.
    pub cached_tokens: usize,
    pub cache_slot: Option<usize>,
    pub cache_copied_tokens: usize,
    pub cache_seconds: f64,
    /// Positions evaluated with vocabulary predictions, including repeated refinements.
    pub evaluated_tokens: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub elapsed_seconds: f64,
    pub tokens_per_second: f64,
    pub total_tokens_per_second: f64,
    pub refinement_steps_per_block: f64,
}
impl Stats {
    fn update(&mut self, cache: &Cache, start: Instant, block_size: usize) {
        self.total_forwards = cache.forwards;
        self.processed_tokens = cache.processed_tokens;
        self.evaluated_tokens = self.denoise_forwards * block_size;
        self.elapsed_seconds = start.elapsed().as_secs_f64();
        self.decode_seconds =
            (self.elapsed_seconds - self.prefill_seconds - self.cache_seconds).max(0.0);
        self.tokens_per_second = rate(self.completion_tokens, self.elapsed_seconds);
        self.total_tokens_per_second = rate(self.evaluated_tokens, self.decode_seconds);
        self.refinement_steps_per_block = rate(self.denoise_forwards, self.blocks as f64);
    }
}
pub fn rate(tokens: usize, seconds: f64) -> f64 {
    if seconds > 0.0 {
        tokens as f64 / seconds
    } else {
        0.0
    }
}
#[derive(Clone, Default, Debug, Serialize)]
pub struct BatchStats {
    pub index: usize,
    pub offset: usize,
    pub completion_tokens: usize,
    pub refinement_steps: usize,
    pub evaluated_tokens: usize,
    pub elapsed_seconds: f64,
    pub tokens_per_second: f64,
    pub total_tokens_per_second: f64,
}
pub enum Progress<'a> {
    Prefill {
        cached: usize,
        slot: Option<usize>,
        processed: usize,
        total: usize,
        seconds: f64,
    },
    Refinement {
        stats: &'a Stats,
        batch: &'a BatchStats,
    },
    /// Only final tokens are exposed. Returning false stops before the next block.
    Block {
        token_ids: &'a [u32],
        stats: &'a Stats,
        batch: &'a BatchStats,
    },
}
#[derive(Debug, Serialize)]
pub struct Generation {
    pub token_ids: Vec<u32>,
    pub finish_reason: String,
    pub stats: Stats,
    pub batches: Vec<BatchStats>,
}

#[derive(Clone, Copy, Debug)]
struct Prediction {
    token: u32,
    confidence: f32,
}

fn sample_row(
    row: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    banned: &[u32],
    rng: &mut StdRng,
) -> Result<Prediction> {
    let mut order: Vec<usize> = (0..row.len())
        .filter(|i| !banned.contains(&(*i as u32)))
        .collect();
    ensure!(
        !order.is_empty() && row.iter().all(|v| !v.is_nan() && *v != f32::INFINITY),
        "invalid logits"
    );
    order.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    let max = row[order[0]];
    ensure!(max.is_finite(), "no finite logits available");
    let denom: f32 = order.iter().map(|&i| (row[i] - max).exp()).sum();
    let mut token = order[0];
    if temperature > 0.0 {
        if top_k > 0 && top_k < order.len() {
            let cutoff = row[order[top_k - 1]];
            order.retain(|&i| row[i] >= cutoff);
        }
        let mut probs: Vec<f32> = order
            .iter()
            .map(|&i| ((row[i] - max) / temperature).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        if top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut keep = 0;
            for p in &probs {
                cumulative += p / sum;
                keep += 1;
                if cumulative > top_p {
                    break;
                }
            }
            probs.truncate(keep);
            order.truncate(keep);
        }
        let mut draw = rng.random::<f32>() * probs.iter().sum::<f32>();
        token = *order.last().unwrap();
        for (&i, p) in order.iter().zip(probs) {
            if draw < p {
                token = i;
                break;
            }
            draw -= p;
        }
    }
    Ok(Prediction {
        token: token as u32,
        confidence: (row[token] - max).exp() / denom,
    })
}

fn greedy_predictions(logits: &Tensor) -> Result<Vec<Prediction>> {
    // Greedy serving transfers only 32 IDs and confidences, not 32 x vocabulary logits.
    #[cfg(feature = "cuda")]
    if logits.device().is_cuda() {
        let greedy: Vec<Prediction> = crate::cuda::greedy_confidence(logits)?
            .to_vec2::<f32>()?
            .into_iter()
            .map(|p| Prediction {
                token: p[0] as u32,
                confidence: p[1],
            })
            .collect();
        return Ok(greedy);
    }
    let ids = logits.argmax(D::Minus1)?;
    let probs = candle_nn::ops::softmax_last_dim(logits)?;
    let conf = probs
        .gather(&ids.unsqueeze(1)?, 1)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let greedy: Vec<Prediction> = ids
        .to_vec1::<u32>()?
        .into_iter()
        .zip(conf)
        .map(|(token, confidence)| Prediction { token, confidence })
        .collect();
    Ok(greedy)
}

fn predictions(
    logits: &Tensor,
    opts: &Options,
    rng: &mut StdRng,
) -> Result<(Vec<Prediction>, Vec<Prediction>)> {
    let greedy = greedy_predictions(logits)?;
    ensure!(
        greedy.iter().all(|p| p.confidence.is_finite()),
        "non-finite model probabilities"
    );
    let sampled = if opts.temperature == 0.0 {
        greedy.clone()
    } else {
        logits
            .to_vec2::<f32>()?
            .iter()
            .map(|row| sample_row(row, opts.temperature, opts.top_k, opts.top_p, &[], rng))
            .collect::<Result<_>>()?
    };
    Ok((sampled, greedy))
}

/// DELETE removes a slot; SPLIT inserts a new mask before the previous token.
pub fn apply_edits(
    tokens: &[u32],
    old: &[u32],
    original: &[bool],
    special: SpecialTokens,
) -> (Vec<u32>, Vec<bool>) {
    let mut out = Vec::new();
    let mut tracking = Vec::new();
    for (i, &token) in tokens.iter().enumerate() {
        if token == special.delete {
            continue;
        }
        if token == special.split {
            out.extend([special.mask, old[i]]);
            tracking.extend([false, original[i]]);
        } else {
            out.push(token);
            tracking.push(original[i]);
        }
    }
    out.resize(tokens.len(), special.mask);
    tracking.resize(tokens.len(), false);
    (out, tracking)
}

pub fn decode_block(
    mut tokens: Vec<u32>,
    opts: &Options,
    special: SpecialTokens,
    rng: &mut StdRng,
    mut forward: impl FnMut(&[u32]) -> Result<Tensor>,
) -> Result<(Vec<u32>, usize)> {
    let n = tokens.len();
    let prompt: Vec<bool> = tokens.iter().map(|&t| t != special.mask).collect();
    ensure!(
        !tokens
            .iter()
            .any(|&t| t == special.delete || t == special.split),
        "reserved edit tokens in prompt"
    );
    let mut original: Vec<bool> = prompt.iter().map(|v| !v).collect();
    let initial = original.iter().filter(|v| **v).count();
    if initial == 0 {
        return Ok((tokens, 0));
    }
    let schedule: Vec<usize> = (0..opts.steps)
        .map(|i| initial / opts.steps + usize::from(i < initial % opts.steps))
        .collect();
    let mut seen = HashSet::new();
    let mut post = 0;
    let mut step = 0;
    loop {
        let old = tokens.clone();
        let masks: Vec<bool> = old
            .iter()
            .enumerate()
            .map(|(i, &t)| t == special.mask && !prompt[i])
            .collect();
        let original_count = (0..n)
            .filter(|&i| original[i] && old[i] == special.mask)
            .count();
        let mask_count = masks.iter().filter(|v| **v).count();
        let new_count = mask_count - original_count;
        post = if original_count == 0 { post + 1 } else { 0 };
        if mask_count == 0 && post > opts.max_post_steps {
            return Ok((tokens, step));
        }
        ensure!(
            step < opts.max_steps_per_block,
            "block did not converge within {} steps ({} masks remain)",
            opts.max_steps_per_block,
            mask_count
        );
        let logits = forward(&old)?;
        ensure!(
            logits.dim(0)? == n,
            "decoder logits have wrong number of positions"
        );
        let (mut sampled, mut greedy) = predictions(&logits, opts, rng)?;
        let mut m2t = vec![false; n];
        if mask_count > 0 {
            if step < schedule.len() {
                let need = schedule[step] + new_count;
                let high: Vec<usize> = (0..n)
                    .filter(|&i| masks[i] && sampled[i].confidence > opts.threshold)
                    .collect();
                if high.len() >= need {
                    for i in high {
                        m2t[i] = true;
                    }
                } else {
                    let mut order: Vec<usize> = (0..n).filter(|&i| masks[i]).collect();
                    order.sort_unstable_by(|&a, &b| {
                        sampled[b]
                            .confidence
                            .total_cmp(&sampled[a].confidence)
                            .then(a.cmp(&b))
                    });
                    for &i in order.iter().take(need) {
                        m2t[i] = true;
                    }
                }
            } else {
                m2t.clone_from(&masks);
            }
        }
        let t2t: Vec<bool> = (0..n)
            .map(|i| {
                !masks[i]
                    && !prompt[i]
                    && greedy[i].confidence > opts.editing_threshold
                    && old[i] != greedy[i].token
            })
            .collect();
        let final_round = if opts.max_post_steps > 0 {
            post >= opts.max_post_steps
        } else {
            original_count > 0 && (0..n).all(|i| !masks[i] || !original[i] || m2t[i])
        };
        if final_round {
            m2t.clone_from(&masks);
            for i in 0..n {
                for (write, p, temperature) in [
                    (m2t[i], &mut sampled[i], opts.temperature),
                    (t2t[i], &mut greedy[i], 0.0),
                ] {
                    if write && [special.delete, special.split].contains(&p.token) {
                        *p = sample_row(
                            &logits.get(i)?.to_vec1::<f32>()?,
                            temperature,
                            opts.top_k,
                            opts.top_p,
                            &[special.delete, special.split],
                            rng,
                        )?;
                    }
                }
            }
        }
        for i in 0..n {
            if m2t[i] {
                tokens[i] = sampled[i].token;
            }
            if t2t[i] {
                tokens[i] = greedy[i].token;
            }
        }
        if !final_round && (0..n).any(|i| m2t[i] || t2t[i]) && seen.contains(&tokens) {
            for _ in 0..5 {
                let changed: Vec<usize> = (0..n).filter(|&i| tokens[i] != old[i]).collect();
                if changed.is_empty() {
                    break;
                }
                let i = changed[rng.random_range(0..changed.len())];
                let temperature = if m2t[i] { opts.temperature } else { 0.0 };
                tokens[i] = sample_row(
                    &logits.get(i)?.to_vec1::<f32>()?,
                    temperature,
                    opts.top_k,
                    opts.top_p,
                    &[tokens[i]],
                    rng,
                )?
                .token;
                if !seen.contains(&tokens) {
                    break;
                }
            }
        }
        seen.insert(tokens.clone());
        (tokens, original) = apply_edits(&tokens, &old, &original, special);
        step += 1;
        if tokens == old && !tokens.contains(&special.mask) {
            return Ok((tokens, step));
        }
    }
}

/// A forward may execute immediately or join other requests at the scheduler.
/// The latter transfers ownership of K/V to the GPU worker for that invocation.
pub trait Executor {
    fn config(&self) -> &crate::config::Config;
    fn prefill_chunk_tokens(&self) -> usize;
    fn forward_tokens(
        &self,
        tokens: &[u32],
        cache: &mut Cache,
        logits: bool,
    ) -> Result<Option<Tensor>>;
    fn synchronize(&self) -> Result<()>;
}
impl Executor for Model {
    fn config(&self) -> &crate::config::Config {
        &self.config
    }
    fn prefill_chunk_tokens(&self) -> usize {
        Model::prefill_chunk_tokens(self)
    }
    fn forward_tokens(
        &self,
        tokens: &[u32],
        cache: &mut Cache,
        logits: bool,
    ) -> Result<Option<Tensor>> {
        self.forward(tokens, cache, logits, None)
    }
    fn synchronize(&self) -> Result<()> {
        Ok(self.device().synchronize()?)
    }
}

/// Build committed K/V for a block-aligned prompt using the serving prefill path.
pub fn prefill(
    model: &(impl Executor + ?Sized),
    prompt: &[u32],
    cache: &mut Cache,
    cancelled: impl Fn() -> bool,
) -> Result<()> {
    prefill_observed(model, prompt, cache, cancelled, |_, _, _| Ok(()))
}

fn prefill_observed(
    model: &(impl Executor + ?Sized),
    prompt: &[u32],
    cache: &mut Cache,
    cancelled: impl Fn() -> bool,
    mut progress: impl FnMut(usize, usize, f64) -> Result<()>,
) -> Result<()> {
    let start = Instant::now();
    let b = model.config().block_size;
    ensure!(
        prompt.len().is_multiple_of(b) && prompt.starts_with(cache.tokens()),
        "prefill requires complete prompt blocks matching the committed cache"
    );
    progress(cache.len(), prompt.len(), 0.0)?;
    while cache.len() < prompt.len() {
        let requested = model.prefill_chunk_tokens().min(prompt.len() - cache.len());
        let chunk = &prompt[cache.len()..cache.len() + requested];
        if cancelled() {
            bail!("request cancelled");
        }
        model.forward_tokens(chunk, cache, false)?;
        cache.commit(chunk)?;
        model.synchronize()?;
        progress(cache.len(), prompt.len(), start.elapsed().as_secs_f64())?;
    }
    model.synchronize()?;
    Ok(())
}

/// Commit a final block, refreshing K/V only when its tokens differ from the
/// last successful forward. Returns whether an extra forward was necessary.
pub fn commit_block(
    model: &(impl Executor + ?Sized),
    tokens: &[u32],
    cache: &mut Cache,
) -> Result<bool> {
    let refresh = !cache.staged_matches(tokens);
    if refresh {
        model.forward_tokens(tokens, cache, false)?;
    }
    cache.commit(tokens)?;
    Ok(refresh)
}

pub fn generate(
    model: &Model,
    prompt: &[u32],
    opts: &Options,
    special: SpecialTokens,
    cancelled: impl Fn() -> bool,
) -> Result<Generation> {
    generate_observed(model, prompt, opts, special, cancelled, |_| Ok(true))
}

pub fn generate_observed(
    model: &Model,
    prompt: &[u32],
    opts: &Options,
    special: SpecialTokens,
    cancelled: impl Fn() -> bool,
    observe: impl FnMut(Progress<'_>) -> Result<bool>,
) -> Result<Generation> {
    let mut pool = CachePool::new(
        model,
        model.config.max_position_embeddings,
        CacheOptions {
            cache_slots: 0,
            ..CacheOptions::default()
        },
    )?;
    generate_cached_observed(
        model,
        prompt,
        opts,
        special,
        (&mut pool, false),
        cancelled,
        observe,
    )
}

/// Reuse committed prefixes from the pool when the request's cache flag is true.
/// Failed generations discard their checked-out slot; other slots remain valid.
pub fn generate_cached_observed(
    model: &Model,
    prompt: &[u32],
    opts: &Options,
    special: SpecialTokens,
    (pool, reuse): (&mut CachePool, bool),
    cancelled: impl Fn() -> bool,
    observe: impl FnMut(Progress<'_>) -> Result<bool>,
) -> Result<Generation> {
    opts.validate(model.config.vocab_size)?;
    ensure!(!prompt.is_empty(), "prompt must not be empty");
    ensure!(
        prompt
            .iter()
            .all(|&t| (t as usize) < model.config.vocab_size
                && ![special.mask, special.delete, special.split].contains(&t)),
        "prompt contains reserved diffusion tokens or invalid token IDs"
    );
    let start = Instant::now();
    let stats = Stats {
        prompt_tokens: prompt.len(),
        ..Stats::default()
    };
    let max_tokens = opts.output_tokens(prompt.len(), pool.max_context())?;
    if max_tokens == 0 {
        return Ok(Generation {
            token_ids: vec![],
            finish_reason: "length".into(),
            stats,
            batches: vec![],
        });
    }
    let b = model.config.block_size;
    let requested = prompt
        .len()
        .checked_add(max_tokens)
        .ok_or_else(|| anyhow::anyhow!("context length overflow"))?;
    let total = requested
        .checked_add(b - 1)
        .ok_or_else(|| anyhow::anyhow!("context length overflow"))?
        / b
        * b;
    ensure!(
        total <= model.config.max_position_embeddings,
        "request exceeds model context"
    );
    let prefill_len = prompt.len() / b * b;
    let capacity = pool.initial_capacity(prompt.len(), max_tokens, b);
    let mut checkout = pool.checkout(model, &prompt[..prefill_len], capacity, reuse)?;
    let result = generate_in_cache_observed(
        &pool.executor(model, &checkout),
        prompt,
        opts,
        special,
        (&mut checkout, start),
        cancelled,
        observe,
    );
    if result.is_ok() {
        pool.checkin(checkout);
    } else {
        pool.discard(checkout);
    }
    result
}

/// Execute an admitted request. The scheduler owns slot admission and returns
/// the reservation on both success and failure; the decoder owns refinement.
pub(crate) fn generate_in_cache_observed(
    model: &(impl Executor + ?Sized),
    prompt: &[u32],
    opts: &Options,
    special: SpecialTokens,
    (checkout, start): (&mut crate::prefix::Checkout, Instant),
    cancelled: impl Fn() -> bool,
    mut observe: impl FnMut(Progress<'_>) -> Result<bool>,
) -> Result<Generation> {
    opts.validate(model.config().vocab_size)?;
    ensure!(
        !prompt.is_empty()
            && prompt
                .iter()
                .all(|&t| (t as usize) < model.config().vocab_size
                    && ![special.mask, special.delete, special.split].contains(&t)),
        "invalid prompt tokens"
    );
    let b = model.config().block_size;
    let max_tokens = opts.output_tokens(prompt.len(), checkout.context_limit)?;
    let requested = prompt
        .len()
        .checked_add(max_tokens)
        .context("context length overflow")?;
    let total = requested
        .checked_add(b - 1)
        .context("context length overflow")?
        / b
        * b;
    ensure!(
        total <= checkout.context_limit,
        "request exceeds configured context"
    );
    let prefill_len = prompt.len() / b * b;
    let mut stats = Stats {
        prompt_tokens: prompt.len(),
        cached_tokens: checkout.cached,
        cache_copied_tokens: checkout.copied,
        cache_slot: checkout.slot,
        cache_seconds: start.elapsed().as_secs_f64(),
        ..Stats::default()
    };
    if max_tokens == 0 {
        return Ok(Generation {
            token_ids: vec![],
            finish_reason: "length".into(),
            stats,
            batches: vec![],
        });
    }

    let cache = &mut checkout.cache;
    let prefill_start = Instant::now();
    prefill_observed(
        model,
        &prompt[..prefill_len],
        cache,
        &cancelled,
        |processed, total, seconds| {
            ensure!(
                observe(Progress::Prefill {
                    cached: stats.cached_tokens,
                    slot: stats.cache_slot,
                    processed,
                    total,
                    seconds
                })?,
                "request cancelled"
            );
            Ok(())
        },
    )?;
    stats.prefill_tokens = prefill_len - stats.cached_tokens;
    stats.prefill_seconds = prefill_start.elapsed().as_secs_f64();
    let mut rng = StdRng::seed_from_u64(opts.seed.unwrap_or_else(rand::random));
    let mut all = prompt[..prefill_len].to_vec();
    let mut finish = "length";
    let mut batches = Vec::new();
    for offset in (prefill_len..total).step_by(b) {
        let batch_start = Instant::now();
        let mut batch = BatchStats {
            index: stats.blocks,
            offset,
            ..BatchStats::default()
        };
        stats.blocks += 1;
        let mut block = vec![special.mask; b];
        if offset < prompt.len() {
            block[..prompt.len() - offset].copy_from_slice(&prompt[offset..]);
        }
        let (block, steps) = decode_block(block, opts, special, &mut rng, |tokens| {
            if cancelled() {
                bail!("request cancelled");
            }
            let logits = model.forward_tokens(tokens, cache, true)?.unwrap();
            // The prediction path synchronizes as well. Synchronize here so live
            // timing events describe completed GPU work rather than dispatch time.
            model.synchronize()?;
            stats.denoise_forwards += 1;
            stats.update(cache, start, b);
            batch.refinement_steps += 1;
            batch.evaluated_tokens += b;
            batch.elapsed_seconds = batch_start.elapsed().as_secs_f64();
            batch.total_tokens_per_second = rate(batch.evaluated_tokens, batch.elapsed_seconds);
            ensure!(
                observe(Progress::Refinement {
                    stats: &stats,
                    batch: &batch
                })?,
                "request cancelled"
            );
            Ok(logits)
        })?;
        all.extend_from_slice(&block);
        let mut end = all.len().min(requested);
        let eos = all[prompt.len()..end]
            .iter()
            .position(|&t| t == special.eos);
        if let Some(eos) = eos {
            end = prompt.len() + eos + 1;
            finish = "stop";
        }
        if eos.is_none() && offset + b < total {
            if cancelled() {
                bail!("request cancelled");
            }
            if commit_block(model, &block, cache)? {
                stats.commit_forwards += 1;
            } else {
                stats.reused_commits += 1;
            }
        }
        model.synchronize()?;
        batch.completion_tokens = end - prompt.len() - stats.completion_tokens;
        stats.completion_tokens = end - prompt.len();
        stats.update(cache, start, b);
        batch.elapsed_seconds = batch_start.elapsed().as_secs_f64();
        batch.tokens_per_second = rate(batch.completion_tokens, batch.elapsed_seconds);
        batch.total_tokens_per_second = rate(batch.evaluated_tokens, batch.elapsed_seconds);
        let proceed = observe(Progress::Block {
            token_ids: &all[prompt.len()..end],
            stats: &stats,
            batch: &batch,
        })?;
        batches.push(batch);
        if !proceed {
            finish = "stop";
        }
        if eos.is_some() || !proceed {
            break;
        }
        tracing::debug!(offset, steps, "decoded block");
    }
    let mut token_ids = all[prompt.len()..all.len().min(requested)].to_vec();
    if let Some(eos) = token_ids.iter().position(|&t| t == special.eos) {
        token_ids.truncate(eos + 1);
    }
    ensure!(
        !token_ids
            .iter()
            .any(|t| [special.mask, special.delete, special.split].contains(t)),
        "decoding left unresolved diffusion tokens"
    );
    model.synchronize()?;
    stats.completion_tokens = token_ids.len();
    stats.update(cache, start, b);
    Ok(Generation {
        token_ids,
        finish_reason: finish.into(),
        stats,
        batches,
    })
}

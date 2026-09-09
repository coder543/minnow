//! Multiple resident conversation prefixes with exclusive active reservations. Only complete,
//! committed blocks are eligible; each slot owns independent writable K/V.
use crate::model::{Cache, Model};
use anyhow::{Result, ensure};
use clap::Args;
use serde::Serialize;
use std::{
    cell::RefCell,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

pub(crate) const KV_CHUNK_TOKENS: usize = 2048;

#[derive(Debug)]
pub(crate) struct CacheBudgetError;
impl std::fmt::Display for CacheBudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            "KV cache budget exhausted by active requests; retry when capacity is available",
        )
    }
}
impl std::error::Error for CacheBudgetError {}

#[derive(Clone, Debug, Args, Serialize)]
pub struct CacheOptions {
    /// Number of resident conversation prefixes (0 disables cross-request reuse).
    #[arg(long, default_value_t = 4)]
    pub cache_slots: usize,
    /// Minimum matching fraction of the shorter prefix to update a slot in place.
    #[arg(long, default_value_t = 0.8)]
    pub cache_reuse_threshold: f64,
    /// Aggregate K/V capacity in MiB; defaults to one full context's K/V size.
    #[arg(long)]
    pub cache_max_mib: Option<usize>,
}
impl Default for CacheOptions {
    fn default() -> Self {
        Self {
            cache_slots: 4,
            cache_reuse_threshold: 0.8,
            cache_max_mib: None,
        }
    }
}

struct Slot {
    cache: Option<Cache>,
    used: u64,
    busy: bool,
    active_cached: usize,
    active_capacity: usize,
}
#[derive(Clone, Debug, Serialize)]
pub struct SlotInfo {
    pub id: usize,
    pub cached_tokens: usize,
    pub capacity_tokens: usize,
    pub busy: bool,
}
pub struct CachePool {
    options: CacheOptions,
    slots: Vec<Slot>,
    clock: u64,
    budget_tokens: usize,
    max_context: usize,
    identity: Arc<()>,
    active_capacity: usize,
}
pub(crate) struct Checkout {
    pub cache: Cache,
    pub slot: Option<usize>,
    pub cached: usize,
    pub copied: usize,
    pub(crate) reservation: Arc<AtomicUsize>,
    pub(crate) context_limit: usize,
}
impl Checkout {
    /// An empty response does not run the model or reserve/evict K/V storage.
    pub(crate) fn empty(model: &Model, capacity: usize) -> Result<Self> {
        Ok(Self {
            cache: Cache::new(&model.config, capacity)?,
            slot: None,
            cached: 0,
            copied: 0,
            reservation: Arc::new(AtomicUsize::new(0)),
            context_limit: model.config.max_position_embeddings,
        })
    }
}
impl CachePool {
    pub fn new(model: &Model, max_context: usize, options: CacheOptions) -> Result<Self> {
        ensure!(
            options.cache_slots <= 64,
            "cache-slots must be between 0 and 64"
        );
        ensure!(
            options.cache_reuse_threshold.is_finite()
                && (0.0..=1.0).contains(&options.cache_reuse_threshold),
            "cache-reuse-threshold must be between zero and one"
        );
        let b = model.config.block_size;
        ensure!(
            max_context > 0
                && max_context <= model.config.max_position_embeddings
                && max_context.is_multiple_of(b),
            "invalid cache context limit"
        );
        let budget_tokens = match options.cache_max_mib {
            Some(mib) => {
                mib.checked_mul(1024 * 1024)
                    .ok_or_else(|| anyhow::anyhow!("cache budget overflow"))?
                    / model.kv_bytes_per_token()
                    / b
                    * b
            }
            None => max_context,
        };
        ensure!(
            budget_tokens >= max_context,
            "cache-max-mib must fit at least one max-context K/V buffer ({} MiB)",
            (max_context * model.kv_bytes_per_token()).div_ceil(1024 * 1024)
        );
        Ok(Self {
            slots: (0..options.cache_slots)
                .map(|_| Slot {
                    cache: None,
                    used: 0,
                    busy: false,
                    active_cached: 0,
                    active_capacity: 0,
                })
                .collect(),
            options,
            clock: 0,
            budget_tokens,
            max_context,
            identity: model.cache_identity.clone(),
            active_capacity: 0,
        })
    }
    pub fn snapshot(&self) -> Vec<SlotInfo> {
        self.slots
            .iter()
            .enumerate()
            .map(|(id, s)| SlotInfo {
                id,
                cached_tokens: s.cache.as_ref().map_or(s.active_cached, Cache::len),
                capacity_tokens: s.cache.as_ref().map_or(s.active_capacity, Cache::capacity),
                busy: s.busy,
            })
            .collect()
    }
    pub fn budget_bytes(&self, model: &Model) -> usize {
        self.budget_tokens * model.kv_bytes_per_token()
    }
    pub(crate) fn max_context(&self) -> usize {
        self.max_context
    }
    pub(crate) fn chunk_capacity(&self, required: usize) -> usize {
        required
            .div_ceil(KV_CHUNK_TOKENS)
            .saturating_mul(KV_CHUNK_TOKENS)
            .min(self.max_context)
    }
    pub(crate) fn initial_capacity(
        &self,
        prompt_tokens: usize,
        output_tokens: usize,
        block: usize,
    ) -> usize {
        let required = (prompt_tokens + output_tokens.min(block)).div_ceil(block) * block;
        self.chunk_capacity(required)
    }
    /// Grow only between forwards, with all active reservations charged to this pool.
    /// False means another active request must release capacity first.
    pub(crate) fn grow(
        &mut self,
        model: &Model,
        cache: &mut Cache,
        reservation: &AtomicUsize,
        slot: Option<usize>,
        required: usize,
    ) -> Result<bool> {
        ensure!(
            required <= self.max_context,
            "request exceeds cache context"
        );
        if required <= cache.capacity() {
            return Ok(true);
        }
        let capacity = self.chunk_capacity(required);
        let previous = reservation.load(Ordering::Relaxed);
        ensure!(
            previous == cache.capacity(),
            "cache reservation differs from allocation"
        );
        let extra = capacity - previous;
        if self.active_capacity + extra > self.budget_tokens {
            return Ok(false);
        }
        while self.capacity() + extra > self.budget_tokens {
            let victim = self
                .slots
                .iter()
                .enumerate()
                .filter(|(_, s)| !s.busy && s.cache.is_some())
                .min_by_key(|(_, s)| s.used)
                .map(|(i, _)| i)
                .expect("idle cache eviction covers growth within active budget");
            self.slots[victim].cache = None;
        }
        cache.resize(&model.config, capacity)?;
        self.active_capacity += extra;
        reservation.store(capacity, Ordering::Relaxed);
        if let Some(slot) = slot {
            self.slots[slot].active_capacity = capacity;
            self.slots[slot].active_cached = cache.len();
        }
        Ok(true)
    }
    pub(crate) fn executor<'a>(
        &'a mut self,
        model: &'a Model,
        checkout: &Checkout,
    ) -> impl crate::decode::Executor + 'a {
        PooledExecutor {
            model,
            pool: RefCell::new(self),
            reservation: checkout.reservation.clone(),
            slot: checkout.slot,
        }
    }
    fn lru(&self, excluded: &[usize]) -> Option<usize> {
        (0..self.slots.len())
            .filter(|i| !excluded.contains(i) && !self.slots[*i].busy)
            .min_by_key(|&i| (self.slots[i].cache.is_some(), self.slots[i].used))
    }
    fn capacity(&self) -> usize {
        self.active_capacity
            + self
                .slots
                .iter()
                .filter_map(|s| s.cache.as_ref())
                .map(Cache::capacity)
                .sum::<usize>()
    }
    pub(crate) fn can_admit(&self, capacity: usize, enabled: bool) -> bool {
        let capacity = self.chunk_capacity(capacity);
        self.active_capacity + capacity <= self.budget_tokens
            && (!enabled || self.slots.is_empty() || self.slots.iter().any(|s| !s.busy))
    }
    pub(crate) fn checkout(
        &mut self,
        model: &Model,
        prompt: &[u32],
        capacity: usize,
        enabled: bool,
    ) -> Result<Checkout> {
        ensure!(
            Arc::ptr_eq(&self.identity, &model.cache_identity),
            "cache pool belongs to a different model"
        );
        ensure!(
            capacity <= self.max_context,
            "request exceeds cache context"
        );
        ensure!(
            self.can_admit(capacity, enabled),
            "cache capacity is occupied by active requests"
        );
        let capacity = self.chunk_capacity(capacity);
        if !enabled || self.slots.is_empty() {
            // A cold request may need the whole memory budget too.
            while self.capacity() + capacity > self.budget_tokens {
                let Some(i) = self
                    .slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.cache.is_some() && !s.busy)
                    .min_by_key(|(_, s)| s.used)
                    .map(|(i, _)| i)
                else {
                    break;
                };
                self.slots[i].cache = None;
                if self.capacity() == self.active_capacity {
                    break;
                }
            }
            let cache = Cache::new(&model.config, capacity)?;
            self.active_capacity += capacity;
            return Ok(Checkout {
                cache,
                slot: None,
                cached: 0,
                copied: 0,
                reservation: Arc::new(AtomicUsize::new(capacity)),
                context_limit: self.max_context,
            });
        }
        let b = model.config.block_size;
        let best = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if s.busy {
                    return None;
                }
                let c = s.cache.as_ref()?;
                let matched = prompt
                    .iter()
                    .zip(c.tokens())
                    .take_while(|(a, b)| a == b)
                    .count()
                    / b
                    * b;
                (matched > 0).then_some((i, matched))
            })
            .max_by_key(|&(i, matched)| (matched, self.slots[i].used));
        let mut dest = match best {
            Some((i, matched))
                if self.slots.len() == 1
                    || matched as f64
                        >= self.options.cache_reuse_threshold
                            * prompt
                                .len()
                                .min(self.slots[i].cache.as_ref().unwrap().len())
                                as f64 =>
            {
                i
            }
            Some((i, _)) => self.lru(&[i]).unwrap_or(i),
            None => self.lru(&[]).unwrap(),
        };
        if let Some((source, _)) = best
            && dest != source
            && self.active_capacity
                + self.slots[source].cache.as_ref().unwrap().capacity()
                + capacity
                > self.budget_tokens
        {
            // Keeping both branches cannot fit: retain the hit in place.
            dest = source;
        }
        let mut excluded = vec![dest];
        if let Some((source, _)) = best {
            excluded.push(source);
        }
        while self.capacity() - self.slots[dest].cache.as_ref().map_or(0, Cache::capacity)
            + capacity
            > self.budget_tokens
        {
            let victim = self
                .lru(&excluded)
                .expect("destination fits after evicting other slots");
            self.slots[victim].cache = None;
            excluded.push(victim);
        }
        let old = self.slots[dest].cache.take();
        let cached = best.map_or(0, |(_, n)| n);
        let copied = best
            .filter(|(source, _)| *source != dest)
            .map_or(0, |(_, n)| n);
        let cache = if copied > 0 {
            drop(old);
            self.slots[best.unwrap().0]
                .cache
                .as_ref()
                .unwrap()
                .copy_prefix(&model.config, cached, capacity)?
        } else {
            let mut cache = match old {
                Some(cache) => cache,
                None => Cache::new(&model.config, capacity)?,
            };
            cache.truncate(cached)?;
            cache.resize(&model.config, capacity)?;
            cache
        };
        model.device().synchronize()?;
        self.clock += 1;
        self.slots[dest].used = self.clock;
        self.slots[dest].busy = true;
        self.slots[dest].active_cached = cached;
        self.slots[dest].active_capacity = capacity;
        self.active_capacity += capacity;
        Ok(Checkout {
            cache,
            slot: Some(dest),
            cached,
            copied,
            reservation: Arc::new(AtomicUsize::new(capacity)),
            context_limit: self.max_context,
        })
    }
    pub(crate) fn checkin(&mut self, checkout: Checkout) {
        self.active_capacity -= checkout.reservation.load(Ordering::Relaxed);
        if let Some(slot) = checkout.slot {
            self.slots[slot].busy = false;
            self.slots[slot].active_cached = 0;
            self.slots[slot].active_capacity = 0;
            self.slots[slot].cache = (!checkout.cache.is_empty()).then_some(checkout.cache);
        }
    }
    pub(crate) fn discard(&mut self, checkout: Checkout) {
        self.active_capacity -= checkout.reservation.load(Ordering::Relaxed);
        if let Some(slot) = checkout.slot {
            self.slots[slot].busy = false;
            self.slots[slot].active_cached = 0;
            self.slots[slot].active_capacity = 0;
        }
    }
}

struct PooledExecutor<'a> {
    model: &'a Model,
    pool: RefCell<&'a mut CachePool>,
    reservation: Arc<AtomicUsize>,
    slot: Option<usize>,
}
impl crate::decode::Executor for PooledExecutor<'_> {
    fn config(&self) -> &crate::config::Config {
        &self.model.config
    }
    fn prefill_chunk_tokens(&self) -> usize {
        self.model.prefill_chunk_tokens()
    }
    fn synchronize(&self) -> Result<()> {
        Ok(self.model.device().synchronize()?)
    }
    fn forward_tokens(
        &self,
        tokens: &[u32],
        cache: &mut Cache,
        logits: bool,
    ) -> Result<Option<candle_core::Tensor>> {
        if !self.pool.borrow_mut().grow(
            self.model,
            cache,
            &self.reservation,
            self.slot,
            cache.len() + tokens.len(),
        )? {
            return Err(CacheBudgetError.into());
        }
        self.model.forward(tokens, cache, logits, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{
        Options, Progress, SpecialTokens, generate, generate_cached_observed, prefill,
    };
    use candle_core::{DType, Device};
    use std::path::Path;

    fn fixture_model() -> Model {
        Model::load(Path::new("tests/fixtures/tiny"), DType::F32, &Device::Cpu).unwrap()
    }
    #[test]
    fn chunk_growth_preserves_kv_and_accounts_for_active_and_idle_caches() {
        let mut model = fixture_model();
        model.config.max_position_embeddings = 8192;
        let mut pool = CachePool::new(&model, 8192, CacheOptions::default()).unwrap();
        assert_eq!(pool.initial_capacity(32, 8160, 32), 2048);
        let mut first = pool.checkout(&model, &[1; 32], 64, true).unwrap();
        let mut second = pool.checkout(&model, &[2; 32], 64, true).unwrap();
        assert_eq!(pool.active_capacity, 4096);
        prefill(&model, &[1; 32], &mut first.cache, || false).unwrap();
        prefill(&model, &[2; 32], &mut second.cache, || false).unwrap();
        let expected = model
            .forward(&[3; 32], &mut first.cache, true, None)
            .unwrap()
            .unwrap();
        for (required, capacity) in [(2049, 4096), (4097, 6144)] {
            assert!(
                pool.grow(
                    &model,
                    &mut first.cache,
                    &first.reservation,
                    first.slot,
                    required
                )
                .unwrap()
            );
            assert_eq!(first.cache.capacity(), capacity);
            assert_eq!(pool.active_capacity, capacity + 2048);
            let actual = model
                .forward(&[3; 32], &mut first.cache, true, None)
                .unwrap()
                .unwrap();
            assert_eq!(
                (actual - &expected)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap(),
                0.0
            );
        }
        assert!(
            !pool
                .grow(
                    &model,
                    &mut first.cache,
                    &first.reservation,
                    first.slot,
                    6145
                )
                .unwrap()
        );
        pool.checkin(second);
        assert!(
            pool.grow(
                &model,
                &mut first.cache,
                &first.reservation,
                first.slot,
                6145
            )
            .unwrap()
        );
        assert_eq!(pool.capacity(), 8192);
        assert_eq!(
            pool.snapshot()
                .iter()
                .filter(|s| s.capacity_tokens != 0)
                .count(),
            1
        );
        pool.checkin(first);
        let reused = pool.checkout(&model, &[1; 32], 64, true).unwrap();
        assert_eq!(reused.cached, 32);
        assert_eq!(
            reused.cache.capacity(),
            2048,
            "short reuse must release unused capacity"
        );
        pool.discard(reused);
        let mut bypass = pool.checkout(&model, &[], 32, false).unwrap();
        assert!(
            pool.grow(
                &model,
                &mut bypass.cache,
                &bypass.reservation,
                bypass.slot,
                2049
            )
            .unwrap()
        );
        pool.discard(bypass);
        assert_eq!(pool.active_capacity, 0);
        assert_eq!(pool.capacity(), 0);
    }

    #[test]
    fn pooled_executor_grows_across_multiple_prefill_and_decode_chunks() {
        use crate::decode::Executor;
        let mut model = fixture_model();
        model.config.max_position_embeddings = 8192;
        model.set_prefill_chunk_tokens(128).unwrap();
        let mut pool = CachePool::new(&model, 8192, CacheOptions::default()).unwrap();
        let mut checkout = pool.checkout(&model, &[], 32, true).unwrap();
        let prompt = vec![1; 4096];
        let actual = {
            let executor = pool.executor(&model, &checkout);
            prefill(&executor, &prompt, &mut checkout.cache, || false).unwrap();
            executor
                .forward_tokens(&[2; 32], &mut checkout.cache, true)
                .unwrap()
                .unwrap()
        };
        assert_eq!(checkout.cache.capacity(), 6144);
        let mut full = Cache::new(&model.config, 6144).unwrap();
        prefill(&model, &prompt, &mut full, || false).unwrap();
        let expected = model
            .forward(&[2; 32], &mut full, true, None)
            .unwrap()
            .unwrap();
        assert_eq!(
            (actual - expected)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap(),
            0.0
        );
        pool.discard(checkout);
        assert_eq!(pool.active_capacity, 0);
    }
    #[test]
    fn active_slots_are_exclusive_and_failed_requests_release_the_budget() {
        let model = fixture_model();
        let mut pool = CachePool::new(
            &model,
            256,
            CacheOptions {
                cache_slots: 2,
                cache_max_mib: Some(1),
                ..CacheOptions::default()
            },
        )
        .unwrap();
        let mut first = pool.checkout(&model, &[1; 64], 96, true).unwrap();
        prefill(&model, &[1; 64], &mut first.cache, || false).unwrap();
        let second = pool.checkout(&model, &[1; 64], 96, true).unwrap();
        assert!(
            pool.snapshot()
                .iter()
                .all(|s| s.busy && s.capacity_tokens == 256)
        );
        assert_ne!(first.slot, second.slot);
        assert_eq!(
            second.cached, 0,
            "an active prefix must not be reused concurrently"
        );
        assert!(!pool.can_admit(96, true), "both cache slots are busy");
        pool.discard(second);
        assert!(pool.can_admit(96, true));
        pool.checkin(first);
        let held = pool.checkout(&model, &[2; 64], 96, true).unwrap();
        let mut branch = vec![1; 64];
        branch[32..].fill(3);
        let fork = pool.checkout(&model, &branch, 96, true).unwrap();
        assert_eq!(fork.cached, 32);
        assert_eq!(fork.copied, 0, "reuse source when every other slot is busy");
        assert_ne!(fork.slot, held.slot);
        pool.discard(fork);
        pool.discard(held);
        let mut first = pool.checkout(&model, &[1; 64], 96, true).unwrap();
        prefill(&model, &[1; 64], &mut first.cache, || false).unwrap();
        pool.checkin(first);
        let before = serde_json::to_value(pool.snapshot()).unwrap();
        let empty = Checkout::empty(&model, 96).unwrap();
        pool.checkin(empty);
        assert_eq!(before, serde_json::to_value(pool.snapshot()).unwrap());
        let hit = pool.checkout(&model, &[1; 64], 96, true).unwrap();
        assert_eq!(hit.cached, 64);
        pool.discard(hit);
        assert_eq!(pool.active_capacity, 0);
        let mut limited = CachePool::new(&model, 256, CacheOptions::default()).unwrap();
        let active = limited.checkout(&model, &[1; 64], 96, false).unwrap();
        assert!(
            !limited.can_admit(192, false),
            "bypass requests still reserve K/V memory"
        );
        limited.discard(active);
        assert!(limited.can_admit(192, false));
    }
    fn fill(pool: &mut CachePool, model: &Model, ids: &[u32]) -> (usize, usize, usize) {
        let mut out = pool.checkout(model, ids, ids.len() + 32, true).unwrap();
        let info = (out.slot.unwrap(), out.cached, out.copied);
        assert_eq!(out.cache.tokens(), &ids[..out.cached]);
        prefill(model, ids, &mut out.cache, || false).unwrap();
        pool.checkin(out);
        info
    }
    #[test]
    fn branches_have_independent_storage_and_lru_preserves_recent_conversations() {
        let model = fixture_model();
        let mut pool = CachePool::new(
            &model,
            256,
            CacheOptions {
                cache_slots: 3,
                cache_max_mib: Some(1),
                ..CacheOptions::default()
            },
        )
        .unwrap();
        let a = vec![1; 128];
        let mut b = a.clone();
        b[40..].fill(2);
        assert_eq!(fill(&mut pool, &model, &a), (0, 0, 0));
        assert_eq!(fill(&mut pool, &model, &b), (1, 32, 32));
        let c = vec![3; 128];
        assert_eq!(fill(&mut pool, &model, &c), (2, 0, 0));
        assert_eq!(fill(&mut pool, &model, &a), (0, 128, 0));
        let d = vec![4; 128];
        assert_eq!(fill(&mut pool, &model, &d), (1, 0, 0));
        // Rewriting branch B must not alter source A's K/V storage.
        let mut warm = pool.checkout(&model, &a, 160, true).unwrap();
        let query = vec![5; 32];
        let actual = model
            .forward(&query, &mut warm.cache, true, None)
            .unwrap()
            .unwrap();
        let mut cold = Cache::new(&model.config, 160).unwrap();
        prefill(&model, &a, &mut cold, || false).unwrap();
        let expected = model
            .forward(&query, &mut cold, true, None)
            .unwrap()
            .unwrap();
        let error = (actual - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(error < 1e-5, "fork corrupted source K/V: {error}");
        pool.checkin(warm);
        assert_eq!(fill(&mut pool, &model, &c), (2, 128, 0));
    }
    #[test]
    fn truncation_growth_and_budget_eviction_retain_only_complete_blocks() {
        let model = fixture_model();
        let mut cache = Cache::new(&model.config, 64).unwrap();
        prefill(&model, &vec![1; 64], &mut cache, || false).unwrap();
        assert!(cache.truncate(33).is_err());
        cache.truncate(32).unwrap();
        cache.reserve(&model.config, 160).unwrap();
        assert_eq!(cache.tokens(), &[1; 32]);
        let mut prompt = vec![1; 128];
        prompt[40..].fill(2);
        prefill(&model, &prompt, &mut cache, || false).unwrap();
        let query = vec![3; 32];
        let actual = model
            .forward(&query, &mut cache, true, None)
            .unwrap()
            .unwrap();
        let mut cold = Cache::new(&model.config, 160).unwrap();
        prefill(&model, &prompt, &mut cold, || false).unwrap();
        let expected = model
            .forward(&query, &mut cold, true, None)
            .unwrap()
            .unwrap();
        let error = (actual - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(error < 1e-5, "growth/truncation corrupted K/V: {error}");
        // Default budget fits only one of these fixture-sized reservations.
        let mut pool = CachePool::new(&model, 256, CacheOptions::default()).unwrap();
        assert_eq!(fill(&mut pool, &model, &vec![1; 128]), (0, 0, 0));
        assert_eq!(fill(&mut pool, &model, &prompt), (0, 32, 0));
        fill(&mut pool, &model, &vec![5; 128]);
        assert_eq!(
            pool.snapshot()
                .iter()
                .filter(|s| s.capacity_tokens > 0)
                .count(),
            1
        );
        assert!(pool.capacity() <= pool.budget_tokens);
    }
    #[test]
    fn uncapped_generation_respects_a_smaller_cache_context() {
        let model = fixture_model();
        let mut pool = CachePool::new(&model, 128, CacheOptions::default()).unwrap();
        let options = Options {
            steps: 1,
            max_post_steps: 0,
            ..Options::default()
        };
        let special = SpecialTokens {
            mask: 258,
            delete: 256,
            split: 257,
            eos: 255,
        };
        let result = generate_cached_observed(
            &model,
            &[1; 32],
            &options,
            special,
            (&mut pool, true),
            || false,
            |_| Ok(true),
        )
        .unwrap();
        assert!(result.token_ids.len() <= 96);
        assert!(pool.snapshot().iter().any(|s| s.capacity_tokens == 128));
    }

    #[test]
    fn warm_generation_matches_cold_and_counts_only_new_work() {
        let model = fixture_model();
        let mut pool = CachePool::new(
            &model,
            256,
            CacheOptions {
                cache_max_mib: Some(1),
                ..CacheOptions::default()
            },
        )
        .unwrap();
        let special = SpecialTokens {
            mask: 258,
            delete: 256,
            split: 257,
            eos: 255,
        };
        let options = Options {
            max_tokens: Some(64),
            steps: 1,
            max_post_steps: 0,
            ..Options::default()
        };
        let mut prompt = vec![1; 99];
        for case in 0..4 {
            if case == 2 {
                prompt[80] = 2;
            }
            if case == 3 {
                prompt.truncate(67);
            }
            let cold = generate(&model, &prompt, &options, special, || false).unwrap();
            let mut progress = Vec::new();
            let warm = generate_cached_observed(
                &model,
                &prompt,
                &options,
                special,
                (&mut pool, true),
                || false,
                |event| {
                    if let Progress::Prefill {
                        processed,
                        total,
                        cached,
                        ..
                    } = event
                    {
                        progress.push((processed, total, cached));
                    }
                    Ok(true)
                },
            )
            .unwrap();
            let cached = [0, 96, 64, 64][case];
            assert_eq!(warm.stats.cached_tokens, cached);
            assert_eq!(warm.stats.prefill_tokens + cached, prompt.len() / 32 * 32);
            assert_eq!(warm.token_ids, cold.token_ids);
            assert_eq!(warm.stats.denoise_forwards, cold.stats.denoise_forwards);
            assert_eq!(
                warm.stats.processed_tokens + cached,
                cold.stats.processed_tokens
            );
            assert_eq!(progress[0].0, cached);
            assert_eq!(progress.last().unwrap().0, prompt.len() / 32 * 32);
        }
        // A failed checkout is never published as a reusable prefix.
        let result = generate_cached_observed(
            &model,
            &prompt,
            &options,
            special,
            (&mut pool, true),
            || true,
            |_| Ok(true),
        );
        assert!(result.is_err());
        assert!(
            pool.snapshot()
                .iter()
                .all(|s| s.cached_tokens == 0 || s.cached_tokens >= 96)
        );
        let other_model = fixture_model();
        assert!(pool.checkout(&other_model, &prompt, 128, true).is_err());
    }
}

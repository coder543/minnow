use crate::config::Config;
use anyhow::{Context, Result, ensure};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
#[cfg(feature = "cuda")]
use std::collections::BTreeSet;
use std::{collections::HashMap, path::Path, sync::Arc};

struct Norm {
    weight: Tensor,
    eps: f64,
    fused: bool,
}
impl Norm {
    fn load(size: usize, eps: f64, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            eps,
            fused: true,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.fused && x.device().is_cuda() && x.dtype() == DType::BF16 {
            return Ok(crate::cuda::rms_norm(x, &self.weight, self.eps)?);
        }
        // Preserve the reference's cast BEFORE the learned scale multiplication.
        let f = x.to_dtype(DType::F32)?;
        let inv = (f.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?
            .sqrt()?
            .recip()?;
        Ok(f.broadcast_mul(&inv)?
            .to_dtype(x.dtype())?
            .broadcast_mul(&self.weight)?)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}
fn silu_mul(gate: &Tensor, up: &Tensor, _fused: bool) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if _fused && gate.device().is_cuda() && gate.dtype() == DType::BF16 {
        return Ok(crate::cuda::silu_mul(gate, up)?);
    }
    let activated = candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?.to_dtype(gate.dtype())?;
    Ok((activated * up)?)
}
impl Mlp {
    fn load(h: usize, i: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            gate: candle_nn::linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up: candle_nn::linear_no_bias(h, i, vb.pp("up_proj"))?,
            down: candle_nn::linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }
    fn forward(&self, x: &Tensor, fused: bool) -> Result<Tensor> {
        let gate = self.gate.forward(x)?;
        Ok(self
            .down
            .forward(&silu_mul(&gate, &self.up.forward(x)?, fused)?)?)
    }
}

struct Moe {
    gate: Linear,
    bias: Vec<f32>,
    #[cfg(feature = "cuda")]
    device_bias: Tensor,
    experts: Vec<Mlp>,
    shared: Option<Mlp>,
    packed: Vec<ExpertProjection>,
}

enum ExpertProjection {
    Float(Tensor),
    Quantized(crate::quant::Weights),
}
impl ExpertProjection {
    fn float(&self) -> &Tensor {
        match self {
            Self::Float(t) => t,
            Self::Quantized(_) => unreachable!("quantized projection in floating execution"),
        }
    }
    fn grouped(&self, x: &Tensor, segments: &[(usize, usize)]) -> Result<Tensor> {
        match self {
            Self::Quantized(w) => w.grouped(x, segments),
            Self::Float(w) => {
                #[cfg(feature = "cuda")]
                if x.device().is_cuda() && x.dtype() == DType::BF16 {
                    return Ok(crate::cuda::grouped_expert_gemm(x, w, segments)?);
                }
                let mut row = 0;
                let mut outputs = Vec::new();
                for &(e, n) in segments {
                    outputs.push(x.narrow(0, row, n)?.matmul(&w.get(e)?.t()?)?);
                    row += n;
                }
                Ok(Tensor::cat(&outputs, 0)?)
            }
        }
    }
}

struct MoeExecution {
    batched_experts: bool,
    compact_decode_experts: bool,
    fused_activation: bool,
    fused_mix: bool,
    device_routing: bool,
    host_routing: bool,
}

fn mix_experts(
    out: &Tensor,
    rows: Vec<u32>,
    weights: Vec<f32>,
    top_k: usize,
    _fused: bool,
) -> Result<Tensor> {
    let dtype = out.dtype();
    #[cfg(feature = "cuda")]
    if _fused && out.device().is_cuda() && out.dtype() == DType::BF16 && top_k <= 32 {
        return Ok(crate::cuda::mix_experts(out, &rows, &weights, top_k)?);
    }
    let tokens = rows.len() / top_k;
    let out = out
        .index_select(&Tensor::from_vec(rows, tokens * top_k, out.device())?, 0)?
        .reshape((tokens, top_k, out.dim(1)?))?
        .to_dtype(DType::F32)?;
    let weights = Tensor::from_vec(weights, (tokens, top_k, 1), out.device())?;
    Ok(out.broadcast_mul(&weights)?.sum(1)?.to_dtype(dtype)?)
}

/// Selection uses biased scores; mixing uses unbiased scores. Blocks never mix.
pub fn route(scores: &[Vec<f32>], bias: &[f32], c: &Config) -> Result<(Vec<u32>, Vec<f32>)> {
    ensure!(
        scores.len().is_multiple_of(c.block_size)
            && scores.iter().all(|s| s.len() == c.num_experts)
            && bias.len() == c.num_experts,
        "invalid routing shape"
    );
    let mut ids = Vec::with_capacity(scores.len() * c.num_experts_per_tok);
    let mut weights = Vec::with_capacity(ids.capacity());
    for block in scores.chunks_exact(c.block_size) {
        let maxima: Vec<f32> = (0..c.num_experts)
            .map(|e| {
                block
                    .iter()
                    .map(|s| s[e] + bias[e])
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();
        let mut allowed: Vec<usize> = (0..c.num_experts).collect();
        allowed.sort_unstable_by(|&a, &b| maxima[b].total_cmp(&maxima[a]).then(a.cmp(&b)));
        allowed.truncate(c.expert_capacity);
        for scores in block {
            allowed.sort_unstable_by(|&a, &b| {
                (scores[b] + bias[b])
                    .total_cmp(&(scores[a] + bias[a]))
                    .then(a.cmp(&b))
            });
            let selected = &allowed[..c.num_experts_per_tok];
            let sum = if c.num_experts_per_tok > 1 {
                selected.iter().map(|&e| scores[e]).sum::<f32>() + 1e-20
            } else {
                1.0
            };
            for &e in selected {
                ids.push(e as u32);
                weights.push(scores[e] / sum * c.routed_scaling_factor as f32);
            }
        }
    }
    Ok((ids, weights))
}

impl Moe {
    fn load(c: &Config, vb: VarBuilder<'_>, loader: &crate::weights::WeightLoader) -> Result<Self> {
        // The streaming backend writes experts into their final packed storage.
        let mut packed = Vec::new();
        for (name, shape) in [
            ("gate_proj", (c.moe_intermediate_size, c.hidden_size)),
            ("up_proj", (c.moe_intermediate_size, c.hidden_size)),
            ("down_proj", (c.hidden_size, c.moe_intermediate_size)),
        ] {
            let shape = (c.num_experts, shape.0, shape.1);
            let name = format!("experts.{name}.weight");
            packed.push(
                match loader.quantized_experts(
                    &format!("{}.{}", vb.prefix(), name),
                    shape,
                    vb.device(),
                )? {
                    Some(weight) => ExpertProjection::Quantized(weight),
                    None => ExpertProjection::Float(vb.get(shape, &name)?),
                },
            );
        }
        let count = if packed
            .iter()
            .all(|p| matches!(p, ExpertProjection::Float(_)))
        {
            c.num_experts
        } else {
            0
        };
        let experts = (0..count)
            .map(|e| {
                Ok(Mlp {
                    gate: Linear::new(packed[0].float().get(e)?, None),
                    up: Linear::new(packed[1].float().get(e)?, None),
                    down: Linear::new(packed[2].float().get(e)?, None),
                })
            })
            .collect::<Result<_>>()?;
        let bias = vb
            .clone()
            .set_device(Device::Cpu)
            .to_dtype(DType::F32)
            .get(c.num_experts, "gate.expert_bias")?
            .to_vec1::<f32>()?;
        Ok(Self {
            gate: Linear::new(
                vb.to_dtype(DType::F32)
                    .get((c.num_experts, c.hidden_size), "gate.weight")?,
                None,
            ),
            #[cfg(feature = "cuda")]
            device_bias: Tensor::from_slice(&bias, c.num_experts, vb.device())?,
            bias,
            experts,
            packed,
            shared: if c.num_shared_experts > 0 {
                Some(Mlp::load(
                    c.hidden_size,
                    c.moe_intermediate_size * c.num_shared_experts,
                    vb.pp("shared_experts"),
                )?)
            } else {
                None
            },
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        c: &Config,
        trace: &mut Option<&mut Trace>,
        name: &str,
        execution: &MoeExecution,
    ) -> Result<Tensor> {
        let fused = execution.fused_activation;
        let quantized = self.experts.is_empty();
        record(trace, format!("{name}.router_input"), x);
        let logits = self.gate.forward(&x.to_dtype(DType::F32)?)?;
        record(trace, format!("{name}.router_logits"), &logits);
        #[cfg(feature = "cuda")]
        if quantized
            && !execution.host_routing
            && execution.batched_experts
            && execution.fused_mix
            && x.device().is_cuda()
            && x.dtype() == DType::BF16
            && x.dim(0)? == 32
            && c.block_size == 32
            && c.num_experts == 256
            && c.num_experts_per_tok == 8
            && c.expert_capacity == 48
            && let [
                ExpertProjection::Quantized(a),
                ExpertProjection::Quantized(b),
                ExpertProjection::Quantized(down),
            ] = self.packed.as_slice()
            && [a, b, down]
                .iter()
                .all(|w| w.encoding == crate::container::Encoding::Nvfp4)
        {
            let plan = crate::cuda::route_mini_compact(
                &logits,
                &self.device_bias,
                c.routed_scaling_factor as f32,
            )?;
            if trace.is_some() {
                record(trace, format!("{name}.router_ids"), &plan.expert_ids()?);
            }
            let hidden = if fused {
                crate::cuda::nvfp4::routed_silu(x, a, b, &plan)?
            } else {
                let (gate, up) = crate::cuda::nvfp4::routed_pair(x, a, b, &plan)?;
                silu_mul(&gate, &up, fused)?
            };
            let experts = crate::cuda::nvfp4::routed(&hidden, down, &plan)?;
            let mut out = crate::cuda::mix_compact_experts(&experts, &plan)?;
            if let Some(shared) = &self.shared {
                out = (out + shared.forward(x, fused)?)?;
            }
            return Ok(out);
        }
        #[cfg(feature = "cuda")]
        if execution.device_routing
            && !execution.host_routing
            && !quantized
            && execution.batched_experts
            && execution.fused_mix
            && x.device().is_cuda()
            && x.dtype() == DType::BF16
            && x.dim(0)? == 32
            && c.block_size == 32
            && c.num_experts == 256
            && c.num_experts_per_tok == 8
            && c.expert_capacity == 48
        {
            let plan = crate::cuda::route_mini(
                &logits,
                &self.device_bias,
                c.routed_scaling_factor as f32,
            )?;
            if trace.is_some() {
                record(trace, format!("{name}.router_ids"), &plan.expert_ids()?);
            }
            let gate = crate::cuda::routed_expert_gemm(x, self.packed[0].float(), &plan, false)?;
            let up = crate::cuda::routed_expert_gemm(x, self.packed[1].float(), &plan, false)?;
            let hidden = silu_mul(&gate, &up, fused)?;
            let experts =
                crate::cuda::routed_expert_gemm(&hidden, self.packed[2].float(), &plan, true)?;
            let mut out = crate::cuda::mix_routed_experts(&experts, &plan)?;
            if let Some(shared) = &self.shared {
                out = (out + shared.forward(x, fused)?)?;
            }
            return Ok(out);
        }
        let scores = candle_nn::ops::sigmoid(&logits)?.to_vec2::<f32>()?;
        let (ids, weights) = route(&scores, &self.bias, c)?;
        if trace.is_some() {
            record(
                trace,
                format!("{name}.router_ids"),
                &Tensor::from_vec(ids.clone(), (x.dim(0)?, c.num_experts_per_tok), x.device())?,
            );
        }
        // Batch each distinct expert once, with all block rows sharing its weights.
        #[cfg(feature = "cuda")]
        if execution.batched_experts
            && !quantized
            && !execution.compact_decode_experts
            && x.device().is_cuda()
            && x.dtype() == DType::BF16
            && x.dim(0)? == c.block_size
        {
            let selected: Vec<usize> = ids
                .iter()
                .map(|&e| e as usize)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let gate = crate::cuda::expert_gemm(x, self.packed[0].float(), &selected, false)?;
            let up = crate::cuda::expert_gemm(x, self.packed[1].float(), &selected, false)?;
            let hidden = silu_mul(&gate, &up, fused)?;
            let out = crate::cuda::expert_gemm(&hidden, self.packed[2].float(), &selected, true)?
                .flatten_to(1)?;
            let mut mapping = vec![0; c.num_experts];
            for (i, &e) in selected.iter().enumerate() {
                mapping[e] = i;
            }
            let rows: Vec<u32> = ids
                .iter()
                .enumerate()
                .map(|(slot, &e)| {
                    (mapping[e as usize] * c.block_size + slot / c.num_experts_per_tok) as u32
                })
                .collect();
            let mut out = mix_experts(
                &out,
                rows,
                weights,
                c.num_experts_per_tok,
                execution.fused_mix,
            )?;
            if let Some(shared) = &self.shared {
                out = (out + shared.forward(x, fused)?)?;
            }
            return Ok(out);
        }
        let mut assignments = vec![Vec::new(); c.num_experts];
        for (slot, &e) in ids.iter().enumerate() {
            assignments[e as usize].push(slot);
        }
        let mut inverse = vec![0u32; ids.len()];
        let mut token_ids = Vec::with_capacity(ids.len());
        let mut segments = Vec::new();
        for (e, slots) in assignments
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.is_empty())
        {
            segments.push((e, slots.len()));
            for &slot in slots {
                inverse[slot] = token_ids.len() as u32;
                token_ids.push((slot / c.num_experts_per_tok) as u32);
            }
        }
        if quantized {
            let evaluate_pair = || -> Result<(Tensor, Tensor)> {
                #[cfg(feature = "cuda")]
                if x.device().is_cuda()
                    && let (ExpertProjection::Quantized(a), ExpertProjection::Quantized(b)) =
                        (&self.packed[0], &self.packed[1])
                    && a.encoding == crate::container::Encoding::Nvfp4
                    && b.encoding == a.encoding
                {
                    return Ok(crate::cuda::nvfp4::grouped_pair_indexed(
                        x, a, b, &segments, &token_ids,
                    )?);
                }
                let selected =
                    x.index_select(&Tensor::from_slice(&token_ids, ids.len(), x.device())?, 0)?;
                Ok((
                    self.packed[0].grouped(&selected, &segments)?,
                    self.packed[1].grouped(&selected, &segments)?,
                ))
            };
            let evaluate_hidden = || -> Result<Tensor> {
                #[cfg(feature = "cuda")]
                if fused
                    && x.device().is_cuda()
                    && let (ExpertProjection::Quantized(a), ExpertProjection::Quantized(b)) =
                        (&self.packed[0], &self.packed[1])
                    && a.encoding == crate::container::Encoding::Nvfp4
                    && b.encoding == a.encoding
                {
                    return Ok(crate::cuda::nvfp4::grouped_silu_indexed(
                        x, a, b, &segments, &token_ids,
                    )?);
                }
                let (gate, up) = evaluate_pair()?;
                silu_mul(&gate, &up, fused)
            };
            let hidden = evaluate_hidden()?;
            let output = self.packed[2].grouped(&hidden, &segments)?;
            let mut output = mix_experts(
                &output,
                inverse,
                weights,
                c.num_experts_per_tok,
                execution.fused_mix,
            )?;
            if let Some(shared) = &self.shared {
                output = (output + shared.forward(x, fused)?)?;
            }
            return Ok(output);
        }
        let selected = x.index_select(&Tensor::from_vec(token_ids, ids.len(), x.device())?, 0)?;
        let evaluate_serial = || -> Result<Tensor> {
            let mut outputs = Vec::new();
            let mut row = 0;
            for &(e, n) in &segments {
                outputs.push(self.experts[e].forward(&selected.narrow(0, row, n)?, fused)?);
                row += n;
            }
            Ok(Tensor::cat(&outputs, 0)?)
        };
        #[cfg(feature = "cuda")]
        let outputs =
            if execution.batched_experts && x.device().is_cuda() && x.dtype() == DType::BF16 {
                let gate =
                    crate::cuda::grouped_expert_gemm(&selected, self.packed[0].float(), &segments)?;
                let up =
                    crate::cuda::grouped_expert_gemm(&selected, self.packed[1].float(), &segments)?;
                let hidden = silu_mul(&gate, &up, fused)?;
                crate::cuda::grouped_expert_gemm(&hidden, self.packed[2].float(), &segments)?
            } else {
                evaluate_serial()?
            };
        #[cfg(not(feature = "cuda"))]
        let outputs = evaluate_serial()?;
        let mut y = mix_experts(
            &outputs,
            inverse,
            weights,
            c.num_experts_per_tok,
            execution.fused_mix,
        )?;
        if let Some(shared) = &self.shared {
            y = (y + shared.forward(x, fused)?)?;
        }
        Ok(y)
    }
}

#[cfg(test)]
#[path = "model/cache_tests.rs"]
mod cache_tests;

enum FeedForward {
    Dense(Mlp),
    Sparse(Moe),
}
struct Layer {
    input_norm: Norm,
    post_norm: Norm,
    qkv: Linear,
    out: Linear,
    qnorm: Option<Norm>,
    knorm: Option<Norm>,
    mlp: FeedForward,
}

/// Scratch slots after `len` may change on every forward. Only `commit` advances it.
pub struct Cache {
    layers: Vec<Option<(Tensor, Tensor)>>,
    len: usize,
    capacity: usize,
    block_size: usize,
    committed_tokens: Vec<u32>,
    staged_tokens: Option<Vec<u32>>,
    rope: Option<(usize, usize, Tensor, Tensor)>,
    pub forwards: usize,
    pub processed_tokens: usize,
}
impl Cache {
    fn copy_heads(source: &Tensor, target: &Tensor, len: usize) -> Result<()> {
        // Each head's prefix is contiguous even when the full prefix view has
        // a capacity-sized stride. Copy those views directly, without packing.
        for head in 0..source.dim(0)? {
            target
                .get(head)?
                .slice_set(&source.get(head)?.narrow(0, 0, len)?, 0, 0)?;
        }
        Ok(())
    }
    pub fn new(c: &Config, capacity: usize) -> Result<Self> {
        ensure!(
            capacity > 0
                && capacity <= c.max_position_embeddings
                && capacity.is_multiple_of(c.block_size),
            "cache capacity must be a positive whole number of blocks within model context"
        );
        Ok(Self {
            layers: vec![None; c.num_hidden_layers],
            len: 0,
            capacity,
            block_size: c.block_size,
            committed_tokens: Vec::new(),
            staged_tokens: None,
            rope: None,
            forwards: 0,
            processed_tokens: 0,
        })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn tokens(&self) -> &[u32] {
        &self.committed_tokens
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// A block is bidirectional, so partial-block K/V cannot be retained.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        ensure!(
            len <= self.len && len.is_multiple_of(self.block_size),
            "cache truncation requires a committed block boundary"
        );
        self.len = len;
        self.committed_tokens.truncate(len);
        self.staged_tokens = None;
        self.rope = None;
        self.forwards = 0;
        self.processed_tokens = 0;
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn reserve(&mut self, c: &Config, capacity: usize) -> Result<()> {
        self.resize(c, capacity.max(self.capacity))
    }
    pub(crate) fn resize(&mut self, c: &Config, capacity: usize) -> Result<()> {
        ensure!(
            capacity >= self.len
                && capacity <= c.max_position_embeddings
                && capacity.is_multiple_of(c.block_size),
            "invalid cache capacity"
        );
        if capacity == self.capacity {
            return Ok(());
        }
        // Replace one layer at a time. Peak growth is at most one layer above
        // the new cache size; never duplicate the full K/V pool or the weights.
        for layer in &mut self.layers {
            if let Some((k, v)) = layer.take() {
                let shape = (c.num_key_value_heads, capacity, c.head_dim);
                let next_k = Tensor::zeros(shape, k.dtype(), k.device())?;
                let next_v = Tensor::zeros(shape, v.dtype(), v.device())?;
                if self.len > 0 {
                    Self::copy_heads(&k, &next_k, self.len)?;
                    Self::copy_heads(&v, &next_v, self.len)?;
                }
                *layer = Some((next_k, next_v));
            }
        }
        self.capacity = capacity;
        Ok(())
    }
    pub(crate) fn copy_prefix(&self, c: &Config, len: usize, capacity: usize) -> Result<Self> {
        ensure!(
            len <= self.len && len <= capacity && len.is_multiple_of(c.block_size),
            "cache copy requires a committed block boundary"
        );
        let mut dest = Self::new(c, capacity)?;
        if len > 0 {
            for (source, target) in self.layers.iter().zip(&mut dest.layers) {
                let (k, v) = source.as_ref().context("missing committed K/V layer")?;
                let shape = (c.num_key_value_heads, capacity, c.head_dim);
                let next_k = Tensor::zeros(shape, k.dtype(), k.device())?;
                let next_v = Tensor::zeros(shape, v.dtype(), v.device())?;
                Self::copy_heads(k, &next_k, len)?;
                Self::copy_heads(v, &next_v, len)?;
                *target = Some((next_k, next_v));
            }
        }
        dest.len = len;
        dest.committed_tokens
            .extend_from_slice(&self.committed_tokens[..len]);
        Ok(dest)
    }
    pub(crate) fn staged_matches(&self, tokens: &[u32]) -> bool {
        self.staged_tokens.as_deref() == Some(tokens)
    }
    pub fn commit(&mut self, tokens: &[u32]) -> Result<()> {
        ensure!(
            self.staged_matches(tokens),
            "cache commit requires a successful forward of the exact final tokens"
        );
        self.len += tokens.len();
        self.committed_tokens.extend_from_slice(tokens);
        self.staged_tokens = None;
        Ok(())
    }
}

pub type Trace = HashMap<String, Tensor>;

#[path = "model/batching.rs"]
mod batching;
pub use batching::Forward;

// Bound each attention tile independently of the transformer prefill batch.
pub const MAX_ATTENTION_ELEMENTS: usize = 128 * 1024 * 1024;
// Also bound token/expert activations independently of the attention tile.
pub const MAX_FORWARD_TOKENS: usize = 8192;
fn record(trace: &mut Option<&mut Trace>, name: String, tensor: &Tensor) {
    if let Some(t) = trace {
        t.insert(name, tensor.clone());
    }
}

pub struct Model {
    pub(crate) cache_identity: Arc<()>,
    pub config: Config,
    embeddings: Tensor,
    layers: Vec<Layer>,
    norm: Norm,
    head: Linear,
    device: Device,
    dtype: DType,
    moe_execution: MoeExecution,
    fused_attention: bool,
    flash_attention: bool,
    fused_qkv: bool,
    prefill_chunk_tokens: usize,
    attention_chunk_tokens: usize,
    // Drop after the weight fields.
    #[cfg(feature = "cuda")]
    workspace_cache: Option<crate::cuda::workspace::WorkspaceCache>,
}
impl Drop for Model {
    fn drop(&mut self) {
        // cudarc's asynchronous CudaSlice destructor frees on the current
        // context. A model can be moved to an idle worker and dropped before
        // that thread has ever submitted CUDA work; bind before fields drop.
        #[cfg(feature = "cuda")]
        if let Device::Cuda(device) = &self.device
            && let Err(error) = device.cuda_stream().context().bind_to_thread()
        {
            tracing::warn!(%error,"binding CUDA context before model destruction");
        }
    }
}
impl Model {
    pub fn load(path: &Path, dtype: DType, device: &Device) -> Result<Self> {
        Self::load_with_memory_reserve(
            path,
            dtype,
            device,
            crate::weights::DEFAULT_MEMORY_RESERVE_MIB,
        )
    }
    pub fn load_with_memory_reserve(
        path: &Path,
        dtype: DType,
        device: &Device,
        reserve_mib: u64,
    ) -> Result<Self> {
        let c = Config::load(path)?;
        let mut loader = crate::weights::WeightLoader::open(path)?;
        loader.set_memory_reserve_mib(reserve_mib)?;
        #[cfg(feature = "cuda")]
        if device.is_cuda()
            && loader
                .inventory()
                .iter()
                .any(|(_, _, e)| *e == crate::container::Encoding::Nvfp4)
        {
            crate::cuda::nvfp4::check_device(device)?;
        }
        ensure!(
            !device.is_cuda()
                || dtype == DType::BF16
                || !loader
                    .inventory()
                    .iter()
                    .any(|(_, _, encoding)| encoding.quantized()),
            "quantized CUDA execution requires BF16 activations; use --dtype bf16"
        );
        loader.check_memory(dtype)?;
        loader.start_allocations(dtype, device)?;
        let loader = Arc::new(loader);
        let vb = VarBuilder::from_backend(
            Box::new(crate::weights::SharedLoader(loader.clone())),
            dtype,
            device.clone(),
        );
        let embeddings = vb.get(
            (c.vocab_size, c.hidden_size),
            "model.word_embeddings.weight",
        )?;
        let mut layers = Vec::new();
        for i in 0..c.num_hidden_layers {
            tracing::info!(layer = i, total = c.num_hidden_layers, "loading layer");
            let v = vb.pp(format!("model.layers.{i}"));
            layers.push(Layer {
                input_norm: Norm::load(c.hidden_size, c.rms_norm_eps, v.pp("input_layernorm"))?,
                post_norm: Norm::load(
                    c.hidden_size,
                    c.rms_norm_eps,
                    v.pp("post_attention_layernorm"),
                )?,
                qkv: candle_nn::linear_no_bias(
                    c.hidden_size,
                    (c.num_attention_heads + 2 * c.num_key_value_heads) * c.head_dim,
                    v.pp("attention.query_key_value"),
                )?,
                out: candle_nn::linear_no_bias(
                    c.num_attention_heads * c.head_dim,
                    c.hidden_size,
                    v.pp("attention.dense"),
                )?,
                qnorm: if c.use_qk_norm {
                    Some(Norm::load(
                        c.head_dim,
                        c.rms_norm_eps,
                        v.pp("attention.query_layernorm"),
                    )?)
                } else {
                    None
                },
                knorm: if c.use_qk_norm {
                    Some(Norm::load(
                        c.head_dim,
                        c.rms_norm_eps,
                        v.pp("attention.key_layernorm"),
                    )?)
                } else {
                    None
                },
                mlp: if i < c.first_k_dense_replace {
                    FeedForward::Dense(Mlp::load(c.hidden_size, c.intermediate_size, v.pp("mlp"))?)
                } else {
                    FeedForward::Sparse(Moe::load(&c, v.pp("mlp"), &loader)?)
                },
            });
        }
        let norm = Norm::load(c.hidden_size, c.rms_norm_eps, vb.pp("model.norm"))?;
        let head = candle_nn::linear_no_bias(c.hidden_size, c.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            cache_identity: Arc::new(()),
            config: c,
            embeddings,
            layers,
            norm,
            head,
            device: device.clone(),
            dtype,
            moe_execution: MoeExecution {
                batched_experts: true,
                compact_decode_experts: false,
                fused_activation: true,
                fused_mix: true,
                device_routing: false,
                host_routing: false,
            },
            fused_attention: true,
            flash_attention: true,
            fused_qkv: true,
            prefill_chunk_tokens: 4096,
            attention_chunk_tokens: 1024,
            #[cfg(feature = "cuda")]
            workspace_cache: crate::cuda::workspace::WorkspaceCache::new(device),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn set_workspace_cache_mib(&self, mib: usize) -> Result<()> {
        let _bytes = mib
            .checked_mul(1024 * 1024)
            .context("workspace cache size overflow")?;
        #[cfg(feature = "cuda")]
        if let Some(cache) = &self.workspace_cache {
            cache.set(_bytes as u64)?;
        }
        Ok(())
    }
    fn refresh_workspace_cache(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Some(cache) = &self.workspace_cache {
            cache.refresh()?;
        }
        Ok(())
    }
    pub fn kv_bytes_per_token(&self) -> usize {
        2 * self.config.num_hidden_layers
            * self.config.num_key_value_heads
            * self.config.head_dim
            * self.dtype.size_in_bytes()
    }
    pub fn set_batched_experts(&mut self, enabled: bool) {
        self.moe_execution.batched_experts = enabled;
    }
    pub fn set_compact_decode_experts(&mut self, enabled: bool) {
        self.moe_execution.compact_decode_experts = enabled;
    }
    pub fn set_fused_activation(&mut self, enabled: bool) {
        self.moe_execution.fused_activation = enabled;
    }
    pub fn set_fused_expert_mix(&mut self, enabled: bool) {
        self.moe_execution.fused_mix = enabled;
    }
    pub fn set_device_routing(&mut self, enabled: bool) {
        self.moe_execution.device_routing = enabled;
    }
    pub fn set_host_routing(&mut self, enabled: bool) {
        self.moe_execution.host_routing = enabled;
    }
    pub fn set_fused_norm(&mut self, enabled: bool) {
        self.norm.fused = enabled;
        for layer in &mut self.layers {
            layer.input_norm.fused = enabled;
            layer.post_norm.fused = enabled;
            if let Some(norm) = &mut layer.qnorm {
                norm.fused = enabled;
            }
            if let Some(norm) = &mut layer.knorm {
                norm.fused = enabled;
            }
        }
    }
    pub fn set_fused_attention(&mut self, enabled: bool) {
        self.fused_attention = enabled;
    }
    pub fn set_flash_attention(&mut self, enabled: bool) {
        self.flash_attention = enabled;
    }
    pub fn set_fused_qkv(&mut self, enabled: bool) {
        self.fused_qkv = enabled;
    }
    pub fn set_prefill_chunk_tokens(&mut self, tokens: usize) -> Result<()> {
        ensure!(
            tokens > 0
                && tokens <= MAX_FORWARD_TOKENS
                && tokens.is_multiple_of(self.config.block_size),
            "prefill chunk size must be a positive whole number of blocks, at most {MAX_FORWARD_TOKENS} tokens"
        );
        self.prefill_chunk_tokens = tokens;
        Ok(())
    }
    pub fn prefill_chunk_tokens(&self) -> usize {
        self.prefill_chunk_tokens
    }

    pub fn set_attention_chunk_tokens(&mut self, tokens: usize) -> Result<()> {
        ensure!(
            tokens > 0 && tokens.is_multiple_of(self.config.block_size),
            "attention chunk size must be a positive whole number of blocks"
        );
        self.attention_chunk_tokens = tokens;
        Ok(())
    }
    pub fn attention_chunk_tokens(&self) -> usize {
        self.attention_chunk_tokens
    }

    fn rope(&self, n: usize, offset: usize) -> Result<(Tensor, Tensor)> {
        let dim = self.config.rotary_dim();
        let mut angles = Vec::with_capacity(n * dim);
        for p in offset..offset + n {
            for i in 0..dim {
                let inv = 1.0f32
                    / (self.config.rope_theta as f32)
                        .powf((2 * (i % (dim / 2))) as f32 / dim as f32);
                angles.push(p as f32 * inv);
            }
        }
        let angles = Tensor::from_vec(angles, (1, n, dim), &self.device)?;
        Ok((
            angles.cos()?.to_dtype(self.dtype)?,
            angles.sin()?.to_dtype(self.dtype)?,
        ))
    }

    fn rotate(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let r = self.config.rotary_dim();
        let rot = x.narrow(2, 0, r)?;
        let half = Tensor::cat(
            &[
                &rot.narrow(2, r / 2, r / 2)?.neg()?,
                &rot.narrow(2, 0, r / 2)?,
            ],
            2,
        )?;
        let y = (rot.broadcast_mul(cos)? + half.broadcast_mul(sin)?)?;
        Ok(if r == self.config.head_dim {
            y
        } else {
            Tensor::cat(&[&y, &x.narrow(2, r, self.config.head_dim - r)?], 2)?
        })
    }

    fn prepare_qkv(
        &self,
        layer: &Layer,
        qkv: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let c = &self.config;
        #[cfg(feature = "cuda")]
        if self.fused_qkv
            && self.device.is_cuda()
            && self.dtype == DType::BF16
            && [16, 32].contains(&c.num_attention_heads)
            && c.num_key_value_heads == 4
            && c.head_dim == 128
            && c.rotary_dim() == 64
            && let (Some(qnorm), Some(knorm)) = (&layer.qnorm, &layer.knorm)
            && qnorm.fused
            && knorm.fused
        {
            let heads = crate::cuda::prepare_qkv(
                qkv,
                &qnorm.weight,
                &knorm.weight,
                cos,
                sin,
                c.rms_norm_eps,
            )?;
            return Ok((
                heads.narrow(0, 0, c.num_attention_heads)?,
                heads.narrow(0, c.num_attention_heads, 4)?,
                heads.narrow(0, c.num_attention_heads + 4, 4)?,
            ));
        }
        let mut q = qkv.narrow(1, 0, c.num_attention_heads)?.transpose(0, 1)?;
        let mut k = qkv
            .narrow(1, c.num_attention_heads, c.num_key_value_heads)?
            .transpose(0, 1)?;
        let v = qkv
            .narrow(
                1,
                c.num_attention_heads + c.num_key_value_heads,
                c.num_key_value_heads,
            )?
            .transpose(0, 1)?
            .contiguous()?;
        if let Some(norm) = &layer.qnorm {
            q = norm.forward(&q)?;
        }
        if let Some(norm) = &layer.knorm {
            k = norm.forward(&k)?;
        }
        Ok((
            self.rotate(&q, cos, sin)?.contiguous()?,
            self.rotate(&k, cos, sin)?.contiguous()?,
            v,
        ))
    }

    /// Tile queries independently from the transformer batch. Every tile sees
    /// only its committed prefix and earlier/current blocks within this batch.
    fn attention(
        &self,
        q: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        let c = &self.config;
        let n = q.dim(1)?;
        #[cfg(feature = "cuda")]
        if self.fused_attention
            && self.device.is_cuda()
            && self.dtype == DType::BF16
            && c.head_dim == 128
            && c.block_size == 32
            && self.flash_attention
        {
            return Ok(crate::cuda::flash::attention(
                &q.contiguous()?,
                keys,
                values,
                offset,
            )?);
        }
        let groups = c.num_attention_heads / c.num_key_value_heads;
        let budget = MAX_ATTENTION_ELEMENTS / c.num_attention_heads / (offset + n);
        let chunk = n
            .min(self.attention_chunk_tokens)
            .min(budget / c.block_size * c.block_size);
        ensure!(
            chunk >= c.block_size,
            "attention workspace cannot fit one block"
        );
        let mut outputs = Vec::new();
        for start in (0..n).step_by(chunk) {
            let len = chunk.min(n - start);
            let query_offset = offset + start;
            let total = query_offset + len;
            // cuBLAS accepts the cache's head stride directly, without copying
            // the prefix. Keys beyond the end of this tile need not be read.
            let keys = keys.narrow(1, 0, total)?;
            let values = values.narrow(1, 0, total)?;
            let q = q.narrow(1, start, len)?.contiguous()?.reshape((
                c.num_key_value_heads,
                groups * len,
                c.head_dim,
            ))?;
            let raw_scores = q.matmul(&keys.t()?)?;
            let eager_softmax = || -> Result<Tensor> {
                // Preserve the FP32 scale and BF16 rounding before softmax.
                let scores = (raw_scores.to_dtype(DType::F32)? * (c.head_dim as f64).powf(-0.5))?
                    .to_dtype(self.dtype)?
                    .reshape((c.num_key_value_heads, groups, len, total))?;
                let scores = if len > c.block_size {
                    let mask: Vec<f32> = (0..len)
                        .flat_map(|q| {
                            (0..total).map(move |k| {
                                if k / c.block_size <= (query_offset + q) / c.block_size {
                                    0.
                                } else {
                                    f32::NEG_INFINITY
                                }
                            })
                        })
                        .collect();
                    let mask = Tensor::from_vec(mask, (1, 1, len, total), &self.device)?
                        .to_dtype(self.dtype)?;
                    scores.broadcast_add(&mask)?
                } else {
                    scores
                };
                Ok(
                    candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
                        .to_dtype(self.dtype)?
                        .reshape((c.num_key_value_heads, groups * len, total))?,
                )
            };
            #[cfg(feature = "cuda")]
            let probs = if self.fused_attention
                && self.device.is_cuda()
                && self.dtype == DType::BF16
                && total <= 131072
            {
                crate::cuda::block_softmax(
                    &raw_scores,
                    len,
                    query_offset,
                    c.block_size,
                    (c.head_dim as f64).powf(-0.5) as f32,
                )?
            } else {
                eager_softmax()?
            };
            #[cfg(not(feature = "cuda"))]
            let probs = eager_softmax()?;
            outputs.push(
                probs
                    .matmul(&values)?
                    .reshape((c.num_attention_heads, len, c.head_dim))?
                    .transpose(0, 1)?
                    .contiguous()?
                    .reshape((len, c.num_attention_heads * c.head_dim))?,
            );
        }
        Ok(Tensor::cat(&outputs, 0)?)
    }

    /// Evaluates aligned blocks at the committed offset, replacing uncommitted K/V.
    /// `logits=false` avoids the large vocabulary projection during prefill/commit.
    pub fn forward(
        &self,
        tokens: &[u32],
        cache: &mut Cache,
        logits: bool,
        mut trace: Option<&mut Trace>,
    ) -> Result<Option<Tensor>> {
        let c = &self.config;
        let n = tokens.len();
        ensure!(
            n > 0 && n.is_multiple_of(c.block_size) && cache.len.is_multiple_of(c.block_size),
            "forward must contain complete aligned blocks"
        );
        ensure!(
            cache.layers.len() == c.num_hidden_layers && cache.len + n <= cache.capacity,
            "cache capacity exceeded or incompatible cache"
        );
        ensure!(
            tokens.iter().all(|&t| (t as usize) < c.vocab_size),
            "token outside vocabulary"
        );
        ensure!(
            n <= MAX_FORWARD_TOKENS,
            "forward workspace limit exceeded; process at most {MAX_FORWARD_TOKENS} tokens per batch"
        );
        cache.staged_tokens = None;
        let offset = cache.len;
        let mut x = self
            .embeddings
            .index_select(&Tensor::from_vec(tokens.to_vec(), n, &self.device)?, 0)?;
        record(&mut trace, "embeddings".into(), &x);
        if !matches!(&cache.rope, Some((position, count, _, _)) if *position==offset && *count==n) {
            let (cos, sin) = self.rope(n, offset)?;
            cache.rope = Some((offset, n, cos, sin));
        }
        let (_, _, cos, sin) = cache.rope.as_ref().unwrap();
        let (cos, sin) = (cos.clone(), sin.clone());
        for (i, layer) in self.layers.iter().enumerate() {
            self.refresh_workspace_cache()?;
            let h = layer.input_norm.forward(&x)?;
            let qkv = layer.qkv.forward(&h)?.reshape((
                n,
                c.num_attention_heads + 2 * c.num_key_value_heads,
                c.head_dim,
            ))?;
            let (q, k, v) = self.prepare_qkv(layer, &qkv, &cos, &sin)?;
            if trace.is_some() {
                record(&mut trace, format!("layers.{i}.query"), &q.transpose(0, 1)?);
                record(&mut trace, format!("layers.{i}.key"), &k.transpose(0, 1)?);
                record(&mut trace, format!("layers.{i}.value"), &v.transpose(0, 1)?);
            }
            if cache.layers[i].is_none() {
                let shape = (c.num_key_value_heads, cache.capacity, c.head_dim);
                cache.layers[i] = Some((
                    Tensor::zeros(shape, self.dtype, &self.device)?,
                    Tensor::zeros(shape, self.dtype, &self.device)?,
                ));
            }
            let (keys, values) = cache.layers[i].as_ref().unwrap();
            keys.slice_set(&k, 1, cache.len)?;
            values.slice_set(&v, 1, cache.len)?;
            let h = self.attention(&q, keys, values, offset)?;
            record(&mut trace, format!("layers.{i}.attention_output"), &h);
            x = (x + layer.out.forward(&h)?)?;
            record(&mut trace, format!("layers.{i}.attention"), &x);
            let h = layer.post_norm.forward(&x)?;
            let y = match &layer.mlp {
                FeedForward::Dense(mlp) => mlp.forward(&h, self.moe_execution.fused_activation)?,
                FeedForward::Sparse(moe) => moe.forward(
                    &h,
                    c,
                    &mut trace,
                    &format!("layers.{i}"),
                    &self.moe_execution,
                )?,
            };
            x = (x + y).with_context(|| format!("layer {i}"))?;
            record(&mut trace, format!("layers.{i}.hidden"), &x);
        }
        let h = self.norm.forward(&x)?;
        record(&mut trace, "normalized".into(), &h);
        let out = if logits {
            Some(self.head.forward(&h)?.to_dtype(DType::F32)?)
        } else {
            None
        };
        cache.staged_tokens = Some(tokens.to_vec());
        cache.forwards += 1;
        cache.processed_tokens += n;
        Ok(out)
    }
}

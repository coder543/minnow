use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Config {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub partial_rotary_factor: f64,
    #[serde(default = "rope_theta")]
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub first_k_dense_replace: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub num_shared_experts: usize,
    pub expert_capacity: usize,
    pub block_size: usize,
    pub routed_scaling_factor: f64,
    pub use_qk_norm: bool,
    pub use_qkv_bias: bool,
    pub use_bias: bool,
    pub hidden_act: String,
    #[serde(default)]
    pub norm_head: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
}

fn rope_theta() -> f64 {
    3_000_000.0
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let mut c: Self = serde_json::from_slice(&fs::read(path.join("config.json"))?)
            .context("reading LLaDA2.2 configuration")?;
        if let Some(p) = &c.rope_parameters {
            ensure!(
                p["rope_type"] == "default",
                "only default partial RoPE is supported"
            );
            if let Some(theta) = p["rope_theta"].as_f64() {
                c.rope_theta = theta;
            }
            if let Some(factor) = p["partial_rotary_factor"].as_f64() {
                c.partial_rotary_factor = factor;
            }
        }
        c.validate()?;
        Ok(c)
    }

    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor) as usize
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "llada2_moe",
            "expected llada2_moe, got {}",
            self.model_type
        );
        ensure!(
            self.hidden_size > 0 && self.num_hidden_layers > 0 && self.vocab_size > 0,
            "empty model dimensions"
        );
        ensure!(
            self.num_key_value_heads > 0 && self.num_attention_heads > 0 && self.head_dim > 0,
            "invalid attention dimensions"
        );
        ensure!(
            self.num_attention_heads
                .is_multiple_of(self.num_key_value_heads),
            "Q heads must be divisible by KV heads"
        );
        ensure!(
            self.rotary_dim() > 0
                && self.rotary_dim().is_multiple_of(2)
                && self.rotary_dim() <= self.head_dim,
            "invalid rotary dimension"
        );
        ensure!(
            self.rope_theta.is_finite()
                && self.rope_theta > 0.0
                && self.rms_norm_eps.is_finite()
                && self.rms_norm_eps > 0.0,
            "invalid RoPE/RMSNorm parameters"
        );
        ensure!(
            self.block_size > 0 && self.max_position_embeddings >= self.block_size,
            "invalid block/context dimensions"
        );
        ensure!(
            self.num_experts_per_tok > 0
                && self.num_experts_per_tok <= self.expert_capacity
                && self.expert_capacity <= self.num_experts,
            "invalid block routing dimensions"
        );
        ensure!(
            self.first_k_dense_replace <= self.num_hidden_layers
                && self.moe_intermediate_size > 0
                && self.intermediate_size > 0,
            "invalid MLP dimensions"
        );
        ensure!(
            self.routed_scaling_factor.is_finite() && self.routed_scaling_factor > 0.0,
            "invalid routing scale"
        );
        ensure!(
            self.hidden_act == "silu"
                && !self.use_qkv_bias
                && !self.use_bias
                && !self.norm_head
                && !self.tie_word_embeddings
                && !self.use_sliding_window
                && self.rope_scaling.is_none(),
            "unsupported model variant: requires SiLU, no projection bias, untied unnormalized head, no sliding window or scaled RoPE"
        );
        Ok(())
    }
}

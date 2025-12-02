use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ContinuumMemConfig {
    pub enabled: bool,
    pub ultra_short_span: usize,
    pub short_span: usize,
    pub mid_span: usize,
    pub long_span: usize,
    pub episodic_span: usize,
}

impl Default for ContinuumMemConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ultra_short_span: 2,
            short_span: 8,
            mid_span: 32,
            long_span: 128,
            episodic_span: 512,
        }
    }
}

impl ContinuumMemConfig {
    pub fn validate(&self) {
        if self.enabled {
            assert!(self.ultra_short_span > 0, "ultra_short_span must be > 0");
            assert!(self.short_span >= self.ultra_short_span, "short_span must be >= ultra_short_span");
            assert!(self.mid_span >= self.short_span, "mid_span must be >= short_span");
            assert!(self.long_span >= self.mid_span, "long_span must be >= mid_span");
            assert!(self.episodic_span >= self.long_span, "episodic_span must be >= long_span");
        }
    }
}

impl fmt::Display for ContinuumMemConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SelfModifyConfig {
    pub enabled: bool,
    pub meta_lr: f32,
    pub update_frequency: usize,
    pub weight_mod_dim: usize,
}

impl Default for SelfModifyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            meta_lr: 1e-5,
            update_frequency: 8,
            weight_mod_dim: 128,
        }
    }
}

impl SelfModifyConfig {
    pub fn validate(&self) {
        if self.enabled {
            assert!(self.meta_lr > 0.0, "meta_lr must be > 0");
            assert!(self.update_frequency > 0, "update_frequency must be > 0");
            assert!(self.weight_mod_dim > 0, "weight_mod_dim must be > 0");
        }
    }
}

impl fmt::Display for SelfModifyConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DeepOptimizerConfig {
    pub enabled: bool,
    pub fast_lr_scale: f32,
    pub slow_lr_scale: f32,
    pub fast_ema: f32,
    pub slow_ema: f32,
    pub sync_interval: usize,
    pub gradient_compression_dim: usize,
}

impl Default for DeepOptimizerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            fast_lr_scale: 1.0,
            slow_lr_scale: 0.1,
            fast_ema: 0.9,
            slow_ema: 0.99,
            sync_interval: 64,
            gradient_compression_dim: 256,
        }
    }
}

impl DeepOptimizerConfig {
    pub fn validate(&self) {
        if self.enabled {
            assert!((0.0..=1.0).contains(&self.fast_ema), "fast_ema must be within [0,1]");
            assert!((0.0..=1.0).contains(&self.slow_ema), "slow_ema must be within [0,1]");
            assert!(self.sync_interval > 0, "sync_interval must be > 0");
            assert!(self.fast_lr_scale > 0.0, "fast_lr_scale must be > 0");
            assert!(self.slow_lr_scale > 0.0, "slow_lr_scale must be > 0");
            assert!(self.gradient_compression_dim > 0, "gradient_compression_dim must be > 0");
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HopeConfig {
    // 基础架构
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub ff_multiplier: f32,
    pub dropout: f64,
    
    // 嵌套层级
    pub num_levels: usize,
    pub level_timescales: Vec<usize>,
    
    // 连续内存
    pub continuum_mem: ContinuumMemConfig,
    
    // 自修改
    pub self_modify: SelfModifyConfig,
    
    // Deep Optimizer
    pub deep_optimizer: DeepOptimizerConfig,
}

impl Default for HopeConfig {
    fn default() -> Self {
        Self {
            hidden_size: 384,
            vocab_size: 512,
            seq_len: 256,
            num_heads: 8,
            num_layers: 4,
            ff_multiplier: 4.0,
            dropout: 0.1,
            num_levels: 3,
            level_timescales: vec![1, 4, 16],
            continuum_mem: ContinuumMemConfig::default(),
            self_modify: SelfModifyConfig::default(),
            deep_optimizer: DeepOptimizerConfig::default(),
        }
    }
}

impl HopeConfig {
    pub fn validate(&self) {
        assert!(self.hidden_size > 0, "hidden_size must be > 0");
        assert!(self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads");
        assert!(self.vocab_size > 0, "vocab_size must be > 0");
        assert!(self.seq_len > 0, "seq_len must be > 0");
        assert!(self.num_heads > 0, "num_heads must be > 0");
        assert!(self.num_layers > 0, "num_layers must be > 0");
        assert!(self.num_levels > 0, "num_levels must be > 0");
        assert!(!self.level_timescales.is_empty(), "level_timescales must not be empty");
        assert_eq!(
            self.level_timescales.len(),
            self.num_levels,
            "level_timescales length must match num_levels"
        );
        self.continuum_mem.validate();
        self.self_modify.validate();
        self.deep_optimizer.validate();
    }

    pub fn feedforward_dim(&self) -> usize {
        (self.hidden_size as f32 * self.ff_multiplier).round() as usize
    }
}

impl fmt::Display for HopeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainingConfig {
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_num_steps")]
    pub num_steps: usize,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f32,
    #[serde(default = "default_log_every")]
    pub log_every: usize,
    #[serde(default = "default_use_random_data")]
    #[allow(dead_code)]
    pub use_random_data: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainConfig {
    pub model: HopeConfig,
    #[serde(rename = "training")]
    pub training: TrainingConfig,
}

impl TrainConfig {
    pub fn batch_size(&self) -> usize {
        self.training.batch_size
    }
    
    pub fn num_steps(&self) -> usize {
        self.training.num_steps
    }
    
    pub fn learning_rate(&self) -> f32 {
        self.training.learning_rate
    }
    
    pub fn log_every(&self) -> usize {
        self.training.log_every
    }
}

fn default_batch_size() -> usize {
    4
}

fn default_num_steps() -> usize {
    1000
}

fn default_learning_rate() -> f32 {
    1e-4
}

fn default_log_every() -> usize {
    10
}

fn default_use_random_data() -> bool {
    true
}


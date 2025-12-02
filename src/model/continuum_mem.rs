use burn::constant;
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::{Tensor, activation, backend::Backend};
use crate::config::ContinuumMemConfig;

constant!(ContinuumMemConfig);

#[derive(Clone, Debug)]
pub struct ContinuumMemoryState<B: Backend> {
    pub ultra_short: Tensor<B, 3>,
    pub short: Tensor<B, 3>,
    pub mid: Tensor<B, 3>,
    pub long: Tensor<B, 3>,
    pub episodic: Tensor<B, 3>,
}

#[derive(Module, Debug)]
pub struct ContinuumMemory<B: Backend> {
    #[module(skip)]
    config: ContinuumMemConfig,
    query_proj: Linear<B>,
    key_proj: Linear<B>,
    value_proj: Linear<B>,
    norm: LayerNorm<B>,
}

impl<B: Backend> ContinuumMemory<B> {
    pub fn new(config: ContinuumMemConfig, hidden_size: usize, device: &B::Device) -> Self {
        config.validate();
        let query_proj = LinearConfig::new(hidden_size, hidden_size).init(device);
        let key_proj = LinearConfig::new(hidden_size, hidden_size).init(device);
        let value_proj = LinearConfig::new(hidden_size, hidden_size).init(device);
        let norm = LayerNormConfig::new(hidden_size).init(device);

        Self {
            config,
            query_proj,
            key_proj,
            value_proj,
            norm,
        }
    }

    pub fn init_state(
        &self,
        batch: usize,
        seq_len: usize,
        hidden_size: usize,
        device: &B::Device,
    ) -> ContinuumMemoryState<B> {
        let zeros = || Tensor::zeros([batch, seq_len, hidden_size], device);
        ContinuumMemoryState {
            ultra_short: zeros(),
            short: zeros(),
            mid: zeros(),
            long: zeros(),
            episodic: zeros(),
        }
    }

    pub fn update(
        &self,
        state: &mut ContinuumMemoryState<B>,
        new_hidden: &Tensor<B, 3>,
    ) {
        if !self.config.enabled {
            return;
        }

        // Ultra-short: direct copy (1-4 steps)
        state.ultra_short = new_hidden.clone();

        // Short: fast EMA (4-16 steps)
        let short_alpha = self.compute_alpha(self.config.short_span);
        state.short = self.ema_update(&state.short, new_hidden, short_alpha);

        // Mid: medium EMA (16-64 steps)
        let mid_alpha = self.compute_alpha(self.config.mid_span);
        state.mid = self.ema_update(&state.mid, new_hidden, mid_alpha);

        // Long: slow EMA (64-256 steps)
        let long_alpha = self.compute_alpha(self.config.long_span);
        state.long = self.ema_update(&state.long, new_hidden, long_alpha);

        // Episodic: very slow EMA (>256 steps)
        let episodic_alpha = self.compute_alpha(self.config.episodic_span);
        state.episodic = self.ema_update(&state.episodic, new_hidden, episodic_alpha);
    }

    pub fn retrieve(
        &self,
        state: &ContinuumMemoryState<B>,
        query: &Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        if !self.config.enabled {
            return query.clone();
        }

        // Compute attention over all memory banks
        let memories = vec![
            &state.ultra_short,
            &state.short,
            &state.mid,
            &state.long,
            &state.episodic,
        ];

        let batch = query.dims()[0];
        let seq_len = query.dims()[1];
        let hidden = query.dims()[2];
        
        // Reshape query to 2D for linear projection
        let query_2d = query.clone().reshape([batch * seq_len, hidden]);
        let query_proj = self.query_proj.forward(query_2d);
        let query_proj = self.norm.forward(query_proj);
        let query_proj = query_proj.reshape([batch, seq_len, hidden]);

        let mut all_keys = Vec::new();
        let mut all_values = Vec::new();

        for memory in &memories {
            let mem_batch = memory.dims()[0];
            let mem_seq_len = memory.dims()[1];
            let mem_hidden = memory.dims()[2];
            
            // Reshape memory to 2D for linear projection
            let mem_clone = (*memory).clone();
            let mem_2d = mem_clone.reshape([mem_batch * mem_seq_len, mem_hidden]);
            let keys_2d = self.key_proj.forward(mem_2d.clone());
            let values_2d = self.value_proj.forward(mem_2d);
            let keys = keys_2d.reshape([mem_batch, mem_seq_len, hidden]);
            let values = values_2d.reshape([mem_batch, mem_seq_len, hidden]);
            all_keys.push(keys);
            all_values.push(values);
        }

        // Concatenate all memories
        let keys = Tensor::cat(all_keys, 1);
        let values = Tensor::cat(all_values, 1);

        // Simplified attention: compute weighted sum over all memory banks
        let _batch = query.dims()[0];
        let _seq_len = query.dims()[1];
        let hidden = query.dims()[2];
        let _mem_seq_len = keys.dims()[1];

        // Compute attention scores: [batch, seq_len, hidden] x [batch, hidden, mem_seq_len]
        let query_expanded = query_proj.clone(); // [batch, seq_len, hidden]
        let keys_transposed = keys.swap_dims(1, 2); // [batch, hidden, mem_seq_len]
        
        // Compute scores: [batch, seq_len, mem_seq_len]
        let scores = query_expanded.matmul(keys_transposed);
        let scale = (hidden as f32).sqrt().recip();
        let scores = scores * scale;
        let attn_weights = activation::softmax(scores, 2);

        // Apply attention to values: [batch, seq_len, mem_seq_len] x [batch, mem_seq_len, hidden]
        let attended = attn_weights.matmul(values); // [batch, seq_len, hidden]

        // Residual connection
        query.clone() + attended
    }

    fn compute_alpha(&self, span: usize) -> f32 {
        if span == 0 {
            1.0
        } else {
            (1.0 / span as f32).clamp(0.0, 1.0)
        }
    }

    fn ema_update(&self, old: &Tensor<B, 3>, new: &Tensor<B, 3>, alpha: f32) -> Tensor<B, 3> {
        let one_minus_alpha = 1.0 - alpha;
        old.clone() * one_minus_alpha + new.clone() * alpha
    }

    #[allow(dead_code)]
    pub fn config(&self) -> &ContinuumMemConfig {
        &self.config
    }
}


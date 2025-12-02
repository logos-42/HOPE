use burn::tensor::{Tensor, backend::AutodiffBackend};
use crate::config::DeepOptimizerConfig;

#[derive(Clone, Debug)]
pub struct DeepOptimizerState<B: AutodiffBackend> {
    pub fast_params: Vec<Tensor<B, 3>>,
    pub slow_params: Vec<Tensor<B, 3>>,
    pub fast_ema: Vec<Tensor<B, 3>>,
    pub slow_ema: Vec<Tensor<B, 3>>,
    pub step_count: usize,
}

pub struct DeepOptimizer {
    config: DeepOptimizerConfig,
}

impl DeepOptimizer {
    pub fn new(config: DeepOptimizerConfig) -> Self {
        config.validate();
        Self { config }
    }

    pub fn init_state<B: AutodiffBackend>(
        &self,
        num_levels: usize,
        batch: usize,
        seq_len: usize,
        hidden_size: usize,
        device: &B::Device,
    ) -> DeepOptimizerState<B> {
        let zeros = || Tensor::zeros([batch, seq_len, hidden_size], device);
        
        DeepOptimizerState {
            fast_params: (0..num_levels).map(|_| zeros()).collect(),
            slow_params: (0..num_levels).map(|_| zeros()).collect(),
            fast_ema: (0..num_levels).map(|_| zeros()).collect(),
            slow_ema: (0..num_levels).map(|_| zeros()).collect(),
            step_count: 0,
        }
    }

    pub fn update_fast_params<B: AutodiffBackend>(
        &self,
        state: &mut DeepOptimizerState<B>,
        gradients: &[Tensor<B, 3>],
        learning_rate: f32,
    ) {
        if !self.config.enabled {
            return;
        }

        let fast_lr = learning_rate * self.config.fast_lr_scale;

        for (level_idx, grad) in gradients.iter().enumerate() {
            if level_idx >= state.fast_params.len() {
                continue;
            }

            // Update fast parameters
            let update = grad.clone() * fast_lr;
            state.fast_params[level_idx] = state.fast_params[level_idx].clone() - update.clone();

            // Update fast EMA
            let ema_alpha = self.config.fast_ema;
            state.fast_ema[level_idx] = state.fast_ema[level_idx].clone() * (1.0 - ema_alpha)
                + state.fast_params[level_idx].clone() * ema_alpha;
        }

        state.step_count += 1;
    }

    pub fn update_slow_params<B: AutodiffBackend>(
        &self,
        state: &mut DeepOptimizerState<B>,
        learning_rate: f32,
    ) {
        if !self.config.enabled {
            return;
        }

        let slow_lr = learning_rate * self.config.slow_lr_scale;

        for level_idx in 0..state.slow_params.len() {
            // Slow parameters are updated from fast EMA
            let diff = state.fast_ema[level_idx].clone() - state.slow_params[level_idx].clone();
            state.slow_params[level_idx] = state.slow_params[level_idx].clone() + diff * slow_lr;

            // Update slow EMA
            let ema_alpha = self.config.slow_ema;
            state.slow_ema[level_idx] = state.slow_ema[level_idx].clone() * (1.0 - ema_alpha)
                + state.slow_params[level_idx].clone() * ema_alpha;
        }
    }

    pub fn compress_gradient<B: AutodiffBackend>(
        &self,
        gradient: &Tensor<B, 3>,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        if !self.config.enabled {
            let batch = gradient.dims()[0];
            return Tensor::zeros([batch, self.config.gradient_compression_dim], device);
        }

        let batch = gradient.dims()[0];
        let seq_len = gradient.dims()[1];
        let hidden = gradient.dims()[2];

        // Average over sequence dimension and compress
        let grad_avg = gradient
            .clone()
            .sum_dim(1)
            .div_scalar(seq_len as f32)
            .reshape([batch, hidden]);

        // Simple compression: take first N dimensions
        let compress_dim = self.config.gradient_compression_dim.min(hidden);
        grad_avg.slice([0..batch, 0..compress_dim])
    }

    pub fn should_sync<B: AutodiffBackend>(&self, state: &DeepOptimizerState<B>) -> bool {
        self.config.enabled && (state.step_count % self.config.sync_interval == 0)
    }

    pub fn sync<B: AutodiffBackend>(
        &self,
        state: &mut DeepOptimizerState<B>,
    ) {
        if !self.config.enabled {
            return;
        }

        // Synchronize slow parameters with fast EMA
        for level_idx in 0..state.slow_params.len() {
            state.slow_params[level_idx] = state.fast_ema[level_idx].clone();
        }
    }

    pub fn get_fast_params<'a, B: AutodiffBackend>(
        &self,
        state: &'a DeepOptimizerState<B>,
        level_idx: usize,
    ) -> Option<&'a Tensor<B, 3>> {
        state.fast_params.get(level_idx)
    }

    pub fn get_slow_params<'a, B: AutodiffBackend>(
        &self,
        state: &'a DeepOptimizerState<B>,
        level_idx: usize,
    ) -> Option<&'a Tensor<B, 3>> {
        state.slow_params.get(level_idx)
    }

    pub fn config(&self) -> &DeepOptimizerConfig {
        &self.config
    }
}


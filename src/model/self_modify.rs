use burn::constant;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::{Tensor, activation, backend::Backend};
use crate::config::SelfModifyConfig;

constant!(SelfModifyConfig);

#[derive(Clone, Debug)]
pub struct SelfModifyState<B: Backend> {
    pub meta_state: Tensor<B, 2>,
    pub update_count: usize,
}

#[derive(Module, Debug)]
pub struct SelfModifyModule<B: Backend> {
    #[module(skip)]
    config: SelfModifyConfig,
    meta_network: MetaNetwork<B>,
    weight_mod_network: WeightModNetwork<B>,
    gradient_compressor: GradientCompressor<B>,
    norm: LayerNorm<B>,
    dropout: Dropout,
}

#[derive(Module, Debug)]
struct MetaNetwork<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    layer3: Linear<B>,
}

#[derive(Module, Debug)]
struct WeightModNetwork<B: Backend> {
    input_proj: Linear<B>,
    hidden: Linear<B>,
    output_proj: Linear<B>,
}

#[derive(Module, Debug)]
struct GradientCompressor<B: Backend> {
    compress: Linear<B>,
    decompress: Linear<B>,
}

impl<B: Backend> SelfModifyModule<B> {
    pub fn new(
        config: SelfModifyConfig,
        hidden_size: usize,
        device: &B::Device,
    ) -> Self {
        config.validate();
        let meta_dim = config.weight_mod_dim;
        let meta_network = MetaNetwork {
            layer1: LinearConfig::new(hidden_size, meta_dim).init(device),
            layer2: LinearConfig::new(meta_dim, meta_dim).init(device),
            layer3: LinearConfig::new(meta_dim, meta_dim).init(device),
        };
        let weight_mod_network = WeightModNetwork {
            input_proj: LinearConfig::new(hidden_size, meta_dim).init(device),
            hidden: LinearConfig::new(meta_dim, meta_dim).init(device),
            output_proj: LinearConfig::new(meta_dim, hidden_size).init(device),
        };
        let gradient_compressor = GradientCompressor {
            compress: LinearConfig::new(hidden_size, meta_dim).init(device),
            decompress: LinearConfig::new(meta_dim, hidden_size).init(device),
        };
        let norm = LayerNormConfig::new(hidden_size).init(device);
        let dropout = DropoutConfig::new(0.1).init();

        Self {
            config,
            meta_network,
            weight_mod_network,
            gradient_compressor,
            norm,
            dropout,
        }
    }

    pub fn init_state(
        &self,
        batch: usize,
        _hidden_size: usize,
        device: &B::Device,
    ) -> SelfModifyState<B> {
        SelfModifyState {
            meta_state: Tensor::zeros([batch, self.config.weight_mod_dim], device),
            update_count: 0,
        }
    }

    pub fn compute_update_rule(
        &self,
        hidden: &Tensor<B, 3>,
        state: &SelfModifyState<B>,
    ) -> Tensor<B, 2> {
        if !self.config.enabled {
            let batch = hidden.dims()[0];
            let device = hidden.device();
            return Tensor::zeros([batch, self.config.weight_mod_dim], &device);
        }

        // Extract first token representation for meta-learning
        let batch = hidden.dims()[0];
        let meta_input = hidden
            .clone()
            .slice([0..batch, 0..1, 0..hidden.dims()[2]])
            .reshape([batch, hidden.dims()[2]]);

        // Meta network generates update rule
        let x = self.meta_network.layer1.forward(meta_input);
        let x = activation::relu(x);
        let x = self.meta_network.layer2.forward(x);
        let x = activation::relu(x);
        let x = self.meta_network.layer3.forward(x);
        let update_rule = activation::tanh(x);

        // Combine with previous meta state
        let meta_state = state.meta_state.clone() * 0.9 + update_rule.clone() * 0.1;
        meta_state
    }

    pub fn apply_weight_modification(
        &self,
        hidden: &Tensor<B, 3>,
        meta_state: &Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        if !self.config.enabled {
            return hidden.clone();
        }

        let batch = hidden.dims()[0];
        let seq_len = hidden.dims()[1];
        let hidden_size = hidden.dims()[2];

        // Project hidden states
        let hidden_flat = hidden.clone().reshape([batch * seq_len, hidden_size]);
        let x = self.weight_mod_network.input_proj.forward(hidden_flat);
        let x = activation::relu(x);

        // Use meta state to modulate
        let meta_expanded = meta_state.clone()
            .unsqueeze_dim::<3>(1)
            .repeat_dim(1, seq_len)
            .reshape([batch * seq_len, self.config.weight_mod_dim]);
        let x = x + meta_expanded;
        let x = self.weight_mod_network.hidden.forward(x);
        let x = activation::relu(x);
        let weight_mod = self.weight_mod_network.output_proj.forward(x);
        let weight_mod = weight_mod.reshape([batch, seq_len, hidden_size]);

        // Apply modification with residual connection
        let modified = hidden.clone() + weight_mod * 0.1; // Small scaling factor
        self.norm.forward(modified)
    }

    #[allow(dead_code)]
    pub fn compress_gradients(
        &self,
        gradients: &Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        if !self.config.enabled {
            let batch = gradients.dims()[0];
            let device = gradients.device();
            return Tensor::zeros([batch, self.config.weight_mod_dim], &device);
        }

        let batch = gradients.dims()[0];
        let seq_len = gradients.dims()[1];
        let hidden_size = gradients.dims()[2];

        // Average over sequence dimension
        let grad_avg = gradients
            .clone()
            .sum_dim(1)
            .div_scalar(seq_len as f32)
            .reshape([batch, hidden_size]);

        // Compress gradients
        let compressed = self.gradient_compressor.compress.forward(grad_avg);
        activation::tanh(compressed)
    }

    #[allow(dead_code)]
    pub fn decompress_gradients(
        &self,
        compressed: &Tensor<B, 2>,
        target_shape: &[usize],
    ) -> Tensor<B, 3> {
        if !self.config.enabled {
            let device = compressed.device();
            return Tensor::zeros(target_shape, &device);
        }

        let _batch = compressed.dims()[0];
        let decompressed = self.gradient_compressor.decompress.forward(compressed.clone());
        let shape: [usize; 3] = [target_shape[0], target_shape[1], target_shape[2]];
        decompressed
            .unsqueeze_dim::<3>(1)
            .repeat_dim(1, target_shape[1])
            .reshape(shape)
    }

    #[allow(dead_code)]
    pub fn should_update(&self, state: &SelfModifyState<B>) -> bool {
        self.config.enabled && (state.update_count % self.config.update_frequency == 0)
    }

    #[allow(dead_code)]
    pub fn config(&self) -> &SelfModifyConfig {
        &self.config
    }
}

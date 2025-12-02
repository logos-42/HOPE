use burn::nn::loss::CrossEntropyLoss;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Int, Tensor, backend::{AutodiffBackend, Backend}};
use crate::config::TrainConfig;
use crate::model::{HopeModel, HopeInput};

#[derive(Clone, Debug)]
pub struct TrainOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub step: usize,
}

impl<B: Backend> TrainOutput<B> {
    pub fn new(loss: Tensor<B, 1>, step: usize) -> Self {
        Self { loss, step }
    }
}

pub struct HopeTrainer<B: AutodiffBackend>
where
    <B as AutodiffBackend>::InnerBackend: AutodiffBackend,
{
    model: HopeModel<B>,
    optimizer: OptimizerAdaptor<Adam, HopeModel<B>, B>,
    loss_fn: CrossEntropyLoss<B>,
    config: TrainConfig,
}

impl<B: AutodiffBackend> HopeTrainer<B>
where
    <B as AutodiffBackend>::InnerBackend: AutodiffBackend,
{
    pub fn new(
        model: HopeModel<B>,
        config: TrainConfig,
        device: &<B as Backend>::Device,
    ) -> Self {
        let optimizer = AdamConfig::new().init::<B, HopeModel<B>>();
        let loss_fn = CrossEntropyLoss::new(None, device);

        Self {
            model,
            optimizer,
            loss_fn,
            config,
        }
    }

    pub fn train_step(
        &mut self,
        batch: BatchData<B>,
    ) -> TrainOutput<B> {
        let device = batch.tokens.device();
        let batch_size = batch.tokens.dims()[0];

        // Initialize carry state
        let carry = self.model.initial_carry(batch_size, &device);

        // Forward pass
        let (_, output) = self.model.forward(
            HopeInput {
                tokens: batch.tokens,
            },
            carry,
        );

        // Compute loss
        let logits = output.logits;
        let targets = batch.targets;

        // Reshape for loss computation: [batch, seq_len, vocab_size] -> [batch * seq_len, vocab_size]
        let batch_size = logits.dims()[0];
        let seq_len = logits.dims()[1];
        let vocab_size = logits.dims()[2];

        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = targets.reshape([batch_size * seq_len]);

        let loss = self.loss_fn.forward(logits_flat.clone(), targets_flat.clone());

        // Backward pass
        let grads = GradientsParams::from_grads(loss.backward(), &self.model);

        // Optimizer step
        let lr = f64::from(self.config.learning_rate);
        self.model = self.optimizer.step(lr, self.model.clone(), grads);

        TrainOutput::new(loss, 1)
    }

    pub fn model(&self) -> &HopeModel<B> {
        &self.model
    }
}

#[derive(Clone, Debug)]
pub struct BatchData<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> BatchData<B> {
    pub fn new(tokens: Tensor<B, 2, Int>, targets: Tensor<B, 2, Int>) -> Self {
        Self { tokens, targets }
    }
}

pub fn generate_random_batch<B: Backend>(
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    device: &<B as Backend>::Device,
) -> BatchData<B> {
    // Generate random tokens using arange and remainder
    let total = batch_size * seq_len;
    let tokens = Tensor::<B, 1, Int>::arange(0..total as i64, device)
        .reshape([batch_size, seq_len])
        .remainder_scalar(vocab_size as i64);

    // Targets are tokens shifted by 1 (next token prediction)
    let targets = tokens.clone().slice([
        0..batch_size,
        1..seq_len,
    ]);
    
    // Pad targets to match seq_len
    let pad_token = Tensor::<B, 2, Int>::zeros([batch_size, 1], device);
    let targets = Tensor::cat(vec![targets, pad_token], 1);

    BatchData::new(tokens, targets)
}


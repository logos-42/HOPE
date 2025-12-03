use anyhow::Result;
use burn::tensor::backend::Backend;
use crate::training::BatchData;

/// Trait for data loading
pub trait DataLoader<B: Backend> {
    /// Get the next batch of data
    fn next_batch(&mut self) -> Result<Option<BatchData<B>>>;
    
    /// Reset the data loader to the beginning
    fn reset(&mut self);
    
    /// Get the total number of batches (if known)
    fn num_batches(&self) -> Option<usize>;
}

/// Random data loader for testing (existing functionality)
pub struct RandomDataLoader<B: Backend> {
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    num_batches: usize,
    current_batch: usize,
    device: B::Device,
}

impl<B: Backend> RandomDataLoader<B> {
    pub fn new(
        batch_size: usize,
        seq_len: usize,
        vocab_size: usize,
        num_batches: usize,
        device: B::Device,
    ) -> Self {
        Self {
            batch_size,
            seq_len,
            vocab_size,
            num_batches,
            current_batch: 0,
            device,
        }
    }
}

impl<B: Backend> DataLoader<B> for RandomDataLoader<B> {
    fn next_batch(&mut self) -> Result<Option<BatchData<B>>> {
        if self.current_batch >= self.num_batches {
            return Ok(None);
        }
        
        self.current_batch += 1;
        
        // Use the existing random batch generation
        let batch = crate::training::generate_random_batch::<B>(
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            &self.device,
        );
        
        Ok(Some(batch))
    }
    
    fn reset(&mut self) {
        self.current_batch = 0;
    }
    
    fn num_batches(&self) -> Option<usize> {
        Some(self.num_batches)
    }
}


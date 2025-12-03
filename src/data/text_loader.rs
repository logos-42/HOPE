use anyhow::{Context, Result};
use burn::tensor::{Int, Tensor, backend::Backend};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;
use walkdir::WalkDir;

use super::loader::DataLoader;
use super::tokenizer::Tokenizer;
use crate::training::BatchData;

/// Text data loader that loads data from text files
pub struct TextDataLoader<B: Backend> {
    tokens: Vec<i64>,
    batch_size: usize,
    seq_len: usize,
    current_pos: usize,
    device: B::Device,
}

impl<B: Backend> TextDataLoader<B> {
    /// Create a new text data loader from a single file
    pub fn from_file<T: Tokenizer>(
        path: &Path,
        tokenizer: &T,
        batch_size: usize,
        seq_len: usize,
        device: B::Device,
    ) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("Failed to read text file: {:?}", path))?;
        
        info!("Loaded text file: {:?} ({} characters)", path, text.len());
        
        let tokens = tokenizer.encode(&text);
        info!("Tokenized to {} tokens", tokens.len());
        
        Ok(Self {
            tokens,
            batch_size,
            seq_len,
            current_pos: 0,
            device,
        })
    }
    
    /// Create a new text data loader from multiple files
    pub fn from_directory<T: Tokenizer>(
        dir_path: &Path,
        tokenizer: &T,
        batch_size: usize,
        seq_len: usize,
        device: B::Device,
    ) -> Result<Self> {
        let mut all_tokens = Vec::new();
        let mut file_count = 0;
        
        for entry in WalkDir::new(dir_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            
            // Only process text files
            if let Some(ext) = path.extension() {
                if ext == "txt" {
                    if let Ok(text) = fs::read_to_string(path) {
                        let tokens = tokenizer.encode(&text);
                        all_tokens.extend(tokens);
                        file_count += 1;
                        
                        if file_count % 10 == 0 {
                            info!("Processed {} files, {} tokens so far", file_count, all_tokens.len());
                        }
                    }
                }
            }
        }
        
        info!("Loaded {} text files from {:?} ({} total tokens)", 
            file_count, dir_path, all_tokens.len());
        
        if all_tokens.is_empty() {
            anyhow::bail!("No text data found in directory: {:?}", dir_path);
        }
        
        Ok(Self {
            tokens: all_tokens,
            batch_size,
            seq_len,
            current_pos: 0,
            device,
        })
    }
    
    /// Create from pre-tokenized data
    pub fn from_tokens(
        tokens: Vec<i64>,
        batch_size: usize,
        seq_len: usize,
        device: B::Device,
    ) -> Self {
        Self {
            tokens,
            batch_size,
            seq_len,
            current_pos: 0,
            device,
        }
    }
}

impl<B: Backend> DataLoader<B> for TextDataLoader<B> {
    fn next_batch(&mut self) -> Result<Option<BatchData<B>>> {
        // Check if we have enough data for a full batch
        let required_len = self.batch_size * (self.seq_len + 1);  // +1 for target
        
        if self.current_pos + required_len > self.tokens.len() {
            return Ok(None);
        }
        
        // Extract batch data
        let mut batch_tokens = Vec::new();
        let mut batch_targets = Vec::new();
        
        for _ in 0..self.batch_size {
            let start = self.current_pos;
            let end = start + self.seq_len + 1;
            
            if end > self.tokens.len() {
                return Ok(None);
            }
            
            let sequence = &self.tokens[start..end];
            
            // Input tokens
            batch_tokens.extend_from_slice(&sequence[..self.seq_len]);
            
            // Target tokens (shifted by 1)
            batch_targets.extend_from_slice(&sequence[1..]);
            
            self.current_pos += self.seq_len;
        }
        
        // Convert to tensors
        let tokens_tensor = Tensor::<B, 1, Int>::from_ints(
            batch_tokens.as_slice(),
            &self.device,
        ).reshape([self.batch_size, self.seq_len]);
        
        let targets_tensor = Tensor::<B, 1, Int>::from_ints(
            batch_targets.as_slice(),
            &self.device,
        ).reshape([self.batch_size, self.seq_len]);
        
        Ok(Some(BatchData {
            tokens: tokens_tensor,
            targets: targets_tensor,
        }))
    }
    
    fn reset(&mut self) {
        self.current_pos = 0;
    }
    
    fn num_batches(&self) -> Option<usize> {
        let required_len = self.batch_size * (self.seq_len + 1);
        if self.tokens.len() < required_len {
            return Some(0);
        }
        
        // Calculate how many complete batches we can make
        let available_sequences = (self.tokens.len() - self.seq_len) / self.seq_len;
        Some(available_sequences / self.batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::CharTokenizer;
    use burn_ndarray::NdArray;
    use tempfile::NamedTempFile;
    use std::io::Write;
    
    type TestBackend = NdArray<f32>;
    
    #[test]
    fn test_text_data_loader() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "Hello, World! This is a test.").unwrap();
        
        let tokenizer = CharTokenizer::from_text("Hello, World! This is a test.");
        let device = Default::default();
        
        let mut loader = TextDataLoader::<TestBackend>::from_file(
            temp_file.path(),
            &tokenizer,
            2,
            5,
            device,
        ).unwrap();
        
        assert!(loader.num_batches().unwrap() > 0);
        
        let batch = loader.next_batch().unwrap();
        assert!(batch.is_some());
        
        let batch_data = batch.unwrap();
        assert_eq!(batch_data.tokens.dims(), [2, 5]);
        assert_eq!(batch_data.targets.dims(), [2, 5]);
    }
}


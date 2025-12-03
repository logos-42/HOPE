use anyhow::{Context, Result};
use burn::tensor::{Int, Tensor, backend::Backend};
use std::path::{Path, PathBuf};
use tracing::{info, warn};
use walkdir::WalkDir;

use super::loader::DataLoader;
use super::tokenizer::Tokenizer;
use crate::training::BatchData;
use crate::utils::{extract_text_from_pdf, extract_text_from_epub, add_structure_markers, clean_text};

/// Book data loader that supports PDF and EPUB files
pub struct BookDataLoader<B: Backend> {
    tokens: Vec<i64>,
    batch_size: usize,
    seq_len: usize,
    current_pos: usize,
    device: B::Device,
    book_files: Vec<PathBuf>,
}

impl<B: Backend> BookDataLoader<B> {
    /// Create a new book data loader from a directory (online mode)
    pub fn from_directory<T: Tokenizer>(
        dir_path: &Path,
        tokenizer: &T,
        batch_size: usize,
        seq_len: usize,
        device: B::Device,
        preserve_structure: bool,
    ) -> Result<Self> {
        info!("Loading books from directory: {:?}", dir_path);
        
        let mut book_files = Vec::new();
        let mut all_text = String::new();
        
        // Find all PDF and EPUB files
        for entry in WalkDir::new(dir_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                
                if ext_str == "pdf" || ext_str == "epub" {
                    book_files.push(path.to_path_buf());
                }
            }
        }
        
        info!("Found {} book files", book_files.len());
        
        // Process each book
        for (idx, book_path) in book_files.iter().enumerate() {
            info!("Processing book {}/{}: {:?}", idx + 1, book_files.len(), book_path);
            
            match Self::extract_book_text(book_path, preserve_structure) {
                Ok(text) => {
                    all_text.push_str(&text);
                    all_text.push_str("\n\n");
                }
                Err(e) => {
                    warn!("Failed to process book {:?}: {}", book_path, e);
                }
            }
        }
        
        if all_text.is_empty() {
            anyhow::bail!("No text extracted from books in {:?}", dir_path);
        }
        
        info!("Total text length: {} characters", all_text.len());
        
        // Tokenize
        let tokens = tokenizer.encode(&all_text);
        info!("Tokenized to {} tokens", tokens.len());
        
        Ok(Self {
            tokens,
            batch_size,
            seq_len,
            current_pos: 0,
            device,
            book_files,
        })
    }
    
    /// Create from preprocessed tokens (offline mode)
    pub fn from_preprocessed(
        tokens: Vec<i64>,
        batch_size: usize,
        seq_len: usize,
        device: B::Device,
    ) -> Self {
        info!("Loading from preprocessed tokens: {} tokens", tokens.len());
        
        Self {
            tokens,
            batch_size,
            seq_len,
            current_pos: 0,
            device,
            book_files: Vec::new(),
        }
    }
    
    /// Extract text from a single book file
    fn extract_book_text(path: &Path, preserve_structure: bool) -> Result<String> {
        let ext = path.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        let text = match ext.as_str() {
            "pdf" => {
                let content = extract_text_from_pdf(path)?;
                
                if !content.has_text {
                    anyhow::bail!("PDF has no extractable text (may need OCR)");
                }
                
                if preserve_structure {
                    // Try to extract structured content
                    match crate::utils::pdf_parser::extract_structured_content(path) {
                        Ok(sections) => add_structure_markers(sections),
                        Err(_) => clean_text(&content.text),
                    }
                } else {
                    clean_text(&content.text)
                }
            }
            "epub" => {
                let content = extract_text_from_epub(path)?;
                
                if preserve_structure {
                    add_structure_markers(content.chapters)
                } else {
                    // Concatenate all chapters
                    content.chapters
                        .into_iter()
                        .map(|(_, text)| clean_text(&text))
                        .collect::<Vec<_>>()
                        .join("\n\n")
                }
            }
            _ => {
                anyhow::bail!("Unsupported file format: {}", ext);
            }
        };
        
        Ok(text)
    }
    
    /// Get list of processed book files
    pub fn book_files(&self) -> &[PathBuf] {
        &self.book_files
    }
}

impl<B: Backend> DataLoader<B> for BookDataLoader<B> {
    fn next_batch(&mut self) -> Result<Option<BatchData<B>>> {
        // Check if we have enough data for a full batch
        let required_len = self.batch_size * (self.seq_len + 1);
        
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
        
        let available_sequences = (self.tokens.len() - self.seq_len) / self.seq_len;
        Some(available_sequences / self.batch_size)
    }
}


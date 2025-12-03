use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Trait for tokenization
pub trait Tokenizer: Send + Sync {
    /// Encode text to token IDs
    fn encode(&self, text: &str) -> Vec<i64>;
    
    /// Decode token IDs to text
    fn decode(&self, tokens: &[i64]) -> String;
    
    /// Get vocabulary size
    fn vocab_size(&self) -> usize;
    
    /// Get the ID for unknown tokens
    fn unk_id(&self) -> i64;
    
    /// Get the ID for padding tokens
    fn pad_id(&self) -> i64;
}

/// Character-level tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharTokenizer {
    char_to_id: HashMap<char, i64>,
    id_to_char: HashMap<i64, char>,
    vocab_size: usize,
    unk_id: i64,
    pad_id: i64,
}

impl CharTokenizer {
    /// Create a new character tokenizer from text
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();
        
        // Reserve IDs for special tokens
        let pad_id = 0;
        let unk_id = 1;
        let mut next_id = 2;
        
        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();
        
        // Add special tokens
        char_to_id.insert('\0', pad_id);  // Padding
        id_to_char.insert(pad_id, '\0');
        
        char_to_id.insert('�', unk_id);  // Unknown
        id_to_char.insert(unk_id, '�');
        
        // Add regular characters
        for ch in chars {
            if ch != '\0' && ch != '�' {
                char_to_id.insert(ch, next_id);
                id_to_char.insert(next_id, ch);
                next_id += 1;
            }
        }
        
        let vocab_size = next_id as usize;
        
        Self {
            char_to_id,
            id_to_char,
            vocab_size,
            unk_id,
            pad_id,
        }
    }
    
    /// Create a tokenizer with a predefined vocabulary
    pub fn from_vocab(vocab: Vec<char>) -> Self {
        let pad_id = 0;
        let unk_id = 1;
        let mut next_id = 2;
        
        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();
        
        // Add special tokens
        char_to_id.insert('\0', pad_id);
        id_to_char.insert(pad_id, '\0');
        
        char_to_id.insert('�', unk_id);
        id_to_char.insert(unk_id, '�');
        
        // Add vocabulary characters
        for ch in vocab {
            if ch != '\0' && ch != '�' && !char_to_id.contains_key(&ch) {
                char_to_id.insert(ch, next_id);
                id_to_char.insert(next_id, ch);
                next_id += 1;
            }
        }
        
        let vocab_size = next_id as usize;
        
        Self {
            char_to_id,
            id_to_char,
            vocab_size,
            unk_id,
            pad_id,
        }
    }
    
    /// Save tokenizer to a JSON file
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .with_context(|| "Failed to serialize tokenizer")?;
        
        fs::write(path, json)
            .with_context(|| format!("Failed to write tokenizer to {:?}", path))?;
        
        Ok(())
    }
    
    /// Load tokenizer from a JSON file
    pub fn load(path: &Path) -> Result<Self> {
        let json = fs::read_to_string(path)
            .with_context(|| format!("Failed to read tokenizer from {:?}", path))?;
        
        let tokenizer: Self = serde_json::from_str(&json)
            .with_context(|| "Failed to deserialize tokenizer")?;
        
        Ok(tokenizer)
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, text: &str) -> Vec<i64> {
        text.chars()
            .map(|ch| *self.char_to_id.get(&ch).unwrap_or(&self.unk_id))
            .collect()
    }
    
    fn decode(&self, tokens: &[i64]) -> String {
        tokens
            .iter()
            .filter_map(|&id| self.id_to_char.get(&id))
            .collect()
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    fn unk_id(&self) -> i64 {
        self.unk_id
    }
    
    fn pad_id(&self) -> i64 {
        self.pad_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_char_tokenizer_encode_decode() {
        let text = "Hello, World!";
        let tokenizer = CharTokenizer::from_text(text);
        
        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);
        
        assert_eq!(text, decoded);
    }
    
    #[test]
    fn test_char_tokenizer_unknown() {
        let tokenizer = CharTokenizer::from_text("abc");
        let encoded = tokenizer.encode("xyz");
        
        // All characters should be unknown
        assert!(encoded.iter().all(|&id| id == tokenizer.unk_id()));
    }
}


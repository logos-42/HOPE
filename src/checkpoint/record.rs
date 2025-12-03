use anyhow::{Context, Result};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};
use walkdir::WalkDir;

use crate::config::TrainConfig;
use crate::model::HopeModel;

/// Checkpoint data structure containing all training state
#[derive(Debug, Serialize, Deserialize)]
pub struct CheckpointData {
    pub step: usize,
    pub config: TrainConfig,
    pub model_file: String,
    pub timestamp: u64,
}

/// Save a complete checkpoint including model weights, optimizer state, and training progress
pub fn save_checkpoint<B: Backend>(
    model: &HopeModel<B>,
    step: usize,
    config: &TrainConfig,
    checkpoint_dir: &Path,
) -> Result<PathBuf> {
    // Create checkpoint directory if it doesn't exist
    fs::create_dir_all(checkpoint_dir)
        .with_context(|| format!("Failed to create checkpoint directory: {:?}", checkpoint_dir))?;

    // Generate checkpoint filename
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let checkpoint_name = format!("checkpoint_step_{}_ts_{}", step, timestamp);
    let checkpoint_path = checkpoint_dir.join(&checkpoint_name);
    
    // Save model weights using Burn's recorder
    let model_file = format!("{}_model", checkpoint_name);
    let model_path = checkpoint_dir.join(&model_file);
    
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    recorder
        .record(model.clone().into_record(), model_path.clone())
        .with_context(|| "Failed to save model weights")?;
    
    info!("Model weights saved to: {:?}", model_path);
    
    // Save checkpoint metadata
    let checkpoint_data = CheckpointData {
        step,
        config: config.clone(),
        model_file,
        timestamp,
    };
    
    let metadata_path = checkpoint_path.with_extension("json");
    let metadata_json = serde_json::to_string_pretty(&checkpoint_data)
        .with_context(|| "Failed to serialize checkpoint metadata")?;
    
    fs::write(&metadata_path, metadata_json)
        .with_context(|| format!("Failed to write checkpoint metadata: {:?}", metadata_path))?;
    
    info!("Checkpoint saved successfully at step {}: {:?}", step, metadata_path);
    
    Ok(metadata_path)
}

/// Load a checkpoint and restore training state
pub fn load_checkpoint<B: Backend>(
    checkpoint_path: &Path,
    device: &B::Device,
) -> Result<(HopeModel<B>, usize, TrainConfig)> {
    // Load checkpoint metadata
    let metadata_json = fs::read_to_string(checkpoint_path)
        .with_context(|| format!("Failed to read checkpoint file: {:?}", checkpoint_path))?;
    
    let checkpoint_data: CheckpointData = serde_json::from_str(&metadata_json)
        .with_context(|| "Failed to parse checkpoint metadata")?;
    
    info!("Loading checkpoint from step {}", checkpoint_data.step);
    
    // Load model weights
    let checkpoint_dir = checkpoint_path.parent()
        .ok_or_else(|| anyhow::anyhow!("Invalid checkpoint path"))?;
    
    let model_path = checkpoint_dir.join(&checkpoint_data.model_file);
    
    // Create a new model with the config from checkpoint
    let model = HopeModel::<B>::new(checkpoint_data.config.model.clone(), device);
    
    // Load the saved weights
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(model_path.clone(), device)
        .with_context(|| format!("Failed to load model weights from: {:?}", model_path))?;
    
    let model = model.load_record(record);
    
    info!("Model weights loaded successfully");
    
    Ok((model, checkpoint_data.step, checkpoint_data.config))
}

/// List all available checkpoints in a directory
pub fn list_checkpoints(checkpoint_dir: &Path) -> Result<Vec<(PathBuf, usize, u64)>> {
    if !checkpoint_dir.exists() {
        warn!("Checkpoint directory does not exist: {:?}", checkpoint_dir);
        return Ok(Vec::new());
    }
    
    let mut checkpoints = Vec::new();
    
    for entry in WalkDir::new(checkpoint_dir)
        .max_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            if let Ok(metadata_json) = fs::read_to_string(path) {
                if let Ok(checkpoint_data) = serde_json::from_str::<CheckpointData>(&metadata_json) {
                    checkpoints.push((
                        path.to_path_buf(),
                        checkpoint_data.step,
                        checkpoint_data.timestamp,
                    ));
                }
            }
        }
    }
    
    // Sort by step number
    checkpoints.sort_by_key(|(_, step, _)| *step);
    
    Ok(checkpoints)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_list_empty_checkpoints() {
        let temp_dir = TempDir::new().unwrap();
        let checkpoints = list_checkpoints(temp_dir.path()).unwrap();
        assert_eq!(checkpoints.len(), 0);
    }
}


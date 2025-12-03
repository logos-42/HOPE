mod checkpoint;
mod config;
mod data;
mod model;
mod training;
mod utils;

use anyhow::{Context, Result};
use burn::backend::Autodiff;
use burn_ndarray::NdArray;
use clap::{Args, Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use tracing::info;
use tracing_subscriber::EnvFilter;

use checkpoint::{save_checkpoint, load_checkpoint, list_checkpoints};
use config::TrainConfig;
use model::HopeModel;
use training::{HopeTrainer, BatchData, generate_random_batch};

// 使用单层 Autodiff 包装 - 模型使用 Backend trait，只在训练时需要 AutodiffBackend
type Backend = Autodiff<NdArray<f32>>;

#[derive(Debug, Parser)]
#[command(author, version, about = "HOPE Model Training CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Train the HOPE model
    Train(TrainArgs),
    /// Evaluate the model (placeholder)
    Eval(EvalArgs),
}

#[derive(Debug, Args)]
struct TrainArgs {
    /// Path to configuration JSON file
    #[arg(long)]
    config: PathBuf,
}

#[derive(Debug, Args)]
struct EvalArgs {
    /// Path to model checkpoint
    #[arg(long)]
    checkpoint: PathBuf,
    /// Path to evaluation data
    #[arg(long)]
    data: PathBuf,
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Train(args) => train_command(args),
        Commands::Eval(args) => {
            info!("Evaluation not yet implemented: {:?}", args);
            Ok(())
        }
    }
}

fn train_command(args: TrainArgs) -> Result<()> {
    info!("Loading configuration from: {:?}", args.config);
    
    let config_str = fs::read_to_string(&args.config)
        .with_context(|| format!("Failed to read config file: {:?}", args.config))?;
    
    let train_config: TrainConfig = serde_json::from_str(&config_str)
        .with_context(|| "Failed to parse config JSON")?;

    info!("Configuration loaded successfully");
    info!("Model config: hidden_size={}, vocab_size={}, seq_len={}", 
        train_config.model.hidden_size,
        train_config.model.vocab_size,
        train_config.model.seq_len);
    info!("Training config: batch_size={}, num_steps={}, learning_rate={}", 
        train_config.training.batch_size,
        train_config.training.num_steps,
        train_config.training.learning_rate);

    // Initialize device (CPU for now)
    let device = Default::default();

    // Check if we should resume from a checkpoint
    let (model, start_step) = if let Some(ref checkpoint_path) = train_config.training.resume_from {
        info!("Resuming training from checkpoint: {:?}", checkpoint_path);
        let (loaded_model, step, loaded_config) = load_checkpoint::<Backend>(checkpoint_path, &device)
            .with_context(|| "Failed to load checkpoint")?;
        
        // Verify configs are compatible (optional, could be relaxed)
        if loaded_config.model.hidden_size != train_config.model.hidden_size ||
           loaded_config.model.vocab_size != train_config.model.vocab_size {
            anyhow::bail!("Checkpoint model config doesn't match current config");
        }
        
        info!("Resumed from step {}", step);
        (loaded_model, step)
    } else {
        // List available checkpoints for information
        if let Ok(checkpoints) = list_checkpoints(&train_config.training.checkpoint_dir) {
            if !checkpoints.is_empty() {
                info!("Found {} existing checkpoint(s) in {:?}", 
                    checkpoints.len(), 
                    train_config.training.checkpoint_dir);
                info!("Latest checkpoint at step: {}", 
                    checkpoints.last().map(|(_, step, _)| *step).unwrap_or(0));
                info!("Starting new training (use resume_from to continue from checkpoint)");
            }
        }
        
        // Create model
        info!("Initializing HOPE model...");
        info!("  - Hidden size: {}", train_config.model.hidden_size);
        info!("  - Vocabulary size: {}", train_config.model.vocab_size);
        info!("  - Sequence length: {}", train_config.model.seq_len);
        info!("  - Number of levels: {}", train_config.model.num_levels);
        info!("  - Number of layers: {}", train_config.model.num_layers);
        
        let start_time = std::time::Instant::now();
        let model = HopeModel::<Backend>::new(train_config.model.clone(), &device);
        let init_duration = start_time.elapsed();
        info!("Model initialized successfully in {:.2}s", init_duration.as_secs_f64());
        
        (model, 0)
    };

    // Create trainer
    info!("Creating trainer...");
    let mut trainer = HopeTrainer::new(model, train_config.clone(), &device);
    info!("Trainer created");

    // Training loop
    info!("Starting training for {} steps...", train_config.training.num_steps);
    info!("  - Batch size: {}", train_config.training.batch_size);
    info!("  - Learning rate: {}", train_config.training.learning_rate);
    info!("  - Logging every {} steps", train_config.training.log_every);
    info!("  - Checkpoint directory: {:?}", train_config.training.checkpoint_dir);
    info!("  - Save checkpoint every {} steps", train_config.training.save_every);
    
    let mut total_loss = 0.0;
    let mut loss_count = 0;
    let training_start = std::time::Instant::now();

    for step in start_step..(start_step + train_config.training.num_steps) {
        let step_start = std::time::Instant::now();
        
        // Generate random batch data for testing
        let batch = generate_random_batch::<Backend>(
            train_config.training.batch_size,
            train_config.model.seq_len,
            train_config.model.vocab_size,
            &device,
        );

        // Use batch data directly
        let batch_data = BatchData {
            tokens: batch.tokens,
            targets: batch.targets,
        };

        // Training step
        let output = trainer.train_step(batch_data);
        let loss_data = output.loss.into_data();
        let loss_value = loss_data.to_vec::<f32>().unwrap_or_default().first().copied().unwrap_or(0.0);
        total_loss += loss_value;
        loss_count += 1;
        
        let step_duration = step_start.elapsed();

        // Logging
        if (step + 1) % train_config.training.log_every == 0 {
            let avg_loss = total_loss / loss_count as f32;
            let elapsed = training_start.elapsed();
            let steps_per_sec = (step + 1 - start_step) as f64 / elapsed.as_secs_f64();
            info!(
                "Step {}/{}: Loss = {:.6} (avg: {:.6}) | Step time: {:.3}s | Speed: {:.2} steps/s",
                step + 1,
                start_step + train_config.training.num_steps,
                loss_value,
                avg_loss,
                step_duration.as_secs_f64(),
                steps_per_sec
            );
            total_loss = 0.0;
            loss_count = 0;
        } else {
            // 每步都输出简单进度（不输出详细日志）
            eprint!(".");
            if (step + 1) % 10 == 0 {
                eprintln!(" {} steps", step + 1);
            }
        }
        
        // Save checkpoint
        if train_config.training.save_every > 0 && (step + 1) % train_config.training.save_every == 0 {
            info!("Saving checkpoint at step {}...", step + 1);
            match save_checkpoint(
                trainer.model(),
                step + 1,
                &train_config,
                &train_config.training.checkpoint_dir,
            ) {
                Ok(checkpoint_path) => {
                    info!("Checkpoint saved: {:?}", checkpoint_path);
                }
                Err(e) => {
                    warn!("Failed to save checkpoint: {}", e);
                }
            }
        }
    }
    
    // Save final checkpoint
    info!("Saving final checkpoint...");
    let final_step = start_step + train_config.training.num_steps;
    match save_checkpoint(
        trainer.model(),
        final_step,
        &train_config,
        &train_config.training.checkpoint_dir,
    ) {
        Ok(checkpoint_path) => {
            info!("Final checkpoint saved: {:?}", checkpoint_path);
        }
        Err(e) => {
            warn!("Failed to save final checkpoint: {}", e);
        }
    }
    
    let total_duration = training_start.elapsed();
    info!("Training completed in {:.2}s", total_duration.as_secs_f64());

    info!("Training completed!");
    Ok(())
}


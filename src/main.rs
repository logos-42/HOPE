mod config;
mod model;
mod training;

use anyhow::{Context, Result};
use burn::backend::Autodiff;
use burn_ndarray::NdArray;
use clap::{Args, Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use tracing::info;
use tracing_subscriber::EnvFilter;

use config::TrainConfig;
use model::HopeModel;
use training::{HopeTrainer, BatchData, generate_random_batch};

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

    // Initialize device (CPU for now)
    let device = Default::default();

    // Create model
    info!("Initializing HOPE model...");
    let model = HopeModel::<Backend>::new(train_config.model.clone(), &device);
    info!("Model initialized successfully");

    // Create trainer
    let mut trainer = HopeTrainer::new(model, train_config.clone(), &device);

    // Training loop
    info!("Starting training for {} steps...", train_config.num_steps);
    
    let mut total_loss = 0.0;
    let mut loss_count = 0;

    for step in 0..train_config.num_steps {
        // Generate random batch data for testing
        let batch = generate_random_batch::<Backend>(
            train_config.batch_size,
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

        // Logging
        if (step + 1) % train_config.log_every == 0 {
            let avg_loss = total_loss / loss_count as f32;
            info!(
                "Step {}/{}: Loss = {:.6} (avg: {:.6})",
                step + 1,
                train_config.num_steps,
                loss_value,
                avg_loss
            );
            total_loss = 0.0;
            loss_count = 0;
        }
    }

    info!("Training completed!");
    Ok(())
}


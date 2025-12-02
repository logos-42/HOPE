#!/bin/bash
# Training script for HOPE model

set -e

echo "Starting HOPE model training..."

# Build the project first
echo "Building project..."
cargo build --release

# Run training with example config
echo "Running training..."
cargo run --release --bin hope-train -- train --config examples/config_hope.json

echo "Training completed!"


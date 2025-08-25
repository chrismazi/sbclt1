#!/usr/bin/env python3
"""
Main training script for SBCLT: Sparse Bayesian Cross-Lingual Transformer

This script provides a clean interface to train the SBCLT model with:
- Professional logging and monitoring
- Reproducible experiments
- Advanced training features
- Research-grade output

Usage:
    python train.py [--config CONFIG] [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.trainer import SBCLTTrainer
from src.config import ModelConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SBCLT model for Kinyarwanda-English translation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to custom configuration file (optional)"
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data",
        help="Directory containing training data"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs",
        help="Directory for saving outputs and checkpoints"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = ModelConfig()
    config.seed = args.seed
    
    # Validate paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("SBCLT: Sparse Bayesian Cross-Lingual Transformer")
    print("Professional Training Pipeline")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {config.seed}")
    print(f"Model parameters: {config.d_model}D, {config.num_layers}L, {config.num_heads}H")
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = SBCLTTrainer(config, str(data_dir), str(output_dir))
        
        # Start training
        trainer.train()
        
        print("=" * 60)
        print("Training completed successfully!")
        print(f"Best BLEU score: {trainer.best_bleu:.2f}")
        print(f"Outputs saved to: {output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

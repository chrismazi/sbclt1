"""
Logging utilities for SBCLT model training.

This module provides:
- Structured logging setup
- Training metrics logging
- Experiment tracking utilities
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def setup_logging(log_file: str, level: int = logging.INFO) -> None:
    """
    Setup professional logging configuration.
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Disable propagation to avoid duplicate logs
    root_logger.propagate = False


def log_metrics(metrics: Dict[str, Any], log_file: str = None) -> None:
    """
    Log training metrics in a structured format.
    
    Args:
        metrics: Dictionary of metrics to log
        log_file: Optional log file path
    """
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "metrics": metrics
    }
    
    # Log to console
    logging.info(f"Metrics: {json.dumps(log_entry, indent=2)}")
    
    # Optionally save to file
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')


def log_experiment_config(config: Any, output_dir: str) -> None:
    """
    Log experiment configuration for reproducibility.
    
    Args:
        config: Configuration object
        output_dir: Output directory for logs
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_file = output_path / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    
    # Log configuration
    logging.info(f"Experiment configuration saved to: {config_file}")
    logging.info(f"Configuration: {json.dumps(config.__dict__, indent=2, default=str)}")


def log_training_summary(
    best_bleu: float,
    total_epochs: int,
    output_dir: str
) -> None:
    """
    Log training summary at the end of training.
    
    Args:
        best_bleu: Best BLEU score achieved
        total_epochs: Total epochs trained
        output_dir: Output directory for logs
    """
    summary = {
        "best_bleu": best_bleu,
        "total_epochs": total_epochs,
        "training_completed": True,
        "completion_time": datetime.now().isoformat()
    }
    
    # Save summary
    output_path = Path(output_dir)
    summary_file = output_path / "training_summary.json"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Log summary
    logging.info("=" * 60)
    logging.info("TRAINING SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Best BLEU Score: {best_bleu:.2f}")
    logging.info(f"Total Epochs: {total_epochs}")
    logging.info(f"Training completed successfully!")
    logging.info(f"Summary saved to: {summary_file}")
    logging.info("=" * 60)

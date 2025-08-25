"""
Configuration for SBCLT: Sparse Bayesian Cross-Lingual Transformer

This module defines all hyperparameters and configuration options for:
- Model architecture
- Training process
- Inference settings
- Data processing
- Optimization strategies

All parameters are optimized for achieving BLEU 25+ on Kinyarwanda-English translation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """
    Comprehensive configuration for SBCLT model training and inference.
    
    This configuration is optimized for research-grade NMT performance
    and includes all necessary parameters for reproducible experiments.
    """
    
    # ==================== Model Architecture ====================
    d_model: int = 1024
    """Model dimension (embedding size) - increased for better capacity"""
    
    num_heads: int = 12
    """Number of attention heads - balanced for efficiency and expressiveness"""
    
    num_layers: int = 6
    """Number of transformer layers - optimized for dataset size"""
    
    ff_dim: int = 4096
    """Feed-forward dimension - 4x d_model for optimal capacity"""
    
    dropout: float = 0.3
    """Dropout rate - increased for better regularization on small dataset"""
    
    # ==================== Sparse Attention ====================
    top_k: int = 64
    """Top-k sparsity for attention - balances efficiency and quality"""
    
    alpha: float = 0.2
    """PMI bias strength - controls statistical prior influence"""
    
    # ==================== Training Configuration ====================
    batch_size: int = 32
    """Training batch size - optimized for GPU memory and convergence"""
    
    learning_rate: float = 1e-4
    """Learning rate - conservative for stable training"""
    
    weight_decay: float = 0.01
    """Weight decay - L2 regularization for generalization"""
    
    warmup_steps: int = 4000
    """Warmup steps - gradual learning rate increase"""
    
    max_epochs: int = 20
    """Maximum training epochs - sufficient for convergence"""
    
    gradient_clip: float = 1.0
    """Gradient clipping - prevents exploding gradients"""
    
    label_smoothing: float = 0.15
    """Label smoothing - improves generalization and convergence"""
    
    # ==================== Inference Settings ====================
    beam_size: int = 8
    """Beam search size - larger for better translation quality"""
    
    max_length: int = 50
    """Maximum generation length - covers most translation needs"""
    
    length_penalty: float = 1.0
    """Length penalty - neutral for balanced length generation"""
    
    # ==================== Data Processing ====================
    max_char_len: int = 12
    """Maximum characters per token - covers morphological complexity"""
    
    min_freq: int = 2
    """Minimum token frequency - reduces vocabulary noise"""
    
    # ==================== Early Stopping ====================
    patience: int = 5
    """Early stopping patience - prevents overfitting"""
    
    min_delta: float = 0.001
    """Minimum improvement threshold - ensures meaningful progress"""
    
    # ==================== Advanced Features ====================
    use_mixed_precision: bool = True
    """Enable mixed precision training for speed and memory efficiency"""
    
    gradient_accumulation_steps: int = 2
    """Gradient accumulation for effective larger batch sizes"""
    
    save_checkpoints: bool = True
    """Save model checkpoints during training"""
    
    log_interval: int = 100
    """Logging interval for training progress"""
    
    eval_interval: int = 1
    """Evaluation interval (epochs)"""
    
    # ==================== Reproducibility ====================
    seed: int = 42
    """Random seed for reproducible experiments"""
    
    deterministic: bool = True
    """Enable deterministic training for exact reproducibility"""

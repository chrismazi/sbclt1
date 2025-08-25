from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Model architecture
    d_model: int = 1024
    num_heads: int = 12
    num_layers: int = 6
    ff_dim: int = 4096
    dropout: float = 0.3
    
    # Sparse attention
    top_k: int = 64
    alpha: float = 0.2
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 4000
    max_epochs: int = 20
    gradient_clip: float = 1.0
    label_smoothing: float = 0.15
    
    # Inference
    beam_size: int = 8
    max_length: int = 50
    length_penalty: float = 1.0
    
    # Data
    max_char_len: int = 12
    min_freq: int = 2
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001 
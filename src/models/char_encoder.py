"""
Character-level encoder for morphological feature extraction.

This module implements a convolutional character encoder that:
- Embeds individual characters
- Extracts n-gram features using 1D convolutions
- Combines character and subword representations
- Handles variable-length character sequences
"""

import torch
import torch.nn as nn


class CharEncoder(nn.Module):
    """
    Character-level encoder for morphological feature extraction.
    
    Combines subword and character-level representations using:
    - Character embeddings
    - 1D convolutions for n-gram features
    - Adaptive pooling for variable-length sequences
    - Linear projection for final representation
    """
    
    def __init__(self, char_vocab_size: int, char_emb_dim: int, out_dim: int, max_char_len: int = 12):
        """
        Initialize the character encoder.
        
        Args:
            char_vocab_size: Number of unique characters
            char_emb_dim: Dimensionality of character embeddings
            out_dim: Output dimension (should match subword embedding dim)
            max_char_len: Maximum characters per token (pad/truncate)
        """
        super().__init__()
        
        self.char_embed = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        
        # 1D Convolution for n-gram feature extraction
        self.conv = nn.Conv1d(
            in_channels=char_emb_dim,
            out_channels=out_dim,
            kernel_size=3,  # tri-gram features
            padding=1
        )
        
        # Adaptive pooling for variable-length sequences
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Projection layer to combine char + subword representations
        self.proj = nn.Linear(out_dim * 2, out_dim)

    def forward(self, subword_emb: torch.Tensor, char_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for character encoding.
        
        Args:
            subword_emb: Subword embeddings (B, T, D)
            char_seq: Character ID sequence for each token (B, T, C)
            
        Returns:
            Final token embeddings with character and subword features merged (B, T, D)
        """
        B, T, C = char_seq.shape
        
        # Flatten batch and sequence dimensions for processing
        chars = char_seq.reshape(-1, C)
        
        # Character embeddings
        char_emb = self.char_embed(chars)  # (B*T, C, char_emb_dim)
        
        # Conv1D expects (B*T, char_emb_dim, C)
        char_emb = char_emb.transpose(1, 2)
        
        # Extract n-gram features
        conv_out = torch.relu(self.conv(char_emb))  # (B*T, out_dim, C)
        pooled = self.pool(conv_out).squeeze(-1)    # (B*T, out_dim)
        
        # Reshape back to batch format
        char_repr = pooled.view(B, T, -1)  # (B, T, out_dim)
        
        # Combine character and subword representations
        combined = torch.cat([subword_emb, char_repr], dim=-1)  # (B, T, 2*D)
        projected = self.proj(combined)  # (B, T, D)
        
        return projected

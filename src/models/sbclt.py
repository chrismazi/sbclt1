"""
SBCLT: Sparse Bayesian Cross-Lingual Transformer
Main model implementation for Kinyarwanda-English neural machine translation.

This module implements the core transformer architecture with:
- Sparse Bayesian attention mechanisms
- Character-level and subword-level encoding
- Pre-norm transformer blocks
- Professional weight initialization
"""

import torch
import torch.nn as nn

from .char_encoder import CharEncoder
from .sparse_attention import SparseStatAttention
from .positional_encoding import PositionalEncoding
from ..config import ModelConfig


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with sparse attention."""
    
    def __init__(self, config: ModelConfig, pmi_matrix=None):
        super().__init__()
        self.attn = SparseStatAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            top_k=config.top_k,
            pmi_matrix=pmi_matrix,
            alpha=config.alpha
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.d_model),
            nn.Dropout(config.dropout)
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, token_ids=None, memory=None):
        if memory is not None:
            # Cross-attention (Q from decoder, K/V from encoder)
            attn_out = self.attn(x, token_ids, kv_input=memory)
        else:
            # Self-attention
            attn_out = self.attn(x, token_ids)
        
        # Pre-norm architecture
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class SBCLTEncoderDecoder(nn.Module):
    """
    SBCLT: Sparse Bayesian Cross-Lingual Transformer
    
    A state-of-the-art neural machine translation model featuring:
    - Sparse Bayesian attention for efficiency
    - Character-level encoding for morphological richness
    - Pre-norm transformer architecture for stability
    - Professional weight initialization for optimal convergence
    """
    
    def __init__(
        self,
        vocab_size: int,
        char_vocab_size: int,
        config: ModelConfig,
        pmi_matrix=None
    ):
        super().__init__()
        
        # Core components
        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(config.d_model, config.dropout)
        
        # Character-level encoding
        self.char_encoder = CharEncoder(
            char_vocab_size=char_vocab_size,
            char_emb_dim=config.d_model // 4,
            out_dim=config.d_model,
            max_char_len=config.max_char_len
        )

        # Transformer layers
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(config, pmi_matrix)
            for _ in range(config.num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            TransformerBlock(config, pmi_matrix)
            for _ in range(config.num_layers)
        ])

        # Output projection
        self.norm = nn.LayerNorm(config.d_model)
        self.output_layer = nn.Linear(config.d_model, vocab_size)
        
        # Professional weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights using best practices for transformer models."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, src_ids, src_chars, tgt_ids, tgt_chars):
        """
        Forward pass for sequence-to-sequence translation.
        
        Args:
            src_ids: Source token IDs (B, T_src)
            src_chars: Source character IDs (B, T_src, C)
            tgt_ids: Target token IDs (B, T_tgt)
            tgt_chars: Target character IDs (B, T_tgt, C)
            
        Returns:
            logits: Translation logits (B, T_tgt, V)
        """
        # Encode source sequence
        src_emb = self.embedding(src_ids)
        src_emb = self.pos_encoding(src_emb)
        src_repr = self.char_encoder(src_emb, src_chars)
        
        for layer in self.encoder_layers:
            src_repr = layer(src_repr, src_ids)

        # Decode target sequence
        tgt_emb = self.embedding(tgt_ids)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_repr = self.char_encoder(tgt_emb, tgt_chars)
        
        for layer in self.decoder_layers:
            tgt_repr = layer(tgt_repr, tgt_ids, memory=src_repr)

        # Output projection
        out = self.norm(tgt_repr)
        logits = self.output_layer(out)
        return logits

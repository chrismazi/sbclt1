"""
Vocabulary utilities for SBCLT model.

This module provides functions for:
- Loading and managing vocabularies
- Creating character vocabularies
- Vocabulary statistics and analysis
"""

import json
from pathlib import Path
from typing import Dict, List


def load_vocab(vocab_path: str) -> Dict[str, int]:
    """
    Load vocabulary from SentencePiece vocabulary file.
    
    Args:
        vocab_path: Path to the .vocab file
        
    Returns:
        Dictionary mapping tokens to IDs
    """
    vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    token = parts[0]
                    try:
                        token_id = int(parts[1])
                        vocab[token] = token_id
                    except ValueError:
                        continue
    
    return vocab


def create_char_vocab() -> Dict[str, int]:
    """
    Create character vocabulary for morphological encoding.
    
    Returns:
        Dictionary mapping characters to IDs
    """
    # Basic Latin characters
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Digits and punctuation
    chars += "0123456789.,?!-_:;'\""
    
    # Special tokens
    char_vocab = {"<unk>": 0, "<pad>": 1}
    
    # Add character mappings
    for i, char in enumerate(chars):
        char_vocab[char] = i + 2
    
    return char_vocab


def get_vocab_stats(vocab: Dict[str, int]) -> Dict[str, int]:
    """
    Get vocabulary statistics.
    
    Args:
        vocab: Vocabulary dictionary
        
    Returns:
        Dictionary with vocabulary statistics
    """
    stats = {
        "total_tokens": len(vocab),
        "unk_id": vocab.get("<unk>", -1),
        "pad_id": vocab.get("<pad>", -1),
        "bos_id": vocab.get("<s>", -1),
        "eos_id": vocab.get("</s>", -1)
    }
    
    return stats


def save_vocab(vocab: Dict[str, int], output_path: str):
    """
    Save vocabulary to file.
    
    Args:
        vocab: Vocabulary dictionary
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\t{token_id}\n")

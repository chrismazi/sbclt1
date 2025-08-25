"""
Dataset and data loading utilities for SBCLT model.

This module provides:
- Translation dataset class with noise injection
- Professional data collation functions
- Data preprocessing and augmentation utilities
"""

import torch
from torch.utils.data import Dataset
import random
from typing import List, Tuple, Dict
from pathlib import Path


class TranslationDataset(Dataset):
    """
    Professional dataset for Kinyarwanda-English translation.
    
    Features:
    - Parallel sentence pairs
    - Character-level encoding
    - Word dropout for robustness
    - Professional error handling
    """
    
    def __init__(
        self, 
        src_file: str, 
        tgt_file: str, 
        vocab: Dict[str, int], 
        char_vocab: Dict[str, int], 
        max_char_len: int = 12,
        word_dropout: float = 0.0
    ):
        """
        Initialize the translation dataset.
        
        Args:
            src_file: Source language file path
            tgt_file: Target language file path
            vocab: Token vocabulary
            char_vocab: Character vocabulary
            max_char_len: Maximum characters per token
            word_dropout: Probability of replacing words with <unk>
        """
        self.src_file = Path(src_file)
        self.tgt_file = Path(tgt_file)
        self.vocab = vocab
        self.char_vocab = char_vocab
        self.max_char_len = max_char_len
        self.word_dropout = word_dropout
        
        # Load data
        self._load_data()
        
        # Validate data
        self._validate_data()
    
    def _load_data(self):
        """Load source and target data files."""
        try:
            with open(self.src_file, 'r', encoding='utf-8') as f:
                self.src_lines = f.read().splitlines()
            
            with open(self.tgt_file, 'r', encoding='utf-8') as f:
                self.tgt_lines = f.read().splitlines()
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data file not found: {e}")
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(f"Encoding error in data file: {e}")
    
    def _validate_data(self):
        """Validate that source and target have matching lengths."""
        if len(self.src_lines) != len(self.tgt_lines):
            raise ValueError(
                f"Source and target files have different lengths: "
                f"src={len(self.src_lines)}, tgt={len(self.tgt_lines)}"
            )
        
        # Check for empty lines
        empty_src = sum(1 for line in self.src_lines if not line.strip())
        empty_tgt = sum(1 for line in self.tgt_lines if not line.strip())
        
        if empty_src > 0 or empty_tgt > 0:
            print(f"Warning: Found {empty_src} empty source lines and {empty_tgt} empty target lines")
    
    def __len__(self) -> int:
        """Return the number of sentence pairs."""
        return len(self.src_lines)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx: Index of the sentence pair
            
        Returns:
            Tuple of (src_ids, src_chars, tgt_ids, tgt_chars)
        """
        src_tokens = self.src_lines[idx].split()
        tgt_tokens = self.tgt_lines[idx].split()
        
        # Apply word dropout to source
        if self.word_dropout > 0:
            src_tokens = [
                tok if random.random() > self.word_dropout else '<unk>' 
                for tok in src_tokens
            ]
        
        # Convert tokens to IDs
        src_ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in src_tokens]
        tgt_ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tgt_tokens]
        
        # Convert to character IDs
        src_chars = self._tokens_to_chars(src_tokens)
        tgt_chars = self._tokens_to_chars(tgt_tokens)
        
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(src_chars, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
            torch.tensor(tgt_chars, dtype=torch.long)
        )
    
    def _tokens_to_chars(self, tokens: List[str]) -> List[List[int]]:
        """
        Convert tokens to character ID sequences.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of character ID sequences
        """
        char_sequences = []
        
        for token in tokens:
            # Convert token to character IDs
            chars = [self.char_vocab.get(c, self.char_vocab["<unk>"]) for c in list(token)]
            
            # Truncate or pad to max_char_len
            if len(chars) > self.max_char_len:
                chars = chars[:self.max_char_len]
            else:
                chars += [self.char_vocab["<pad>"]] * (self.max_char_len - len(chars))
            
            char_sequences.append(chars)
        
        return char_sequences


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for batching translation data.
    
    Args:
        batch: List of (src_ids, src_chars, tgt_ids, tgt_chars) tuples
        
    Returns:
        Batched tensors with proper padding
    """
    src_ids, src_chars, tgt_ids, tgt_chars = zip(*batch)
    
    # Find maximum lengths
    max_src_len = max(len(ids) for ids in src_ids)
    max_tgt_len = max(len(ids) for ids in tgt_ids)
    
    # Pad sequences
    padded_src_ids = _pad_sequences(src_ids, max_src_len, pad_value=0)
    padded_tgt_ids = _pad_sequences(tgt_ids, max_tgt_len, pad_value=0)
    
    # Pad character sequences
    padded_src_chars = _pad_char_sequences(src_chars, max_src_len)
    padded_tgt_chars = _pad_char_sequences(tgt_chars, max_tgt_len)
    
    return (
        padded_src_ids,
        padded_src_chars,
        padded_tgt_ids,
        padded_tgt_chars
    )


def _pad_sequences(sequences: List[torch.Tensor], max_len: int, pad_value: int = 0) -> torch.Tensor:
    """
    Pad a list of sequences to the same length.
    
    Args:
        sequences: List of 1D tensors
        max_len: Target length
        pad_value: Value to use for padding
        
    Returns:
        Padded tensor of shape (batch_size, max_len)
    """
    padded = []
    
    for seq in sequences:
        if len(seq) < max_len:
            # Pad with pad_value
            padding = torch.full((max_len - len(seq),), pad_value, dtype=seq.dtype)
            padded_seq = torch.cat([seq, padding])
        else:
            padded_seq = seq
        
        padded.append(padded_seq)
    
    return torch.stack(padded)


def _pad_char_sequences(char_sequences: List[List[List[int]]], max_len: int) -> torch.Tensor:
    """
    Pad character sequences to the same length.
    
    Args:
        char_sequences: List of character ID sequences
        max_len: Target sequence length
        
    Returns:
        Padded tensor of shape (batch_size, max_len, max_char_len)
    """
    padded = []
    
    for char_seq in char_sequences:
        if len(char_seq) < max_len:
            # Pad with empty character sequences
            padding = [[0] * len(char_seq[0])] * (max_len - len(char_seq))
            padded_seq = char_seq + padding
        else:
            padded_seq = char_seq
        
        padded.append(padded_seq)
    
    return torch.tensor(padded, dtype=torch.long)

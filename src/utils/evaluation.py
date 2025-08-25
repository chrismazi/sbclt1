"""
Evaluation utilities for SBCLT model.

This module provides functions for:
- BLEU score calculation
- Translation quality assessment
- Model evaluation pipeline
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List
import logging

from ..models.beam_search import beam_search
from ..data.dataset import TranslationDataset, collate_batch


def evaluate_bleu(
    model: torch.nn.Module,
    dataloader: DataLoader,
    vocab: Dict[str, int],
    char_vocab: Dict[str, int],
    config,
    device: torch.device
) -> float:
    """
    Evaluate model performance using BLEU score.
    
    Args:
        model: Trained SBCLT model
        dataloader: Validation data loader
        vocab: Token vocabulary
        char_vocab: Character vocabulary
        config: Model configuration
        device: Device to run evaluation on
        
    Returns:
        BLEU score (0-100)
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except ImportError:
        logging.error("NLTK not installed. Please install: pip install nltk")
        return 0.0
    
    model.eval()
    refs = []
    hyps = []
    
    with torch.no_grad():
        for src_ids, src_chars, tgt_ids, tgt_chars in dataloader:
            src_ids, src_chars = src_ids.to(device), src_chars.to(device)
            
            # Use beam search for decoding
            pred_ids = []
            for i in range(src_ids.size(0)):
                pred = beam_search(
                    model=model,
                    src_ids=src_ids[i:i+1],
                    src_chars=src_chars[i:i+1],
                    vocab=vocab,
                    char_vocab=char_vocab,
                    config=config,
                    device=device
                )
                pred_ids.append(pred)
            
            # Convert predictions and references to tokens
            for pred, tgt in zip(pred_ids, tgt_ids):
                pred_tokens = [k for k, v in vocab.items() if v in pred and v != 0]
                tgt_tokens = [k for k, v in vocab.items() if v in tgt.tolist() and v != 0]
                
                hyps.append(pred_tokens)
                refs.append([tgt_tokens])
    
    # Calculate BLEU score
    smoothie = SmoothingFunction().method4
    bleu = corpus_bleu(refs, hyps, smoothing_function=smoothie) * 100
    
    return bleu


def generate_translations(
    model: torch.nn.Module,
    src_sentences: List[str],
    vocab: Dict[str, int],
    char_vocab: Dict[str, int],
    config,
    device: torch.device,
    sp_model=None
) -> List[str]:
    """
    Generate translations for a list of source sentences.
    
    Args:
        model: Trained SBCLT model
        src_sentences: List of source sentences
        vocab: Token vocabulary
        char_vocab: Character vocabulary
        config: Model configuration
        device: Device to run inference on
        sp_model: SentencePiece model for detokenization
        
    Returns:
        List of translated sentences
    """
    model.eval()
    translations = []
    
    with torch.no_grad():
        for src_sentence in src_sentences:
            # Tokenize source sentence
            if sp_model:
                src_tokens = sp_model.encode(src_sentence, out_type=str)
            else:
                src_tokens = src_sentence.split()
            
            # Convert to IDs
            src_ids = [vocab.get(tok, vocab["<unk>"]) for tok in src_tokens]
            src_chars = []
            
            for token in src_tokens:
                chars = [char_vocab.get(c, char_vocab["<unk>"]) for c in list(token)]
                chars += [char_vocab["<pad>"]] * (config.max_char_len - len(chars))
                src_chars.append(chars)
            
            # Convert to tensors
            src_ids = torch.tensor([src_ids], device=device)
            src_chars = torch.tensor([src_chars], device=device)
            
            # Generate translation
            pred = beam_search(
                model=model,
                src_ids=src_ids,
                src_chars=src_chars,
                vocab=vocab,
                char_vocab=char_vocab,
                config=config,
                device=device
            )
            
            # Convert to tokens
            pred_tokens = [k for k, v in vocab.items() if v in pred and v != 0]
            
            # Detokenize
            if sp_model:
                translation = sp_model.decode(pred_tokens)
            else:
                translation = " ".join(pred_tokens)
            
            translations.append(translation)
    
    return translations

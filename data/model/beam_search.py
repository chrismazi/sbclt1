import torch
import torch.nn.functional as F
from typing import List, Tuple
from dataclasses import dataclass
from config import ModelConfig

@dataclass
class BeamHypothesis:
    tokens: List[int]
    score: float
    char_ids: List[List[int]]

def beam_search(
    model,
    src_ids: torch.Tensor,
    src_chars: torch.Tensor,
    vocab: dict,
    char_vocab: dict,
    config: ModelConfig,
    device: torch.device
) -> List[int]:
    """
    Beam search decoding for the translation model.
    """
    batch_size = src_ids.size(0)
    vocab_size = len(vocab)
    
    # Initialize beam
    beam = [BeamHypothesis(
        tokens=[vocab["<s>"]],
        score=0.0,
        char_ids=[[char_vocab.get(c, 0) for c in list("<s>")[:config.max_char_len]] + [0] * (config.max_char_len - 1)]
    ) for _ in range(config.beam_size)]
    
    # Expand source tensors for beam search
    src_ids = src_ids.repeat(config.beam_size, 1)
    src_chars = src_chars.repeat(config.beam_size, 1, 1)
    
    for t in range(config.max_length - 1):
        candidates = []
        
        # Process each hypothesis in the beam
        for hyp in beam:
            if hyp.tokens[-1] == vocab["</s>"]:
                candidates.append(hyp)
                continue
                
            # Prepare input tensors
            tgt_ids = torch.tensor([hyp.tokens], device=device)
            tgt_chars = torch.tensor([hyp.char_ids], device=device)
            
            # Get model predictions
            with torch.no_grad():
                logits = model(src_ids[:1], src_chars[:1], tgt_ids, tgt_chars)
                log_probs = F.log_softmax(logits[:, -1], dim=-1)
            
            # Get top k candidates
            topk_probs, topk_ids = torch.topk(log_probs[0], config.beam_size)
            
            for prob, token_id in zip(topk_probs, topk_ids):
                token_str = list(vocab.keys())[list(vocab.values()).index(token_id.item())]
                char_ids = [char_vocab.get(c, 0) for c in list(token_str)[:config.max_char_len]]
                char_ids += [0] * (config.max_char_len - len(char_ids))
                
                new_hyp = BeamHypothesis(
                    tokens=hyp.tokens + [token_id.item()],
                    score=hyp.score + prob.item(),
                    char_ids=hyp.char_ids + [char_ids]
                )
                candidates.append(new_hyp)
        
        # Select top beam_size candidates
        candidates.sort(key=lambda x: x.score / len(x.tokens) ** config.length_penalty, reverse=True)
        beam = candidates[:config.beam_size]
        
        # Check if all hypotheses end with </s>
        if all(hyp.tokens[-1] == vocab["</s>"] for hyp in beam):
            break
    
    # Return best hypothesis
    best_hyp = max(beam, key=lambda x: x.score / len(x.tokens) ** config.length_penalty)
    return best_hyp.tokens 
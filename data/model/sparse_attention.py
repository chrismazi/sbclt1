import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseStatAttention(nn.Module):
    def __init__(self, d_model, num_heads, top_k=32, pmi_matrix=None, alpha=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_head = d_model // num_heads
        self.h = num_heads
        self.k = top_k
        self.alpha = alpha

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.pmi_matrix = pmi_matrix  # precomputed PMI tensor (V, V)

    def forward(self, x, token_ids=None, kv_input=None):
        """
        Args:
            x (B, T_q, D): Query input embeddings (decoder or encoder input)
            token_ids (B, T_q): Query token IDs for PMI (optional)
            kv_input (B, T_kv, D): Key/Value input embeddings (encoder memory for decoder)
        Returns:
            (B, T_q, D): Output tensor after sparse PMI-biased attention
        """
        B, T_q, D = x.size()
        H = self.h
        d_h = self.d_head

        Q = self.W_Q(x).view(B, T_q, H, d_h).transpose(1, 2)  # (B, H, T_q, d_h)

        if kv_input is None:
            K = self.W_K(x).view(B, T_q, H, d_h).transpose(1, 2)  # (B, H, T_q, d_h)
            V = self.W_V(x).view(B, T_q, H, d_h).transpose(1, 2)  # (B, H, T_q, d_h)
            token_ids_k = token_ids
        else:
            T_kv = kv_input.size(1)
            K = self.W_K(kv_input).view(B, T_kv, H, d_h).transpose(1, 2)  # (B, H, T_kv, d_h)
            V = self.W_V(kv_input).view(B, T_kv, H, d_h).transpose(1, 2)  # (B, H, T_kv, d_h)
            token_ids_k = None  # PMI bias skipped for cross-attn unless added for both sides

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_h ** 0.5)  # (B, H, T_q, T_kv)

        # Add PMI bias if provided and not in cross-attention
        if self.pmi_matrix is not None and token_ids is not None and token_ids_k is None and self.alpha > 0.0:
            PMI = self.pmi_matrix[token_ids.unsqueeze(2), token_ids.unsqueeze(1)]  # (B, T_q, T_q)
            PMI = PMI.unsqueeze(1).expand(-1, H, -1, -1)
            scores = scores + self.alpha * PMI

        k = min(self.k, scores.size(-1))
        topk_vals, topk_idx = torch.topk(scores, k, dim=-1)

        mask = torch.full_like(scores, float('-inf'))
        b_idx = torch.arange(B).view(B, 1, 1, 1).expand(B, H, T_q, k)
        h_idx = torch.arange(H).view(1, H, 1, 1).expand(B, H, T_q, k)
        t_idx = torch.arange(T_q).view(1, 1, T_q, 1).expand(B, H, T_q, k)

        mask[b_idx, h_idx, t_idx, topk_idx] = topk_vals

        attn_weights = F.softmax(mask, dim=-1)  # (B, H, T_q, T_kv)
        out = torch.matmul(attn_weights, V)     # (B, H, T_q, d_h)
        out = out.transpose(1, 2).contiguous().view(B, T_q, D)  # (B, T_q, D)

        return self.W_O(out)

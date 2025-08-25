import torch
import torch.nn as nn

class CharEncoder(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, out_dim, max_char_len=12):
        """
        Args:
            char_vocab_size (int): Number of unique characters.
            char_emb_dim (int): Dimensionality of character embeddings.
            out_dim (int): Output dimension (should match subword emb dim).
            max_char_len (int): Max characters per token (pad/truncate).
        """
        super(CharEncoder, self).__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        
        # 1D Convolution over characters
        self.conv = nn.Conv1d(in_channels=char_emb_dim,
                              out_channels=out_dim,
                              kernel_size=3,  # tri-gram features
                              padding=1)

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Linear(out_dim * 2, out_dim)  # combine char + subword

    def forward(self, subword_emb, char_seq):
        """
        Args:
            subword_emb (B, T, D): Subword embeddings
            char_seq (B, T, C): Character ID sequence for each token

        Returns:
            (B, T, D): Final token embeddings with char + subword merged
        """
        B, T, C = char_seq.shape
        # Flatten: (B*T, C)
        chars = char_seq.reshape(-1, C)

        # Embed: (B*T, C, char_emb_dim)
        char_emb = self.char_embed(chars)

        # Conv1D expects (B*T, char_emb_dim, C)
        char_emb = char_emb.transpose(1, 2)

        conv_out = torch.relu(self.conv(char_emb))  # (B*T, out_dim, C)
        pooled = self.pool(conv_out).squeeze(-1)    # (B*T, out_dim)

        char_repr = pooled.view(B, T, -1)           # (B, T, out_dim)

        # Concatenate with subword
        combined = torch.cat([subword_emb, char_repr], dim=-1)  # (B, T, 2*D)
        projected = self.proj(combined)                         # (B, T, D)

        return projected

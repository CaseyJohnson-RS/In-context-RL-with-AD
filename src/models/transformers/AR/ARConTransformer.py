import torch
import torch.nn as nn
from torch import Tensor

from src.layers import PositionalEncoding
from .ARTransformerBase import ARTransformerBase


class ARConTransformer(ARTransformerBase):
    """
    Autoregressive transformer with interleaved action-reward tokens.
    Creates doubled sequence [a1, r1, a2, r2, ..., aL, rL] for explicit 
    action-reward pairing in attention.
    
    Architecture:
        actions -> Embedding(n_actions→d_model)
        rewards -> Linear(1→d_model)
        Interleave [action_emb, reward_emb] -> (B, 2L, d_model)
        -> PosEnc(2*max_len) -> Causal Transformer -> head(-2)
    
    Args:
        Same as ARTransformerBase; max_len is action steps (sequence 2*max_len).
    """
    
    def __init__(
        self,
        n_actions: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        max_len: int = 200,
        dropout: float = 0.1,
    ) -> None:
        # Double max_len for interleaved sequence
        super().__init__(
            n_actions, d_model, nhead, num_layers, 
            dim_feedforward, max_len * 2, dropout
        )
        
        self.action_emb = nn.Embedding(n_actions, d_model)
        self.reward_emb = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len * 2)
    
    def sequence_encode(self, actions: Tensor, rewards: Tensor) -> Tensor:
        """
        Interleave action/reward embeddings into doubled sequence.
        
        Args:
            actions: (B, L) or (L,) action indices [0, n_actions).
            rewards: (B, L) or (L,) reward scalars.
            
        Returns:
            Interleaved embeddings (B, 2L, d_model): [a1,r1,a2,r2,...].
        """
        # Batch normalization
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
            rewards = rewards.unsqueeze(0)
        
        B, L = actions.shape
        
        # Embeddings
        a_emb = self.action_emb(actions.long())      # (B, L, d_model)
        r_emb = self.reward_emb(rewards.unsqueeze(-1))  # (B, L, 1) -> (B, L, d_model)
        
        # Interleave: [a1, r1, a2, r2, ...]
        interleaved = torch.stack([a_emb, r_emb], dim=2)  # (B, L, 2, d_model)
        x = interleaved.reshape(B, L * 2, self.d_model)   # (B, 2L, d_model)
        
        # Positional encoding (handles 2*max_len)
        x = self.pos_enc(x)
        
        return x
    
    def decode_action(self, x: Tensor) -> Tensor:
        """
        Predict next action from last action token (position -2).
        
        Args:
            x: (B, 2L, d_model) transformer output.
            
        Returns:
            Logits (B, n_actions) from final action embedding.
        """
        # Sequence: [... a_L-1, r_L-1, a_L, r_L]
        # Predict from a_L (index -2)
        last_action_emb = x[:, -2]
        return self.action_head(last_action_emb)

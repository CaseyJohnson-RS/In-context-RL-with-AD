import torch
import torch.nn as nn
from torch import Tensor

from src.layers import PositionalEncoding
from .ARTransformerBase import ARTransformerBase


class ARMulTransformer(ARTransformerBase):
    """
    Autoregressive transformer encoding actions as scaled one-hot vectors.
    One-hot actions multiplied by rewards, projected to embeddings, then transformed.
    
    Architecture:
        actions/rewards -> one-hot * rewards -> Linear(n_actions→d_model) 
        -> PosEnc -> Causal Transformer -> LN -> Linear(d_model→n_actions)
    
    Args:
        Same as ARTransformerBase.
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
        super().__init__(
            n_actions, d_model, nhead, num_layers, 
            dim_feedforward, max_len, dropout
        )
        
        # Action embedding: one-hot rewards -> d_model
        self.action_emb = nn.Linear(n_actions, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
    
    def sequence_encode(self, actions: Tensor, rewards: Tensor) -> Tensor:
        """
        Encode action/reward sequence via reward-weighted one-hot.
        
        Args:
            actions: (B, L) or (L,) action indices [0, n_actions).
            rewards: (B, L) or (L,) reward scalars.
            
        Returns:
            Embeddings (B, L, d_model).
        """
        # Ensure 3D: (B, L) -> (B, L, 1) for broadcast
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
            rewards = rewards.unsqueeze(0)
        
        B, L = actions.shape
        
        # One-hot: (B, L, n_actions)
        one_hot = torch.zeros(B, L, self.n_actions, device=actions.device, dtype=torch.float)
        one_hot.scatter_(2, actions.unsqueeze(-1).long(), 1.0)
        
        # Reward scaling: (B, L, n_actions)
        weighted = one_hot * rewards.unsqueeze(-1)
        
        # Project to embeddings: (B, L, d_model)
        x = self.action_emb(weighted)
        
        # Positional encoding (assuming src.layers.PositionalEncoding)
        x = self.pos_enc(x)
        
        return x
    
    def decode_action(self, x: Tensor) -> Tensor:
        """
        Extract final timestep for next-action prediction.
        
        Args:
            x: (B, L, d_model) transformer output.
            
        Returns:
            Logits (B, n_actions).
        """
        # Last position only (autoregressive next-action)
        last_x = x[:, -1]
        return self.action_head(last_x)

import torch.nn as nn
from torch import Tensor

from src.layers import PositionalEncoding  # type: ignore[import]
from .ARTransformerBase import ARTransformerBase


class ARAddTransformer(ARTransformerBase):
    """
    Autoregressive transformer fusing action+reward via element-wise addition.
    Creates compact sequence where each timestep embeds action_t + reward_t.
    
    Architecture:
        actions -> Embedding(n_actions→d_model)
        rewards -> Linear(1→d_model)
        Add: action_emb + reward_emb -> (B, L, d_model)
        -> PosEnc -> Causal Transformer -> head(-1)
    
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
        
        self.action_emb = nn.Embedding(n_actions, d_model)
        self.reward_emb = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
    
    def sequence_encode(self, actions: Tensor, rewards: Tensor) -> Tensor:
        """
        Fuse action/reward via addition into per-step embedding.
        
        Args:
            actions: (B, L) or (L,) action indices [0, n_actions).
            rewards: (B, L) or (L,) reward scalars.
            
        Returns:
            Fused embeddings (B, L, d_model).
        """
        # Batch normalization
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
            rewards = rewards.unsqueeze(0)
        
        # Embeddings
        a_emb = self.action_emb(actions.long())         # (B, L, d_model)
        r_emb = self.reward_emb(rewards.unsqueeze(-1))  # (B, L, 1) -> (B, L, d_model)
        
        # Element-wise fusion
        x = a_emb + r_emb  # (B, L, d_model)
        
        # Positional encoding
        x = self.pos_enc(x)
        
        return x
    
    def decode_action(self, x: Tensor) -> Tensor:
        """
        Predict next action from final fused timestep.
        
        Args:
            x: (B, L, d_model) transformer output.
            
        Returns:
            Logits (B, n_actions).
        """
        # Last position (a_L + r_L)
        final_emb = x[:, -1]
        return self.action_head(final_emb)

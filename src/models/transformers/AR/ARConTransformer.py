import torch
import torch.nn as nn
from torch import Tensor

from src.layers import PositionalEncoding
from .ARTransformerBase import ARTransformerBase


class ARConTransformer(ARTransformerBase):
    """
    Autoregressive transformer with interleaved action-reward tokens.
    Predicts ONLY the next action.
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
            n_actions=n_actions,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len * 2,   # interleaved
            dropout=dropout,
        )

        self.action_emb = nn.Embedding(n_actions, d_model)
        self.reward_emb = nn.Linear(1, d_model)

        self.pos_enc = PositionalEncoding(d_model, max_len=max_len * 2)

    def sequence_encode(self, X: Tensor) -> Tensor:
        """
        Args:
            X: (B, 2, L) or (2, L)
                X[:,0] = actions (int)
                X[:,1] = rewards (float)

        Returns:
            (B, 2L, d_model)
        """
        if X.dim() == 2:
            X = X.unsqueeze(0)

        B, _, L = X.shape

        actions = X[:, 0].long()          # (B, L)
        rewards = X[:, 1].unsqueeze(-1)   # (B, L, 1)

        a_emb = self.action_emb(actions)  # (B, L, D)
        r_emb = self.reward_emb(rewards)  # (B, L, D)

        out = torch.empty(B, 2 * L, self.d_model, device=X.device)
        out[:, 0::2] = a_emb
        out[:, 1::2] = r_emb

        out = self.pos_enc(out)

        return out

    def decode_action(self, X: Tensor) -> Tensor:
        """
        Use the last ACTION token (position -2).

        Args:
            X: (B, 2L, d_model)

        Returns:
            (B, n_actions)
        """
        last_action_token = X[:, -2, :]      # (B, D)
        logits = self.action_head(last_action_token)
        return logits

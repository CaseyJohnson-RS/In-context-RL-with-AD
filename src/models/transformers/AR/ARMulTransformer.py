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
        actions/rewards -> one-hot * rewards -> Linear(n_actions → d_model)
        → PositionalEncoding → Transformer → LN → Linear(d_model → n_actions)
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
            max_len=max_len,
            dropout=dropout,
        )

        # Linear projection for reward-weighted one-hot vectors
        self.action_emb = nn.Linear(n_actions, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

    def sequence_encode(self, X: Tensor) -> Tensor:
        """
        Encode a sequence of actions and rewards as reward-weighted one-hot embeddings.

        Args:
            X: (B, 2, L) tensor, X[:, 0, :] = actions, X[:, 1, :] = rewards

        Returns:
            Tensor of shape (B, L, d_model) ready for transformer input.
        """
        if X.dim() == 2:
            X = X.unsqueeze(0)  # (1, 2, L)

        actions = X[:, 0, :]  # (B, L)
        rewards = X[:, 1, :]  # (B, L)

        B, L = actions.shape

        # One-hot encoding: (B, L, n_actions)
        one_hot = torch.zeros(B, L, self.n_actions, device=actions.device, dtype=torch.float)
        one_hot.scatter_(2, actions.unsqueeze(-1).long(), 1.0)

        # Scale by rewards: (B, L, n_actions)
        weighted = one_hot * rewards.unsqueeze(-1)

        # Project to embeddings: (B, L, d_model)
        x = self.action_emb(weighted)

        # Add positional encoding
        x = self.pos_enc(x)
        return x

    def decode_action(self, X: Tensor) -> Tensor:
        """
        Decode transformer output to next-action logits.

        Args:
            X: (B, L, d_model) transformer output

        Returns:
            Tensor of shape (B, n_actions)
        """
        return self.action_head(X[:, -1])

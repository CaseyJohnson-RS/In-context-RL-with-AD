import torch.nn as nn
from torch import Tensor

from src.layers import PositionalEncoding  # type: ignore[import]
from .ARTransformerBase import ARTransformerBase


class ARAddTransformer(ARTransformerBase):
    """
    Autoregressive transformer fusing action+reward via element-wise addition.

    Each timestep embeds `action_t + reward_t` as a single vector, then passes
    through positional encoding and a causal transformer.

    Architecture:
        actions -> Embedding(n_actions → d_model)
        rewards -> Linear(1 → d_model)
        fused: action_emb + reward_emb → (B, L, d_model)
        → PositionalEncoding → Transformer → action_head
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

        self.action_emb = nn.Embedding(n_actions, d_model)
        self.reward_emb = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

    def sequence_encode(self, X: Tensor) -> Tensor:
        """
        Encode a sequence of actions and rewards into fused embeddings.

        Args:
            X: (B, 2, L) tensor, where X[:, 0, :] = actions, X[:, 1, :] = rewards

        Returns:
            Tensor of shape (B, L, d_model) ready for transformer input.
        """
        if X.dim() == 2:
            X = X.unsqueeze(0)  # (1, 2, L)

        actions = X[:, 0, :]  # (B, L)
        rewards = X[:, 1, :]  # (B, L)

        a_emb = self.action_emb(actions.long())               # (B, L, d_model)
        r_emb = self.reward_emb(rewards.unsqueeze(-1))       # (B, L, 1) → (B, L, d_model)
        x = a_emb + r_emb                                     # fused embedding

        x = self.pos_enc(x)
        return x

    def decode_action(self, X: Tensor) -> Tensor:
        """
        Decode transformer output into next-action logits.

        Args:
            X: (B, L, d_model) transformer output

        Returns:
            Tensor of shape (B, n_actions)
        """
        return self.action_head(X[:, -1])

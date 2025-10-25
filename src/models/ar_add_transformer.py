import torch
import torch.nn as nn
from .transformer_base import TransformerBase
from src.layers import PositionalEncoding


class ARAddTransformer(TransformerBase):
    """
    Трансформер, который суммирует эмбеддинги действий и наград.
    Action и Reward кодируются в отдельные векторы, затем складываются
    для формирования единого представления каждого шага.

    Параметры
    ----------
    n_actions : int
        Количество возможных действий (рукавов бандита).
    d_model : int, default=64
        Размер эмбеддингов и скрытых состояний трансформера.
    nhead : int, default=4
        Количество голов в Multi-Head Attention.
    num_layers : int, default=4
        Количество слоёв TransformerEncoder.
    dim_feedforward : int, default=2048
        Размер скрытого слоя в feedforward-блоке трансформера.
    max_len : int, default=200
        Максимальная длина последовательности.
    """

    def __init__(
        self,
        n_actions: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        max_len: int = 200
    ):
        super().__init__(n_actions, d_model, nhead, num_layers, dim_feedforward, max_len)

        # ----------------------
        # Эмбеддинги действий и наград
        # ----------------------
        self.action_emb = nn.Embedding(n_actions, d_model)  # действия -> вектор d_model
        self.reward_emb = nn.Linear(1, d_model)             # награда -> вектор d_model

        # ----------------------
        # Positional encoding
        # ----------------------
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

    def forward(self, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Параметры
        ----------
        actions : torch.Tensor, shape (B, L)
            Индексы действий.
        rewards : torch.Tensor, shape (B, L, 1)
            Значения наград.

        Возвращает
        ----------
        logits : torch.Tensor, shape (B, n_actions)
            Предсказанные логиты для следующего действия.
        """
        B, L = actions.shape

        a_e = self.action_emb(actions)       # (B, L, D)
        r_e = self.reward_emb(rewards)       # (B, L, D)

        x = a_e + r_e  # (B, L, D)

        x = self.pos_enc(x)

        x = self.transformer(x)
        x = self.ln(x)

        # ----------------------
        # Берём последний токен действия для предсказания следующего
        # ----------------------
        logits = self.head(x[:, -1, :])  # (B, n_actions)
        return logits
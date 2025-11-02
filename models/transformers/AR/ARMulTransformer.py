import torch
import torch.nn as nn
from src.layers import PositionalEncoding
from .ARTransformerBase import ARTransformerBase


class ARMulTransformer(ARTransformerBase):
    """
    Трансформер, который кодирует действия через one-hot вектора, масштабирует их наградами
    и проецирует в пространство эмбеддингов перед подачей в Transformer.

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
        max_len: int = 200,
    ):
        super().__init__(
            n_actions, d_model, nhead, num_layers, dim_feedforward, max_len
        )

        # Линейная проекция one-hot действий в пространство эмбеддингов
        self.emb = nn.Linear(n_actions, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

    def sequence_encode(self, actions, rewards):
        B, L = actions.shape

        # One-hot кодирование действий
        x = torch.zeros((B, L, self.n_actions), device=actions.device)
        x.scatter_(2, actions.unsqueeze(-1).long(), 1.0)
        x = x * rewards  # (B, L, n_actions)

        x = self.emb(x)  # (B, L, d_model)
        x = self.pos_enc(x)

        return x
    
    def decode_action(self, x):
        # Берём последний токен для предсказания следующего
        return self.head(x[:, -1, :])  # (B, n_actions)

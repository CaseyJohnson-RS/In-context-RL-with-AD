import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class TransformerBase(nn.Module, ABC):
    """
    Базовый класс трансформера для последовательностей действий.
    Используется для предсказания следующего действия по истории.

    Параметры
    ----------
    n_actions : int
        Количество возможных действий (рукавов бандита).
    d_model : int, default=64
        Размер эмбеддингов и скрытого состояния трансформера.
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
        super().__init__()

        self.n_actions = n_actions
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_len = max_len

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # LayerNorm и линейная проекция на действия
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_actions)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass для предсказания следующего действия."""
        pass
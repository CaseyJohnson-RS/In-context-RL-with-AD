import torch
import torch.nn as nn
from abc import ABC, abstractmethod


Tensor = torch.Tensor


class ARTransformerBase(nn.Module, ABC):
    """
    Базовый класс трансформера для последовательностей действий.
    Используется для предсказания следующего действия по истории
    действий и наград (actions, rewards).

    Параметры
    ----------
    n_actions : int
        Количество возможных действий (рук бандита).
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
        max_len: int = 200,
    ):
        super().__init__()

        # Всё запоминаем
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
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # LayerNorm и линейная проекция на действия
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_actions)

    @abstractmethod
    def sequence_encode(self, actions: Tensor, rewards: Tensor) -> Tensor:
        """
        Функция для кодировки тензоров actions и rewards в 
        последовательность эмбеддингов.
        """
        pass

    @abstractmethod
    def decode_action(self, x: Tensor) -> Tensor:
        """
        Функция для определения действия. x - выход LayerNorm слоя.
        """
        pass

    def forward(self, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:

        # Кодируем последовательность
        x = self.sequence_encode(actions, rewards)

        seq_len = x.size(1)

        # Маска размера (seq_len, seq_len)
        causal_mask = torch.ones(
            (seq_len, seq_len), dtype=torch.bool, device=x.device
        ).triu_(1)

        # Для TransformerEncoder маска должна быть float и иметь значения:
        # 0.0 = разрешено смотреть
        # -inf = запрещено смотреть
        attn_mask = torch.zeros((seq_len, seq_len), device=x.device)
        attn_mask.masked_fill_(causal_mask, float("-inf"))

        # Проход через трансформер с маской
        x = self.transformer(x, mask=attn_mask)  # (B, L, D)
        x = self.ln(x)

        # Выбираем действие
        return self.decode_action(x)

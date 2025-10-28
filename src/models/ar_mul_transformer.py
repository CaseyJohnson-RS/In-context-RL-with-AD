import torch
import torch.nn as nn
from .transformer_base import TransformerBase
from src.layers import PositionalEncoding


class ARMultiplyTransformer(TransformerBase):
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

        # ----------------------
        # Линейная проекция one-hot действий в пространство эмбеддингов
        # ----------------------
        self.emb = nn.Linear(n_actions, d_model)

        # ----------------------
        # Positional encoding
        # ----------------------
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)


    def forward(self, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """
        Forward pass с каузальной маской.

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

        # ----------------------
        # One-hot кодирование действий
        # ----------------------
        x = torch.zeros((B, L, self.n_actions), device=actions.device)
        x.scatter_(2, actions.unsqueeze(-1).long(), 1.0)
        x = x * rewards  # (B, L, n_actions)

        x = self.emb(x)  # (B, L, d_model)

        # ----------------------
        # Создание каузальной маски для attention
        # ----------------------
        seq_len = x.size(1)  # L
        # В PyTorch, для маски attention: 
        # False (0) = разрешено смотреть
        # True (1) = запрещено смотреть
        # Маска размера (seq_len, seq_len)
        causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device).triu_(1)
        
        # Для TransformerEncoder маска должна быть float и иметь значения:
        # 0.0 = разрешено смотреть
        # -inf = запрещено смотреть
        attn_mask = torch.zeros((seq_len, seq_len), device=x.device)
        attn_mask.masked_fill_(causal_mask, float('-inf'))

        # ----------------------
        # Проход через трансформер с маской
        # ----------------------
        x = self.transformer(x, mask=attn_mask)  # (B, L, D)
        x = self.ln(x)  # LayerNorm

        # ----------------------
        # Берём последний токен ДЕЙСТВИЯ для предсказания следующего
        # Последний токен действия находится на позиции -1
        # ----------------------
        logits = self.head(x[:, -1, :])  # (B, n_actions)
        return logits
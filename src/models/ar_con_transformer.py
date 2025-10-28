from .transformer_base import TransformerBase
import torch
import torch.nn as nn
from src.layers import PositionalEncoding


class ARConcatTransformer(TransformerBase):
    """
    Трансформер с чередующимися токенами действия и награды.
    Action и Reward кодируются в отдельные векторы, затем конкатенируются
    для формирования удвоенной последовательности: [a_1, r_1, a_2, r_2, ...].

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
        Максимальная длина последовательности действий (не удвоенная). 
        Значение должно быть **ЧЁТНЫМ**.
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
        # Эмбеддинги
        # ----------------------
        self.action_emb = nn.Embedding(n_actions, d_model)  # действия -> вектор d_model
        self.reward_emb = nn.Linear(1, d_model)             # награда -> вектор d_model

        # ----------------------
        # Positional encoding для удвоенной последовательности
        # (чередуем action и reward)
        # длина max_len * 2
        # ----------------------
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len * 2)

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
        D = self.d_model

        # ----------------------
        # Эмбеддинги действий и наград
        # ----------------------
        a_e = self.action_emb(actions)      # (B, L, D)
        r_e = self.reward_emb(rewards)      # (B, L, D)

        # ----------------------
        # Конкатенация action-reward в чередующуюся последовательность
        # (B, L, 2, D) -> (B, 2*L, D)
        # Порядок: [a₁, r₁, a₂, r₂, ..., a_L, r_L]
        # ----------------------
        x = torch.stack([a_e, r_e], dim=2)  # (B, L, 2, D)
        x = x.reshape(B, L * 2, D)          # (B, 2L, D)

        x = self.pos_enc(x)  # Позиционная кодировка

        # ----------------------
        # Создание каузальной маски для attention
        # ----------------------
        seq_len = x.size(1)  # 2L
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
        x = self.transformer(x, mask=attn_mask)  # (B, 2L, D)
        x = self.ln(x)  # LayerNorm

        # ----------------------
        # Берём последний токен ДЕЙСТВИЯ для предсказания следующего
        # Последний токен действия находится на позиции -2 (перед последним reward)
        # ----------------------
        logits = self.head(x[:, -2, :])  # (B, n_actions)
        return logits

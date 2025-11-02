import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Позиционное кодирование для трансформера.
    Прямо как в статье "Attention is All You Need".

    Цель:
    ----------------------
    - Трансформеры не имеют внутреннего представления порядка последовательности.
    - PositionalEncoding добавляет информацию о позиции каждого элемента в последовательности,
      позволяя модели учитывать порядок действий и наград.

    Формула:
    ----------------------
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    где:
        pos - индекс позиции в последовательности (0..max_len-1)
        i   - индекс размерности (0..d_model/2)
    """

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # матрица размером (max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()  # позиции 0..max_len-1
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        # sin для четных измерений
        pe[:, 0::2] = torch.sin(pos * div)
        # cos для нечетных измерений
        pe[:, 1::2] = torch.cos(pos * div)
        # добавляем размер batch как 1-й измерение и сохраняем как буфер
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        Добавляет позиционное кодирование к входу.

        Параметры:
        ----------------------
        x: torch.Tensor, размер (B, L, D)
           B - batch size
           L - длина последовательности
           D - размерность модели

        Возвращает:
        ----------------------
        torch.Tensor, размер (B, L, D) с добавленным positional encoding
        """
        L = x.size(1)
        return x + self.pe[:, :L, :]

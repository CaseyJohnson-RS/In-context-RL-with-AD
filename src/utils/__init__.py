from .k_armer_bandit import generate_karmedbandit_traces

__all__ = [
    "generate_karmedbandit_traces"
]

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Common functions
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


import numpy as np
import random
import torch


def set_seed(seed: int) -> None:
    """
    Устанавливает фиксированное значение случайного генератора для всех используемых библиотек,
    чтобы обеспечить воспроизводимость экспериментов.

    Аргументы:
        seed (int): Целое число, используемое как зерно (seed) для генераторов случайных чисел.

    Действует на:
        - стандартный генератор случайных чисел Python
        - генератор случайных чисел NumPy
        - генератор случайных чисел PyTorch (CPU и CUDA)
    """
    random.seed(seed)  # Для стандартного модуля random Python
    np.random.seed(seed)  # Для NumPy
    torch.manual_seed(seed)  # Для PyTorch на CPU
    torch.cuda.manual_seed_all(seed)  # Для всех доступных GPU

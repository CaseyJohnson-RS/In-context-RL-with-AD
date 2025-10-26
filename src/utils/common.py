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


def batch_iterator(
    seqs: list,
    batch_size: int,
    device: str = "cpu",
    shuffle: bool = True,
):
    """
    Создаёт итератор по батчам для обучения трансформера.

    Поддерживает оба формата входных данных:
      - (actions, rewards, target)
      - (states, actions, rewards, target)

    Параметры
    ----------
    seqs : list of tuples
        Список обучающих примеров:
        (actions, rewards, target) или (states, actions, rewards, target)
    batch_size : int
        Размер батча.
    device : str или torch.device, optional
        Устройство для размещения тензоров ('cpu' или 'cuda').
    shuffle : bool, optional
        Перемешивать ли данные перед созданием батчей. По умолчанию True.

    Возвращает
    ----------
    generator
        Генератор, выдающий кортеж тензоров:
        - если вход (a, r, t): (actions_b, rewards_b, targets_b)
        - если вход (s, a, r, t): (states_b, actions_b, rewards_b, targets_b)

        где:
        actions_b : torch.LongTensor (B, L)
        rewards_b : torch.FloatTensor (B, L, 1)
        targets_b : torch.LongTensor (B,)
        states_b  : torch.LongTensor (B, L) — если присутствует
    """
    if shuffle:
        random.shuffle(seqs)

    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        first = batch[0]

        if len(first) == 3:
            # Формат (actions, rewards, target)
            actions_b = torch.tensor(
                [b[0] for b in batch], dtype=torch.long, device=device
            )
            rewards_b = torch.tensor(
                [b[1] for b in batch], dtype=torch.float32, device=device
            ).unsqueeze(-1)
            targets_b = torch.tensor(
                [b[2] for b in batch], dtype=torch.long, device=device
            )

            yield actions_b, rewards_b, targets_b

        elif len(first) == 4:
            # Формат (states, actions, rewards, target)
            states_b = torch.tensor(
                [b[0] for b in batch], dtype=torch.long, device=device
            )
            actions_b = torch.tensor(
                [b[1] for b in batch], dtype=torch.long, device=device
            )
            rewards_b = torch.tensor(
                [b[2] for b in batch], dtype=torch.float32, device=device
            ).unsqueeze(-1)
            targets_b = torch.tensor(
                [b[3] for b in batch], dtype=torch.long, device=device
            )

            yield states_b, actions_b, rewards_b, targets_b

        else:
            raise ValueError(
                f"Некорректный формат данных: ожидалось 3 или 4 элемента, получено {len(first)}."
            )

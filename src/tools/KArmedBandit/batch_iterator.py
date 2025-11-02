import random
import torch


def batch_iterator(
    seqs: list,
    batch_size: int,
    device: str = "cpu",
    shuffle: bool = True,
):
    """
    Создаёт итератор по батчам для обучения трансформера.

    Формат входных данных:
      - (actions, rewards, target)

    Параметры
    ----------
    seqs : list of tuples
        Список обучающих примеров:
        (actions, rewards, target)
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
        - (actions_b, rewards_b, targets_b)

        где:
        actions_b : torch.LongTensor (B, L)
        rewards_b : torch.FloatTensor (B, L, 1)
        targets_b : torch.LongTensor (B,)
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

        else:
            raise ValueError(
                f"Некорректный формат данных: ожидалось 3 или 4 элемента, получено {len(first)}."
            )
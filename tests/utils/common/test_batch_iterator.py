import pytest
import torch
from src.utils import batch_iterator


def make_example_data():
    """Вспомогательная функция для генерации тестовых данных."""
    seqs_3 = [
        ([1, 2, 3], [0.1, 0.2, 0.3], 2),
        ([3, 1, 2], [0.4, 0.5, 0.6], 1),
        ([2, 2, 1], [0.7, 0.8, 0.9], 3),
    ]

    seqs_4 = [
        ([10, 11, 12], [1, 2, 3], [0.1, 0.2, 0.3], 2),
        ([20, 21, 22], [2, 1, 3], [0.4, 0.5, 0.6], 1),
        ([30, 31, 32], [3, 3, 2], [0.7, 0.8, 0.9], 3),
    ]
    return seqs_3, seqs_4


# --- Тесты для формата (actions, rewards, target) --- #
def test_batch_iterator_3tuple_shapes():
    seqs_3, _ = make_example_data()
    batch_size = 2

    batches = list(batch_iterator(seqs_3, batch_size=batch_size, device="cpu", shuffle=False))
    assert len(batches) == 2  # ожидается 2 батча (3 примера → 2+1)
    for actions_b, rewards_b, targets_b in batches:
        assert actions_b.shape[1] == 3
        assert rewards_b.shape[1:] == (3, 1)
        assert targets_b.ndim == 1
        assert actions_b.dtype == torch.long
        assert rewards_b.dtype == torch.float32
        assert targets_b.dtype == torch.long


def test_batch_iterator_3tuple_content_consistency():
    seqs_3, _ = make_example_data()
    batch = next(batch_iterator(seqs_3, batch_size=2, shuffle=False))

    actions_b, rewards_b, targets_b = batch
    assert torch.allclose(rewards_b[0].squeeze(), torch.tensor([0.1, 0.2, 0.3]))
    assert targets_b.tolist() == [2, 1]


# --- Тесты для формата (states, actions, rewards, target) --- #
def test_batch_iterator_4tuple_shapes():
    _, seqs_4 = make_example_data()
    batch_size = 2

    batches = list(batch_iterator(seqs_4, batch_size=batch_size, device="cpu", shuffle=False))
    assert len(batches) == 2
    for states_b, actions_b, rewards_b, targets_b in batches:
        assert states_b.shape[1] == 3
        assert actions_b.shape[1] == 3
        assert rewards_b.shape[1:] == (3, 1)
        assert targets_b.ndim == 1
        assert states_b.dtype == torch.long
        assert actions_b.dtype == torch.long
        assert rewards_b.dtype == torch.float32


def test_batch_iterator_4tuple_content_consistency():
    _, seqs_4 = make_example_data()
    batch = next(batch_iterator(seqs_4, batch_size=1, shuffle=False))

    states_b, actions_b, rewards_b, targets_b = batch
    assert torch.equal(states_b[0], torch.tensor([10, 11, 12]))
    assert torch.equal(actions_b[0], torch.tensor([1, 2, 3]))
    assert torch.allclose(rewards_b[0].squeeze(), torch.tensor([0.1, 0.2, 0.3]))
    assert targets_b.item() == 2


# --- Общие тесты --- #
def test_batch_iterator_shuffle_does_not_crash():
    seqs_3, _ = make_example_data()
    list(batch_iterator(seqs_3, batch_size=2, shuffle=True))  # просто проверка, что работает без ошибок


def test_batch_iterator_invalid_format_raises():
    """Функция должна выбросить ошибку, если формат данных неправильный."""
    invalid_data = [([1, 2, 3], [0.1, 0.2, 0.3])]  # только 2 поля вместо 3 или 4
    with pytest.raises(ValueError):
        next(batch_iterator(invalid_data, batch_size=1))

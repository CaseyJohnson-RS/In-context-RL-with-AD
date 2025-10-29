import torch
from torch import nn
from src.utils import karmedbandit_run_in_context, set_seed


# ======== Фиктивная модель трансформера ========
class DummyTransformer(nn.Module):
    """
    Простая фиктивная модель, которая всегда возвращает фиксированные логиты.
    Позволяет тестировать karmedbandit_run_in_context без обучения.
    """
    def __init__(self, K):
        super().__init__()
        self.K = K

    def forward(self, actions, rewards):
        batch_size = actions.shape[0]
        return torch.ones((batch_size, self.K), dtype=torch.float32)


# ======== Тесты ========

def test_total_reward_type_and_range():
    """Проверяем, что функция возвращает float и награда не отрицательна."""
    K = 3
    mus = [0.1, 0.5, 0.9]
    T = 10
    model = DummyTransformer(K)

    reward_history, total_reward = karmedbandit_run_in_context(model, mus, T, seq_len=5)
    assert isinstance(total_reward, float)
    assert 0.0 <= total_reward <= T * max(mus)
    assert isinstance(reward_history, list)
    assert len(reward_history) == T


def test_deterministic_with_fixed_seed():
    """Проверка, что фиксированная случайность даёт воспроизводимые результаты."""
    K = 2
    mus = [0.2, 0.8]
    T = 5
    model = DummyTransformer(K)

    set_seed(42)
    reward_history1, total_reward1 = karmedbandit_run_in_context(model, mus, T, seq_len=3)
    set_seed(42)
    reward_history2, total_reward2 = karmedbandit_run_in_context(model, mus, T, seq_len=3)

    assert reward_history1 == reward_history2
    assert total_reward1 == total_reward2


def test_zero_length_trajectory():
    """Если T=0, награда должна быть равна нулю и история пустая."""
    K = 3
    mus = [0.1, 0.5, 0.9]
    model = DummyTransformer(K)

    reward_history, total_reward = karmedbandit_run_in_context(model, mus, T=0, seq_len=5)
    assert total_reward == 0.0
    assert reward_history == []


def test_various_K_and_seq_len():
    """Проверка работы с разными числами рук и seq_len."""
    for K, seq_len in [(2, 1), (5, 10), (3, 3)]:
        mus = [i / K for i in range(K)]
        model = DummyTransformer(K)
        reward_history, total_reward = karmedbandit_run_in_context(model, mus, T=7, seq_len=seq_len)
        assert isinstance(total_reward, float)
        assert isinstance(reward_history, list)
        assert len(reward_history) == 7


def test_non_integer_rewards():
    """Проверка работы с плавающими средними наградами."""
    mus = [0.1, 0.25, 0.5, 0.75]
    model = DummyTransformer(len(mus))
    reward_history, total_reward = karmedbandit_run_in_context(model, mus, T=8, seq_len=4)
    assert 0.0 <= total_reward <= 8 * max(mus)
    assert len(reward_history) == 8

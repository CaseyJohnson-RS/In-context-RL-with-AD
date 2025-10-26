import numpy as np
import pytest

from src.utils import karmedbandit_generate_traces 


# ======== Вспомогательные классы для тестов ========

class DummyKArmedBandit:
    """Простая заглушка для среды KArmedBandit."""
    def __init__(self, K: int, mus: np.ndarray):
        self.K = K
        self.mus = mus
        self._calls = 0

    def step(self, action: int) -> float:
        """Возвращает фиксированную награду на основе среднего."""
        assert 0 <= action < self.K
        self._calls += 1
        return float(self.mus[action])  # детерминированный результат


class DummyThompsonSampling:
    """Простая заглушка для агента ThompsonSampling."""
    def __init__(self, K: int):
        self.K = K
        self.last_action = 0

    def select(self) -> int:
        """Циклический выбор действий."""
        action = self.last_action
        self.last_action = (self.last_action + 1) % self.K
        return action

    def update(self, action: int, reward: float):
        """Фиктивное обновление."""
        pass


# ======== Monkeypatch фикстура ========

@pytest.fixture(autouse=True)
def patch_bandit_and_agent(monkeypatch):
    """Патчит реальные классы в целевой функции на фиктивные."""
    monkeypatch.setattr("src.environments.KArmedBandit", DummyKArmedBandit)
    monkeypatch.setattr("src.models.ThompsonSampling", DummyThompsonSampling)


# ======== Тесты ========

def test_output_structure():
    """Проверяет, что функция возвращает корректную структуру данных."""
    trajectories = karmedbandit_generate_traces(
        num_tasks=3,
        K=5,
        T_per_task=10,
        mu_sampler=lambda K: np.linspace(0, 1, K),
        seed=42
    )

    assert isinstance(trajectories, list)
    assert len(trajectories) == 3
    for traj in trajectories:
        assert isinstance(traj, list)
        assert len(traj) == 10
        for step in traj:
            assert isinstance(step, tuple)
            assert len(step) == 2
            a, r = step
            assert isinstance(a, int)
            assert isinstance(r, float)


def test_reproducibility():
    """Проверяет воспроизводимость при одинаковом сиде."""
    def sampler(K):
        return np.random.normal(0, 1, size=K)

    traj1 = karmedbandit_generate_traces(2, 3, 5, sampler, seed=123)
    traj2 = karmedbandit_generate_traces(2, 3, 5, sampler, seed=123)

    assert traj1 == traj2, "Результаты должны быть идентичны при одинаковом сиде"


def test_different_seeds_produce_different_results():
    """Проверяет, что разные сиды дают разные результаты."""
    def sampler(K):
        return np.random.normal(0, 1, size=K)

    traj1 = karmedbandit_generate_traces(1, 3, 5, sampler, seed=1)
    traj2 = karmedbandit_generate_traces(1, 3, 5, sampler, seed=2)

    assert traj1 != traj2, "Разные сиды должны давать разные траектории"


def test_actions_within_range():
    """Проверяет, что все действия в допустимом диапазоне."""
    K = 4
    trajectories = karmedbandit_generate_traces(
        num_tasks=1,
        K=K,
        T_per_task=10,
        mu_sampler=lambda K: np.ones(K),
        seed=0
    )

    for traj in trajectories[0]:
        action, _ = traj
        assert 0 <= action < K, "Действие должно быть в диапазоне [0, K-1]"


def test_no_side_effects_between_tasks():
    """Проверяет, что агент и среда не влияют на следующие задачи."""
    trajectories = karmedbandit_generate_traces(
        num_tasks=2,
        K=3,
        T_per_task=5,
        mu_sampler=lambda K: np.arange(K),
        seed=0
    )

    # Проверяем, что обе траектории независимы
    assert trajectories[0] != trajectories[1]

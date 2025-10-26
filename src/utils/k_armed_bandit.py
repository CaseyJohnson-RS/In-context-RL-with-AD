from typing import List, Tuple

from src.models import ThompsonSampling
from src.environments import KArmedBandit
from .common import set_seed


def generate_karmedbandit_traces(
    num_tasks: int,
    K: int,
    T_per_task: int,
    mu_sampler,
    seed: int = 0
) -> List[List[Tuple[int, float]]]:
    """
    Генерация траекторий "учителя" для задач K-armed bandit
    с использованием модели Thompson Sampling.

    Параметры
    ----------
    num_tasks : int
        Количество задач (траекторий), которые нужно сгенерировать.
    K : int
        Количество рук (arms) в каждой задаче.
    T_per_task : int
        Количество шагов взаимодействия (длина каждой траектории).
    mu_sampler : callable
        Функция для генерации средних наград для K рук.
        default: lambda K: np.random.normal(0, 1, size=K)
    seed : int, optional
        Сид для воспроизводимости. По умолчанию 0.

    Возвращает
    ----------
    List[List[Tuple[int, float]]]
        Список траекторий. Каждая траектория — это список кортежей (действие, награда),
        длиной T_per_task.
    """
    set_seed(seed)

    trajectories: List[List[Tuple[int, float]]] = []

    for _ in range(num_tasks):
        # 1. Генерация средних наград
        mus = mu_sampler(K)

        # 2. Инициализация среды и агента
        env = KArmedBandit(K, mus)
        agent = ThompsonSampling(K)

        # 3. Генерация одной траектории
        trajectory: List[Tuple[int, float]] = []
        for _ in range(T_per_task):
            action = agent.select()
            reward = env.step(action)
            agent.update(action, reward)
            trajectory.append((action, float(reward)))

        trajectories.append(trajectory)

    return trajectories


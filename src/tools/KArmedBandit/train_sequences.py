from typing import List, Tuple

from src.environments import KArmedBandit
from models.teachers import ThompsonSampling


def __generate_traces(
    num_tasks: int, 
    K: int, 
    T_per_task: int, 
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
    seed : int, optional
        Сид для воспроизводимости. По умолчанию 0.

    Возвращает
    ----------
    List[List[Tuple[int, float]]]
        Список траекторий. Каждая траектория — это список кортежей (действие, награда),
        длиной T_per_task.
    """

    traces: List[List[Tuple[int, float]]] = []

    for _ in range(num_tasks):
        # 1. Инициализация среды и агента
        env = KArmedBandit(K)
        agent = ThompsonSampling(K)

        # 2. Генерация одной траектории
        trajectory: List[Tuple[int, float]] = []
        for _ in range(T_per_task):
            action = agent.select()
            reward = env.step(action)
            agent.update(action, reward)
            trajectory.append((action, float(reward)))

        traces.append(trajectory)

    return traces


def __traces_to_sequences(
    traces: List[List[Tuple[int, float]]], seq_len: int
) -> List[Tuple[List[int], List[float], int]]:
    """
    Формирует обучающие последовательности для трансформера из списка траекторий,
    сгенерированных функцией `generate_karmedbandit_traces`.

    Для каждой траектории создаются скользящие окна длиной `seq_len`, где:
      - вход (input) — это предыдущие действия и награды,
      - цель (target) — следующее действие.

    Параметры
    ----------
    trajectories : List[List[Tuple[int, float]]]
        Список траекторий, каждая — это список кортежей (действие, награда),
        возвращаемых функцией `generate_karmedbandit_traces`.
    seq_len : int
        Максимальная длина входной последовательности для трансформера.
        Если траектория короче `seq_len`, используется padding слева.

    Возвращает
    ----------
    List[Tuple[List[int], List[float], int]]
        Список обучающих примеров. Каждый пример содержит:
        (
            actions : List[int]   — последовательность действий (длина seq_len),
            rewards : List[float] — последовательность наград (длина seq_len),
            target  : int         — следующее действие для предсказания
        )

    Особенности
    -----------
    - Используется скользящее окно по всей траектории (начиная с t=1).
    - Padding слева: (action=0, reward=0.0) для недостающих элементов.
    - Цель (target) — действие на позиции t.
    """
    sequences: List[Tuple[List[int], List[float], int]] = []

    for traj in traces:
        traj_len = len(traj)

        for t in range(1, traj_len):
            start = max(0, t - seq_len)
            window = traj[start:t]  # предыдущие шаги (≤ seq_len)

            # Цель — действие в момент t
            target_action = traj[t][0]

            # Паддинг слева
            pad_len = seq_len - len(window)
            actions = [0] * pad_len + [a for a, _ in window]
            rewards = [0.0] * pad_len + [r for _, r in window]

            sequences.append((actions, rewards, target_action))

    return sequences


def generate_train_sequences(
        num_tasks: int,
        K: int,
        T_per_task: int,
        seq_len: int
):
    traces = __generate_traces(num_tasks=num_tasks, K=K, T_per_task=T_per_task)
    sequences = __traces_to_sequences(traces=traces, seq_len=seq_len)

    return sequences
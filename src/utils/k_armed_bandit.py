from typing import List, Tuple
import torch

from src.models import ThompsonSampling
from src.environments import KArmedBandit
from .common import set_seed


def karmedbandit_generate_traces(
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


def karmedbandit_trajectories_to_sequences(
    trajectories: List[List[Tuple[int, float]]],
    seq_len: int
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

    for traj in trajectories:
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


def karmedbandit_run_in_context(
    model: torch.nn.Module,
    mus: List[float],
    T: int,
    device: str = "cpu",
    seq_len: int = 20
) -> float:
    """
    Запуск обученного трансформера в среде KArmedBandit, возвращает суммарную награду.

    Параметры
    ----------
    model : nn.Module
        Обученный трансформер, принимающий тензоры (actions, rewards) и выдающий логиты.
    mus : list or array_like, shape (K,)
        Средние награды для K рук (arms) в задаче.
    T : int
        Длина генерируемой траектории.
    device : str или torch.device, optional
        Устройство для вычислений ('cpu' или 'cuda').
    seq_len : int, optional
        Максимальная длина контекстной последовательности для модели.

    Возвращает
    ----------
    total_reward : float
        Суммарная награда за все T шагов в среде.
    """
    model.eval()
    K = len(mus)
    env = KArmedBandit(K, mus)

    # Инициализация контекста
    actions_seq = [0] * seq_len
    rewards_seq = [0.0] * seq_len
    total_reward = 0.0

    for _ in range(T):
        # Подготовка батча размером 1
        a_tensor = torch.tensor([actions_seq], dtype=torch.long, device=device)
        r_tensor = torch.tensor([rewards_seq], dtype=torch.float32, device=device).unsqueeze(-1)

        # Предсказание действия трансформером
        with torch.no_grad():
            logits = model(a_tensor, r_tensor)  # (1, K)
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, num_samples=1).item())

        # Шаг в среде и накопление награды
        reward = env.step(action)
        total_reward += reward

        # Обновление контекста
        actions_seq = actions_seq[1:] + [action]
        rewards_seq = rewards_seq[1:] + [reward]

    return total_reward
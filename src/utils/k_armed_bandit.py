from typing import List, Tuple
from collections import defaultdict
from torch import nn, optim
import torch
import numpy as np
import time

from src.models import ThompsonSampling
from src.environments import KArmedBandit
from .common import set_seed, batch_iterator


def karmedbandit_generate_traces(
    num_tasks: int, K: int, T_per_task: int, mu_sampler, seed: int = 0
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
    trajectories: List[List[Tuple[int, float]]], seq_len: int
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
    seq_len: int = 20,
) -> Tuple[List[float], float]:
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
    reward_history = []

    for _ in range(T):
        # Подготовка батча размером 1
        a_tensor = torch.tensor([actions_seq], dtype=torch.long, device=device)
        r_tensor = torch.tensor(
            [rewards_seq], dtype=torch.float32, device=device
        ).unsqueeze(-1)

        # Предсказание действия трансформером
        with torch.no_grad():
            logits = model(a_tensor, r_tensor)  # (1, K)
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, num_samples=1).item())

        # Шаг в среде и накопление награды
        reward = env.step(action)
        total_reward += reward
        reward_history.append(total_reward)

        # Обновление контекста
        actions_seq = actions_seq[1:] + [action]
        rewards_seq = rewards_seq[1:] + [reward]

    return reward_history, total_reward


def karmedbandit_evaluate(
    model,
    mu_sampler,       # callable, возвращает вектор средних наград длиной K
    K: int,
    T_test: int,
    n_tasks: int = 10,
    device='cpu',
    seq_len: int = 256,
    verbose: bool = True,
):
    """
    Оценивает качество обученной модели (трансформера) на сгенерированных задачах,
    сравнивая её с учителем (Thompson Sampling) и случайной стратегией.

    Параметры
    ----------
    model : torch.nn.Module
        Трансформерная модель для оценки.
    mu_sampler : callable
        Функция, принимающая K и возвращающая np.array со средними наградами для каждой руки.
    K : int
        Количество "рукавов" (действий) в многорукавом бандите.
    T_test : int
        Количество шагов взаимодействия с каждой задачей.
    n_tasks : int
        Количество задач для оценки.
    device : torch.device | str | None
        Устройство для вычислений.
    seq_len : int
        Максимальная длина контекстной последовательности для модели.
    verbose : bool
        Если True — печатает результаты оценки.

    Возвращает
    ----------
    dict[str, tuple[float, float]]
        Среднее и стандартная ошибка для каждой стратегии:
        {'teacher': (mean, stderr), 'model': (mean, stderr), 'random': (mean, stderr)}
    """
    results = defaultdict(list)

    for _ in range(n_tasks):
        mus = mu_sampler(K)

        # --- 1. Учитель (Thompson Sampling)
        teacher = ThompsonSampling(K)
        env_karmedbandit = KArmedBandit(K, mus)
        teacher_reward = 0.0
        for _ in range(T_test):
            a = teacher.select()
            r = env_karmedbandit.step(a)
            teacher.update(a, r)
            teacher_reward += r

        # --- 2. Случайная стратегия
        env_random = KArmedBandit(K, mus)
        rand_reward = 0.0
        for _ in range(T_test):
            a = np.random.randint(0, K)
            rand_reward += env_random.step(a)

        # --- 3. Модель (трансформер)
        _, model_reward = karmedbandit_run_in_context(
            model, mus, T_test, device=device, seq_len=seq_len
        )

        results['teacher'].append(teacher_reward)
        results['random'].append(rand_reward)
        results['model'].append(model_reward)

    # --- Вычисление среднего и стандартной ошибки
    def mean_std(xs):
        xs = np.array(xs)
        return float(np.mean(xs)), float(np.std(xs) / np.sqrt(len(xs)))

    summary = {k: mean_std(v) for k, v in results.items()}

    if verbose:
        print("Results (mean ± stderr) over test tasks:")
        for k, (m, se) in summary.items():
            print(f" {k:<7}: {m:.3f} ± {se:.3f}")

    return summary


class KArmedBanditTrainer:
    def __init__(
        self,
        model,
        seqs,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
        cosine_decay: bool = True,
        T_max: int = 100
    ):
        """
        Инициализация тренера для трансформера в режиме Action-Decision (AD).
        """
        self.model = model.to(device)
        self.seqs = seqs
        self.batch_size = batch_size
        self.device = device

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = (
            optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=lr * 0.1
            )
            if cosine_decay
            else None
        )

        self.loss_history: list = []
        self.epoch = 0

    def train_epoch(self, verbose: bool = True):
        """
        Выполняет одну эпоху обучения и возвращает средний лосс.
        """
        self.model.train()
        start_time = time.time()
        total_loss = 0.0
        batch_count = 0

        for actions_b, rewards_b, targets_b in batch_iterator(
            self.seqs, self.batch_size, device=self.device, shuffle=True
        ):
            logits = self.model(actions_b, rewards_b)  # (B, n_actions)
            
            # Модель предсказывает только следующее действие, значит цель — это targets_b[:, -1]
            if targets_b.ndim > 1:
                targets_b = targets_b[:, -1]
            
            loss = self.criterion(logits, targets_b)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1

        if self.scheduler:
            self.scheduler.step()

        avg_loss = total_loss / max(batch_count, 1)
        self.loss_history.append(avg_loss)
        self.epoch += 1

        if verbose:
            elapsed = time.time() - start_time
            print(
                f"Epoch {self.epoch:3d} | Loss: {avg_loss:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

        return avg_loss

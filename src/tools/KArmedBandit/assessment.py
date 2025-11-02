import numpy as np
import torch
from collections import defaultdict
from typing import List, Tuple

from src.environments import KArmedBandit
from models.teachers import ThompsonSampling


def run_in_inference(
    model: torch.nn.Module,
    env: KArmedBandit,
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
    env : KArmedBandit
        Окружение.
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

    # Инициализация контекста
    actions_seq = [0] * seq_len
    rewards_seq = [0.0] * seq_len
    total_reward = 0.0

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

        # Обновление контекста
        actions_seq = actions_seq[1:] + [action]
        rewards_seq = rewards_seq[1:] + [reward]

    return total_reward


def evaluate(
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
        env_tranformer = KArmedBandit(K, mus)
        model_reward = run_in_inference(
            model, env_tranformer, T_test, device=device, seq_len=seq_len
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
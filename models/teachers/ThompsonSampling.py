import math
import numpy as np


class ThompsonSampling:
    """
    Класс для алгоритма Thompson Sampling с гауссовским вознаграждением.

    Особенности:
    - Вознаграждение каждой руки (arm) имеет гауссовское распределение с неизвестным средним μ и известной дисперсией σ².
    - Апостериорное распределение среднего награды каждой руки также гауссовское, благодаря конъюгированному приору.
    - Цель: на каждом шаге выбрать руку, чтобы максимизировать ожидаемую сумму наград, балансируя исследование и эксплуатацию.

    Формулировка:
    ----------------------
    Среднее вознаграждение руки i: μ_i ~ Normal(prior_mean, prior_var)
    Наблюдаемое вознаграждение: r_i ~ Normal(μ_i, obs_var)
    Постериорное распределение после n_i наблюдений:
        var_post = 1 / (1/prior_var + n_i / obs_var)
        mu_post  = var_post * (prior_mean/prior_var + sum_rewards_i / obs_var)
    """

    def __init__(self, K, prior_mean=0.0, prior_var=1.0, obs_var=1.0):
        """
        Инициализация учителя Thompson Sampling.

        Параметры:
        ----------------------
        K (int): число рук (arms) в многоруком бандите.
        prior_mean (float): среднее априорного нормального распределения для μ_i.
        prior_var (float): дисперсия априорного нормального распределения.
        obs_var (float): известная дисперсия наблюдаемого вознаграждения.
        """
        self.K = K
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.obs_var = obs_var
        self.reset()  # инициализация счётчиков и сумм наград

    def reset(self):
        """
        Сбрасывает внутреннее состояние учителя для новой задачи:
        - counts: количество выборов каждой руки.
        - sum_rewards: суммарная награда, полученная каждой рукой.
        """
        self.counts = np.zeros(self.K, dtype=np.int32)
        self.sum_rewards = np.zeros(self.K, dtype=np.float32)

    def select(self):
        """
        Выбор действия по алгоритму Thompson Sampling.

        Для каждой руки:
        1. Вычисляется постериорное распределение среднего вознаграждения:
            - var_post = 1 / (1/prior_var + n / obs_var)
            - mu_post  = var_post * (prior_mean/prior_var + sum_rewards / obs_var)
        2. Семплируется случайное значение из Normal(mu_post, sqrt(var_post)).
        3. Выбирается рука с максимальным семплированным средним.

        Возвращает:
        ----------------------
        int: индекс выбранной руки (от 0 до K-1)
        """
        samples = []
        for a in range(self.K):
            n = self.counts[a]
            # Постериорная дисперсия для средней награды этой руки
            post_var = 1.0 / (1.0 / self.prior_var + n / self.obs_var)
            # Постериорное среднее
            post_mean = post_var * (
                self.prior_mean / self.prior_var + self.sum_rewards[a] / self.obs_var
            )
            # Семплируем случайное среднее награды
            s = np.random.normal(post_mean, math.sqrt(post_var))
            samples.append(s)
        # Выбираем руку с максимальным семплированным средним
        return int(np.argmax(samples))

    def update(self, action, reward):
        """
        Обновление статистики после того, как агент сделал выбор.

        Параметры:
        ----------------------
        action (int): индекс выбранной руки
        reward (float): полученная награда
        """
        self.counts[action] += 1
        self.sum_rewards[action] += reward

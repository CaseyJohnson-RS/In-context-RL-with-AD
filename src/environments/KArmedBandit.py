import numpy as np


class KArmedBandit:
    """
    Модель многорукого бандита с K рычагами.

    Каждый рычаг (action) имеет свою среднюю награду (mu) и общую дисперсию (sigma).

    Attributes:
        K (int): Количество рычагов.
        mus (np.ndarray): Средние значения награды для каждого рычага.
        sigma (float): Стандартное отклонение награды для всех рычагов.
    """

    def __init__(self, K: int, mus: list[float] = None, sigma: float = 1.0):
        """
        Инициализация многорукого бандита.

        Args:
            K (int): Количество рычагов.
            mus (list[float]): Список средних значений награды для каждого рычага.
            sigma (float, optional): Стандартное отклонение награды. По умолчанию 1.0.

        Raises:
            ValueError: Если длина списка mus не совпадает с K.
        """
        if mus is None:
            mus = np.random.normal(0.0, 1.0, size=K).astype(np.float32)
        elif len(mus) != K:
            raise ValueError(f"Длина mus ({len(mus)}) должна совпадать с K ({K}).")
        self.K: int = K
        self.mus: np.ndarray = np.array(mus, dtype=np.float32)
        self.sigma: float = sigma

    def step(self, action: int) -> float:
        """
        Выполнение действия (выбор рычага) и получение случайной награды.

        Награда для выбранного рычага генерируется по нормальному распределению
        с средним значением mus[action] и стандартным отклонением sigma.

        Args:
            action (int): Индекс выбранного рычага (0 <= action < K).

        Returns:
            float: Случайная награда для выбранного рычага.

        Raises:
            IndexError: Если action выходит за пределы допустимого диапазона.
        """
        if not 0 <= action < self.K:
            raise IndexError(
                f"Выбранный action ({action}) выходит за допустимый диапазон 0-{self.K - 1}."
            )
        reward = np.random.normal(self.mus[action], self.sigma)
        return reward

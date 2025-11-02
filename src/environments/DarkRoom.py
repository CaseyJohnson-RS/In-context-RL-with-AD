import numpy as np
from typing import Tuple, Dict, Any


class DarkRoom:
    """
    Класс среды 'Dark Room' — 2D дискретная POMDP, где агент должен найти невидимую цель.
    Агент наблюдает только свои координаты (x, y) и получает вознаграждение r=1 при достижении цели.
    """

    def __init__(self, size: int = 9, hard: bool = False, max_steps: int = 20):
        """
        :param size: Размер комнаты (size x size)
        :param hard: Если True — используется сложный вариант с редким вознаграждением
        :param max_steps: Максимальная длина эпизода
        """
        self.size = size
        self.hard = hard
        self.max_steps = max_steps

        self.reset()

    # --- Интерфейс среды ---
    def reset(self) -> np.ndarray:
        """Сбрасывает среду и возвращает начальное наблюдение (координаты агента)."""
        self.agent_pos = np.array([self.size // 2, self.size // 2], dtype=np.int32)
        self.goal_pos = np.random.randint(0, self.size, size=2, dtype=np.int32)
        self.steps = 0
        self.goal_found = False
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Выполняет одно действие агента.
        :param action: 0=вверх, 1=вниз, 2=влево, 3=вправо, 4=ничего не делать
        :return: (наблюдение, вознаграждение, done, информация)
        """
        if not (0 <= action <= 4):
            raise ValueError("Действие должно быть числом от 0 до 4")

        # Двигаем агента
        if action == 0:  # вверх
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # вниз
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # влево
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # вправо
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        # действие 4 — no-op (ничего не делает)

        self.steps += 1
        done = False
        reward = 0.0

        # Проверяем, достиг ли агент цели
        if np.array_equal(self.agent_pos, self.goal_pos):
            if not self.hard or not self.goal_found:
                reward = 1.0
            self.goal_found = True
            if self.hard:
                done = True  # В сложной версии эпизод завершается после нахождения цели

        # Проверяем лимит по шагам
        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {"goal": self.goal_pos.copy()}

    def _get_obs(self) -> np.ndarray:
        """Возвращает наблюдение — только текущие координаты агента."""
        return self.agent_pos.copy()

    # --- Вспомогательные методы ---
    def render(self) -> None:
        """Простая текстовая визуализация комнаты."""
        grid = np.full((self.size, self.size), '.', dtype=str)
        x, y = self.agent_pos
        gx, gy = self.goal_pos
        grid[gx, gy] = 'G'  # цель
        grid[x, y] = 'A'    # агент
        print("\n".join(" ".join(row) for row in grid))
        print(f"Шаг: {self.steps} | Позиция агента: {self.agent_pos.tolist()} | Цель: {self.goal_pos.tolist()}")


if __name__ == "__main__":
    env = DarkRoom(size=9, hard=False)

    obs = env.reset()
    print("Начальное наблюдение:", obs)

    for t in range(20):
        action = np.random.randint(0, 5)  # случайное действие
        obs, reward, done, info = env.step(action)
        print(f"Шаг {t+1}: действие={action}, наблюдение={obs}, награда={reward}")
        env.render()
        if reward > 0:
            print("Цель достигнута!")
        if done:
            print("Эпизод завершён.")
            break

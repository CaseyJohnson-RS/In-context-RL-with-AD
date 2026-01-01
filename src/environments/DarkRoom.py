import numpy as np
from typing import Tuple, Dict, Any, Set

from .Environment import Environment



class DarkRoom(Environment):
    """
    A grid‑based environment where an agent must find a goal in a dark room.
    Supports 'easy' (exploration‑reward) and 'hard' (single‑goal) modes.
    """

    def __init__(self, size: int = 9, mode: str = 'easy', max_steps: int = 20) -> None:
        """
        Initialize the DarkRoom environment.

        Args:
            size: Grid size (must be ≥ 1).
            mode: Game mode ('easy' or 'hard').
            max_steps: Maximum steps before truncation (must be ≥ 1).
        """
        if size < 1:
            raise ValueError(f"Room size must be ≥ 1, got {size}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be ≥ 1, got {max_steps}")
        if mode not in ('easy', 'hard'):
            raise ValueError(f"Mode must be 'easy' or 'hard', got {mode}")

        self.size: int = size
        self.mode: str = mode
        self.max_steps: int = max_steps

        # Track visited positions for exploration rewards in 'easy' mode
        self.visited_positions: Set[Tuple[int, int]] = set()

        self.reset()

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns:
            Observation and info dictionary.
        """
        # Center agent
        center: int = self.size // 2
        self.agent_pos: Tuple[int, int] = (center, center)

        # Random goal position
        self.goal_pos: np.ndarray = np.random.randint(
            0, self.size, size=2, dtype=np.int32
        )

        self.steps: int = 0
        self.goal_found: bool = False

        # Reset visited positions
        self.visited_positions.clear()
        self.visited_positions.add(self.agent_pos)

        info: Dict[str, Any] = {
            "initial_position": self.agent_pos,
            "goal_position": tuple(self.goal_pos),
            "mode": self.mode,
            "max_steps": self.max_steps,
            "visited_count": 1,
        }
        return self._get_obs(), info

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step in the environment.

        Args:
            action: Integer action (0–4).

        Returns:
            observation, reward, terminated, truncated, info
        
        Moves:
         - 0 -> STAY
         - 1 -> UP
         - 2 -> DOWN
         - 3 -> LEFT
         - 4 -> RIGHT
        """

        # Define action effects: (dx, dy)
        actions: Tuple[Tuple[int, int], ...] = (
            (0, 0),  # stay
            (0, -1), # up
            (0, 1),  # down
            (-1, 0), # left
            (1, 0),  # right
        )

        if action >= len(actions):
            raise ValueError(f"Action must be 0–{len(actions) - 1}, got {action}")        

        # Apply action and clamp to grid
        new_x: int = self.__clamp(
            self.agent_pos[0] + actions[action][0], 0, self.size - 1
        )
        new_y: int = self.__clamp(
            self.agent_pos[1] + actions[action][1], 0, self.size - 1
        )
        self.agent_pos = (new_x, new_y)

        self.steps += 1

        terminated: bool = False
        truncated: bool = False
        reward: float = 0.0

        # Check goal
        if np.array_equal(self.agent_pos, self.goal_pos):
            if self.mode == 'hard':
                terminated = True
                if not self.goal_found:
                    reward += 1.0
            else:  # 'easy' mode
                reward += 1.0
            self.goal_found = True

        # Exploration reward in 'easy' mode
        if self.mode == 'easy':
            is_new: bool = self.agent_pos not in self.visited_positions
            if is_new:
                reward += 0.05
                self.visited_positions.add(self.agent_pos)
            else:
                reward -= 0.05

        # Clamp reward
        reward = float(np.clip(reward, -0.1, 1.0))

        # Check step limit
        if self.steps >= self.max_steps:
            truncated = True

        info: Dict[str, Any] = {
            "step": self.steps,
            "agent_position": self.agent_pos,
            "goal_position": tuple(self.goal_pos),
            "goal_found": self.goal_found,
            "reward": reward,
            "is_new_position": self.agent_pos not in self.visited_positions,
            "visited_count": len(self.visited_positions),
        }

        return self._get_obs(), reward, terminated, truncated, info

    def observation(self) -> Tuple[int, int]:
        """Get current observation."""
        return self._get_obs()

    @staticmethod
    def __clamp(x: int, minimum: int, maximum: int) -> int:
        """Clamp value between minimum and maximum."""
        return max(minimum, min(x, maximum))

    def _get_obs(self) -> Tuple[int, int]:
        """Return agent position as observation."""
        return self.agent_pos


__all__ = ["DarkRoom"]

import numpy as np
from typing import Tuple, Dict, Any
from .Environment import Environment


class DarkRoom(Environment):

    def __init__(self, size: int = 9, mode: str = 'easy', max_steps: int = 20):
        if size < 1:
            raise ValueError(f"Room size must be ≥ 1, got {size}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be ≥ 1, got {max_steps}")

        self.size = size
        self.mode = mode
        self.max_steps = max_steps
        self.visited_positions = set()  # Track visited positions
        self.reset()

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        self.agent_pos = (self.size // 2, self.size // 2)
        self.goal_pos = np.random.randint(0, self.size, size=2, dtype=np.int32)
        self.steps = 0
        self.goal_found = False
        self.visited_positions.clear()  # Reset visited positions
        self.visited_positions.add(self.agent_pos)  # Add starting position
        
        info = {
            "initial_position": self.agent_pos,
            "goal_position": tuple(self.goal_pos),
            "mode": self.mode,
            "max_steps": self.max_steps,
            "visited_count": 1
        }
        return (self._get_obs(), info)

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        if not (0 <= action <= 4):
            raise ValueError("Action must be an integer from 0 to 4 (inclusive)")

        # Define action effects: (dx, dy) - 0:stay, 1:up, 2:down, 3:left, 4:right
        actions = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        
        # Apply action and clamp to grid boundaries
        self.agent_pos = (
            self.__clamp(self.agent_pos[0] + actions[action][0], 0, self.size - 1),
            self.__clamp(self.agent_pos[1] + actions[action][1], 0, self.size - 1),
        )

        self.steps += 1
        terminated = False
        truncated = False
        reward = 0.0

        # Goal reward
        if np.array_equal(self.agent_pos, self.goal_pos):
            if not self.mode == 'hard' or not self.goal_found:
                reward += 1.0
            self.goal_found = True
            if self.mode == 'hard':
                terminated = True

        current_pos_tuple = tuple(self.agent_pos)
        
        if self.mode == 'easy':
            # EXPLORATION REWARD: New position bonus vs revisit penalty
            if current_pos_tuple not in self.visited_positions:
                # NEW POSITION: exploration bonus
                reward += 0.05
                self.visited_positions.add(current_pos_tuple)
            else:
                # REVISITED POSITION: penalty
                reward -= 0.05

        # Clamp total reward
        reward = np.clip(reward, -0.1, 1.0)

        # Check step limit
        if self.steps >= self.max_steps:
            truncated = True

        info = {
            "step": self.steps,
            "agent_position": self.agent_pos,
            "goal_position": tuple(self.goal_pos),
            "goal_found": self.goal_found,
            "reward": reward,
            "is_new_position": current_pos_tuple not in self.visited_positions,
            "visited_count": len(self.visited_positions),
        }

        return self._get_obs(), reward, terminated, truncated, info

    def observation(self):
        return self._get_obs()

    @staticmethod
    def __clamp(x, minimum, maximum):
        return max(minimum, min(x, maximum))

    def _get_obs(self) -> Tuple[int, int]:
        return self.agent_pos

__all__ = ["DarkRoom"]

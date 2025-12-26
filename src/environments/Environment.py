from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict


class Environment(ABC):
    """
    Abstract base class for RL environments.

    Defines an API compatible with Gym v0.26+:
    - reset() -> (observation, info)
    - step(action) -> (observation, reward, terminated, truncated, info)
    """

    @abstractmethod
    def observation(self) -> Any:
        pass

    @abstractmethod
    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Resets the environment to the initial state.

        Returns:
            observation (Any): the initial state of the environment.
            info (dict): metadata about the reset.

        Notes:
            - Does not return reward/terminated/truncated.
            - The observation format must match the format used in step().
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Executes a single environment step.

        Args:
            action: agent action. The type depends on the specific environment.

        Returns:
            observation (Any): the next environment state.
            reward (float): reward for the transition.
            terminated (bool): episode ended naturally.
            truncated (bool): episode was cut short (timeout, external limit, etc.).
            info (dict): diagnostic information.

        Notes:
            - terminated and truncated must not both be True.
        """
        pass

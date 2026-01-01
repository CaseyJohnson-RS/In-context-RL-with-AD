import numpy as np
from typing import Tuple, Dict, Any

from .Environment import Environment


class KArmedBandit(Environment):
    """
    K‑armed bandit environment.

    Each arm (action) has its own mean reward (mu) and a shared standard deviation (sigma).
    The environment is stateless: each step depends only on the chosen action.
    """

    def __init__(self, k: int, sigma: float = 1.0) -> None:
        """
        Initialize the k‑armed bandit.

        Args:
            k: The number of arms (actions) available in the bandit.
            sigma: The standard deviation of the reward distribution, shared across all arms.
                   Defaults to 1.0.

        Note:
            The mean rewards for each arm (mus) are randomly sampled from N(0, 1) during reset().
            They are not provided as an argument to __init__ to ensure proper encapsulation
            and alignment with typical bandit problem formulations.
        """
        if k < 1:
            raise ValueError(f"Number of arms (k) must be ≥ 1, got {k}")
        if sigma < 0:
            raise ValueError(f"Standard deviation (sigma) must be ≥ 0, got {sigma}")

        self.k: int = k
        self.sigma: float = sigma

        # Will be initialized in reset()
        self.mus: np.ndarray = np.empty(0, dtype=np.float32)

        self.reset()

    def reset(self) -> Tuple[None, Dict[str, Any]]:
        """
        Reset the environment to a new initial state.

        This method samples new mean rewards for each arm from a standard normal distribution
        N(0, 1), effectively creating a new instance of the k‑armed bandit problem.

        Returns:
            Tuple containing:
                - observation: None (the bandit provides no observations)
                - info: A dictionary with environment metadata:
                    * "k": number of arms
                    * "mus": array of mean rewards for each arm
                    * "sigma": standard deviation of reward distribution

        Note:
            Since the k‑armed bandit is stateless, reset primarily serves to initialize
            the mean rewards for the arms.
        """
        self.mus = np.random.normal(
            0.0, 1.0, size=self.k
        ).astype(np.float32)

        info: Dict[str, Any] = {
            "k": self.k,
            "mus": self.mus,
            "sigma": self.sigma,
        }
        return None, info

    def step(self, action: int) -> Tuple[None, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment by pulling a specific arm.

        The reward is sampled from a normal distribution with parameters:
        N(mu[action], sigma), where mu[action] is the mean reward for the selected arm.

        Args:
            action: The index of the arm to pull (integer in range [0, k−1]).

        Returns:
            Tuple containing:
                - observation: None (the k‑armed bandit provides no observational feedback)
                - reward: The stochastic reward sampled from N(mu[action], sigma)
                - terminated: False (k‑armed bandit episodes run indefinitely)
                - truncated: False (no built‑in truncation mechanism)
                - info: Empty dictionary (no additional information provided)

        Raises:
            IndexError: If the action index is outside the valid range [0, k−1].

        Example:
            >>> env = KArmedBandit(k=3)
            >>> env.reset()
            >>> reward = env.step(0)[1]  # Pull first arm and get reward
        """
        if not (0 <= action < self.k):
            raise IndexError(
                f"Action {action} is out of valid range [0, {self.k - 1}]."
            )

        reward: float = float(np.random.normal(self.mus[action], self.sigma))
        return None, reward, False, False, {}

    def observation(self) -> None:
        """
        Get current observation.

        Returns:
            None — the k‑armed bandit provides no observational feedback.
        """
        return None


__all__ = ["KArmedBandit"]

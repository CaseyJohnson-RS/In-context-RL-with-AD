import numpy as np
import math
from typing import Any, Callable, Dict, List, Tuple


from src.environments import KArmedBandit
from .RLAgent import RLAgent


class ThompsonSamplingAgent(RLAgent):
    """Thompson Sampling agent for multi‑armed bandit problems.

    Implements a Bayesian approach: for each arm, maintains a posterior distribution
    over rewards and selects the arm with the highest random sample from this
    distribution. Assumes normally distributed rewards with known observation variance.
    """

    trace_format = ("action", "reward")
    trace_zeroed = (0, 0.0)

    def __init__(
        self,
        k: int,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
        obs_var: float = 1.0,
    ):
        """Constructs a Thompson Sampling agent.

        Args:
            k: number of arms (actions) in the bandit
            prior_mean: prior belief about mean reward (default: 0.0)
            prior_var: prior variance of mean reward (default: 1.0)
            obs_var: known observation variance (same for all arms, default: 1.0)
        """
        self.k: int = k
        self.prior_mean: float = prior_mean
        self.prior_var: float = prior_var
        self.obs_var: float = obs_var

        self.reset()

    def _select(self) -> int:
        """Selects an action using Thompson Sampling.

        For each arm:
        1. Computes posterior mean and variance given observed data
        2. Draws a random sample from the posterior distribution
        3. Selects the arm with the highest sample value

        Returns:
            Index of the selected arm (0 to k-1)
        """
        samples = []
        for arm in range(self.k):
            n_pulls = self.counts[arm]

            post_var = 1.0 / (1.0 / self.prior_var + n_pulls / self.obs_var)
            post_mean = post_var * (
                self.prior_mean / self.prior_var + self.sum_rewards[arm] / self.obs_var
            )

            sample = np.random.normal(post_mean, math.sqrt(post_var))
            samples.append(sample)

        return int(np.argmax(samples))

    def _update(self, action: int, reward: float) -> None:
        """Updates agent statistics after receiving a reward.

        Increments:
        - pull count for the selected arm
        - cumulative reward for the selected arm


        Args:
            action: index of the selected arm (0 to k-1)
            reward: observed reward

        Raises:
            IndexError: if action is out of valid range
            TypeError: if reward is not numeric
        """
        if not 0 <= action < self.k:
            raise IndexError(f"Action {action} is out of range [0, {self.k - 1}]")

        if not isinstance(reward, (int, float)):
            raise TypeError(f"Reward must be numeric, got {type(reward)}")

        self.counts[action] += 1
        self.sum_rewards[action] += float(reward)
    
    def _run(
        self, environment: KArmedBandit, steps: int, update_agent: bool
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Runs the agent in the environment for a specified number of steps.

        Args:
            environment: environment for interaction
            steps: maximum number of steps
            update_agent: whether to update agent parameters (for training/testing)

        Returns:
            metrics: summary metrics (average reward, entropy)
            info: detailed information (reward history, actions, etc.)
        """
        cum_reward: float = 0.0
        total_steps: int = 0
        
        actions = []
        cumulative_reward_history = []

        for _ in range(steps):
            total_steps += 1

            action = self._select()
            actions.append(action)

            observation, reward, terminated, t_runcated, info = environment.step(action)

            if update_agent:
                self._update(action, reward)

            cum_reward += reward
            cumulative_reward_history.append(cum_reward)

            if terminated or t_runcated:
                break

        avg_reward = cum_reward / total_steps if total_steps > 0 else 0.0

        unique_actions, counts = np.unique(actions, return_counts=True)
        action_counts = np.zeros(environment.k, dtype=int)
        for act, cnt in zip(unique_actions, counts):
            action_counts[act] = cnt
        action_frequencies = action_counts / total_steps

        nonzero_freqs = action_frequencies[action_frequencies > 0]
        entropy = (
            -np.sum(nonzero_freqs * np.log(nonzero_freqs))
            if len(nonzero_freqs) > 0
            else 0.0
        )

        metrics = {
            "Average reward": avg_reward,
            "Entropy": entropy,
        }

        info = {
            "Reward": round(avg_reward, 4),
            "Cumulative reward history": cumulative_reward_history,
            "Actions": actions,
            "Action counts": action_counts.tolist(),
            "Action frequencies": [round(f, 4) for f in action_frequencies],
            "Entropy of action distribution": round(entropy, 4),
        }

        return metrics, info

    # ========== API ==========

    def train(
        self, env_constructor: Callable[[], KArmedBandit], episodes: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Trains the agent in the given environment.

        Args:
            environment: environment for interaction
            steps: number of training steps

        Returns:
            metrics: primary training metrics
            info: detailed information about the process
        """
        metrics, info = self._run(
            environment=env_constructor(), steps=episodes, update_agent=True
        )
        return metrics, {"Reward": info["Reward"]}

    def test(
        self, env_constructor: Callable[[], KArmedBandit], episodes: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Tests the agent in the environment (without updating parameters).

        Args:
            environment: environment for interaction
            steps: number of testing steps

        Returns:
            metrics: primary testing metrics
            info: detailed information about the process
        """
        metrics, info = self._run(
            environment=env_constructor(), steps=episodes, update_agent=False
        )
        return metrics, {"Reward": info["Reward"]}

    def trace(
        self, env_constructor: Callable[[], KArmedBandit], trace_len: int
    ) -> List[Tuple]:
        """Collects a sequence of actions and rewards.

        Args:
            env_constructor: function to create the environment
            trace_len: length of the trace (number of steps)

        Returns:
            List of (action, reward) pairs for the specified steps
        """
        trace: List[Tuple[int, float]] = []

        environment = env_constructor()
        for _ in range(trace_len):
            action = self._select()
            observation, reward, terminated, t_runcated, info = environment.step(action)
            self._update(action, reward)
            trace.append((action, reward))

        return trace

    def reset(self) -> None:
        """Resets agent statistics to initial state.

        Resets:
        - counts: number of times each arm has been pulled (all zeros)
        - sum_rewards: cumulative rewards for each arm (all zeros)
        """
        self.counts: np.ndarray = np.zeros(self.k, dtype=np.int32)
        self.sum_rewards: np.ndarray = np.zeros(self.k, dtype=np.float32)

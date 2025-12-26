from abc import ABC, abstractmethod
from typing import Any, Callable, List, Tuple, Dict, Optional

from src.environments import Environment


class RLAgent(ABC):
    """Abstract base class for reinforcement learning agents.

    Defines the core interface that all RL agents must implement, including methods for
    training, testing, tracing, and state management. Agents should override abstract
    methods to provide specific implementations.
    """

    trace_format: Optional[Tuple[str, ...]] = None
    trace_zeroed: Optional[Tuple[Any, ...]] = None

    @abstractmethod
    def train(self, env_constructor: Callable[[], Environment], episodes: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Train the agent in a given environment for a specified number of steps.

        Args:
            env_constructor: Function that creates a new environment instance

        Returns:
            Tuple containing:
            - metrics: Dictionary of quantitative performance metrics
            - info: Dictionary with additional information/metadata

        Note:
            Implementation should include learning/update logic specific to the agent.
        """
        pass

    @abstractmethod
    def test(self, env_constructor: Callable[[], Environment], episodes: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Test the agent in a given environment (typically without parameter updates).

        Args:
            env_constructor: Function that creates a new environment instance

        Returns:
            Tuple containing:
            - metrics: Dictionary of quantitative performance metrics
            - info: Dictionary with additional information/metadata

        Note:
            Implementation may skip learning updates to evaluate pre-trained behavior.
        """
        pass

    @abstractmethod
    def trace(self, env_constructor: Callable[[], Environment], trace_len: int) -> List[Tuple]:
        """Generate a trace of agent-environment interactions.

        Args:
            env_constructor: Function that creates a new environment instance
            trace_len: Number of steps to include in the trace

        Returns:
            List of tuples representing interaction steps. Each tuple should follow
            the format defined in `trace_format`.

        Note:
            The structure of returned tuples should match `trace_format` if defined.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the agent's internal state for a new episode.

        Note:
            This method should clear any episode-dependent variables and prepare the
            agent for a new interaction sequence.
        """
        pass

    def get_trace_format(self) -> Optional[Tuple[str, ...]]:
        """Get the format specification for trace steps.

        Returns:
            Tuple of field names describing the structure of trace entries,
            or None if trace format is not defined.

        Example:
            >>> agent.get_trace_format()
            ('action', 'reward', 'state')
        """
        return tuple(self.trace_format) if self.trace_format is not None else None

    def get_trace_zeroed(self) -> Optional[Tuple[Any, ...]]:
        """Get a zero-valued trace step matching the trace format.

        Returns:
            Tuple with default/zero values for each field in the trace format,
            or None if trace format is not defined.

        Purpose:
            Provides a template for initializing trace entries or filling missing data
            while maintaining consistent structure with `trace_format`.

        Example:
            >>> agent.get_trace_zeroed()
            (0, 0.0, None)
        """
        return tuple(self.trace_zeroed) if self.trace_zeroed is not None else None

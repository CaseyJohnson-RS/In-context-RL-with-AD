from .RLAgent import RLAgent # noqa: F401
from .ThompsonSampling import ThompsonSamplingAgent
from .DQNAgent import DQNAgent

agent_collection = {
    "ThompsonSamplingAgent": ThompsonSamplingAgent,
    "DQNAgent": DQNAgent
}

def create_agent(agent_name: str, agent_args: dict) -> RLAgent:
    return agent_collection[agent_name](**agent_args)
import pytest

from src.workflow.dataset_generator import create_dataset, __generate_traces, __traces_to_sequences
from src.workflow.SequenceDataset import SequenceDataset

# Mock classes without unittest
class MockRLAgent:
    def __init__(self, trace_format=None, trace_zeroed=None):
        self._trace_format = trace_format or ["state", "action", "reward"]
        self._trace_zeroed = trace_zeroed or [0, 0, 0]
        self.__class__.__name__ = "TestAgent"
    
    def get_trace_format(self):
        return self._trace_format
    
    def get_trace_zeroed(self):
        return self._trace_zeroed
    
    def trace(self, env_constructor, trace_len):
        return [(1, 2, 3)] * trace_len  # Mock trace

class MockEnv:
    pass

def mock_agent_constructor():
    return MockRLAgent()

def mock_env_constructor():
    return MockEnv()


def test_generate_traces_valid():
    traces = __generate_traces(mock_agent_constructor, mock_env_constructor, trace_count=2, trace_len=5)
    assert len(traces) == 2
    assert len(traces[0]) == 5


def test_traces_to_sequences_valid():
    mock_agent = MockRLAgent()
    traces = [[(1,2,3), (4,5,6), (7,8,9)]]
    sequences, targets = __traces_to_sequences(traces=traces, agent=mock_agent, seq_len=2, seq_per_trace=2)
    assert len(sequences) >= 1
    assert len(targets) >= 1
    assert isinstance(sequences[0][0], list)


def test_traces_to_sequences_no_action():
    mock_agent = MockRLAgent(trace_format=["state", "reward"])
    with pytest.raises(ValueError, match="hasn't 'action'"):
        __traces_to_sequences([], mock_agent, 1, 1)


def test_traces_to_sequences_none_format():
    mock_agent = MockRLAgent()
    mock_agent._trace_format = None
    print(mock_agent.get_trace_format())
    with pytest.raises(ValueError, match="None trace format"):
        __traces_to_sequences([], mock_agent, 1, 1)


def test_traces_to_sequences_short_trace():
    mock_agent = MockRLAgent()
    short_trace = [[(1,2)]]
    with pytest.raises(ValueError, match="lower than sequence length"):
        __traces_to_sequences(short_trace, mock_agent, seq_len=2, seq_per_trace=1)


def test_create_dataset_full_flow():
    traces = __generate_traces(mock_agent_constructor, mock_env_constructor, 1, 5)
    assert len(traces) == 1
    dataset = create_dataset(mock_agent_constructor, mock_env_constructor, 1, 5, 2, 1)
    assert isinstance(dataset, SequenceDataset)
    assert len(dataset) > 0

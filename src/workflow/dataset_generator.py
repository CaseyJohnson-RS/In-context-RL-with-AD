from typing import List, Tuple, Callable, Any
from src.environments import Environment 
from src.models.agents import RLAgent
from .SequenceDataset import SequenceDataset


def __generate_traces(
        agent_constructor: Callable[[], RLAgent],
        env_constructor: Callable[[], Environment],
        trace_count: int,
        trace_len: int
    ) -> List[List[Tuple]]:

    traces: List[List[Tuple]] = []

    for _ in range(trace_count):

        agent: RLAgent = agent_constructor()
        trace = agent.trace(env_constructor=env_constructor, trace_len=trace_len)

        traces.append(trace)

    return traces


def __traces_to_sequences(traces: List[List[Tuple]], agent: RLAgent,  seq_len: int, seq_per_trace: int) -> Tuple[List[List[List]], List[Any]]:

    sequences: List[List[List]] = []
    targets: List[Any] = []

    trace_format = agent.get_trace_format()
    trace_zeroed = agent.get_trace_zeroed()
    
    if trace_format is None:
        raise ValueError(f"Agent '{agent.__class__.__name__}' has None trace format!")
    
    if trace_zeroed is None:
        raise ValueError(f"Agent '{agent.__class__.__name__}' has None zeroed trace!")
    
    if "action" not in trace_format:
        raise ValueError(f"Agent's ({agent.__class__.__name__}) trace format hasn't 'action' in trace!")

    action_idx = trace_format.index("action")
    trace_step_len = len(trace_format)

    for trace in traces:
        trace_len = len(trace)

        if trace_len // seq_per_trace == 0:
            raise ValueError(
                f"Length of traces ({trace_len}) is larger than amount of generating sequences ({seq_per_trace})!"
            )
        
        if trace_len < seq_len:
            raise ValueError(
                f"Length of trace ({trace_len}) is lower than sequence length ({seq_len})!"
            )

        for t in range(trace_len - 1, 0, -(trace_len // seq_per_trace)):
            start = max(0, t - seq_len)
            window = trace[start:t]

            target = trace[t][action_idx]
            sequence: List[List] = [[] for _ in range(trace_step_len)]

            for _ in range(seq_len - t + start):
                for i in range(trace_step_len):
                    sequence[i].append(trace_zeroed[i])

            for trace_step in window:
                for i in range(trace_step_len):
                    sequence[i].append(trace_step[i])
            
            sequences.append(sequence)
            targets.append(target)

    return sequences, targets


def create_dataset(
        agent_constructor: Callable[[], RLAgent],
        env_constructor: Callable[[], Environment],
        trace_count: int,
        trace_len: int,
        seq_len: int,
        seq_per_trace: int
    ) -> SequenceDataset:

    traces = __generate_traces(
        agent_constructor=agent_constructor, 
        env_constructor=env_constructor,
        trace_count=trace_count,
        trace_len=trace_len
    )

    sequences, targets = __traces_to_sequences(
        traces=traces,
        agent=agent_constructor(),
        seq_len=seq_len,
        seq_per_trace=seq_per_trace
    )

    dataset = SequenceDataset(sequences=sequences, targets=targets)

    return dataset


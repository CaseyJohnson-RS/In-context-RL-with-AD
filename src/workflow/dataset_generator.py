from typing import List, Tuple, Callable
from src.environments import Environment 
from src.models.agents import RLAgent
from .SequenceDataset import SequenceDataset


def __generate_traces(
        agent_constructor: Callable[[], RLAgent],
        env_constructor: Callable[[], Environment],
        trace_count: int,
        trace_len: int
    ) -> List[List[Tuple[int, float]]]:

    traces: List[List[Tuple[int, float]]] = []

    for _ in range(trace_count):

        agent: RLAgent = agent_constructor()
        trace = agent.trace(env_constructor=env_constructor, trace_len=trace_len)

        traces.append(trace)

    return traces


def __traces_to_sequences(traces: List, agent: RLAgent,  seq_len: int, seq_per_trace: int) -> List[Tuple]:

    sequences: List[Tuple] = []
    action_idx = agent.get_trace_format().index("action")
    trace_step_len = len(agent.get_trace_format())
    get_trace_zeroed = agent.get_trace_zeroed()

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

            sequence = ([[] for _ in range(trace_step_len)] + [target])

            for _ in range(seq_len - t + start):
                for i in range(trace_step_len):
                    sequence[i].append(get_trace_zeroed[i])

            for trace_step in window:
                for i in range(trace_step_len):
                    sequence[i].append(trace_step[i])
            
            for i in range(action_idx):
                sequence[i].append(trace[t][i])
            
            sequences.append(sequence)

    return sequences


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

    sequences = __traces_to_sequences(
        traces=traces,
        agent=agent_constructor(),
        seq_len=seq_len,
        seq_per_trace=seq_per_trace
    )

    dataset = SequenceDataset(sequences=sequences)

    return dataset


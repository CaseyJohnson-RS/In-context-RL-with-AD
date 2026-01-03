from collections import deque
from rich.console import Console
from typing import Callable, List, Tuple
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


from src import set_seed
from src.workflow import MLFlowManager
from src.models.transformers.AR import (ARAddTransformer, ARConTransformer, ARMulTransformer, ARTransformerBase)
from src.models.agents import ThompsonSamplingAgent
from src.environments import KArmedBandit
from src.workflow import create_dataset
from src.workflow.SequenceDataset import SequenceDataset


console = Console()


# ---------------------- INIT ---------------------- #

def init_mlflow(CONFIG: dict) -> MLFlowManager:
    mlflow = MLFlowManager(
        experiment_name=CONFIG['experiment_name'],
        track_experiment=CONFIG['track_experiment']
    )
    mlflow.connect()
    return mlflow

def create_experiment_components(CONFIG: dict) -> Tuple[Callable[[], ThompsonSamplingAgent], Callable[[], ARTransformerBase], Callable[[], KArmedBandit]]:

    def model_constructor() -> ARTransformerBase:
        models = {
            "ARAddTransformer": ARAddTransformer,
            "ARConTransformer": ARConTransformer,
            "ARMulTransformer": ARMulTransformer
        }

        if "model" in CONFIG and CONFIG["model"] in models:
            model = models[CONFIG["model"]](**CONFIG['model_args']) # type: ignore
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            return model
        else:
            raise ValueError(f"Check model name. Got '({None if 'model' not in CONFIG else CONFIG['model']})'")
    
    def agent_constructor() -> ThompsonSamplingAgent:
        return ThompsonSamplingAgent(**CONFIG['agent_args'])

    def env_constructor() -> KArmedBandit:
        return KArmedBandit(**CONFIG['env_args'])

    return agent_constructor, model_constructor, env_constructor

# ---------------------- EXPERIMENT ---------------------- #

def rollout_transformer(
    model: ARTransformerBase,
    env,
    seq_len: int,
    max_steps: int,
    pad_action: int,
) -> float:
    model.eval()
    device = next(model.parameters()).device

    a_buffer: deque[int] = deque(maxlen=seq_len)
    r_buffer: deque[float] = deque(maxlen=seq_len)

    total_reward = 0.0
    steps = 0

    for _ in range(max_steps):
        steps += 1

        a_seq = [pad_action] * (seq_len - len(a_buffer)) + list(a_buffer)
        r_seq = [0.0] * (seq_len - len(r_buffer)) + list(r_buffer)

        X = torch.stack([
            torch.tensor(a_seq, dtype=torch.long, device=device),
            torch.tensor(r_seq, dtype=torch.float32, device=device),
        ], dim=0).unsqueeze(0)  # (1, 2, L)

        with torch.no_grad():
            action = int(torch.argmax(model(X), dim=-1).item())

        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        a_buffer.append(action)
        r_buffer.append(reward)

        if terminated or truncated:
            break

    return total_reward / steps


def rollout_agent(
    agent: ThompsonSamplingAgent,
    env,
    max_steps: int,
) -> float:
    total_reward = 0.0
    steps = 0

    for _ in range(max_steps):
        steps += 1

        action = agent._select()
        _, reward, terminated, truncated, _ = env.step(action)

        agent._update(action, reward)
        total_reward += reward

        if terminated or truncated:
            break

    return total_reward / steps


def evaluate_parallel(
    model: ARTransformerBase,
    agent: ThompsonSamplingAgent,
    env_constructor,
    CONFIG: dict,
) -> Tuple[float, float]:

    env_model = env_constructor()
    env_agent = env_constructor()

    model_avg = rollout_transformer(
        model=model,
        env=env_model,
        seq_len=CONFIG["seq_len"],
        max_steps=CONFIG["test_episode_len"],
        pad_action=CONFIG["pad_action"],
    )

    agent_avg = rollout_agent(
        agent=agent,
        env=env_agent,
        max_steps=CONFIG["test_episode_len"],
    )

    return model_avg, agent_avg



def experiment(
        mlflow: MLFlowManager,
        model_constructor: Callable[[], ARTransformerBase],
        agent_constructor: Callable[[], ThompsonSamplingAgent],
        env_constructor: Callable[[], KArmedBandit],
        dataset: SequenceDataset,
        CONFIG: dict
    ):

    mlflow.start_run(CONFIG['run_name'])

    model = model_constructor()

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()
    epochs: int = CONFIG['epochs']

    dataloader: DataLoader = DataLoader(dataset, **CONFIG['dataloader_args'])
    
    for epoch in range(1, epochs + 1):

        # TRAIN
        
        model.train()

        total_loss = 0.0
        avg_loss = 0.0
        batch_count = 0

        tdataloader = tqdm(dataloader)

        tdataloader.set_description("Train")
        for x, y in tdataloader: # (B, 2, L), (B)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_count += 1
            total_loss += float(loss.item())
            avg_loss = total_loss / max(batch_count, 1)

            tdataloader.set_postfix({"Average loss": avg_loss})

        # TEST

        model_avg_rewards: List[float] = []
        agent_avg_rewards: List[float] = []
        trange = tqdm(range(1, CONFIG['test_episodes'] + 1))
        trange.set_description("Test")

        for t in trange:
            model_avg, agent_avg = evaluate_parallel(model, agent_constructor(), env_constructor, CONFIG)
            model_avg_rewards.append(model_avg)
            agent_avg_rewards.append(agent_avg)

            trange.set_postfix({"Average reward": sum(model_avg_rewards) / t})
            
        
        # LOGGING

        metrics = {
            "AVG model reward": sum(model_avg_rewards) / len(model_avg_rewards),
            "AVG agent reward": sum(agent_avg_rewards) / len(agent_avg_rewards),
            "AVG train loss": avg_loss
        }

        mlflow.log_metrics(metrics, step=epoch)
    
    mlflow.end_run()
    

# ---------------------- MAIN ---------------------- #


def run(CONFIG: dict):
    set_seed(CONFIG['seed'])
    mlflow = init_mlflow(CONFIG)
    agent_constructor, model_constructor, env_constructor = create_experiment_components(CONFIG)

    dataset = create_dataset(
        agent_constructor=agent_constructor,
        env_constructor=env_constructor,
        trace_count=CONFIG['trace_count'],
        trace_len=CONFIG['trace_len'],
        seq_len=CONFIG['seq_len'],
        seq_per_trace=CONFIG['seq_per_trace']
    )

    experiment(mlflow, model_constructor, agent_constructor, env_constructor, dataset, CONFIG)

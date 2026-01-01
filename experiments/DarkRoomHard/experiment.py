from typing import Callable, Tuple
from rich.console import Console
from src import set_seed
from src.workflow import MLFlowManager
from src.models.agents import DQNAgent
from src.environments import DarkRoom
from tqdm import tqdm

console = Console()


# ---------------------- INIT ---------------------- #

def initialize(CONFIG: dict) -> Tuple[MLFlowManager, Callable[[], DQNAgent], Callable[[], DarkRoom]]:

    mlflow = MLFlowManager(
        experiment_name=CONFIG['experiment_name'],
        track_experiment=CONFIG['track_experiment']
    )
    mlflow.connect()

    def agent_constructor() -> DQNAgent:
        return DQNAgent(**CONFIG['agent_args'])
    
    def env_constructor() -> DarkRoom:
        return DarkRoom(**CONFIG['env_args'])

    agent = agent_constructor()
    agent.reset()
    agent.train(env_constructor=env_constructor, episodes=1)
    agent.test(env_constructor=env_constructor, episodes=1)

    console.print("[green]✓[/green] Agent and DarkRoom are checked")

    return mlflow, agent_constructor, env_constructor

# ---------------------- TRAIN ---------------------- #

def train_agent(
        mlflow: MLFlowManager,
        agent_constructor: Callable[[], DQNAgent],
        env_constructor: Callable[[], DarkRoom],
        CONFIG: dict
    ):
    mlflow.start_run(CONFIG['run_name'])
    mlflow.log_params(CONFIG)

    train_episodes = CONFIG['train_episodes']
    test_episodes = CONFIG['test_episodes']
    log_episodes_interval = CONFIG['log_episodes_interval']

    agent = agent_constructor()

    with tqdm(total=train_episodes, unit=" episode") as pbar:

        for step in range(1, train_episodes + 1, log_episodes_interval):

            pbar.set_description("Training...")
            metrics, info = agent.train(env_constructor=env_constructor, episodes=log_episodes_interval)
            mlflow.log_metrics(metrics, step)

            pbar.update(log_episodes_interval)

            pbar.set_description("Testing...")
            metrics, info = agent.test(env_constructor=env_constructor, episodes=test_episodes)

            mlflow.log_metrics(metrics, step)
            pbar.set_postfix(info)

    mlflow.end_run()
    console.rule("[green]Train complete[/green]", style="bright black")


# ---------------------- MAIN ---------------------- #

def run(CONFIG: dict):
    set_seed(CONFIG['seed'])
    mlflow, agent_constructor, env_constructor = initialize(CONFIG)
    train_agent(mlflow, agent_constructor, env_constructor, CONFIG)
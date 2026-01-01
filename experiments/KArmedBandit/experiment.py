from rich.console import Console
from typing import Callable, Tuple
from tqdm import tqdm

from src import set_seed
from src.workflow import MLFlowManager
from src.models.agents import ThompsonSamplingAgent
from src.environments import KArmedBandit


console = Console()


# ---------------------- INIT ---------------------- #

def initialize(CONFIG: dict) -> Tuple[MLFlowManager, Callable[[], ThompsonSamplingAgent], Callable[[], KArmedBandit]]:

    mlflow = MLFlowManager(
        experiment_name=CONFIG['experiment_name'],
        track_experiment=CONFIG['track_experiment']
    )
    mlflow.connect()

    def agent_constructor() -> ThompsonSamplingAgent:
        return ThompsonSamplingAgent(**CONFIG['agent_args'])
    
    gen_environment = KArmedBandit(**CONFIG['env_args'])
    def env_constructor() -> KArmedBandit:
        return gen_environment

    agent = agent_constructor()
    agent.reset()
    agent.train(env_constructor=env_constructor, episodes=100)

    console.print("[green]✓[/green] Agent and Environment are checked")

    return mlflow, agent_constructor, env_constructor

# ---------------------- TRAIN ---------------------- #

def train_agent(
        mlflow: MLFlowManager,
        agent_constructor: Callable[[], ThompsonSamplingAgent],
        env_constructor: Callable[[], KArmedBandit],
        CONFIG: dict
    ):

    mlflow.start_run(CONFIG['run_name'])
    mlflow.log_params(CONFIG)

    train_steps = CONFIG['train_steps']
    log_interval = CONFIG['log_interval']

    agent = agent_constructor()

    with tqdm(total=train_steps, desc="Training", unit=" step") as pbar:

        for step in range(1, train_steps + 1, log_interval):

            agent.train(env_constructor=env_constructor, episodes=log_interval)
            metrics, info = agent.test(env_constructor=env_constructor, episodes=5000)

            mlflow.log_metrics(metrics, step)

            pbar.update(log_interval)
            pbar.set_postfix(info)

    mlflow.end_run()
    console.rule("[green]Train complete[/green]", style="bright black")


# ---------------------- MAIN ---------------------- #


def run(CONFIG: dict):
    set_seed(CONFIG['seed'])
    mlflow, agent_constructor, env_constructor = initialize(CONFIG)
    train_agent(mlflow, agent_constructor, env_constructor, CONFIG)

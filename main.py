import importlib.util
import os
from pathlib import Path
from typing import Dict

from questionary import questionary
from rich.console import Console

console = Console()

# --- CONF LOAD --- 

def load_environ() -> str:
    from dotenv import load_dotenv
    console.print("→ Load environment variables")

    load_dotenv(override=True)

    os.environ["USERNAME"] = "Unknown" if not os.getenv("USERNAME") else os.environ["USERNAME"]

def pasre_args() -> Dict:
    import argparse
    console.print("→ Check general configs")

    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment', '-e', type=str)
    parser.add_argument('--seed', '-s', type=int)
    parser.add_argument('--autoconfirm', '-a', action='store_true')
    parser.add_argument('--track', '-t', action='store_true')
    parser.add_argument('--run_name', '-r', type=str)

    args = parser.parse_args()
    return {k: v for k, v in vars(args).items() if v is not None}

def load_experiments() -> Dict:
    from src import load_yaml
    console.print("→ Load experiments' configs")

    exp_dirs = [entry for entry in  Path('experiments').iterdir() if entry.is_dir()]

    experiments = []
    for exp_dir in exp_dirs:
        try:
            path = exp_dir / 'config.yaml'
            config = load_yaml(path)
            experiments.append((config, exp_dir))
            console.print(f"\t[green]✓[/green] {config['experiment_name']}")
        except Exception as e:
            console.print(f"\t[red]X[/red] {str(exp_dir)}: {e}")
    
    return experiments
    
# --- INIT ---

def choose_experiment(GENERAL_CONFIG: Dict, EXP_CONFIGS: Dict) -> int:
    names = [rec[0]['experiment_name'] for rec in EXP_CONFIGS]
    question = questionary.select("Choose the experiment", names)
    name = GENERAL_CONFIG.get("experiment", False)

    if not name:
        name = question.ask()
    elif name not in names:
        console.print("! Experiment with name {GENERAL_CONFIG.get('experiment')} does not exist")
        name = question.ask()
        del GENERAL_CONFIG['experiment']

    return names.index(name)

def preprocess_experiment_configs(GENERAL_CONFIG: Dict, CONFIG: Dict) -> Dict:
    CONFIG.update(GENERAL_CONFIG)
    
    import torch
    CONFIG['device'], color = ('cuda', 'green') if torch.cuda.is_available() else ('cpu', 'red')
    console.print(f'→ Device [{color}]{CONFIG["device"]}[/{color}]')

    CONFIG['track_experiment'] = bool(CONFIG.get('track_experiment', False))
    status, color = ('enabled', 'green') if CONFIG['track_experiment'] else ('disabled', 'red')
    console.print(f'→ Experiment tracking [{color}]{status}[/{color}]')

    if CONFIG['track_experiment']:
        name = CONFIG.get('run_name')
        if not isinstance(name, str) or not name.strip():
            CONFIG['run_name'] = questionary.text(
                'Enter run name:',
                default='Test'
            ).ask()
        console.print(f"→ Run name: [green]{CONFIG['run_name']}[/green]")

    if not CONFIG.get("autoconfirm", False):
        from src import pretty_print_yaml
        pretty_print_yaml(CONFIG, console, 4, 5)
        if not questionary.confirm("Is configuration correct?", default=True).ask():
            exit()
    
    return CONFIG

# === MAIN LOGIC ===

console.rule('[magenta]Load configs and environment variables[/magenta]')

load_environ()
GENERAL_CONFIG = pasre_args()
EXP_CONFIGS = load_experiments()

console.rule('[magenta]Initializing[/magenta]')

exp_index = choose_experiment(GENERAL_CONFIG, EXP_CONFIGS)
CONFIG = preprocess_experiment_configs(GENERAL_CONFIG, EXP_CONFIGS[exp_index][0])

console.rule('[magenta]Start experiment[/magenta]')

file_path = str(EXP_CONFIGS[exp_index][1] / 'experiment.py')
module_name = 'experiment'

spec = importlib.util.spec_from_file_location(module_name, file_path)
if spec is None:
    console.print(f'[red]X[/red] Cannot find experiment script: {file_path}')
else:
    my_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(my_module)
    my_module.run(CONFIG)
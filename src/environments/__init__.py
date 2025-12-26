from .Environment import Environment # noqa: F401
from .KArmedBandit import KArmedBandit
from .DarkRoom import DarkRoom

environment_collection = {
    "KArmedBandit": KArmedBandit,
    "DarkRoom": DarkRoom
}

def create_environment(env_name: str, env_args: dict) -> Environment:
    return environment_collection[env_name](**env_args)
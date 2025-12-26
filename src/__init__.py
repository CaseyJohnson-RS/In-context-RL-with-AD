import numpy as np
from pathlib import Path
import random
from rich.console import Console
from rich.text import Text
import torch
from typing import Optional, Any, Dict, List, Union
import yaml


def set_seed(seed: int) -> None:
    """
    Устанавливает фиксированное значение случайного генератора для всех используемых библиотек,
    чтобы обеспечить воспроизводимость экспериментов.

    Аргументы:
        seed (int): Целое число, используемое как зерно (seed) для генераторов случайных чисел.

    Действует на:
        - стандартный генератор случайных чисел Python
        - генератор случайных чисел NumPy
        - генератор случайных чисел PyTorch (CPU и CUDA)
    """
    random.seed(seed)  # Для стандартного модуля random Python
    np.random.seed(seed)  # Для NumPy
    torch.manual_seed(seed)  # Для PyTorch на CPU
    torch.cuda.manual_seed_all(seed)  # Для всех доступных GPU


def load_yaml(filepath: str | Path) -> Optional[dict]:
    """
    Loads a YAML file and returns its contents as a dictionary.

    Args:
        filepath: path to YAML file (string or Path object)

    Returns:
        Dictionary with YAML file data or None on error

    Examples:
        >>> data = load_yaml_file("config.yaml")
        >>> if data is not None:
        ...     print(data)
    """
    file_path = Path(filepath)

    with file_path.open('r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise FileExistsError(f"File content is not a dictionary: {file_path}")
        
    return data
    

def pretty_print_yaml(
    data: Union[Dict, List, str, Path],
    console: Console = None,
    indent: int = 0,
    max_depth: int = 10,
    current_depth: int = 0
) -> None:
    """
    Красиво выводит YAML-данные с подсветкой типов и визуализацией вложенности.

    Args:
        data: Словарь, список, путь к файлу YAML или строка с YAML.
        console: Объект Console из rich (если None — создаётся новый).
        indent: Начальный отступ (в пробелах).
        max_depth: Максимальная глубина рекурсии.
        current_depth: Текущая глубина для рекурсивного вызова.
    """
    if console is None:
        console = Console()

    # Загрузка данных, если это путь или строка YAML
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.exists() and path.is_file():
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            # Пытаемся загрузить как строку YAML
            data = yaml.safe_load(data)

    if current_depth >= max_depth:
        console.print(Text("... (max depth reached)", style="dim italic"))
        return

    indent_str = "  " * indent  # Отступ для текущего уровня

    if isinstance(data, dict):
        for key, value in data.items():
            # Ключ (всегда строковый, выделяем жёлтым)
            key_text = Text(f"{indent_str}{key}: ", style="bold yellow")
            
            if isinstance(value, (dict, list)):
                # Вложенная структура — просто ключ и переход на следующую строку
                console.print(key_text)
                pretty_print_yaml(
                    value, console, indent + 1, max_depth, current_depth + 1
                )
            else:
                # Простое значение — ключ и значение в одной строке
                value_text = format_value(value)
                console.print(key_text + value_text)
    
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                console.print(Text(f"{indent_str}-", style="dim"))
                pretty_print_yaml(
                    item, console, indent + 1, max_depth, current_depth + 1
                )
            else:
                value_text = format_value(item)
                console.print(Text(f"{indent_str}- ", style="dim") + value_text)
    
    else:
        # Одиночное значение (не контейнер)
        value_text = format_value(data)
        console.print(Text(f"{indent_str}", style="dim") + value_text)


def format_value(value: Any) -> Text:
    """Формирует Text-объект с подсветкой типа значения."""
    text = Text()
    
    if value is None:
        text.append("null", style="bold magenta")
    elif isinstance(value, bool):
        text.append(str(value).lower(), style="bold cyan")
    elif isinstance(value, (int, float)):
        text.append(str(value), style="bold green")
    elif isinstance(value, str):
        # Проверяем, похоже ли на путь/файл
        if any(ext in value.lower() for ext in [".py", ".txt", ".csv", ".json", ".yaml", ".yml"]):
            text.append(f'"{value}"', style="italic blue")
        else:
            text.append(f'"{value}"', style="white")
    else:
        # Для прочих типов — просто строковое представление
        text.append(str(value), style="dim")
    
    return text


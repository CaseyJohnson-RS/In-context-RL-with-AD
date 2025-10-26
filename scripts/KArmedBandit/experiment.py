# ============================================
# experiment.py — эксперимент с AD K-Armed Bandit
# ============================================
import os
import sys
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch

# --------------------------------------------
# Устанавливаем корень проекта, чтобы Python видел 'src'
# --------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)

from src.utils import (  # noqa: E402
    karmedbandit_generate_traces,
    karmedbandit_trajectories_to_sequences,
    karmedbandit_evaluate,
    KArmedBanditTrainer,
    set_seed,
)
from src.models import ARAddTransformer, ARConcatTransformer, ARMultiplyTransformer  # noqa: E402

# --------------------------------------------
# Типизация конфигурации
# --------------------------------------------
class Config:
    seed: int
    device: str

    K: int
    num_train_tasks: int
    num_val_tasks: int
    num_test_tasks: int
    T_train: int
    T_test: int
    seq_len: int

    model_type: str
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int

    batch_size: int
    epochs: int
    lr: float
    eval_every_epochs: int

    mlflow_server: str
    mlflow_port: int
    experiment_name: str

# --------------------------------------------
# Настройки гиперпараметров
# --------------------------------------------
CONFIG: Config = Config()
CONFIG.seed = 42
CONFIG.device = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG.K = 10
CONFIG.num_train_tasks = 50
CONFIG.num_val_tasks = 50
CONFIG.num_test_tasks = 50
CONFIG.T_train = 50
CONFIG.T_test = 50
CONFIG.seq_len = 50

CONFIG.model_type = "ARAddTransformer"  # ARAddTransformer, ARConcatTransformer, ARMultiplyTransformer
CONFIG.d_model = 64
CONFIG.nhead = 4
CONFIG.num_layers = 4
CONFIG.dim_feedforward = 2048

CONFIG.batch_size = 128
CONFIG.epochs = 20
CONFIG.lr = 1e-3
CONFIG.eval_every_epochs = 5

CONFIG.mlflow_server = "http://127.0.0.1"
CONFIG.mlflow_port = 5000
CONFIG.experiment_name = "AD K-Armed Bandit"

USER: str = "Casey Johnson"

# --------------------------------------------
# Инициализация окружения
# --------------------------------------------
os.environ["USER"] = USER
set_seed(CONFIG.seed)
print(f"Using device: {CONFIG.device}")

# --------------------------------------------
# Типы
# --------------------------------------------
MuSamplerType = Callable[[int], np.ndarray]
ModelClassType = type[ARAddTransformer] | type[ARConcatTransformer] | type[ARMultiplyTransformer]

# --------------------------------------------
# Фабрика модели
# --------------------------------------------
def create_model(
    model_type: str,
    n_actions: int,
    d_model: int,
    nhead: int,
    dim_feedforward: int,
    num_layers: int,
    max_len: int,
    device: str
) -> nn.Module:
    model_classes: dict[str, ModelClassType] = {
        "ARAddTransformer": ARAddTransformer,
        "ARConcatTransformer": ARConcatTransformer,
        "ARMultiplyTransformer": ARMultiplyTransformer,
    }
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    model_class = model_classes[model_type]
    model: nn.Module = model_class(
        n_actions=n_actions,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        max_len=max_len,
    ).to(device)
    return model

# --------------------------------------------
# Генерация данных
# --------------------------------------------
print("\n[1/5] Генерация данных...")
def generate_train_data(num_tasks: int, K: int, T: int, seq_len: int, seed: int) -> np.ndarray:
    traces: np.ndarray = karmedbandit_generate_traces(
        num_tasks,
        K,
        T,
        mu_sampler=lambda K: np.random.normal(0.0, 1.0, size=K).astype(np.float32),
        seed=seed,
    )
    sequences: np.ndarray = karmedbandit_trajectories_to_sequences(traces, seq_len=seq_len)
    return sequences

train_dataset: np.ndarray = generate_train_data(
    CONFIG.num_train_tasks, CONFIG.K, CONFIG.T_train, CONFIG.seq_len, CONFIG.seed
)

# --------------------------------------------
# Создание модели
# --------------------------------------------
print("[2/5] Инициализация модели...")
model: nn.Module = create_model(
    CONFIG.model_type,
    n_actions=CONFIG.K,
    d_model=CONFIG.d_model,
    nhead=CONFIG.nhead,
    dim_feedforward=CONFIG.dim_feedforward,
    num_layers=CONFIG.num_layers,
    max_len=CONFIG.seq_len,
    device=CONFIG.device
)

# --------------------------------------------
# Настройка MLflow
# --------------------------------------------
print("[3/5] Подключение к MLflow...")
mlflow_uri: str = f"{CONFIG.mlflow_server}:{CONFIG.mlflow_port}"
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment(CONFIG.experiment_name)

run_name: str = f"{CONFIG.model_type}_run_{int(time.time())}"

mlflow.autolog(
    log_input_examples=False,
    log_model_signatures=True,
    log_models=True,
    disable=False,
    exclusive=True,
    disable_for_unsupported_versions=True,
    silent=True
)

# --------------------------------------------
# Тренировка и оценка
# --------------------------------------------
with mlflow.start_run(run_name=run_name):
    mlflow.log_params(vars(CONFIG))  # логируем все параметры

    print("[4/5] Запуск обучения...")
    trainer: KArmedBanditTrainer = KArmedBanditTrainer(
        model=model,
        seqs=train_dataset,
        lr=CONFIG.lr,
        batch_size=CONFIG.batch_size,
        device=CONFIG.device
    )

    for epoch in range(1, CONFIG.epochs + 1):
        avg_loss: float = trainer.train_epoch(verbose=True)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        if epoch % CONFIG.eval_every_epochs == 0 or epoch == CONFIG.epochs:
            print(f"\n--- Evaluation at epoch {epoch} ---")
            summary: dict[str, tuple[float, float]] = karmedbandit_evaluate(
                model=trainer.model,
                K=CONFIG.K,
                T_test=CONFIG.T_test,
                n_tasks=CONFIG.num_test_tasks,
                device=CONFIG.device,
                mu_sampler=lambda K: np.random.normal(0.0, 1.0, size=K).astype(float),
                seq_len=CONFIG.seq_len,
            )

            for key in ["model", "teacher", "random"]:
                mlflow.log_metric(f"{key}_avg_test_reward", summary[key][0], step=epoch)
                mlflow.log_metric(f"{key}_stderr_test_reward", summary[key][1], step=epoch)

    # ----------------------------------------
    # Сохранение модели в MLflow
    # ----------------------------------------
    print("[5/5] Сохранение модели...")
    mlflow.pytorch.log_model(trainer.model, name="model")
    print(f"\n✅ Эксперимент завершён: {run_name}")
    print(f"✅ Модель успешно сохранена в MLflow под run: {run_name}")

mlflow.end_run()

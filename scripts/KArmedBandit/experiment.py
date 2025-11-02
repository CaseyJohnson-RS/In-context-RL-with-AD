# ============================================
# experiment.py — эксперимент с AD K-Armed Bandit
# ============================================
import os
import sys

import numpy as np
import torch.nn as nn
import mlflow
import mlflow.pytorch

# Устанавливаем корень проекта, чтобы Python видел 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)

from src.tools import set_seed # noqa: E402
from src.tools.KArmedBandit import TransformerTrainer, generate_train_sequences, evaluate # noqa: E402
from scripts.KArmedBandit.config import (  # noqa: E402
    SEED, 
    DEVICE, 
    TRAIN_SEQUENCES_CONFIG, 
    TRANSFORMER_TRAINER_CONFIG, 
    MODEL_CLASS,
    MODEL_CONFIG,
    MLFLOW_URI,
    EXPERIMENT_NAME,
    LOGGING_PARAMS,
    EVALUATE_CONFIG,
    EPOCHS,
    EVAL_EVERY_EPOCHS
)

# --------------------------------------------
# Инициализация окружения
# --------------------------------------------
os.environ["USER"] = input("Enter your name: ")
set_seed(SEED)
print(f"Using device: {DEVICE}")

# --------------------------------------------
# Генерация данных
# --------------------------------------------
print("\n[1/5] Data generation...")
train_dataset: np.ndarray = generate_train_sequences(**TRAIN_SEQUENCES_CONFIG)

# --------------------------------------------
# Создание модели
# --------------------------------------------
print("[2/5] Model initialization...")
model: nn.Module = MODEL_CLASS(**MODEL_CONFIG)

# --------------------------------------------
# Настройка MLflow
# --------------------------------------------
print("[3/5] Connecting to MLflow...")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

run_name: str = input(f"Enter run name (default=\'{MODEL_CLASS.__name__ }\'): ")
run_name = MODEL_CLASS.__name__ if len(run_name.strip()) == 0 else run_name.strip()

mlflow.autolog(silent=True)

# --------------------------------------------
# Тренировка и оценка
# --------------------------------------------
with mlflow.start_run(run_name=run_name):
    mlflow.log_params(LOGGING_PARAMS)  # логируем все параметры

    print("[4/5] Start learning...")
    trainer: TransformerTrainer = TransformerTrainer(
        model=model,
        seqs=train_dataset,
        **TRANSFORMER_TRAINER_CONFIG
    )

    for epoch in range(1, EPOCHS + 1):
        avg_loss: float = trainer.train_epoch(verbose=True)
        mlflow.log_metric("train loss", avg_loss, step=epoch)
        mlflow.log_metric("learning rate", trainer.optimizer.param_groups[0]['lr'], step=epoch)

        if epoch % EVAL_EVERY_EPOCHS == 0 or epoch == EPOCHS:
            print(f"\n--- Evaluation at epoch {epoch} ---")
            summary: dict[str, tuple[float, float]] = evaluate(
                model=trainer.model,
                **EVALUATE_CONFIG
            )

            for key in ["model", "teacher", "random"]:
                mlflow.log_metric(f"{key} avg test reward", summary[key][0], step=epoch)
                # mlflow.log_metric(f"{key} stderr test reward", summary[key][1], step=epoch)

    # ----------------------------------------
    # Сохранение модели в MLflow
    # ----------------------------------------
    print("[5/5] Saving model...")
    mlflow.pytorch.log_model(trainer.model, name="model")
    print(f"\nExperiment is complete: {run_name}")
    print(f"The model was successfully saved in MLflow under run: {run_name}")

mlflow.end_run()

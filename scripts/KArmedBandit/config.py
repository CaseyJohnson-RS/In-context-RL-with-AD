import numpy as np
import torch
from models.transformers.AR import ARAddTransformer, ARConTransformer, ARMulTransformer  # noqa: F401

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Параметры эксперимента
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

K = 10                  # Количество рук бандита
num_train_tasks = 196   # Количество обучающих траекторий
num_test_tasks = 128    # Количество тестовых траекторий (чем больше, тем лучше)
T_train = 128           # Длина обучающих траекторий
T_test = 128            # Длина тествых траекторий
seq_len = 128           # Длина контекста, который видит трансформер

model_class = ARAddTransformer         # Один из: ARConTransformer, ARAddTransformer, ARMulTransformer
d_model = 64                           # Размерность эмбеддинга
nhead = 4                              # Количество голов трансформера
num_layers = 4                         # Количество слоёв
dim_feedforward = 2048                 # Размерность полносвязного перцептрона

batch_size = 128                      # Размер батча
epochs = 150                          # Количество эпох
eval_every_epochs = 5                 # Производить оценку модели каждые N эпох
lr = 1e-3                             # Learning Rate (LR)
cosine_decay = True                   # Планировщик Learning Rate (True/False)
eta_min = 1e-5                        # Если планировщик LR включен, то к последней 
                                      # эпохе скорость обучения будет равна eta_min

mlflow_uri = "http://127.0.0.1:5000"    # URI MLFlow сервер
experiment_name = "AD K-Armed Bandit"   # Название эксперимента


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# DANGER ZONE
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


SEED = seed
DEVICE = device
MODEL_CLASS = model_class
MLFLOW_URI = mlflow_uri
EXPERIMENT_NAME = experiment_name
EPOCHS = epochs
EVAL_EVERY_EPOCHS = eval_every_epochs

TRAIN_SEQUENCES_CONFIG = {
  "num_tasks": num_train_tasks, 
  "K": K, 
  "T_per_task": T_train, 
  "seq_len": seq_len,
}

MODEL_CONFIG = {
  "n_actions": K,
  "d_model": d_model,
  "nhead": nhead,
  "num_layers": num_layers,
  "dim_feedforward": dim_feedforward,
  "max_len": seq_len,
}

TRANSFORMER_TRAINER_CONFIG = {
  "lr": lr,
  "batch_size": batch_size,
  "device": device,
  "cosine_decay": cosine_decay,
  "T_max": epochs,
  "eta_min": eta_min,
}

EVALUATE_CONFIG = {
    "K":K,
    "T_test":T_test,
    "n_tasks":num_test_tasks,
    "device":device,
    "mu_sampler":lambda K: np.random.normal(0.0, 1.0, size=K).astype(float),
    "seq_len":seq_len,
}

LOGGING_PARAMS = {
  "seed": seed,
  "K": K,
  "num_train_tasks": num_train_tasks,
  "num_test_tasks": num_test_tasks,
  "T_train": T_train,
  "T_test": T_test,            
  "seq_len": seq_len,

  "model_class": MODEL_CLASS.__name__,
  "d_model": d_model,
  "nhead": nhead,
  "num_layers": num_layers,
  "dim_feedforward": dim_feedforward,

  "batch_size": batch_size,
  "epochs": epochs,
  "eval_every_epochs": eval_every_epochs,
  "lr": lr,
  "cosine_decay": cosine_decay,
  "eta_min": eta_min,
}
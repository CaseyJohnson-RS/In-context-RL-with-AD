import pytest
import torch
import torch.nn as nn
import numpy as np

from src.utils import KArmedBanditTrainer


# --- Вспомогательная простая модель для тестов ---
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 10 (actions) + 10 (rewards) = 20 признаков
        self.fc = nn.Linear(20, 5)

    def forward(self, actions, rewards):
        actions = actions.float()
        rewards = rewards.squeeze(-1)
        x = torch.cat([actions, rewards], dim=-1)
        return self.fc(x)


# --- Вспомогательная функция для создания фейковых данных ---
def generate_dummy_seqs(n_samples=100, K=5, seq_len=10):
    seqs = []
    for _ in range(n_samples):
        actions = np.random.randint(0, K, size=seq_len)
        rewards = np.random.randn(seq_len)
        targets = np.random.randint(0, K, size=seq_len)
        seqs.append((actions, rewards, targets))
    return seqs


# --- Тест 1: корректная инициализация ---
def test_trainer_initialization():
    model = DummyModel()
    seqs = generate_dummy_seqs()
    trainer = KArmedBanditTrainer(model, seqs, lr=1e-3, batch_size=16, device="cpu")

    assert isinstance(trainer.model, nn.Module)
    assert trainer.batch_size == 16
    assert trainer.device == "cpu"
    assert isinstance(trainer.optimizer, torch.optim.Optimizer)
    assert hasattr(trainer, "train_epoch")


# --- Тест 2: одна эпоха обучения возвращает float и обновляет историю ---
def test_train_epoch_runs():
    model = DummyModel()
    seqs = generate_dummy_seqs()
    trainer = KArmedBanditTrainer(model, seqs, lr=1e-3, batch_size=16, device="cpu")

    loss = trainer.train_epoch(verbose=False)
    assert isinstance(loss, float)
    assert len(trainer.loss_history) == 1
    assert trainer.epoch == 1


# --- Тест 3: несколько эпох подряд ---
def test_train_epochs_runs_multiple_times(monkeypatch):
    model = DummyModel()
    seqs = generate_dummy_seqs()
    trainer = KArmedBanditTrainer(model, seqs, lr=1e-3, batch_size=16, device="cpu")

    # Подменяем train_epoch, чтобы ускорить тест
    called = []
    def fake_train_epoch(verbose):
        called.append(True)
        return 0.123
    monkeypatch.setattr(trainer, "train_epoch", fake_train_epoch)

    trainer.train_epochs(epochs=3)
    assert len(called) == 3


# --- Тест 4: устройство cuda при наличии GPU ---
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_trainer_on_cuda():
    model = DummyModel().to("cuda")
    seqs = generate_dummy_seqs()
    trainer = KArmedBanditTrainer(model, seqs, lr=1e-3, batch_size=16, device="cuda")

    loss = trainer.train_epoch(verbose=False)
    assert isinstance(loss, float)
    assert trainer.model.fc.weight.is_cuda

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

# ===== DummyTransformer для тестов =====
class DummyTransformer(nn.Module):
    def __init__(self, K=4):
        super().__init__()
        self.fc = nn.Linear(K * 2, K)  # actions + rewards

    def forward(self, actions, rewards):
        # Сплющиваем rewards до shape [batch_size, K]
        rewards_flat = rewards.view(rewards.size(0), -1)
        x = torch.cat([actions, rewards_flat], dim=-1)
        return self.fc(x)

# ===== Пример данных =====
@pytest.fixture
def seqs():
    return [
        ([1, 2, 3, 0], [0.5, 0.1, 0.7, 0.0], 2),
        ([1, 2, 3, 0], [0.5, 0.1, 0.7, 0.0], 2),
        ([1, 2, 3, 0], [0.5, 0.1, 0.7, 0.0], 2),
    ]

@pytest.fixture
def model():
    return DummyTransformer(K=4)

# ===== Тесты =====

def test_train_basic(model, seqs):
    from src.utils.k_armed_bandit import karmedbandit_train

    trained_model = karmedbandit_train(
        model,
        seqs,
        batch_size=2,
        epochs=2,
        lr=1e-3,
        device="cpu",
        verbose=False,
    )

    assert isinstance(trained_model, nn.Module)

def test_train_no_scheduler(model, seqs):
    from src.utils.k_armed_bandit import karmedbandit_train

    trained_model = karmedbandit_train(
        model,
        seqs,
        epochs=1,
        cosine_decay=False,
        verbose=False,
    )
    assert isinstance(trained_model, nn.Module)

# ===== Проверка логирования в MLflow =====
@patch("mlflow.active_run", return_value=True)
@patch("mlflow.log_metric")
def test_train_with_mlflow(mock_log_metric, mock_active_run, model, seqs):
    from src.utils.k_armed_bandit import karmedbandit_train

    karmedbandit_train(model, seqs, epochs=1, verbose=False)

    # Проверяем, что метрики логируются
    assert mock_log_metric.called
    mock_active_run.assert_called_once()

def test_train_small_batch(model, seqs):
    from src.utils.k_armed_bandit import karmedbandit_train

    trained_model = karmedbandit_train(
        model,
        seqs,
        batch_size=1,
        epochs=1,
        verbose=False,
    )

    assert isinstance(trained_model, nn.Module)

def test_train_verbose_output(model, seqs, capsys):
    from src.utils.k_armed_bandit import karmedbandit_train

    karmedbandit_train(model, seqs, epochs=1, verbose=True)

    captured = capsys.readouterr()
    assert "Loss" in captured.out
    assert "LR" in captured.out

def test_train_empty_dataset(model):
    from src.utils.k_armed_bandit import karmedbandit_train

    trained_model = karmedbandit_train(model, [], epochs=1, verbose=False)
    assert isinstance(trained_model, nn.Module)

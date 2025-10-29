import pytest
import torch
from src.models.a3c import A3CNet


@pytest.fixture
def base_model():
    """Базовая модель без LSTM"""
    return A3CNet(obs_dim=2, action_dim=5, use_lstm=False)


@pytest.fixture
def lstm_model():
    """Модель с LSTM"""
    return A3CNet(obs_dim=2, action_dim=5, use_lstm=True)


def test_forward_no_lstm(base_model):
    """Проверка базового прохода без LSTM"""
    obs = torch.randn(4, 2)
    logits, value, hx = base_model(obs)
    assert logits.shape == (4, 5)
    assert value.shape == (4, 1)
    assert hx is None


def test_forward_with_lstm(lstm_model):
    """Проверка прохода с LSTM"""
    obs = torch.randn(3, 4, 2)  # (T=3, B=4)
    logits, value, hx = lstm_model(obs)
    assert logits.shape == (3, 4, 5)
    assert value.shape == (3, 4, 1)
    assert isinstance(hx, tuple) and len(hx) == 2


@pytest.mark.parametrize("deterministic", [True, False])
def test_act_function(base_model, deterministic):
    """Проверка работы метода act"""
    obs = torch.randn(1, 2)
    action, log_prob, value, hx = base_model.act(obs, deterministic=deterministic)
    assert action.shape == (1,)
    assert isinstance(log_prob, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    assert (hx is None), "Без LSTM hx должно быть None"


def test_act_stochastic_behavior(base_model):
    """Стохастический выбор должен давать разные действия"""
    obs = torch.randn(1, 2)
    actions = [base_model.act(obs, deterministic=False)[0].item() for _ in range(30)]
    assert len(set(actions)) > 1, "Стохастический выбор не работает (все действия одинаковые)"


def test_evaluate_actions_shapes(base_model):
    """Проверка evaluate_actions"""
    obs = torch.randn(6, 2)
    actions = torch.randint(0, 5, (6,))
    log_probs, entropy, value = base_model.evaluate_actions(obs, actions)
    assert log_probs.shape == (6,)
    assert isinstance(entropy.item(), float)
    assert value.shape == (6,)


def test_lstm_hidden_state_persistence(lstm_model):
    """Проверка, что hidden состояние LSTM переносится между шагами"""
    hx = lstm_model.init_hidden(batch_size=1)
    obs1 = torch.randn(1, 2)
    obs2 = torch.randn(1, 2)
    _, _, hx1 = lstm_model(obs1, hx)
    _, _, hx2 = lstm_model(obs2, hx1)
    h1, c1 = hx1
    h2, c2 = hx2
    assert not torch.allclose(h1, h2), "Hidden состояния не должны быть одинаковыми"


def test_backward_pass(base_model):
    """Проверка корректности градиентов"""
    obs = torch.randn(8, 2)
    logits, value, _ = base_model(obs)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
    loss = -entropy + value.mean()
    loss.backward()
    for name, param in base_model.named_parameters():
        assert param.grad is not None, f"Градиент отсутствует у параметра {name}"
        assert not torch.isnan(param.grad).any(), f"NaN в градиентах параметра {name}"

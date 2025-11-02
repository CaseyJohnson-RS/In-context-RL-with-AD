import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

Tensor = torch.Tensor


class A3CNet(nn.Module):
    """
    Actor-Critic network for A3C-style agents.

    Поддерживает:
      - MLP shared body + отдельные головы для policy logits и value.
      - Опционально: LSTM между shared body и головами для работы с POMDP / историей.
    """

    def __init__(
        self,
        obs_dim: int = 2,
        action_dim: int = 5,
        hidden_size: int = 256,
        use_lstm: bool = False,
        lstm_layers: int = 1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.use_lstm = use_lstm
        self.device = device or torch.device("cpu")

        # Общая (shared) часть
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Опциональный LSTM (для учёта истории)
        if use_lstm:
            self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=lstm_layers)
            self._lstm_layers = lstm_layers

        # Головки: логиты для политики и предсказание value
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

        self.to(self.device)
        self._init_weights()

    def _init_weights(self):
        # Простая инициализация весов (адаптируйте при необходимости)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        if self.use_lstm:
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward(self, x: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass.

        Параметры:
          x: Tensor формы (B, obs_dim) или (T, B, obs_dim) — позволяет подавать последовательности.
          hx: скрытые состояния LSTM (h0, c0) каждый из формы (num_layers, B, hidden_size) — опционально.

        Возвращает:
          logits: (T, B, action_dim) или (B, action_dim)
          value:  (T, B, 1) или (B, 1)
          new_hx: новое скрытое состояние (если используется LSTM), иначе None
        """
        is_sequence = (x.dim() == 3)  # (T, B, obs_dim)
        if not is_sequence:
            # (B, obs_dim) -> (B, hidden)
            x = x.to(self.device)
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            if self.use_lstm:
                # LSTM ожидает (T=1, B, H)
                h_seq = h.unsqueeze(0)
                if hx is None:
                    hx = self.init_hidden(batch_size=h.size(0))
                lstm_out, new_hx = self.lstm(h_seq, hx)
                h_final = lstm_out.squeeze(0)
            else:
                new_hx = None
                h_final = h
            logits = self.policy_head(h_final)
            value = self.value_head(h_final)
            return logits, value, new_hx
        else:
            # Sequence mode
            # x: (T, B, obs_dim)
            T, B, _ = x.shape
            x = x.to(self.device)
            h = F.relu(self.fc1(x.view(-1, self.obs_dim)))
            h = F.relu(self.fc2(h))
            h = h.view(T, B, -1)  # (T, B, hidden)
            if self.use_lstm:
                if hx is None:
                    hx = self.init_hidden(batch_size=B)
                lstm_out, new_hx = self.lstm(h, hx)
                h_final = lstm_out  # (T, B, hidden)
            else:
                new_hx = None
                h_final = h
            logits = self.policy_head(h_final)   # (T, B, action_dim)
            value = self.value_head(h_final)     # (T, B, 1)
            return logits, value, new_hx

    def act(self, obs: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None, deterministic: bool = False):
        """
        Выбор действия (для взаимодействия с средой).
        Возвращает:
          action (torch.LongTensor): (B,) или (1,) с выбранными индексами действий
          log_prob (Tensor): лог-вероятность выбранного действия
          value (Tensor): предсказанный value
          new_hx: новое скрытое состояние LSTM или None
        """
        logits, value, new_hx = self.forward(obs, hx)
        # logits: (B, action_dim) или (T, B, action_dim) в seq-mode; здесь ожидаем single-step (B, action_dim)
        if logits.dim() == 3:
            # если последовательность, возьмём последний временной шаг
            logits = logits[-1]
            value = value[-1]
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1), new_hx

    def evaluate_actions(self, obs: Tensor, actions: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None):
        """
        Для вычисления loss: возвращает log_probs, entropy и value для заданных действий.
        `obs` может быть (B, obs_dim) или (T, B, obs_dim); `actions` должен соответствовать форме.
        """
        logits, value, _ = self.forward(obs, hx)
        if logits.dim() == 3:
            # (T, B, action_dim)
            T, B, _ = logits.shape
            logits_flat = logits.view(T * B, -1)
            actions_flat = actions.view(T * B)
        else:
            logits_flat = logits
            actions_flat = actions
        probs = F.softmax(logits_flat, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions_flat)
        entropy = dist.entropy().mean()
        # value -> flatten to match
        value_flat = value.view(-1)
        return log_probs, entropy, value_flat

    def init_hidden(self, batch_size: int = 1) -> Tuple[Tensor, Tensor]:
        """Инициализация нулевых hidden и cell для LSTM: формы (num_layers, B, H)."""
        if not self.use_lstm:
            return None
        h0 = torch.zeros(self._lstm_layers, batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros(self._lstm_layers, batch_size, self.hidden_size, device=self.device)
        return (h0, c0)

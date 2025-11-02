from torch import nn, optim
import torch
import time

from .batch_iterator import batch_iterator


class TransformerTrainer:
    def __init__(
        self,
        model,
        seqs,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
        cosine_decay: bool = True,
        T_max: int = 100,
        eta_min: float = 1e-4,
    ):
        """
        Инициализация тренера для трансформера в режиме Action-Decision (AD).
        """
        self.model = model.to(device)
        self.seqs = seqs
        self.batch_size = batch_size
        self.device = device

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = (
            optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min
            )
            if cosine_decay
            else None
        )

        self.loss_history: list = []
        self.epoch = 0

    def train_epoch(self, verbose: bool = True):
        """
        Выполняет одну эпоху обучения и возвращает средний лосс.
        """
        self.model.train()
        start_time = time.time()
        total_loss = 0.0
        batch_count = 0

        for actions_b, rewards_b, targets_b in batch_iterator(
            self.seqs, self.batch_size, device=self.device, shuffle=True
        ):
            logits = self.model(actions_b, rewards_b)  # (B, n_actions)
            
            # Модель предсказывает только следующее действие, значит цель — это targets_b[:, -1]
            if targets_b.ndim > 1:
                targets_b = targets_b[:, -1]
            
            loss = self.criterion(logits, targets_b)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1

        if self.scheduler:
            self.scheduler.step()

        avg_loss = total_loss / max(batch_count, 1)
        self.loss_history.append(avg_loss)
        self.epoch += 1

        if verbose:
            elapsed = time.time() - start_time
            print(
                f"Epoch {self.epoch:3d} | Loss: {avg_loss:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

        return avg_loss
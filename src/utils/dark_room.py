import torch
from torch.optim import Adam

class SharedAdam(Adam):
    """
    Реализация Adam для A3C с поддержкой shared memory.
    Каждое состояние (m, v) хранится в общей памяти для всех процессов.
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # Переносим внутренние состояния (m, v) в shared memory
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

                # перемещаем тензоры в общую память
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()
                state["step"].share_memory_()

    def share_memory(self):
        """Совместимо с API других shared объектов."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                for v in state.values():
                    if isinstance(v, torch.Tensor):
                        v.share_memory_()

from typing import Any, List
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, sequences: List[List[List]], targets: List[Any], device: str | None = None):

        if len(sequences) != len(targets):
            raise ValueError(f"Length of sequences ({len(sequences)}) and targets (({len(targets)})) must be same!")

        if len(sequences) <= 0:
            raise ValueError(f"Length of sequences and targets must be larger zero (got {len(sequences)})")

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.sequences = torch.Tensor(sequences.copy()).to(device)
        self.targets = torch.LongTensor(targets).to(device)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
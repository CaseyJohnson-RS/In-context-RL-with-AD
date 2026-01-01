from typing import Any, List
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, sequences: List[List[List]], targets: List[Any]):

        if len(sequences) != len(targets):
            raise ValueError(f"Length of sequences ({len(sequences)}) and targets (({len(targets)})) must be same!")

        if len(sequences) <= 0:
            raise ValueError(f"Length of sequences and targets must be larger zero (got {len(sequences)})")

        self.sequences = sequences.copy()
        self.targets = targets.copy()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
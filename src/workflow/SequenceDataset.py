import numpy as np
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, sequences):

        self.data = []
        self.target = []

        for sequence in sequences:
            self.data.append([])
            for i in range(len(sequence) - 1):
                self.data[-1].append(np.array(sequence[i]))
            self.target.append(sequence[-1])

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
import pytest
from typing import List, Any
# Assuming the code is in a module named sequence_dataset
from src.workflow.SequenceDataset import SequenceDataset  # Adjust import as needed


def test_init_valid_inputs():
    sequences = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    targets = [10, 20]
    dataset = SequenceDataset(sequences, targets)
    assert len(dataset.sequences) == 2
    assert dataset.sequences == sequences
    assert dataset.targets == targets


def test_init_length_mismatch():
    sequences = [[[1, 2]]]
    targets = [10, 20]
    with pytest.raises(ValueError, match="Length of sequences.*must be same"):
        SequenceDataset(sequences, targets)


def test_init_empty_sequences():
    sequences: List[List[List]] = []
    targets: List[Any] = []
    with pytest.raises(ValueError, match="must be larger zero"):
        SequenceDataset(sequences, targets)


def test_len_method():
    sequences = [[[1]], [[2]]]
    targets = [10, 20]
    dataset = SequenceDataset(sequences, targets)
    assert len(dataset) == 2


def test_getitem_valid_index():
    sequences = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    targets = [10, 20]
    dataset = SequenceDataset(sequences, targets)
    # Note: code has bug - tests assume fix to self.sequences[idx], self.targets[idx]
    seq, tgt = dataset[1]
    assert seq == [[5, 6], [7, 8]]
    assert tgt == 20


def test_getitem_out_of_range():
    sequences = [[[1]]]
    targets = [10]
    dataset = SequenceDataset(sequences, targets)
    with pytest.raises(IndexError):
        _ = dataset[1]

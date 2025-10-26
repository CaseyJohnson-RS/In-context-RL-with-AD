import pytest
from typing import List, Tuple
from src.utils import trajectories_to_sequences 


# ======== Вспомогательные данные ========

@pytest.fixture
def simple_trajectory() -> List[List[Tuple[int, float]]]:
    """Одна траектория длиной 5 для базовых тестов."""
    return [[(1, 0.5), (2, 1.0), (3, 0.2), (1, 0.8), (2, 0.3)]]


@pytest.fixture
def multiple_trajectories() -> List[List[Tuple[int, float]]]:
    """Несколько коротких траекторий."""
    return [
        [(1, 0.1), (2, 0.2), (3, 0.3)],
        [(2, 0.5), (1, 0.4)]
    ]


# ======== Тесты ========

def test_output_structure(simple_trajectory):
    """Функция должна возвращать список кортежей (actions, rewards, target)."""
    seqs = trajectories_to_sequences(simple_trajectory, seq_len=3)

    assert isinstance(seqs, list)
    assert all(isinstance(item, tuple) and len(item) == 3 for item in seqs)

    for actions, rewards, target in seqs:
        assert isinstance(actions, list)
        assert isinstance(rewards, list)
        assert isinstance(target, int)
        assert len(actions) == len(rewards) == 3


def test_sliding_window(simple_trajectory):
    """Проверка, что окно скользит корректно."""
    seqs = trajectories_to_sequences(simple_trajectory, seq_len=2)

    # Исходная траектория длиной 5 → должно быть 4 обучающих примера
    assert len(seqs) == 4

    # Проверим правильность первого окна
    actions, rewards, target = seqs[0]
    assert actions == [0, 1]      # pad + первое действие
    assert rewards == [0.0, 0.5]  # pad + первая награда
    assert target == 2            # следующее действие


def test_padding_behavior(simple_trajectory):
    """Проверяет корректный padding, когда seq_len больше длины истории."""
    seqs = trajectories_to_sequences(simple_trajectory, seq_len=10)

    actions, rewards, target = seqs[0]
    pad_count = 10 - 1  # при первом шаге только 1 действие в истории

    assert actions[:pad_count] == [0] * pad_count
    assert rewards[:pad_count] == [0.0] * pad_count
    assert actions[-1] == 1
    assert rewards[-1] == 0.5
    assert target == 2


def test_multiple_trajectories(multiple_trajectories):
    """Проверяет, что функция корректно обрабатывает несколько траекторий."""
    seqs = trajectories_to_sequences(multiple_trajectories, seq_len=2)

    # Траектория 1 длиной 3 → 2 примера, траектория 2 длиной 2 → 1 пример
    assert len(seqs) == 3

    # Проверим, что для каждой пары длина совпадает
    for actions, rewards, target in seqs:
        assert len(actions) == len(rewards) == 2
        assert isinstance(target, int)


def test_target_correctness(simple_trajectory):
    """Проверяет, что целевое действие соответствует следующему по времени."""
    seqs = trajectories_to_sequences(simple_trajectory, seq_len=3)
    traj = simple_trajectory[0]

    # Цель каждого примера должна совпадать с действием из оригинальной траектории на позиции t
    for i, (_, _, target) in enumerate(seqs):
        expected_target = traj[i + 1][0]
        assert target == expected_target


def test_empty_trajectory_handling():
    """Проверяет корректную обработку пустых или коротких траекторий."""
    trajectories = [[]]  # одна пустая траектория
    seqs = trajectories_to_sequences(trajectories, seq_len=5)
    assert seqs == [], "Пустая траектория не должна порождать обучающих примеров"

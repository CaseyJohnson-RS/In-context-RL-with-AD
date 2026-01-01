import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import (Any, Callable, Tuple, Dict, List)

from src.environments import DarkRoom
from .RLAgent import RLAgent


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])



class QNetwork(nn.Module):
    """Q-Network for DarkRoom with neighbor state encoding."""
    
    def __init__(self, state_size: int = 4, action_size: int = 5, hidden_size: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, action_size)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent(RLAgent):
    """DQN agent using local neighbor visitations for DarkRoom exploration."""
    
    state_size: int = 4
    action_size: int = 5
    
    trace_format: Tuple[str, ...] = ("state", "action", "reward")
    trace_zeroed: Tuple[Any, ...] = ((0.0, 0.0, 0.0, 0.0), 0, 0.0)
    
    def __init__(
        self,
        env_size: int = 9,
        lr: float = 5e-4,
        gamma: float = 0.95,
        buffer_size: int = 10000,
        batch_size: int = 32,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9998,
        target_update: int = 200,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.size = env_size
        self.grid_size = env_size ** 2
        self.gamma = gamma
        self.target_update = target_update
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_net = QNetwork(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr, eps=1e-5)
        
        self.memory: deque[Experience] = deque(maxlen=buffer_size)
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.visited_matrix: np.ndarray = np.zeros((env_size, env_size), dtype=np.float32)
        
        self.target_update_counter: int = 0
        self.steps: int = 0
    
    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.visited_matrix.fill(0.0)
        self.target_update_counter = 0
    
    def _mark_visited(self, pos: Tuple[int, int]) -> None:
        """Mark position as visited."""
        x, y = pos
        self.visited_matrix[y, x] = 1.0
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> np.ndarray:
        """Get normalized visitation counts of 4 neighbors."""
        x, y = pos
        neighbors = np.full(4, -0.5, dtype=np.float32)  # [up, down, left, right]
        
        if y > 0:
            neighbors[0] += self.visited_matrix[y - 1, x]
        if y < self.size - 1:
            neighbors[1] += self.visited_matrix[y + 1, x]
        if x > 0:
            neighbors[2] += self.visited_matrix[y, x - 1]
        if x < self.size - 1:
            neighbors[3] += self.visited_matrix[y, x + 1]
            
        return neighbors
    
    def _encode_state(self, pos: Tuple[int, int]) -> np.ndarray:
        """Encode position to neighbor state vector."""
        self._mark_visited(pos)
        return self._get_neighbors(pos)
    
    def _epsilon_greedy(self, state: np.ndarray) -> int:
        """Select action via epsilon-greedy policy."""
        if random.random() < max(self.epsilon, self.epsilon_end):
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax(1).item()
    
    def _remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store experience tuple in replay buffer."""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def _replay(self) -> None:
        """Train Q-network via Double DQN on random batch."""
        if len(self.memory) < self.batch_size:
            return
        
        batch: List[Experience] = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.array([e.state for e in batch], dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in batch], dtype=np.int64)).to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in batch], dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array([e.next_state for e in batch], dtype=np.float32)).to(self.device)
        dones = torch.from_numpy(np.array([e.done for e in batch], dtype=bool)).to(self.device)
        
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.q_net(next_states).max(1)[1].unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (self.gamma * next_q * (~dones).float())
        
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=0.5)
        self.optimizer.step()
    
    def _update_target(self) -> None:
        """Soft/hard update target network periodically."""
        self.target_update_counter += 1
        if self.target_update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
    
    def train(
        self, env_constructor: Callable[[], DarkRoom], episodes: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Train agent over episodes and return metrics."""
        rewards: List[float] = []
        lengths: List[int] = []
        successes: List[bool] = []
        
        for episode in range(episodes):
            env = env_constructor()
            obs, info = env.reset()
            self.reset()
            self._mark_visited(obs)
            
            state = self._encode_state(obs)
            total_reward = 0.0
            
            while True:
                action = self._epsilon_greedy(state)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                next_state = self._encode_state(next_obs)
                
                done = terminated or truncated
                self._remember(state, action, reward, next_state, done)
                self._replay()
                self._update_target()
                
                state = next_state
                total_reward += reward
                self.steps += 1
                
                if done:
                    break
            
            rewards.append(total_reward)
            lengths.append(env.steps)
            successes.append(next_info.get('goal_found', False))
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        metrics = {
            'train_avg_reward': np.mean(rewards),
            'train_avg_length': np.mean(lengths),
            'train_success_rate': np.mean(successes),
            'epsilon': self.epsilon,
        }
        
        info = {'unique_visited': int(np.sum(self.visited_matrix))}
        return metrics, info
    
    def test(
        self, env_constructor: Callable[[], DarkRoom], episodes: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Evaluate greedy agent over episodes."""
        epsilon_backup = self.epsilon
        self.epsilon = 0.0
        
        rewards: List[float] = []
        lengths: List[int] = []
        successes: List[bool] = []
        explored_norm: List[float] = []
        
        for episode in range(episodes):
            env = env_constructor()
            obs, info = env.reset()
            self.reset()
            self._mark_visited(obs)
            state = self._encode_state(obs)
            total_reward = 0.0
            
            while True:
                action = self._epsilon_greedy(state)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                state = self._encode_state(next_obs)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            rewards.append(total_reward)
            lengths.append(env.steps)
            successes.append(next_info.get('goal_found', False))
            explored_norm.append(np.sum(self.visited_matrix) / env.max_steps)
        
        self.epsilon = epsilon_backup
        
        metrics = {
            'test_avg_reward': np.mean(rewards),
            'test_avg_length': np.mean(lengths),
            'test_test_success_rate': np.mean(successes),
            'test_avg_exp_norm': np.mean(explored_norm),
        }
        
        info = {
            'AEN': np.mean(explored_norm),
        }
        return metrics, info
    
    def trace(self, env_constructor: Callable[[], DarkRoom], trace_len: int) -> List[Tuple[Tuple[int, int], int, float]]:
        """Generate deterministic trace (greedy policy)."""
        # FIXME: Implement trace method
        return [self.trace_zeroed + ()]

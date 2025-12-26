import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Any, Callable, Tuple, Dict

from src.environments import Environment
from .RLAgent import RLAgent

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class QNetwork(nn.Module):
    """Q-Network для DarkRoom."""
    
    def __init__(self, state_size: int = 4, action_size: int = 5, hidden_size: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent(RLAgent):

    def __init__(self, env_size: int = 9, lr: float = 5e-4, gamma: float = 0.95,
                 buffer_size: int = 10000, batch_size: int = 32,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.9998, target_update: int = 200,
                 hidden_size: int = 32):
        
        self.size = env_size
        self.grid_size = env_size * env_size
        self.state_size = 4  # neighbors(4)
        self.action_size = 5
        
        self.gamma = gamma
        self.target_update = target_update
        
        # Сети (только локальное состояние)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_net = QNetwork(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Оптимизатор
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr, eps=1e-5)
        
        # Replay buffer
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.visited_matrix = np.zeros((env_size, env_size), dtype=np.float32)
        
        # Stats
        self.target_update_counter = 0
        self.steps = 0
        
    def reset(self) -> None:
        self.visited_matrix.fill(0)
        
    def _mark_visited(self, pos: Tuple[int, int]):
        x, y = pos
        self.visited_matrix[y, x] = 1
        
    def _get_neighbors(self, pos: Tuple[int, int]) -> np.ndarray:
        x, y = pos
        neighbors = np.zeros(4, dtype=np.float32) - 0.5 # [up, down, left, right]
        
        if y > 0:
            neighbors[0] += self.visited_matrix[y-1, x]
        if y < self.size - 1:
            neighbors[1] += self.visited_matrix[y+1, x]
        if x > 0:
            neighbors[2] += self.visited_matrix[y, x-1]
        if x < self.size - 1:
            neighbors[3] += self.visited_matrix[y, x+1]
            
        return neighbors
    
    def _encode_state(self, pos: Tuple[int, int]) -> np.ndarray:
        state = self._get_neighbors(pos)
        self._mark_visited(pos)
        return state
    
    def _epsilon_greedy(self, state: np.ndarray) -> int:
        if np.random.random() < max(self.epsilon, self.epsilon_end):
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()
    
    def _remember(self, state: np.ndarray, action: int, reward: float,
                  next_state: np.ndarray, done: bool):
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def _replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([e.state for e in batch], dtype=np.float32)
        actions = np.array([e.action for e in batch], dtype=np.int64)
        rewards = np.array([e.reward for e in batch], dtype=np.float32)
        next_states = np.array([e.next_state for e in batch], dtype=np.float32)
        dones = np.array([e.done for e in batch], dtype=bool)
        
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).bool().to(self.device)
        
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.q_net(next_states).max(1)[1].unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (self.gamma * next_q * ~dones)
        
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.optimizer.step()
    
    def _update_target(self):
        self.target_update_counter += 1
        if self.target_update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
    
    def train(self, env_constructor: Callable[[], Environment], episodes: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        
        rewards, lengths, successes = [], [], []
        
        for episode in range(episodes):
            env = env_constructor()
            obs, info = env.reset()
            self.reset()
            
            self._mark_visited(obs)
            
            state = self._encode_state(obs)
            total_reward = 0
            
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
            # 'Average reward': np.mean(rewards),
            'epsilon': self.epsilon
        }
        
        info = {
            'unique_visited': int(np.sum(self.visited_matrix))
        }
        
        return metrics, info
    
    def test(self, env_constructor: Callable[[], Environment], episodes: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        
        epsilon_buf = self.epsilon
        self.epsilon = 0.0
        
        rewards, lengths, successes, explored_norm = [], [], [], []
        
        for episode in range(episodes):
            env = env_constructor()
            obs, info = env.reset()
            self.reset()
            self._mark_visited(obs)
            state = self._encode_state(obs)
            total_reward = 0
            
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
        
        self.epsilon = epsilon_buf
        
        metrics = {
            'Success rate': np.mean(successes),
            'Explored': np.mean(explored_norm)
        }   
        
        info = metrics.copy()
        return metrics, info
    
    def trace(self, env_constructor, trace_len):
        return super().trace(env_constructor, trace_len)

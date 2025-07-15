"""
DDQN (Double Deep Q-Network) Algorithm Implementation
Implements double Q-learning for offline RL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import namedtuple
from torch.utils.data import DataLoader, IterableDataset
import datetime

from .base_algorithm import BaseRLAlgorithm, Experience
from src.models.cql_network import GeneralQNetwork as QNetwork, ReplayBuffer
from config.base_config import ALL_BEGIN_TIME_DT, NUMERIC_FEATURES

class DDQNDataset(IterableDataset):
    """Iterable dataset for DDQN training with DataLoader."""
    def __init__(self, replay_buffer: ReplayBuffer):
        self.replay_buffer = replay_buffer
    def __iter__(self):
        while self.replay_buffer.is_ready():
            yield self.replay_buffer.sample()

class DDQNAlgorithm(BaseRLAlgorithm):
    """
    Double Deep Q-Network (DDQN) algorithm implementation.
    Extends BaseRLAlgorithm with DDQN-specific functionality.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # DDQN-specific parameters
        self.gamma = config.get('GAMMA', 0.99)
        self.tau = config.get('TAU', 1e-3)
        self.learning_rate = config.get('LEARNING_RATE', 1e-4)
        self.update_every = config.get('UPDATE_EVERY', 4)
        self.target_update_every = config.get('TARGET_UPDATE_EVERY', 100)
        # Training parameters
        self.batch_size = config.get('TRAINING_CONFIG', {}).get('batch_size', 64)
        self.buffer_size = config.get('TRAINING_CONFIG', {}).get('buffer_size', int(5e4))
        self.epsilon_start = config.get('DDQN_TRAINING_CONFIG', {}).get('epsilon_start', 1.0)
        self.epsilon_end = config.get('DDQN_TRAINING_CONFIG', {}).get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('DDQN_TRAINING_CONFIG', {}).get('epsilon_decay', 0.995)
        # Networks and training components
        self.q_local = None
        self.q_target = None
        self.optimizer = None
        self.replay_buffer = None
        # Training state
        self.epsilon = self.epsilon_start
        self.step_count = 0
        print(f"DDQN Algorithm initialized.")

    def build_network(self) -> None:
        if not self.embedding_sizes:
            raise ValueError("Embedding sizes not available. Load data first.")
        network_config = self.config.get('DDQN_NETWORK_CONFIG', {})
        embedding_dims = self.config.get('EMBEDDING_DIMS', {})
        self.q_local = QNetwork(
            num_categories_embed_size=self.embedding_sizes['num_categories'],
            num_sub_categories_embed_size=self.embedding_sizes['num_sub_categories'],
            num_industries_embed_size=self.embedding_sizes['num_industries'],
            cat_embed_dim=embedding_dims.get('category', 5),
            sub_cat_embed_dim=embedding_dims.get('sub_category', 8),
            ind_embed_dim=embedding_dims.get('industry', 5),
            total_numeric_features=sum(NUMERIC_FEATURES.values()),
            seed=network_config.get('seed', 42),
            fc1_units=network_config.get('fc1_units', 128),
            fc2_units=network_config.get('fc2_units', 32),
            dropout_rate=network_config.get('dropout_rate', 0.4)
        ).to(self.device)
        self.q_target = QNetwork(
            num_categories_embed_size=self.embedding_sizes['num_categories'],
            num_sub_categories_embed_size=self.embedding_sizes['num_sub_categories'],
            num_industries_embed_size=self.embedding_sizes['num_industries'],
            cat_embed_dim=embedding_dims.get('category', 5),
            sub_cat_embed_dim=embedding_dims.get('sub_category', 8),
            ind_embed_dim=embedding_dims.get('industry', 5),
            total_numeric_features=sum(NUMERIC_FEATURES.values()),
            seed=network_config.get('seed', 42),
            fc1_units=network_config.get('fc1_units', 128),
            fc2_units=network_config.get('fc2_units', 32),
            dropout_rate=network_config.get('dropout_rate', 0.4)
        ).to(self.device)
        self.q_target.load_state_dict(self.q_local.state_dict())
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DDQN")
            self.q_local = nn.DataParallel(self.q_local)
            self.q_target = nn.DataParallel(self.q_target)
        self.network = self.q_local
        print("DDQN networks built successfully")

    def act(self, state_options: List[Tuple], epsilon: float = 0.0) -> Optional[int]:
        if not state_options:
            return None
        current_epsilon = epsilon if epsilon > 0 else self.epsilon
        if random.random() > current_epsilon:
            self.q_local.eval()
            with torch.no_grad():
                id_features = []
                numeric_features = []
                for project_id, state_tuple in state_options:
                    id_features.append([state_tuple[0], state_tuple[1], state_tuple[2]])
                    numeric_features.append(state_tuple[3])
                id_features_tensor = torch.LongTensor(id_features).to(self.device)
                numeric_features_tensor = torch.FloatTensor(np.array(numeric_features)).to(self.device)
                q_values = self.q_local(id_features_tensor, numeric_features_tensor)
                action_idx = torch.argmax(q_values).item()
                selected_project_id = state_options[action_idx][0]
            self.q_local.train()
            return selected_project_id
        else:
            return random.choice(state_options)[0]

    def learn(self, experiences: List[Experience]) -> float:
        if not experiences:
            return 0.0
        batch_data = self._ddqn_collate_fn([experiences])
        for key in batch_data:
            if isinstance(batch_data[key], torch.Tensor):
                batch_data[key] = batch_data[key].to(self.device)
        # DDQN target calculation
        q_values = self.q_local(batch_data["id_features"], batch_data["numeric_features"])
        with torch.no_grad():
            next_q_local = self.q_local(batch_data["next_id_features"], batch_data["next_numeric_features"])
            next_q_target = self.q_target(batch_data["next_id_features"], batch_data["next_numeric_features"])
            next_actions = torch.argmax(next_q_local, dim=1, keepdim=True)
            next_q = next_q_target.gather(1, next_actions)
        q_targets = batch_data["rewards"] + (self.gamma * next_q * (1 - batch_data["dones"]))
        loss = F.mse_loss(q_values, q_targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_local.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def soft_update(self, local_model: nn.Module, target_model: nn.Module, tau: float) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def train(self, train_events: List[Dict], val_events: List[Dict]) -> Dict[str, Any]:
        if not self.q_local:
            self.build_network()
        print(f"Starting DDQN training with {len(train_events)} train events, {len(val_events)} val events")
        num_epochs = self.config.get('TRAINING_CONFIG', {}).get('num_epochs', 200)
        print_frequency = self.config.get('LOGGING_CONFIG', {}).get('print_frequency', 10)
        project_info = self.data_dict['project_info']
        entry_info = self.data_dict['entry_info']
        worker_quality_map = self.data_dict['worker_quality_map']
        worker_global_stats = self.data_dict['worker_global_stats']
        worker_cat_performance = self.data_dict['worker_cat_performance']
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            random.shuffle(train_events)
            for event in train_events:
                worker_id = event['worker_id']
                current_time = event['arrival_time_dt']
                available_projects = self.get_available_projects(current_time, project_info, entry_info)
                if not available_projects:
                    continue
                state_options = []
                for project_id in available_projects:
                    state_tuple = self.feature_processor.get_state_tuple(
                        worker_id, project_id, current_time, project_info,
                        worker_quality_map, worker_global_stats, worker_cat_performance
                    )
                    if state_tuple:
                        state_options.append((project_id, state_tuple))
                if not state_options:
                    continue
                selected_project_id = self.act(state_options, self.epsilon)
                if selected_project_id is None:
                    continue
                reward = self.calculate_reward(
                    selected_project_id, worker_id, project_info, entry_info
                )
                # Next state options: for DDQN, we can use the same as CQL (empty, episode ends)
                next_state_options = []
                done = True
                current_state_tuple = None
                for pid, state_tuple in state_options:
                    if pid == selected_project_id:
                        current_state_tuple = state_tuple
                        break
                if current_state_tuple:
                    self.replay_buffer.add(
                        current_state_tuple, selected_project_id, reward, next_state_options, done
                    )
                if self.replay_buffer.is_ready() and self.step_count % self.update_every == 0:
                    experiences = self.replay_buffer.sample()
                    loss = self.learn(experiences)
                    epoch_loss += loss
                    epoch_steps += 1
                if self.step_count % self.target_update_every == 0:
                    self.soft_update(self.q_local, self.q_target, self.tau)
                self.step_count += 1
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            val_metrics = self.evaluate(val_events, epsilon=0.0) if val_events else {'avg_reward': 0.0}
            avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
            self.training_history['epochs'].append(epoch)
            self.training_history['losses'].append(avg_loss)
            self.training_history['val_rewards'].append(val_metrics['avg_reward'])
            if epoch % print_frequency == 0:
                print(f"Epoch {epoch}/{num_epochs}: Loss={avg_loss:.4f}, "
                      f"Val Reward={val_metrics['avg_reward']:.4f}, Epsilon={self.epsilon:.4f}")
        self.is_trained = True
        print("DDQN training completed")
        return {
            'final_loss': self.training_history['losses'][-1] if self.training_history['losses'] else 0.0,
            'final_val_reward': self.training_history['val_rewards'][-1] if self.training_history['val_rewards'] else 0.0,
            'training_history': self.training_history
        }

    def _ddqn_collate_fn(self, experiences_list: List[List[Experience]]) -> Dict[str, torch.Tensor]:
        experiences = experiences_list[0]
        state_tuples = [e.state_tuple for e in experiences]
        id_features = torch.LongTensor([[s[0], s[1], s[2]] for s in state_tuples])
        numeric_features = torch.FloatTensor(np.array([s[3] for s in state_tuples]))
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float).unsqueeze(1)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float).unsqueeze(1)
        # For DDQN, next state is not used (offline, episode ends), so fill with zeros
        next_id_features = torch.zeros_like(id_features)
        next_numeric_features = torch.zeros_like(numeric_features)
        return {
            "id_features": id_features,
            "numeric_features": numeric_features,
            "rewards": rewards,
            "dones": dones,
            "next_id_features": next_id_features,
            "next_numeric_features": next_numeric_features
        } 
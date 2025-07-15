"""
DPO (Direct Policy Optimization) Algorithm Implementation
Implements preference-based policy optimization for offline RL.
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

class DPODataset(IterableDataset):
    """Iterable dataset for DPO training with DataLoader."""
    def __init__(self, replay_buffer: ReplayBuffer):
        self.replay_buffer = replay_buffer
    def __iter__(self):
        while self.replay_buffer.is_ready():
            yield self.replay_buffer.sample()

class DPOAlgorithm(BaseRLAlgorithm):
    """
    Direct Policy Optimization (DPO) algorithm implementation.
    Extends BaseRLAlgorithm with DPO-specific functionality.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.beta = config.get('DPO_BETA', 1.0)
        self.learning_rate = config.get('LEARNING_RATE', 1e-4)
        self.batch_size = config.get('TRAINING_CONFIG', {}).get('batch_size', 64)
        self.buffer_size = config.get('TRAINING_CONFIG', {}).get('buffer_size', int(5e4))
        self.policy_net = None
        self.optimizer = None
        self.replay_buffer = None
        self.step_count = 0
        print(f"DPO Algorithm initialized.")

    def build_network(self) -> None:
        if not self.embedding_sizes:
            raise ValueError("Embedding sizes not available. Load data first.")
        network_config = self.config.get('DPO_NETWORK_CONFIG', {})
        embedding_dims = self.config.get('EMBEDDING_DIMS', {})
        self.policy_net = QNetwork(
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
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DPO")
            self.policy_net = nn.DataParallel(self.policy_net)
        self.network = self.policy_net
        print("DPO network built successfully")

    def act(self, state_options: List[Tuple], epsilon: float = 0.0) -> Optional[int]:
        if not state_options:
            return None
        self.policy_net.eval()
        with torch.no_grad():
            id_features = []
            numeric_features = []
            for project_id, state_tuple in state_options:
                id_features.append([state_tuple[0], state_tuple[1], state_tuple[2]])
                numeric_features.append(state_tuple[3])
            id_features_tensor = torch.LongTensor(id_features).to(self.device)
            numeric_features_tensor = torch.FloatTensor(np.array(numeric_features)).to(self.device)
            logits = self.policy_net(id_features_tensor, numeric_features_tensor)
            # 标准DPO: logits shape [num_options], softmax采样
            probs = torch.softmax(logits, dim=0)
            action_idx = torch.multinomial(probs, 1)[0].item()
            selected_project_id = state_options[action_idx][0]
        self.policy_net.train()
        return selected_project_id

    def learn(self, experiences: List[Experience]) -> float:
        if not experiences:
            return 0.0
        batch_data = self._dpo_collate_fn([experiences])
        for key in batch_data:
            if isinstance(batch_data[key], torch.Tensor):
                batch_data[key] = batch_data[key].to(self.device)
        # DPO loss: cross-entropy between policy and preference
        logits = self.policy_net(batch_data["id_features"], batch_data["numeric_features"])
        # 标准DPO: labels为偏好action的索引
        labels = batch_data["labels"].long().squeeze(1)
        loss = F.cross_entropy(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def train(self, train_events: List[Dict], val_events: List[Dict]) -> Dict[str, Any]:
        if not self.policy_net:
            self.build_network()
        print(f"Starting DPO training with {len(train_events)} train events, {len(val_events)} val events")
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
                selected_project_id = self.act(state_options)
                if selected_project_id is None:
                    continue
                # DPO reward: 1 if selected is preferred, else 0 (for demo, treat as 1)
                reward = 1.0
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
                if self.replay_buffer.is_ready():
                    experiences = self.replay_buffer.sample()
                    loss = self.learn(experiences)
                    epoch_loss += loss
                    epoch_steps += 1
                self.step_count += 1
            val_metrics = self.evaluate(val_events, epsilon=0.0) if val_events else {'avg_reward': 0.0}
            avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
            self.training_history['epochs'].append(epoch)
            self.training_history['losses'].append(avg_loss)
            self.training_history['val_rewards'].append(val_metrics['avg_reward'])
            if epoch % print_frequency == 0:
                print(f"Epoch {epoch}/{num_epochs}: Loss={avg_loss:.4f}, "
                      f"Val Reward={val_metrics['avg_reward']:.4f}")
        self.is_trained = True
        print("DPO training completed")
        return {
            'final_loss': self.training_history['losses'][-1] if self.training_history['losses'] else 0.0,
            'final_val_reward': self.training_history['val_rewards'][-1] if self.training_history['val_rewards'] else 0.0,
            'training_history': self.training_history
        }

    def _dpo_collate_fn(self, experiences_list: List[List[Experience]]) -> Dict[str, torch.Tensor]:
        # 标准DPO: 构造偏好对，labels为偏好action索引
        experiences = experiences_list[0]
        # 假设每个experience包含一个偏好对: (state_tuples, label)
        # 这里简化为单步多action，label为0（即第一个action为偏好）
        # 实际项目应采样真实偏好对
        state_tuples = [e.state_tuple for e in experiences]
        id_features = torch.LongTensor([[s[0], s[1], s[2]] for s in state_tuples])
        numeric_features = torch.FloatTensor(np.array([s[3] for s in state_tuples]))
        labels = torch.zeros(len(experiences), dtype=torch.long).unsqueeze(1)  # 偏好索引
        return {
            "id_features": id_features,
            "numeric_features": numeric_features,
            "labels": labels
        } 
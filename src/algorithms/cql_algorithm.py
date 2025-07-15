"""
CQL (Conservative Q-Learning) Algorithm Implementation
Implements conservative Q-learning for offline reinforcement learning.
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


class CQLDataset(IterableDataset):
    """Iterable dataset for CQL training with DataLoader."""
    
    def __init__(self, replay_buffer: ReplayBuffer):
        """
        Initialize CQL dataset.
        
        Args:
            replay_buffer: Replay buffer containing experiences
        """
        self.replay_buffer = replay_buffer
    
    def __iter__(self):
        """Iterate over batches of experiences."""
        while self.replay_buffer.is_ready():
            yield self.replay_buffer.sample()


class CQLAlgorithm(BaseRLAlgorithm):
    """
    Conservative Q-Learning (CQL) algorithm implementation.
    Extends BaseRLAlgorithm with CQL-specific functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CQL algorithm.
        
        Args:
            config: Configuration dictionary with CQL-specific parameters
        """
        super().__init__(config)
        
        # CQL-specific parameters
        self.cql_alpha = config.get('CQL_ALPHA', 0.15)
        self.gamma = config.get('GAMMA', 0.99)
        self.tau = config.get('TAU', 1e-3)
        self.learning_rate = config.get('LEARNING_RATE', 1e-5)
        self.update_every = config.get('UPDATE_EVERY', 4)
        self.target_update_every = config.get('TARGET_UPDATE_EVERY', 100)
        
        # Training parameters
        self.batch_size = config.get('TRAINING_CONFIG', {}).get('batch_size', 64)
        self.buffer_size = config.get('TRAINING_CONFIG', {}).get('buffer_size', int(5e4))
        self.epsilon_start = config.get('CQL_TRAINING_CONFIG', {}).get('epsilon_start', 1.0)
        self.epsilon_end = config.get('CQL_TRAINING_CONFIG', {}).get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('CQL_TRAINING_CONFIG', {}).get('epsilon_decay', 0.995)
        
        # Networks and training components
        self.q_local = None
        self.q_target = None
        self.optimizer = None
        self.replay_buffer = None
        
        # Training state
        self.epsilon = self.epsilon_start
        self.step_count = 0
        
        print(f"CQL Algorithm initialized with alpha={self.cql_alpha}")
    
    def build_network(self) -> None:
        """Build CQL Q-networks (local and target)."""
        if not self.embedding_sizes:
            raise ValueError("Embedding sizes not available. Load data first.")
        
        # Network configuration
        network_config = self.config.get('CQL_NETWORK_CONFIG', {})
        embedding_dims = self.config.get('EMBEDDING_DIMS', {})
        
        # Create local Q-network
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
        
        # Create target Q-network (copy of local)
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
        
        # Initialize target network with local network weights
        self.q_target.load_state_dict(self.q_local.state_dict())
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=self.learning_rate)
        
        # Setup replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        
        # Use DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for CQL")
            self.q_local = nn.DataParallel(self.q_local)
            self.q_target = nn.DataParallel(self.q_target)
        
        self.network = self.q_local  # For base class compatibility
        
        print("CQL networks built successfully")
        print(f"Network info: {self.q_local.get_network_info() if hasattr(self.q_local, 'get_network_info') else 'Info not available'}")
    
    def act(self, state_options: List[Tuple], epsilon: float = 0.0) -> Optional[int]:
        """
        Select action using epsilon-greedy policy with Q-values.
        
        Args:
            state_options: List of (project_id, state_tuple) pairs
            epsilon: Exploration rate (overrides internal epsilon if provided)
            
        Returns:
            Selected project ID or None if no valid options
        """
        if not state_options:
            return None
        
        # Use provided epsilon or internal epsilon
        current_epsilon = epsilon if epsilon > 0 else self.epsilon
        
        # Epsilon-greedy action selection
        if random.random() > current_epsilon:
            # Greedy action selection
            self.q_local.eval()
            with torch.no_grad():
                # Prepare batch data
                id_features = []
                numeric_features = []
                
                for project_id, state_tuple in state_options:
                    id_features.append([state_tuple[0], state_tuple[1], state_tuple[2]])
                    numeric_features.append(state_tuple[3])
                
                # Convert to tensors
                id_features_tensor = torch.LongTensor(id_features).to(self.device)
                numeric_features_tensor = torch.FloatTensor(np.array(numeric_features)).to(self.device)
                
                # Get Q-values
                q_values = self.q_local(id_features_tensor, numeric_features_tensor)
                
                # Select action with highest Q-value
                action_idx = torch.argmax(q_values).item()
                selected_project_id = state_options[action_idx][0]
                
            self.q_local.train()
            return selected_project_id
        else:
            # Random action selection
            return random.choice(state_options)[0]
    
    def learn(self, experiences: List[Experience]) -> float:
        """
        Learn from a batch of experiences using CQL loss.
        
        Args:
            experiences: List of experience tuples
            
        Returns:
            Loss value
        """
        if not experiences:
            return 0.0
        
        # Prepare batch data using collate function
        batch_data = self._cql_collate_fn([experiences])
        
        # Move to device
        for key in batch_data:
            if isinstance(batch_data[key], torch.Tensor):
                batch_data[key] = batch_data[key].to(self.device)
        
        # Compute Q-values for current states
        current_q_values = self.q_local(
            batch_data["id_features"], 
            batch_data["numeric_features"]
        )
        
        # Compute target Q-values for next states
        next_q_values = torch.zeros(len(experiences), 1).to(self.device)
        
        if batch_data["next_id_features"].size(0) > 0:
            with torch.no_grad():
                next_q_all = self.q_target(
                    batch_data["next_id_features"],
                    batch_data["next_numeric_features"]
                )
                
                # Handle variable number of next state options
                start_idx = 0
                for i, count in enumerate(batch_data["next_options_counts"]):
                    if count > 0:
                        end_idx = start_idx + count
                        next_q_values[i] = torch.max(next_q_all[start_idx:end_idx])
                        start_idx = end_idx
        
        # Compute target values using Bellman equation
        q_targets = batch_data["rewards"] + (self.gamma * next_q_values * (1 - batch_data["dones"]))
        
        # Compute Bellman loss (MSE)
        loss_bellman = F.mse_loss(current_q_values, q_targets.detach())
        
        # Compute CQL loss (conservative regularization)
        # Q-values for out-of-distribution actions
        ood_q_values = self.q_local(
            batch_data["ood_id_features"],
            batch_data["ood_numeric_features"]
        )
        
        # CQL regularization term
        # Encourage lower Q-values for OOD actions, higher for in-distribution actions
        loss_cql = torch.logsumexp(ood_q_values, dim=0).mean() - current_q_values.mean()
        
        # Combined loss
        total_loss = loss_bellman + self.cql_alpha * loss_cql
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_local.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def soft_update(self, local_model: nn.Module, target_model: nn.Module, tau: float) -> None:
        """
        Soft update target network parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            local_model: Local network
            target_model: Target network
            tau: Interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def train(self, train_events: List[Dict], val_events: List[Dict]) -> Dict[str, Any]:
        """
        Train CQL algorithm on provided events.
        
        Args:
            train_events: Training events
            val_events: Validation events
            
        Returns:
            Training results dictionary
        """
        if not self.q_local:
            self.build_network()
        
        print(f"Starting CQL training with {len(train_events)} train events, {len(val_events)} val events")
        
        # Training parameters
        num_epochs = self.config.get('TRAINING_CONFIG', {}).get('num_epochs', 200)
        print_frequency = self.config.get('LOGGING_CONFIG', {}).get('print_frequency', 10)
        
        # Data components
        project_info = self.data_dict['project_info']
        entry_info = self.data_dict['entry_info']
        worker_quality_map = self.data_dict['worker_quality_map']
        worker_global_stats = self.data_dict['worker_global_stats']
        worker_cat_performance = self.data_dict['worker_cat_performance']
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            # Shuffle training events
            random.shuffle(train_events)
            
            for event in train_events:
                worker_id = event['worker_id']
                current_time = event['arrival_time_dt']
                
                # Get available projects
                available_projects = self.get_available_projects(current_time, project_info, entry_info)
                
                if not available_projects:
                    continue
                
                # Generate state options
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
                
                # Select action
                selected_project_id = self.act(state_options, self.epsilon)
                
                if selected_project_id is None:
                    continue
                
                # Calculate reward
                reward = self.calculate_reward(
                    selected_project_id, worker_id, project_info, entry_info
                )
                
                # Get next state options (simplified - assume episode ends)
                next_state_options = []
                done = True
                
                # Store experience
                current_state_tuple = None
                for pid, state_tuple in state_options:
                    if pid == selected_project_id:
                        current_state_tuple = state_tuple
                        break
                
                if current_state_tuple:
                    self.replay_buffer.add(
                        current_state_tuple, selected_project_id, reward, next_state_options, done
                    )
                
                # Learn from experience
                if self.replay_buffer.is_ready() and self.step_count % self.update_every == 0:
                    experiences = self.replay_buffer.sample()
                    loss = self.learn(experiences)
                    epoch_loss += loss
                    epoch_steps += 1
                
                # Update target network
                if self.step_count % self.target_update_every == 0:
                    self.soft_update(self.q_local, self.q_target, self.tau)
                
                self.step_count += 1
            
            # Update epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Validation
            val_metrics = self.evaluate(val_events, epsilon=0.0) if val_events else {'avg_reward': 0.0}
            
            # Record training history
            avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
            self.training_history['epochs'].append(epoch)
            self.training_history['losses'].append(avg_loss)
            self.training_history['val_rewards'].append(val_metrics['avg_reward'])
            
            # Print progress
            if epoch % print_frequency == 0:
                print(f"Epoch {epoch}/{num_epochs}: Loss={avg_loss:.4f}, "
                      f"Val Reward={val_metrics['avg_reward']:.4f}, Epsilon={self.epsilon:.4f}")
        
        self.is_trained = True
        print("CQL training completed")
        
        return {
            'final_loss': self.training_history['losses'][-1] if self.training_history['losses'] else 0.0,
            'final_val_reward': self.training_history['val_rewards'][-1] if self.training_history['val_rewards'] else 0.0,
            'training_history': self.training_history
        }
    
    def _cql_collate_fn(self, experiences_list: List[List[Experience]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for CQL training with OOD sampling.
        
        Args:
            experiences_list: List containing list of experiences
            
        Returns:
            Dictionary with batched tensors
        """
        experiences = experiences_list[0]  # DataLoader wraps in extra list
        
        # Prepare in-distribution data
        state_tuples = [e.state_tuple for e in experiences]
        id_features = torch.LongTensor([[s[0], s[1], s[2]] for s in state_tuples])
        numeric_features = torch.FloatTensor(np.array([s[3] for s in state_tuples]))
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float).unsqueeze(1)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float).unsqueeze(1)
        
        # Prepare out-of-distribution data for CQL
        num_random_actions = self.config.get('CQL_DATA_CONFIG', {}).get('num_random_actions', 10)
        batch_size = len(experiences)
        
        # Sample random projects for OOD actions
        all_project_ids = list(self.data_dict['project_info'].keys())
        random_project_ids = np.random.choice(all_project_ids, size=batch_size * num_random_actions, replace=True)
        
        # Generate OOD state tuples
        ood_id_features_list = []
        ood_numeric_features_list = []
        
        # Use current time from experiences for OOD sampling
        min_t, max_t = self.feature_processor.feature_min_max.get('current_time_val', (0, 1))
        scaled_timestamps = numeric_features[:, -1]  # Last feature is time
        unscaled_seconds = (scaled_timestamps * (max_t - min_t)) + min_t
        batch_datetimes = [ALL_BEGIN_TIME_DT + datetime.timedelta(seconds=s.item()) for s in unscaled_seconds]
        
        for i, project_id in enumerate(random_project_ids):
            original_state_idx = i // num_random_actions
            current_time = batch_datetimes[original_state_idx]
            
            # Get project features
            project_data = self.data_dict['project_info'].get(project_id, {})
            cat_id = project_data.get("category", 0)
            sub_cat_id = project_data.get("sub_category", 0)
            ind_id = project_data.get("industry_id", 0)
            
            ood_id_features_list.append([cat_id, sub_cat_id, ind_id])
            
            # Generate numeric features (simplified)
            _, _, _, numeric_proj_features = self.feature_processor.get_project_features(
                project_id, self.data_dict['project_info'], current_time
            )
            
            if numeric_proj_features is None:
                numeric_proj_features = np.zeros(NUMERIC_FEATURES['base_project'])
            
            # Use worker features from original state
            worker_features = numeric_features[original_state_idx, :NUMERIC_FEATURES['worker']]
            interaction_features = np.zeros(NUMERIC_FEATURES['interaction'])
            context_features = numeric_features[original_state_idx, -NUMERIC_FEATURES['context']:]
            
            ood_numeric_features = np.concatenate([
                worker_features.numpy(),
                numeric_proj_features,
                interaction_features,
                context_features.numpy()
            ])
            
            ood_numeric_features_list.append(ood_numeric_features)
        
        ood_id_features = torch.LongTensor(ood_id_features_list)
        ood_numeric_features = torch.FloatTensor(np.array(ood_numeric_features_list))
        
        # Prepare next state data (simplified - empty for now)
        next_id_features = torch.empty(0, 3, dtype=torch.long)
        next_numeric_features = torch.empty(0, sum(NUMERIC_FEATURES.values()), dtype=torch.float)
        next_options_counts = torch.zeros(batch_size, dtype=torch.int)
        
        return {
            "id_features": id_features,
            "numeric_features": numeric_features,
            "rewards": rewards,
            "dones": dones,
            "ood_id_features": ood_id_features,
            "ood_numeric_features": ood_numeric_features,
            "next_id_features": next_id_features,
            "next_numeric_features": next_numeric_features,
            "next_options_counts": next_options_counts
        }


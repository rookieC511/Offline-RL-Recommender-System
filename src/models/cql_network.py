"""
Neural network architecture for CQL algorithm
Q-Network with embeddings for categorical features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GeneralQNetwork(nn.Module):
    """
    Q-Network with embeddings for CQL algorithm.
    Handles categorical and numeric features with dropout regularization.
    """
    
    def __init__(self, num_categories_embed_size: int, num_sub_categories_embed_size: int,
                 num_industries_embed_size: int, cat_embed_dim: int, sub_cat_embed_dim: int,
                 ind_embed_dim: int, total_numeric_features: int, seed: int = 42,
                 fc1_units: int = 128, fc2_units: int = 32, dropout_rate: float = 0.4):
        """
        Initialize CQL Q-Network.
        
        Args:
            num_categories_embed_size: Number of categories for embedding
            num_sub_categories_embed_size: Number of sub-categories for embedding
            num_industries_embed_size: Number of industries for embedding
            cat_embed_dim: Category embedding dimension
            sub_cat_embed_dim: Sub-category embedding dimension
            ind_embed_dim: Industry embedding dimension
            total_numeric_features: Total number of numeric features
            seed: Random seed for reproducibility
            fc1_units: First fully connected layer units
            fc2_units: Second fully connected layer units
            dropout_rate: Dropout rate for regularization
        """
        super(GeneralQNetwork, self).__init__()
        
        # Set random seed for reproducibility
        self.seed = torch.manual_seed(seed)
        
        # Store dimensions
        self.cat_embed_dim = cat_embed_dim
        self.sub_cat_embed_dim = sub_cat_embed_dim
        self.ind_embed_dim = ind_embed_dim
        self.total_numeric_features = total_numeric_features
        
        # Define embedding layers
        self.category_embedding = nn.Embedding(num_categories_embed_size, cat_embed_dim)
        self.sub_category_embedding = nn.Embedding(num_sub_categories_embed_size, sub_cat_embed_dim)
        self.industry_embedding = nn.Embedding(num_industries_embed_size, ind_embed_dim)
        
        # Calculate total input dimension for fully connected layers
        total_embed_dim = cat_embed_dim + sub_cat_embed_dim + ind_embed_dim
        fc_input_dim = total_embed_dim + total_numeric_features
        
        # Define fully connected layers with dropout
        self.fc1 = nn.Linear(fc_input_dim, fc1_units)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(fc2_units, 1)  # Q-value output
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, id_features_batch: torch.Tensor, numeric_features_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            id_features_batch: Batch of categorical ID features [batch_size, 3]
                              (category_id, sub_category_id, industry_id)
            numeric_features_batch: Batch of numeric features [batch_size, total_numeric_features]
            
        Returns:
            Q-values [batch_size, 1]
        """
        # Validate input dimensions
        if id_features_batch.size(1) != 3:
            raise ValueError(f"Expected id_features_batch to have 3 columns, got {id_features_batch.size(1)}")
        
        if numeric_features_batch.size(1) != self.total_numeric_features:
            raise ValueError(f"Expected {self.total_numeric_features} numeric features, got {numeric_features_batch.size(1)}")
        
        # Split categorical ID features
        cat_ids, sub_cat_ids, ind_ids = id_features_batch.split(1, dim=1)
        
        # Remove extra dimension and ensure proper data type
        cat_ids = cat_ids.squeeze(-1).long()
        sub_cat_ids = sub_cat_ids.squeeze(-1).long()
        ind_ids = ind_ids.squeeze(-1).long()
        
        # Get embedding vectors
        cat_embed = self.category_embedding(cat_ids)
        sub_cat_embed = self.sub_category_embedding(sub_cat_ids)
        ind_embed = self.industry_embedding(ind_ids)
        
        # Concatenate all embeddings
        embedded_features = torch.cat([cat_embed, sub_cat_embed, ind_embed], dim=1)
        
        # Concatenate embeddings with numeric features
        all_features = torch.cat([embedded_features, numeric_features_batch], dim=1)
        
        # Forward pass through fully connected layers
        x = F.relu(self.fc1(all_features))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        q_values = self.fc3(x)
        
        return q_values
    
    def get_embedding_weights(self) -> dict:
        """
        Get embedding layer weights for analysis.
        
        Returns:
            Dictionary containing embedding weights
        """
        return {
            'category_embeddings': self.category_embedding.weight.data.cpu().numpy(),
            'sub_category_embeddings': self.sub_category_embedding.weight.data.cpu().numpy(),
            'industry_embeddings': self.industry_embedding.weight.data.cpu().numpy()
        }
    
    def freeze_embeddings(self):
        """Freeze embedding layers to prevent updates during training."""
        self.category_embedding.weight.requires_grad = False
        self.sub_category_embedding.weight.requires_grad = False
        self.industry_embedding.weight.requires_grad = False
    
    def unfreeze_embeddings(self):
        """Unfreeze embedding layers to allow updates during training."""
        self.category_embedding.weight.requires_grad = True
        self.sub_category_embedding.weight.requires_grad = True
        self.industry_embedding.weight.requires_grad = True
    
    def get_network_info(self) -> dict:
        """
        Get information about network architecture.
        
        Returns:
            Dictionary with network information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'embedding_dimensions': {
                'category': self.cat_embed_dim,
                'sub_category': self.sub_cat_embed_dim,
                'industry': self.ind_embed_dim
            },
            'numeric_features': self.total_numeric_features,
            'architecture': [
                f"Embeddings -> FC({self.fc1.in_features}, {self.fc1.out_features})",
                f"FC({self.fc2.in_features}, {self.fc2.out_features})",
                f"FC({self.fc3.in_features}, {self.fc3.out_features})"
            ]
        }


class ReplayBuffer:
    """
    Experience replay buffer for CQL algorithm.
    Stores and samples experiences for training.
    """
    
    def __init__(self, buffer_size: int, batch_size: int, seed: int = 42):
        """
        Initialize replay buffer.
        
        Args:
            buffer_size: Maximum number of experiences to store
            batch_size: Batch size for sampling
            seed: Random seed for reproducibility
        """
        from collections import deque
        import random
        
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        random.seed(seed)
        np.random.seed(seed)
    
    def add(self, state_tuple, action_project_id, reward, next_state_options_tuples, done):
        """
        Add experience to buffer.
        
        Args:
            state_tuple: Current state tuple
            action_project_id: Selected action (project ID)
            reward: Received reward
            next_state_options_tuples: Available next state options
            done: Whether episode is done
        """
        from src.algorithms.base_algorithm import Experience
        
        experience = Experience(state_tuple, action_project_id, reward, next_state_options_tuples, done)
        self.memory.append(experience)
    
    def sample(self):
        """
        Sample batch of experiences from buffer.
        
        Returns:
            List of sampled experiences
        """
        import random
        return random.sample(self.memory, k=self.batch_size)
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.memory)
    
    def is_ready(self):
        """Check if buffer has enough experiences for sampling."""
        return len(self.memory) >= self.batch_size


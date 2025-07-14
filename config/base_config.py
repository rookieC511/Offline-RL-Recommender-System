"""
Base configuration for RL Crowdsourcing System
Contains shared constants and default values used across all algorithms.
"""

import torch
from dateutil.parser import parse

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data paths
DATA_PATHS = {
    'worker_quality': 'worker_quality.csv',
    'project_list': 'project_list.csv',
    'project_dir': 'project/',
    'entry_dir': 'entry/',
}

# Time reference
ALL_BEGIN_TIME_DT = parse("2018-01-01T0:0:0Z")

# Feature dimensions
EMBEDDING_DIMS = {
    'category': 5,
    'sub_category': 8,
    'industry': 5,
    'worker_id': 8,
}

# Numeric feature counts
NUMERIC_FEATURES = {
    'worker': 5,
    'base_project': 4,
    'interaction': 4,
    'context': 1,
}

# Calculate total numeric features
TOTAL_NUMERIC_FEATURES = sum(NUMERIC_FEATURES.values())

# Network architecture
NETWORK_CONFIG = {
    'fc1_units': 128,
    'fc2_units': 32,
    'dropout_rate': 0.4,
    'seed': 42,
}

# Training parameters (shared defaults)
TRAINING_CONFIG = {
    'batch_size': 64,
    'buffer_size': int(5e4),
    'num_epochs': 200,
    'reward_scaling_factor': 10,
    'reward_scale_reference': 20.0,  # Default, will be updated from data
}

# Evaluation parameters
EVALUATION_CONFIG = {
    'train_split': 0.7,
    'validation_split': 0.15,
    'test_split': 0.15,
    'min_events_threshold': 100,
}

# Logging configuration
LOGGING_CONFIG = {
    'print_frequency': 10,
    'save_frequency': 50,
    'plot_frequency': 20,
}

# Data processing
DATA_PROCESSING = {
    'entry_page_size': 24,
    'feature_min_max_default': (0, 1),
    'quality_normalization_factor': 100.0,
    'participation_normalization_factor': 50.0,
    'category_diversity_normalization_factor': 10.0,
    'score_max_value': 5.0,
}

# Reward calculation
REWARD_CONFIG = {
    'action_cost': -0.01,
    'winner_bonus': 1.0,
    'quality_threshold': 0.5,
}


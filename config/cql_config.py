"""
CQL (Conservative Q-Learning) specific configuration
Inherits from base config and adds CQL-specific parameters.
"""

from .base_config import *

# Algorithm identifier
ALGORITHM_NAME = "CQL"

# CQL-specific hyperparameters
CQL_CONFIG = {
    'alpha': 0.15,  # Conservative loss weight - key hyperparameter for CQL
    'gamma': 0.99,  # Discount factor
    'tau': 1e-3,    # Soft update rate for target network
    'learning_rate': 1e-5,
    'update_every': 4,
    'target_update_every': 100,
}

# CQL-specific network configuration
CQL_NETWORK_CONFIG = {
    **NETWORK_CONFIG,
    'has_target_network': True,
    'double_q_learning': True,
    'output_size': 1,  # Q-value output
}

# CQL-specific training parameters
CQL_TRAINING_CONFIG = {
    **TRAINING_CONFIG,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'experience_replay': True,
}

# CQL data processing
CQL_DATA_CONFIG = {
    'num_random_actions': 10,  # For OOD sampling
    'ood_sampling_enabled': True,
    'conservative_regularization': True,
    'next_state_processing': True,
}

# CQL evaluation
CQL_EVALUATION_CONFIG = {
    **EVALUATION_CONFIG,
    'q_value_logging': True,
    'conservative_loss_logging': True,
    'target_network_sync_logging': True,
}

# Override base config with CQL specifics
LEARNING_RATE = CQL_CONFIG['learning_rate']
GAMMA = CQL_CONFIG['gamma']
TAU = CQL_CONFIG['tau']
CQL_ALPHA = CQL_CONFIG['alpha']
UPDATE_EVERY = CQL_CONFIG['update_every']
TARGET_UPDATE_EVERY = CQL_CONFIG['target_update_every']


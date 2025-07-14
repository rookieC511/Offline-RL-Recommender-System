"""
DDQN (Double Deep Q-Network) specific configuration
Inherits from base config and adds DDQN-specific parameters.
"""

from .base_config import *

# Algorithm identifier
ALGORITHM_NAME = "DDQN"

# DDQN-specific hyperparameters
DDQN_CONFIG = {
    'gamma': 0.99,  # Discount factor
    'tau': 1e-3,    # Soft update rate for target network
    'learning_rate': 1e-4,  # Higher learning rate than CQL
    'update_every': 4,
    'target_update_every': 100,
}

# DDQN-specific network configuration
DDQN_NETWORK_CONFIG = {
    **NETWORK_CONFIG,
    'has_target_network': True,
    'double_q_learning': True,  # Key feature of DDQN
    'output_size': 1,  # Q-value output
}

# DDQN-specific training parameters
DDQN_TRAINING_CONFIG = {
    **TRAINING_CONFIG,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'experience_replay': True,
}

# DDQN data processing
DDQN_DATA_CONFIG = {
    'standard_replay_buffer': True,
    'no_ood_sampling': True,  # Unlike CQL, DDQN doesn't use OOD sampling
    'no_conservative_regularization': True,
    'simple_bellman_updates': True,
}

# DDQN evaluation
DDQN_EVALUATION_CONFIG = {
    **EVALUATION_CONFIG,
    'q_value_logging': True,
    'target_network_sync_logging': True,
    'exploration_rate_logging': True,
}

# Override base config with DDQN specifics
LEARNING_RATE = DDQN_CONFIG['learning_rate']
GAMMA = DDQN_CONFIG['gamma']
TAU = DDQN_CONFIG['tau']
UPDATE_EVERY = DDQN_CONFIG['update_every']
TARGET_UPDATE_EVERY = DDQN_CONFIG['target_update_every']


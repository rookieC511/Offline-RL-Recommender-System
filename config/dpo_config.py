"""
DPO (Direct Policy Optimization) specific configuration
Inherits from base config and adds DPO-specific parameters.
"""

from .base_config import *

# Algorithm identifier
ALGORITHM_NAME = "DPO"

# DPO-specific hyperparameters
DPO_CONFIG = {
    'beta': 0.3,    # Policy deviation control - key hyperparameter for DPO
    'learning_rate': 3e-5,
    'num_rejected_samples': 3,  # Rejected samples per chosen sample
    'preference_threshold': 0.1,  # Threshold for preference pair generation
}

# DPO-specific network configuration
DPO_NETWORK_CONFIG = {
    **NETWORK_CONFIG,
    'has_target_network': False,  # DPO uses single policy network
    'double_q_learning': False,
    'output_size': 1,  # Preference logit output
    'dropout_rate': 0.3,  # Different dropout rate for DPO
}

# DPO-specific training parameters
DPO_TRAINING_CONFIG = {
    **TRAINING_CONFIG,
    'deterministic_policy': True,  # No epsilon-greedy exploration
    'preference_based_learning': True,
    'experience_replay': False,  # DPO doesn't use standard replay buffer
}

# DPO data processing
DPO_DATA_CONFIG = {
    'preference_pair_generation': True,
    'reward_threshold_percentile': 0.6,  # For chosen vs rejected classification
    'negative_sampling_strategy': 'random',
    'preference_augmentation': True,
}

# DPO evaluation
DPO_EVALUATION_CONFIG = {
    **EVALUATION_CONFIG,
    'preference_accuracy_logging': True,
    'policy_deviation_logging': True,
    'logit_distribution_logging': True,
}

# Override base config with DPO specifics
LEARNING_RATE = DPO_CONFIG['learning_rate']
DPO_BETA = DPO_CONFIG['beta']
NUM_REJECTED_SAMPLES = DPO_CONFIG['num_rejected_samples']

# DPO doesn't use these Q-learning parameters
GAMMA = None  # Not used in DPO
TAU = None    # Not used in DPO


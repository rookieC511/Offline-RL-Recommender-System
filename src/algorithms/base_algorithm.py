"""
Base algorithm class for RL Crowdsourcing System
Provides shared functionality for all RL algorithms.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from collections import namedtuple
import matplotlib.pyplot as plt
import os

from src.data.data_loader import DataLoader
from src.data.feature_processor import FeatureProcessor
from src.data.data_validator import DataValidator


# Experience tuple for replay buffer
Experience = namedtuple("Experience", 
                       field_names=["state_tuple", "action_project_id", "reward", 
                                   "next_state_options_tuples", "done"])


class BaseRLAlgorithm(ABC):
    """
    Abstract base class for RL algorithms.
    Provides shared functionality while allowing algorithm-specific implementations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base RL algorithm.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config.get('DEVICE', torch.device('cpu'))
        self.algorithm_name = config.get('ALGORITHM_NAME', 'BaseRL')
        
        # Data components
        self.data_loader = None
        self.feature_processor = None
        self.data_validator = None
        
        # Loaded data
        self.data_dict = {}
        self.embedding_sizes = {}
        
        # Training state
        self.is_trained = False
        self.training_history = {
            'train_rewards': [],
            'val_rewards': [],
            'losses': [],
            'epochs': []
        }
        
        # Model components (to be set by subclasses)
        self.network = None
        self.optimizer = None
        
        print(f"Initialized {self.algorithm_name} algorithm")
    
    def load_data(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load and validate data using unified pipeline.
        
        Args:
            use_cache: Whether to use data caching
            
        Returns:
            Dictionary containing loaded data
        """
        print(f"Loading data for {self.algorithm_name}...")
        
        # Initialize data components
        self.data_loader = DataLoader(self.config, use_cache=use_cache)
        self.feature_processor = FeatureProcessor(self.config)
        self.data_validator = DataValidator(self.config)
        
        # Load all data
        self.data_dict = self.data_loader.load_all_data()
        
        # Validate data
        validation_results = self.data_validator.validate_all_data(self.data_dict)
        
        # Get embedding sizes
        self.embedding_sizes = self.data_dict['embedding_sizes']
        
        print(f"Data loaded successfully for {self.algorithm_name}")
        return self.data_dict
    
    def prepare_features(self, arrival_events: List[Dict]) -> None:
        """
        Prepare feature scaling using training data.
        
        Args:
            arrival_events: List of arrival events for feature scaling
        """
        if not self.feature_processor:
            raise ValueError("Data must be loaded before preparing features")
        
        print("Preparing feature scaling...")
        self.feature_processor.fit_feature_scaling(
            arrival_events,
            self.data_dict['project_info'],
            self.data_dict['entry_info'],
            self.data_dict['worker_quality_map']
        )
        print("Feature scaling prepared")
    
    def get_available_projects(self, current_time, project_info: Dict, entry_info: Dict) -> List[int]:
        """
        Get list of available projects at given time.
        
        Args:
            current_time: Current timestamp
            project_info: Project information dictionary
            entry_info: Entry information dictionary
            
        Returns:
            List of available project IDs
        """
        available = []
        
        for project_id, project_data in project_info.items():
            # Check if project is active
            if (project_data.get("start_date_dt") <= current_time and
                project_data.get("deadline_dt") > current_time and
                project_data.get("status", "open").lower() != "completed"):
                
                # Check if project needs more entries
                accepted_count = 0
                if project_id in entry_info:
                    for _, entry_data in entry_info[project_id].items():
                        if (not entry_data.get("withdrawn", False) and
                            entry_data.get("entry_created_at_dt") <= current_time):
                            accepted_count += 1
                
                required_answers = project_data.get("required_answers", 1)
                if accepted_count < required_answers:
                    available.append(project_id)
        
        return available
    
    def calculate_reward(self, chosen_project_id: int, current_worker_id: int,
                        project_info_map: Dict, entry_info_map: Dict) -> float:
        """
        Calculate reward for a worker-project interaction.
        
        Args:
            chosen_project_id: Selected project ID
            current_worker_id: Worker ID
            project_info_map: Project information mapping
            entry_info_map: Entry information mapping
            
        Returns:
            Calculated reward value
        """
        # Base implementation - can be overridden by subclasses
        reward_config = self.config.get('REWARD_CONFIG', {})
        scale_reference = self.data_dict.get('reward_scale_reference', 20.0)
        scaling_factor = self.config.get('TRAINING_CONFIG', {}).get('reward_scaling_factor', 10)
        
        # Look for historical performance
        actual_award_hist = 0.0
        hist_score = 0
        is_winner_hist = False
        worker_participated_hist = False
        
        if chosen_project_id in entry_info_map:
            for _, entry_data in entry_info_map[chosen_project_id].items():
                if entry_data["worker_id"] == current_worker_id and not entry_data.get("withdrawn", False):
                    worker_participated_hist = True
                    current_score = entry_data.get("score", 0)
                    if current_score > hist_score:
                        hist_score = current_score
                    
                    award_val_raw = entry_data.get("award_value")
                    if award_val_raw is not None:
                        try:
                            award_val_float = float(award_val_raw)
                            if award_val_float > 0:
                                actual_award_hist = award_val_float
                                if entry_data.get("winner"):
                                    is_winner_hist = True
                                    break
                        except (ValueError, TypeError):
                            pass
        
        # Calculate base reward
        base_reward = 0.0
        action_cost = reward_config.get('action_cost', -0.01)
        
        if worker_participated_hist:
            if is_winner_hist and actual_award_hist > 0:
                # Winner with actual award
                base_reward = (actual_award_hist / scale_reference) * scaling_factor
            elif hist_score > 0:
                # Participated with score
                base_reward = (hist_score / 5.0) * scaling_factor * 0.5
            else:
                # Participated but no clear outcome
                base_reward = scaling_factor * 0.1
        else:
            # No participation - small penalty
            base_reward = action_cost
        
        return base_reward + action_cost
    
    def evaluate(self, test_events: List[Dict], epsilon: float = 0.0) -> Dict[str, float]:
        """
        Evaluate algorithm performance on test data.
        
        Args:
            test_events: List of test events
            epsilon: Exploration rate for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            print("Warning: Evaluating untrained model")
        
        total_reward = 0.0
        total_actions = 0
        successful_actions = 0
        
        project_info = self.data_dict['project_info']
        entry_info = self.data_dict['entry_info']
        worker_quality_map = self.data_dict['worker_quality_map']
        worker_global_stats = self.data_dict['worker_global_stats']
        worker_cat_performance = self.data_dict['worker_cat_performance']
        
        for event in test_events:
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
            
            # Select action using algorithm
            selected_project_id = self.act(state_options, epsilon)
            
            if selected_project_id is not None:
                # Calculate reward
                reward = self.calculate_reward(
                    selected_project_id, worker_id, project_info, entry_info
                )
                
                total_reward += reward
                total_actions += 1
                
                if reward > 0:
                    successful_actions += 1
        
        # Calculate metrics
        avg_reward = total_reward / total_actions if total_actions > 0 else 0.0
        success_rate = successful_actions / total_actions if total_actions > 0 else 0.0
        
        metrics = {
            'avg_reward': avg_reward,
            'total_reward': total_reward,
            'total_actions': total_actions,
            'success_rate': success_rate
        }
        
        print(f"Evaluation results: Avg Reward = {avg_reward:.4f}, Success Rate = {success_rate:.4f}")
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.network:
            raise ValueError("No network to save")
        
        save_dict = {
            'algorithm_name': self.algorithm_name,
            'config': self.config,
            'network_state_dict': self.network.state_dict(),
            'embedding_sizes': self.embedding_sizes,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        if self.optimizer:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from file.
        
        Args:
            filepath: Path to load model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Verify algorithm compatibility
        if checkpoint.get('algorithm_name') != self.algorithm_name:
            print(f"Warning: Loading {checkpoint.get('algorithm_name')} model into {self.algorithm_name}")
        
        # Load network state
        if self.network and 'network_state_dict' in checkpoint:
            self.network.load_state_dict(checkpoint['network_state_dict'])
        
        # Load optimizer state
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load other attributes
        self.embedding_sizes = checkpoint.get('embedding_sizes', {})
        self.training_history = checkpoint.get('training_history', {})
        self.is_trained = checkpoint.get('is_trained', False)
        
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path: Optional path to save plot
        """
        if not self.training_history['epochs']:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot rewards
        epochs = self.training_history['epochs']
        if self.training_history['train_rewards']:
            axes[0].plot(epochs, self.training_history['train_rewards'], label='Train Reward', color='blue')
        if self.training_history['val_rewards']:
            axes[0].plot(epochs, self.training_history['val_rewards'], label='Val Reward', color='orange')
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Average Reward')
        axes[0].set_title(f'{self.algorithm_name} - Reward Progress')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot losses
        if self.training_history['losses']:
            axes[1].plot(epochs, self.training_history['losses'], label='Loss', color='red')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title(f'{self.algorithm_name} - Loss Progress')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")
        
        plt.show()
    
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def build_network(self) -> None:
        """Build the neural network architecture."""
        pass
    
    @abstractmethod
    def act(self, state_options: List[Tuple], epsilon: float = 0.0) -> Optional[int]:
        """
        Select action given state options.
        
        Args:
            state_options: List of (project_id, state_tuple) pairs
            epsilon: Exploration rate
            
        Returns:
            Selected project ID or None
        """
        pass
    
    @abstractmethod
    def train(self, train_events: List[Dict], val_events: List[Dict]) -> Dict[str, Any]:
        """
        Train the algorithm.
        
        Args:
            train_events: Training events
            val_events: Validation events
            
        Returns:
            Training results dictionary
        """
        pass
    
    @abstractmethod
    def learn(self, experiences: List[Experience]) -> float:
        """
        Learn from a batch of experiences.
        
        Args:
            experiences: List of experience tuples
            
        Returns:
            Loss value
        """
        pass


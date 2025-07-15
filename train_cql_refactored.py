"""
Training script for refactored CQL algorithm
Demonstrates usage of the new modular architecture.
"""

import os
import sys
import argparse
from datetime import datetime
import torch

# Add src to path
sys.path.insert(0, 'src')

from config import load_config, create_arg_parser
from src.algorithms.cql_algorithm import CQLAlgorithm
from src.algorithms.ddqn_algorithm import DDQNAlgorithm
from src.algorithms.dqn_algorithm import DQNAlgorithm
from src.algorithms.dpo_algorithm import DPOAlgorithm
from src.data.data_loader import DataLoader
from src.data.feature_processor import FeatureProcessor
from src.data.data_validator import DataValidator


def load_arrival_events(data_dict, split_ratio=(0.7, 0.15, 0.15)):
    """
    Create arrival events from loaded data for training.
    This is a simplified version - in practice, you'd load actual arrival events.
    
    Args:
        data_dict: Loaded data dictionary
        split_ratio: Train/validation/test split ratios
        
    Returns:
        Tuple of (train_events, val_events, test_events)
    """
    print("Creating arrival events from data...")
    
    # Create synthetic arrival events from entry data
    all_events = []
    
    for project_id, entries in data_dict['entry_info'].items():
        for entry_id, entry_data in entries.items():
            event = {
                'worker_id': entry_data['worker_id'],
                'arrival_time_dt': entry_data['entry_created_at_dt'],
                'project_id': project_id,  # For reference
                'entry_id': entry_id
            }
            all_events.append(event)
    
    # Sort by arrival time
    all_events.sort(key=lambda x: x['arrival_time_dt'])
    
    # Split data
    total_events = len(all_events)
    train_end = int(total_events * split_ratio[0])
    val_end = train_end + int(total_events * split_ratio[1])
    
    train_events = all_events[:train_end]
    val_events = all_events[train_end:val_end]
    test_events = all_events[val_end:]
    
    print(f"Created {len(train_events)} train, {len(val_events)} val, {len(test_events)} test events")
    
    return train_events, val_events, test_events


def main():
    """Main training function."""
    # Parse arguments
    parser = create_arg_parser()
    parser.add_argument('--save-model', type=str, default='rl_model.pth',
                       help='Path to save trained model')
    parser.add_argument('--plot-results', action='store_true',
                       help='Plot training results')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.algorithm, {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'cql_alpha': getattr(args, 'cql_alpha', None),
        'gamma': args.gamma
    } if any([args.learning_rate, args.batch_size, args.num_epochs, 
              getattr(args, 'cql_alpha', None), args.gamma]) else None)
    
    print(f"Configuration loaded for {config['ALGORITHM_NAME']}")
    print(f"Learning Rate: {config.get('LEARNING_RATE', 'Not set')}")
    
    # Initialize algorithm
    if args.algorithm.lower() == 'cql':
        print("Initializing CQL algorithm...")
        agent = CQLAlgorithm(config)
    elif args.algorithm.lower() == 'ddqn':
        print("Initializing DDQN algorithm...")
        agent = DDQNAlgorithm(config)
    elif args.algorithm.lower() == 'dqn':
        print("Initializing DQN algorithm...")
        agent = DQNAlgorithm(config)
    elif args.algorithm.lower() == 'dpo':
        print("Initializing DPO algorithm...")
        agent = DPOAlgorithm(config)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")
    
    # Load data
    print("Loading data...")
    data_dict = agent.load_data(use_cache=True)
    # Create arrival events
    train_events, val_events, test_events = load_arrival_events(data_dict)
    # Prepare features
    print("Preparing features...")
    agent.prepare_features(train_events)
    # Build network
    print("Building network...")
    agent.build_network()
    # Train algorithm
    print("Starting training...")
    start_time = datetime.now()
    training_results = agent.train(train_events, val_events)
    end_time = datetime.now()
    training_duration = end_time - start_time
    print(f"Training completed in {training_duration}")
    print(f"Final validation reward: {training_results['final_val_reward']:.4f}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = agent.evaluate(test_events, epsilon=0.0)
    
    print("Test Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    if args.save_model:
        print(f"Saving model to {args.save_model}...")
        agent.save_model(args.save_model)
    
    # Plot results
    if args.plot_results:
        print("Plotting training results...")
        plot_path = args.save_model.replace('.pth', '_training_plot.png') if args.save_model else None
        agent.plot_training_history(plot_path)
    
    # Print summary
    print("\n" + "="*60)
    print("RL TRAINING SUMMARY")
    print("="*60)
    print(f"Algorithm: {config['ALGORITHM_NAME']}")
    print(f"Training Duration: {training_duration}")
    print(f"Training Events: {len(train_events)}")
    print(f"Validation Events: {len(val_events)}")
    print(f"Test Events: {len(test_events)}")
    print(f"Final Test Reward: {test_metrics['avg_reward']:.4f}")
    print(f"Test Success Rate: {test_metrics['success_rate']:.4f}")
    print("="*60)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run training
    main()


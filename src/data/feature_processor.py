"""
Feature processing and engineering for RL Crowdsourcing System
Handles feature extraction, normalization, and state tuple generation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dateutil.parser import parse
import datetime

from config.base_config import ALL_BEGIN_TIME_DT, DATA_PROCESSING, NUMERIC_FEATURES


class FeatureProcessor:
    """
    Centralized feature processing with caching and vectorization.
    Replaces duplicated feature engineering logic across training files.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature processor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_min_max = {}
        self.is_fitted = False
        
        # Feature dimensions from config
        self.num_worker_features = NUMERIC_FEATURES['worker']
        self.num_project_features = NUMERIC_FEATURES['base_project']
        self.num_interaction_features = NUMERIC_FEATURES['interaction']
        self.num_context_features = NUMERIC_FEATURES['context']
        self.total_numeric_features = sum(NUMERIC_FEATURES.values())
    
    def fit_feature_scaling(self, arrival_events: List[Dict], project_info: Dict, 
                           entry_info: Dict, worker_quality_map: Dict) -> None:
        """
        Compute feature scaling parameters from training data.
        
        Args:
            arrival_events: List of arrival events for training
            project_info: Project information dictionary
            entry_info: Entry information dictionary
            worker_quality_map: Worker quality mapping
        """
        print("Computing feature scaling parameters...")
        
        # Collect feature values for scaling
        feature_values = {
            "worker_quality": [],
            "time_until_deadline_sec": [],
            "task_age_sec": [],
            "project_duration_sec": [],
            "reward_per_slot": [],
            "current_time_val": []
        }
        
        for event in arrival_events:
            worker_id = event['worker_id']
            current_time = event['arrival_time_dt']
            
            # Worker quality
            feature_values["worker_quality"].append(
                worker_quality_map.get(worker_id, 0.0)
            )
            
            # Current time
            feature_values["current_time_val"].append(
                (current_time - ALL_BEGIN_TIME_DT).total_seconds()
            )
            
            # Project features
            available_projects = self._get_available_projects(
                current_time, project_info, entry_info
            )
            
            for project_id in available_projects:
                project_data = project_info.get(project_id)
                if not project_data:
                    continue
                
                # Time features
                time_until_deadline = (project_data["deadline_dt"] - current_time).total_seconds()
                task_age = (current_time - project_data["start_date_dt"]).total_seconds()
                project_duration = (project_data["deadline_dt"] - project_data["start_date_dt"]).total_seconds()
                
                feature_values["time_until_deadline_sec"].append(time_until_deadline)
                feature_values["task_age_sec"].append(max(0, task_age))
                feature_values["project_duration_sec"].append(max(0, project_duration))
                
                # Reward per slot
                required_answers = project_data.get("required_answers", 1)
                total_awards = project_data.get("total_awards", 0)
                reward_per_slot = (total_awards / required_answers) if required_answers > 0 and total_awards > 0 else 0
                feature_values["reward_per_slot"].append(reward_per_slot)
        
        # Compute min/max for each feature
        for feature_name, values in feature_values.items():
            if values:
                min_val, max_val = np.min(values), np.max(values)
                
                # Handle edge cases
                if min_val == max_val:
                    if min_val == 0:
                        min_val, max_val = 0.0, 1.0
                    else:
                        max_val = min_val * 1.01
                
                self.feature_min_max[feature_name] = (min_val, max_val)
            else:
                self.feature_min_max[feature_name] = DATA_PROCESSING['feature_min_max_default']
        
        self.is_fitted = True
        print("Feature scaling parameters computed.")
    
    def min_max_scale(self, value: float, feature_name: str) -> float:
        """
        Apply min-max scaling to a feature value.
        
        Args:
            value: Raw feature value
            feature_name: Name of the feature
            
        Returns:
            Scaled feature value (0-1 range)
        """
        if not self.is_fitted:
            raise ValueError("FeatureProcessor must be fitted before scaling")
        
        min_val, max_val = self.feature_min_max.get(
            feature_name, DATA_PROCESSING['feature_min_max_default']
        )
        
        if max_val == min_val:
            return 0.5
        
        return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)
    
    def get_worker_features(self, worker_id: int, worker_quality_map: Dict,
                           worker_global_stats: Dict) -> Tuple[int, np.ndarray]:
        """
        Extract worker features including global statistics.
        
        Args:
            worker_id: Worker ID
            worker_quality_map: Worker quality mapping
            worker_global_stats: Worker global statistics
            
        Returns:
            Tuple of (worker_id, numeric_features_array)
        """
        worker_stats = worker_global_stats.get(worker_id, {})
        
        # Original quality score (normalized)
        worker_quality_scaled = self.min_max_scale(
            worker_quality_map.get(worker_id, 0.0), "worker_quality"
        )
        
        # Global participation count
        global_participation_count = worker_stats.get('count', 0)
        global_participation_scaled = np.clip(
            global_participation_count / DATA_PROCESSING['participation_normalization_factor'], 
            0, 1
        )
        
        # Global average score
        global_avg_score = 0
        if global_participation_count > 0:
            global_avg_score = worker_stats.get('total_score', 0) / global_participation_count
        global_avg_score_scaled = global_avg_score / DATA_PROCESSING['score_max_value']
        
        # Global win rate
        global_win_rate = 0
        if global_participation_count > 0:
            global_win_rate = worker_stats.get('wins', 0) / global_participation_count
        
        # Category diversity
        category_diversity = len(worker_stats.get('categories', set()))
        category_diversity_scaled = np.clip(
            category_diversity / DATA_PROCESSING['category_diversity_normalization_factor'], 
            0, 1
        )
        
        numeric_features = np.array([
            worker_quality_scaled,
            global_participation_scaled,
            global_avg_score_scaled,
            global_win_rate,
            category_diversity_scaled
        ])
        
        return worker_id, numeric_features
    
    def get_project_features(self, project_id: int, project_info: Dict, 
                           current_time: datetime.datetime) -> Tuple[int, int, int, Optional[np.ndarray]]:
        """
        Extract project features including categorical and numeric components.
        
        Args:
            project_id: Project ID
            project_info: Project information dictionary
            current_time: Current timestamp
            
        Returns:
            Tuple of (category_id, sub_category_id, industry_id, numeric_features_array)
        """
        project_data = project_info.get(project_id)
        if not project_data:
            return 0, 0, 0, None
        
        # Categorical features
        category_id = max(0, project_data.get("category", 0))
        sub_category_id = max(0, project_data.get("sub_category", 0))
        industry_id = max(0, project_data.get("industry_id", 0))
        
        # Numeric features
        time_until_deadline_raw = (project_data["deadline_dt"] - current_time).total_seconds()
        task_age_raw = (current_time - project_data["start_date_dt"]).total_seconds()
        project_duration_raw = max(0, (project_data["deadline_dt"] - project_data["start_date_dt"]).total_seconds())
        
        # Reward per slot
        required_answers = project_data.get("required_answers", 1)
        total_awards = project_data.get("total_awards", 0)
        reward_per_slot_raw = 0
        if required_answers > 0 and total_awards > 0:
            reward_per_slot_raw = total_awards / required_answers
        
        # Scale numeric features
        numeric_features = np.array([
            self.min_max_scale(time_until_deadline_raw, "time_until_deadline_sec"),
            self.min_max_scale(task_age_raw, "task_age_sec"),
            self.min_max_scale(project_duration_raw, "project_duration_sec"),
            self.min_max_scale(reward_per_slot_raw, "reward_per_slot")
        ])
        
        return category_id, sub_category_id, industry_id, numeric_features
    
    def get_interaction_features(self, worker_id: int, category_id: int,
                               worker_cat_performance: Dict) -> np.ndarray:
        """
        Extract worker-category interaction features.
        
        Args:
            worker_id: Worker ID
            category_id: Project category ID
            worker_cat_performance: Worker category performance mapping
            
        Returns:
            Interaction features array
        """
        key = (worker_id, category_id)
        cat_stats = worker_cat_performance.get(key, {})
        
        cat_participation_count = cat_stats.get('count', 0)
        
        # Category-specific average score
        cat_avg_score = 0
        if cat_participation_count > 0:
            cat_avg_score = cat_stats.get('total_score', 0) / cat_participation_count
        cat_avg_score_scaled = cat_avg_score / DATA_PROCESSING['score_max_value']
        
        # Category-specific participation count
        cat_participation_scaled = np.clip(cat_participation_count / 10.0, 0, 1)
        
        # Category-specific win rate
        cat_win_rate = 0
        if cat_participation_count > 0:
            cat_win_rate = cat_stats.get('wins', 0) / cat_participation_count
        
        # New to category indicator
        is_new_to_category = 1.0 if cat_participation_count == 0 else 0.0
        
        return np.array([
            cat_avg_score_scaled,
            cat_participation_scaled,
            cat_win_rate,
            is_new_to_category
        ])
    
    def get_context_features(self, current_time: datetime.datetime) -> np.ndarray:
        """
        Extract context features (time-based).
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Context features array
        """
        current_time_raw = (current_time - ALL_BEGIN_TIME_DT).total_seconds()
        current_time_scaled = self.min_max_scale(current_time_raw, "current_time_val")
        
        return np.array([current_time_scaled])
    
    def get_state_tuple(self, worker_id: int, project_id: int, current_time: datetime.datetime,
                       project_info: Dict, worker_quality_map: Dict, worker_global_stats: Dict,
                       worker_cat_performance: Dict) -> Optional[Tuple]:
        """
        Generate complete state tuple for a worker-project pair.
        
        Args:
            worker_id: Worker ID
            project_id: Project ID
            current_time: Current timestamp
            project_info: Project information dictionary
            worker_quality_map: Worker quality mapping
            worker_global_stats: Worker global statistics
            worker_cat_performance: Worker category performance mapping
            
        Returns:
            State tuple (category_id, sub_category_id, industry_id, numeric_features)
        """
        # Get project features
        cat_id, sub_cat_id, ind_id, numeric_project_features = self.get_project_features(
            project_id, project_info, current_time
        )
        
        if numeric_project_features is None:
            return None
        
        # Get worker features
        _, numeric_worker_features = self.get_worker_features(
            worker_id, worker_quality_map, worker_global_stats
        )
        
        # Get interaction features
        numeric_interaction_features = self.get_interaction_features(
            worker_id, cat_id, worker_cat_performance
        )
        
        # Get context features
        numeric_context_features = self.get_context_features(current_time)
        
        # Combine all numeric features
        final_numeric_features = np.concatenate([
            numeric_worker_features,
            numeric_project_features,
            numeric_interaction_features,
            numeric_context_features
        ])
        
        return (max(0, cat_id), max(0, sub_cat_id), max(0, ind_id), final_numeric_features)
    
    def batch_get_state_tuples(self, worker_project_pairs: List[Tuple], current_time: datetime.datetime,
                              project_info: Dict, worker_quality_map: Dict, worker_global_stats: Dict,
                              worker_cat_performance: Dict) -> List[Optional[Tuple]]:
        """
        Vectorized state tuple generation for multiple worker-project pairs.
        
        Args:
            worker_project_pairs: List of (worker_id, project_id) tuples
            current_time: Current timestamp
            project_info: Project information dictionary
            worker_quality_map: Worker quality mapping
            worker_global_stats: Worker global statistics
            worker_cat_performance: Worker category performance mapping
            
        Returns:
            List of state tuples
        """
        state_tuples = []
        
        for worker_id, project_id in worker_project_pairs:
            state_tuple = self.get_state_tuple(
                worker_id, project_id, current_time, project_info,
                worker_quality_map, worker_global_stats, worker_cat_performance
            )
            state_tuples.append(state_tuple)
        
        return state_tuples
    
    def _get_available_projects(self, current_time: datetime.datetime, 
                               project_info: Dict, entry_info: Dict) -> List[int]:
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
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about feature dimensions and scaling parameters.
        
        Returns:
            Dictionary with feature information
        """
        return {
            'num_worker_features': self.num_worker_features,
            'num_project_features': self.num_project_features,
            'num_interaction_features': self.num_interaction_features,
            'num_context_features': self.num_context_features,
            'total_numeric_features': self.total_numeric_features,
            'feature_min_max': self.feature_min_max,
            'is_fitted': self.is_fitted
        }


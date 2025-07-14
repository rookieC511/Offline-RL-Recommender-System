"""
Unified data loader for RL Crowdsourcing System
Handles loading of worker quality, project info, and entry data with caching.
"""

import json
import os
import csv
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dateutil.parser import parse
from collections import defaultdict
import numpy as np

from config.base_config import DATA_PATHS, ALL_BEGIN_TIME_DT, DATA_PROCESSING


class DataLoader:
    """
    Centralized data loader with caching and validation.
    Replaces duplicated data loading logic across training files.
    """
    
    def __init__(self, config: Dict[str, Any], use_cache: bool = True):
        """
        Initialize data loader with configuration.
        
        Args:
            config: Configuration dictionary
            use_cache: Whether to use caching for loaded data
        """
        self.config = config
        self.use_cache = use_cache
        self.cache_dir = ".cache"
        
        # Data containers
        self.worker_quality_map = {}
        self.project_list_data = {}
        self.project_info = {}
        self.entry_info = {}
        self.industry_map = {}
        self.worker_global_stats = {}
        self.worker_cat_performance = {}
        
        # Computed statistics
        self.reward_scale_reference = config.get('reward_scale_reference', 20.0)
        self.feature_min_max = {}
        
        # Create cache directory
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_all_data(self) -> Dict[str, Any]:
        """
        Load all data components with caching.
        
        Returns:
            Dictionary containing all loaded data
        """
        cache_key = self._generate_cache_key()
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            print("Loading data from cache...")
            self._restore_from_cache(cached_data)
            return self._get_data_dict()
        
        print("Loading data from files...")
        
        # Load data components in order
        self._load_worker_quality()
        self._load_project_list()
        self._load_project_info()
        self._load_entry_info()
        self._compute_worker_statistics()
        self._compute_reward_statistics()
        
        # Cache the loaded data
        if self.use_cache:
            self._save_to_cache(cache_key)
        
        return self._get_data_dict()
    
    def _load_worker_quality(self) -> None:
        """Load worker quality data from CSV file."""
        filepath = self.config.get('DATA_PATHS', {}).get('worker_quality', 'worker_quality.csv')
        
        if not os.path.exists(filepath):
            print(f"Warning: Worker quality file not found: {filepath}")
            return
        
        try:
            with open(filepath, "r", encoding='utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
                next(csvreader)  # Skip header
                
                for line in csvreader:
                    try:
                        worker_id, quality_str = line[0], line[1]
                        quality = float(quality_str)
                        if quality > 0.0:
                            # Normalize quality score
                            normalized_quality = quality / DATA_PROCESSING['quality_normalization_factor']
                            self.worker_quality_map[int(worker_id)] = normalized_quality
                    except (ValueError, IndexError):
                        continue  # Skip malformed lines
                        
        except Exception as e:
            print(f"Error loading worker quality data: {e}")
        
        print(f"Loaded {len(self.worker_quality_map)} worker quality records")
    
    def _load_project_list(self) -> None:
        """Load project list data from CSV file."""
        filepath = self.config.get('DATA_PATHS', {}).get('project_list', 'project_list.csv')
        
        if not os.path.exists(filepath):
            print(f"Warning: Project list file not found: {filepath}")
            return
        
        try:
            with open(filepath, "r", encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines[1:]:  # Skip header
                    parts = line.strip('\\n').split(',')
                    try:
                        project_id = int(parts[0])
                        required_answers = int(parts[1])
                        self.project_list_data[project_id] = required_answers
                    except (IndexError, ValueError):
                        continue
                        
        except Exception as e:
            print(f"Error loading project list data: {e}")
        
        print(f"Loaded {len(self.project_list_data)} project requirements")
    
    def _load_project_info(self) -> None:
        """Load project information from JSON files."""
        project_dir = self.config.get('DATA_PATHS', {}).get('project_dir', 'project/')
        
        if not os.path.exists(project_dir):
            print(f"Warning: Project directory not found: {project_dir}")
            return
        
        project_files = [f for f in os.listdir(project_dir) 
                        if f.startswith("project_") and f.endswith(".txt")]
        
        industry_counter = 0
        loaded_count = 0
        
        for project_filename in project_files:
            if project_filename == ".DS_Store":
                continue
                
            try:
                project_id = int(project_filename.replace("project_", "").replace(".txt", ""))
                
                if project_id not in self.project_list_data:
                    continue
                
                filepath = os.path.join(project_dir, project_filename)
                with open(filepath, "r", encoding='utf-8') as file:
                    text = json.load(file)
                
                # Process industry mapping
                industry_str = text.get("industry", "unknown_industry")
                if industry_str not in self.industry_map:
                    self.industry_map[industry_str] = industry_counter
                    industry_counter += 1
                
                # Store project information
                self.project_info[project_id] = {
                    "id": project_id,
                    "sub_category": int(text.get("sub_category", -1)),
                    "category": int(text.get("category", -1)),
                    "start_date_dt": parse(text.get("start_date", "1970-01-01T00:00:00Z")),
                    "deadline_dt": parse(text.get("deadline", "1970-01-01T00:00:00Z")),
                    "total_awards": float(text.get("total_awards", 0.0)),
                    "status": text.get("status", "unknown").lower(),
                    "required_answers": self.project_list_data.get(project_id, 1),
                    "industry_id": self.industry_map[industry_str]
                }
                
                loaded_count += 1
                
            except Exception as e:
                print(f"Warning: Error processing project file {project_filename}: {e}")
                continue
        
        print(f"Loaded {loaded_count} project info records, {len(self.industry_map)} industries")
    
    def _load_entry_info(self) -> None:
        """Load entry information from JSON files with pagination."""
        entry_dir = self.config.get('DATA_PATHS', {}).get('entry_dir', 'entry/')
        
        if not os.path.exists(entry_dir):
            print(f"Warning: Entry directory not found: {entry_dir}")
            return
        
        page_size = DATA_PROCESSING['entry_page_size']
        loaded_projects = 0
        total_entries = 0
        
        for project_id in self.project_info.keys():
            self.entry_info[project_id] = {}
            page_k = 0
            project_has_entries = False
            
            while True:
                entry_filename = os.path.join(entry_dir, f"entry_{project_id}_{page_k}.txt")
                
                if not os.path.exists(entry_filename):
                    break
                
                try:
                    with open(entry_filename, "r", encoding='utf-8') as efile:
                        entry_text_data = json.load(efile)
                    
                    for item in entry_text_data.get("results", []):
                        # Extract score with safe handling
                        score_val = 0
                        if (item.get("revisions") and 
                            isinstance(item["revisions"], list) and 
                            len(item["revisions"]) > 0 and
                            isinstance(item["revisions"][0], dict)):
                            score_val = item["revisions"][0].get("score", 0)
                        
                        entry_number = int(item["entry_number"])
                        self.entry_info[project_id][entry_number] = {
                            "entry_created_at_dt": parse(item.get("entry_created_at", "1970-01-01T00:00:00Z")),
                            "worker_id": int(item["author"]),
                            "withdrawn": item.get("withdrawn", False),
                            "award_value": item.get("award_value"),
                            "score": score_val,
                            "winner": item.get("winner", False)
                        }
                        
                        project_has_entries = True
                        total_entries += 1
                
                except Exception as e:
                    print(f"Warning: Error processing entry file {entry_filename}: {e}")
                
                page_k += page_size
            
            if project_has_entries:
                loaded_projects += 1
        
        print(f"Loaded entries for {loaded_projects} projects, {total_entries} total entries")
    
    def _compute_worker_statistics(self) -> None:
        """Compute worker global statistics and category performance."""
        print("Computing worker statistics...")
        
        # Initialize containers
        self.worker_global_stats = {}
        self.worker_cat_performance = {}
        
        for proj_id, entries in self.entry_info.items():
            proj_details = self.project_info.get(proj_id)
            if not proj_details:
                continue
            
            proj_cat = proj_details.get("category")
            if proj_cat is None:
                continue
            
            for _, entry_data in entries.items():
                worker_id = entry_data["worker_id"]
                score = entry_data.get("score", 0)
                is_winner = entry_data.get("winner", False)
                
                # Update category-specific performance
                key = (worker_id, proj_cat)
                if key not in self.worker_cat_performance:
                    self.worker_cat_performance[key] = {
                        'total_score': 0.0, 
                        'count': 0, 
                        'wins': 0
                    }
                
                self.worker_cat_performance[key]['total_score'] += score
                self.worker_cat_performance[key]['count'] += 1
                if is_winner:
                    self.worker_cat_performance[key]['wins'] += 1
                
                # Update global statistics
                if worker_id not in self.worker_global_stats:
                    self.worker_global_stats[worker_id] = {
                        'total_score': 0.0,
                        'count': 0,
                        'wins': 0,
                        'categories': set()
                    }
                
                self.worker_global_stats[worker_id]['total_score'] += score
                self.worker_global_stats[worker_id]['count'] += 1
                self.worker_global_stats[worker_id]['categories'].add(proj_cat)
                if is_winner:
                    self.worker_global_stats[worker_id]['wins'] += 1
        
        print(f"Computed statistics for {len(self.worker_global_stats)} workers, "
              f"{len(self.worker_cat_performance)} worker-category pairs")
    
    def _compute_reward_statistics(self) -> None:
        """Compute reward scaling reference from historical data."""
        print("Computing reward statistics...")
        
        all_historical_awards = []
        
        for project_id, entries_in_project in self.entry_info.items():
            for entry_number, entry_data in entries_in_project.items():
                award_val_raw = entry_data.get("award_value")
                if award_val_raw is not None:
                    try:
                        award_val_float = float(award_val_raw)
                        if award_val_float > 0:
                            all_historical_awards.append(award_val_float)
                    except (ValueError, TypeError):
                        continue
        
        if all_historical_awards:
            awards_array = np.array(all_historical_awards)
            self.reward_scale_reference = np.median(awards_array)
            
            if self.reward_scale_reference <= 0:
                mean_award = np.mean(awards_array)
                self.reward_scale_reference = mean_award if mean_award > 0 else 20.0
            
            print(f"Computed reward scale reference: {self.reward_scale_reference:.2f}")
        else:
            print("No valid historical awards found, using default scale reference")
            self.reward_scale_reference = 20.0
    
    def get_embedding_sizes(self) -> Dict[str, int]:
        """
        Calculate embedding sizes based on loaded data.
        
        Returns:
            Dictionary with embedding sizes for categories, industries, workers
        """
        max_cat_id = max((p["category"] for p in self.project_info.values() 
                         if "category" in p), default=0)
        max_sub_cat_id = max((p["sub_category"] for p in self.project_info.values() 
                             if "sub_category" in p), default=0)
        max_industry_id = max((p["industry_id"] for p in self.project_info.values() 
                              if "industry_id" in p), default=0)
        
        all_workers = set()
        for entries in self.entry_info.values():
            for ed in entries.values():
                all_workers.add(ed["worker_id"])
        max_worker_id = max(all_workers) if all_workers else 0
        
        return {
            'num_categories': max_cat_id + 1,
            'num_sub_categories': max_sub_cat_id + 1,
            'num_industries': max_industry_id + 1,
            'num_workers': max_worker_id + 1
        }
    
    def _get_data_dict(self) -> Dict[str, Any]:
        """Get dictionary containing all loaded data."""
        return {
            'worker_quality_map': self.worker_quality_map,
            'project_list_data': self.project_list_data,
            'project_info': self.project_info,
            'entry_info': self.entry_info,
            'industry_map': self.industry_map,
            'worker_global_stats': self.worker_global_stats,
            'worker_cat_performance': self.worker_cat_performance,
            'reward_scale_reference': self.reward_scale_reference,
            'embedding_sizes': self.get_embedding_sizes()
        }
    
    def _generate_cache_key(self) -> str:
        """Generate cache key based on data file timestamps."""
        file_paths = []
        data_paths = self.config.get('DATA_PATHS', {})
        
        # Add file paths and their modification times
        for key, path in data_paths.items():
            if key.endswith('_dir'):
                if os.path.exists(path):
                    # For directories, include all relevant files
                    if key == 'project_dir':
                        files = [f for f in os.listdir(path) if f.startswith('project_')]
                    elif key == 'entry_dir':
                        files = [f for f in os.listdir(path) if f.startswith('entry_')]
                    else:
                        files = os.listdir(path)
                    
                    for f in sorted(files)[:100]:  # Limit to avoid huge keys
                        full_path = os.path.join(path, f)
                        if os.path.isfile(full_path):
                            file_paths.append(f"{full_path}:{os.path.getmtime(full_path)}")
            else:
                if os.path.exists(path):
                    file_paths.append(f"{path}:{os.path.getmtime(path)}")
        
        # Create hash of file paths and timestamps
        cache_content = "|".join(sorted(file_paths))
        return hashlib.md5(cache_content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load data from cache if available."""
        if not self.use_cache:
            return None
        
        cache_file = os.path.join(self.cache_dir, f"data_cache_{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str) -> None:
        """Save current data to cache."""
        if not self.use_cache:
            return
        
        cache_file = os.path.join(self.cache_dir, f"data_cache_{cache_key}.pkl")
        
        try:
            cache_data = {
                'worker_quality_map': self.worker_quality_map,
                'project_list_data': self.project_list_data,
                'project_info': self.project_info,
                'entry_info': self.entry_info,
                'industry_map': self.industry_map,
                'worker_global_stats': self.worker_global_stats,
                'worker_cat_performance': self.worker_cat_performance,
                'reward_scale_reference': self.reward_scale_reference,
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            print(f"Data cached to {cache_file}")
            
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def _restore_from_cache(self, cached_data: Dict[str, Any]) -> None:
        """Restore data from cached dictionary."""
        self.worker_quality_map = cached_data['worker_quality_map']
        self.project_list_data = cached_data['project_list_data']
        self.project_info = cached_data['project_info']
        self.entry_info = cached_data['entry_info']
        self.industry_map = cached_data['industry_map']
        self.worker_global_stats = cached_data['worker_global_stats']
        self.worker_cat_performance = cached_data['worker_cat_performance']
        self.reward_scale_reference = cached_data['reward_scale_reference']


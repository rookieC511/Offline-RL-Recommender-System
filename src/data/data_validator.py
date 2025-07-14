"""
Data validation utilities for RL Crowdsourcing System
Provides comprehensive data validation and quality checks.
"""

import os
import json
import csv
from typing import Dict, List, Tuple, Optional, Any, Set
from dateutil.parser import parse
import numpy as np
from collections import defaultdict, Counter


class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass


class DataValidator:
    """
    Comprehensive data validator with detailed reporting.
    Replaces silent error handling with explicit validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data validator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.validation_results = {}
        self.warnings = []
        self.errors = []
    
    def validate_all_data(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive validation on all loaded data.
        
        Args:
            data_dict: Dictionary containing all loaded data
            
        Returns:
            Validation results dictionary
            
        Raises:
            DataValidationError: If critical validation errors found
        """
        print("Starting comprehensive data validation...")
        
        # Reset validation state
        self.validation_results = {}
        self.warnings = []
        self.errors = []
        
        # Validate individual components
        self._validate_worker_quality(data_dict.get('worker_quality_map', {}))
        self._validate_project_list(data_dict.get('project_list_data', {}))
        self._validate_project_info(data_dict.get('project_info', {}))
        self._validate_entry_info(data_dict.get('entry_info', {}))
        self._validate_industry_mapping(data_dict.get('industry_map', {}))
        
        # Cross-validation between components
        self._validate_data_consistency(data_dict)
        self._validate_feature_completeness(data_dict)
        self._validate_temporal_consistency(data_dict)
        
        # Generate validation report
        self._generate_validation_report()
        
        # Raise error if critical issues found
        if self.errors:
            error_summary = f"Found {len(self.errors)} critical errors"
            raise DataValidationError(error_summary)
        
        return self.validation_results
    
    def _validate_worker_quality(self, worker_quality_map: Dict[int, float]) -> None:
        """Validate worker quality data."""
        print("Validating worker quality data...")
        
        if not worker_quality_map:
            self.errors.append("Worker quality map is empty")
            return
        
        # Check value ranges
        invalid_qualities = []
        for worker_id, quality in worker_quality_map.items():
            if not isinstance(worker_id, int):
                self.errors.append(f"Invalid worker ID type: {type(worker_id)}")
            
            if not isinstance(quality, (int, float)):
                self.errors.append(f"Invalid quality type for worker {worker_id}: {type(quality)}")
            elif not (0 <= quality <= 1):
                invalid_qualities.append((worker_id, quality))
        
        if invalid_qualities:
            self.warnings.append(f"Found {len(invalid_qualities)} workers with quality outside [0,1] range")
        
        # Statistics
        qualities = list(worker_quality_map.values())
        self.validation_results['worker_quality'] = {
            'count': len(worker_quality_map),
            'mean_quality': np.mean(qualities),
            'std_quality': np.std(qualities),
            'min_quality': np.min(qualities),
            'max_quality': np.max(qualities),
            'invalid_count': len(invalid_qualities)
        }
    
    def _validate_project_list(self, project_list_data: Dict[int, int]) -> None:
        """Validate project list data."""
        print("Validating project list data...")
        
        if not project_list_data:
            self.errors.append("Project list data is empty")
            return
        
        # Check required answers values
        invalid_requirements = []
        for project_id, required_answers in project_list_data.items():
            if not isinstance(project_id, int):
                self.errors.append(f"Invalid project ID type: {type(project_id)}")
            
            if not isinstance(required_answers, int) or required_answers <= 0:
                invalid_requirements.append((project_id, required_answers))
        
        if invalid_requirements:
            self.warnings.append(f"Found {len(invalid_requirements)} projects with invalid required_answers")
        
        # Statistics
        requirements = list(project_list_data.values())
        self.validation_results['project_list'] = {
            'count': len(project_list_data),
            'mean_required': np.mean(requirements),
            'max_required': np.max(requirements),
            'min_required': np.min(requirements),
            'invalid_count': len(invalid_requirements)
        }
    
    def _validate_project_info(self, project_info: Dict[int, Dict]) -> None:
        """Validate project information data."""
        print("Validating project info data...")
        
        if not project_info:
            self.errors.append("Project info is empty")
            return
        
        required_fields = ['id', 'category', 'sub_category', 'start_date_dt', 'deadline_dt', 'total_awards']
        missing_fields = defaultdict(int)
        invalid_dates = []
        negative_awards = []
        
        for project_id, project_data in project_info.items():
            # Check required fields
            for field in required_fields:
                if field not in project_data:
                    missing_fields[field] += 1
            
            # Validate dates
            try:
                start_date = project_data.get('start_date_dt')
                deadline = project_data.get('deadline_dt')
                
                if start_date and deadline:
                    if start_date >= deadline:
                        invalid_dates.append(project_id)
            except Exception:
                invalid_dates.append(project_id)
            
            # Check awards
            total_awards = project_data.get('total_awards', 0)
            if total_awards < 0:
                negative_awards.append(project_id)
        
        # Report issues
        for field, count in missing_fields.items():
            if count > 0:
                self.warnings.append(f"Missing {field} in {count} projects")
        
        if invalid_dates:
            self.warnings.append(f"Found {len(invalid_dates)} projects with invalid date ranges")
        
        if negative_awards:
            self.warnings.append(f"Found {len(negative_awards)} projects with negative awards")
        
        # Statistics
        awards = [p.get('total_awards', 0) for p in project_info.values()]
        categories = [p.get('category', -1) for p in project_info.values()]
        
        self.validation_results['project_info'] = {
            'count': len(project_info),
            'mean_awards': np.mean(awards),
            'unique_categories': len(set(categories)),
            'missing_fields': dict(missing_fields),
            'invalid_dates_count': len(invalid_dates),
            'negative_awards_count': len(negative_awards)
        }
    
    def _validate_entry_info(self, entry_info: Dict[int, Dict[int, Dict]]) -> None:
        """Validate entry information data."""
        print("Validating entry info data...")
        
        if not entry_info:
            self.errors.append("Entry info is empty")
            return
        
        total_entries = 0
        missing_workers = 0
        invalid_scores = 0
        future_entries = 0
        
        for project_id, entries in entry_info.items():
            for entry_id, entry_data in entries.items():
                total_entries += 1
                
                # Check worker ID
                if 'worker_id' not in entry_data:
                    missing_workers += 1
                
                # Check score range
                score = entry_data.get('score', 0)
                if not isinstance(score, (int, float)) or score < 0 or score > 5:
                    invalid_scores += 1
                
                # Check entry creation date
                try:
                    entry_date = entry_data.get('entry_created_at_dt')
                    if entry_date and entry_date > parse("2025-01-01T00:00:00Z"):
                        future_entries += 1
                except Exception:
                    pass
        
        # Report issues
        if missing_workers > 0:
            self.warnings.append(f"Found {missing_workers} entries without worker_id")
        
        if invalid_scores > 0:
            self.warnings.append(f"Found {invalid_scores} entries with invalid scores")
        
        if future_entries > 0:
            self.warnings.append(f"Found {future_entries} entries with future dates")
        
        # Statistics
        self.validation_results['entry_info'] = {
            'total_entries': total_entries,
            'projects_with_entries': len(entry_info),
            'missing_workers': missing_workers,
            'invalid_scores': invalid_scores,
            'future_entries': future_entries
        }
    
    def _validate_industry_mapping(self, industry_map: Dict[str, int]) -> None:
        """Validate industry mapping."""
        print("Validating industry mapping...")
        
        if not industry_map:
            self.warnings.append("Industry map is empty")
            return
        
        # Check for duplicate IDs
        id_counts = Counter(industry_map.values())
        duplicates = [industry_id for industry_id, count in id_counts.items() if count > 1]
        
        if duplicates:
            self.errors.append(f"Found duplicate industry IDs: {duplicates}")
        
        self.validation_results['industry_mapping'] = {
            'unique_industries': len(industry_map),
            'duplicate_ids': len(duplicates)
        }
    
    def _validate_data_consistency(self, data_dict: Dict[str, Any]) -> None:
        """Validate consistency between data components."""
        print("Validating data consistency...")
        
        project_info = data_dict.get('project_info', {})
        project_list_data = data_dict.get('project_list_data', {})
        entry_info = data_dict.get('entry_info', {})
        worker_quality_map = data_dict.get('worker_quality_map', {})\n        
        # Check project consistency
        projects_in_info_not_list = set(project_info.keys()) - set(project_list_data.keys())
        projects_in_list_not_info = set(project_list_data.keys()) - set(project_info.keys())
        
        if projects_in_info_not_list:
            self.warnings.append(f"Found {len(projects_in_info_not_list)} projects in info but not in list")
        
        if projects_in_list_not_info:
            self.warnings.append(f"Found {len(projects_in_list_not_info)} projects in list but not in info")
        
        # Check entry-project consistency
        projects_with_entries = set(entry_info.keys())
        projects_without_entries = set(project_info.keys()) - projects_with_entries
        
        if projects_without_entries:
            self.warnings.append(f"Found {len(projects_without_entries)} projects without entries")
        
        # Check worker consistency
        workers_in_entries = set()
        for entries in entry_info.values():
            for entry_data in entries.values():
                worker_id = entry_data.get('worker_id')
                if worker_id:
                    workers_in_entries.add(worker_id)
        
        workers_without_quality = workers_in_entries - set(worker_quality_map.keys())
        if workers_without_quality:
            self.warnings.append(f"Found {len(workers_without_quality)} workers without quality scores")
        
        self.validation_results['consistency'] = {
            'project_info_list_overlap': len(set(project_info.keys()) & set(project_list_data.keys())),
            'projects_without_entries': len(projects_without_entries),
            'workers_without_quality': len(workers_without_quality)
        }
    
    def _validate_feature_completeness(self, data_dict: Dict[str, Any]) -> None:
        """Validate feature completeness for ML training."""
        print("Validating feature completeness...")
        
        project_info = data_dict.get('project_info', {})
        entry_info = data_dict.get('entry_info', {})
        
        # Check categorical feature coverage
        categories = set()
        sub_categories = set()
        industries = set()
        
        for project_data in project_info.values():
            categories.add(project_data.get('category', -1))
            sub_categories.add(project_data.get('sub_category', -1))
            industries.add(project_data.get('industry_id', -1))
        
        # Check for missing categorical values
        missing_categories = -1 in categories
        missing_sub_categories = -1 in sub_categories
        missing_industries = -1 in industries
        
        if missing_categories:
            self.warnings.append("Some projects have missing category information")
        
        if missing_sub_categories:
            self.warnings.append("Some projects have missing sub_category information")
        
        if missing_industries:
            self.warnings.append("Some projects have missing industry information")
        
        # Check numeric feature availability
        projects_with_awards = sum(1 for p in project_info.values() if p.get('total_awards', 0) > 0)
        entries_with_scores = 0
        entries_with_awards = 0
        
        for entries in entry_info.values():
            for entry_data in entries.values():
                if entry_data.get('score', 0) > 0:
                    entries_with_scores += 1
                if entry_data.get('award_value'):
                    entries_with_awards += 1
        
        self.validation_results['feature_completeness'] = {
            'unique_categories': len(categories) - (1 if missing_categories else 0),
            'unique_sub_categories': len(sub_categories) - (1 if missing_sub_categories else 0),
            'unique_industries': len(industries) - (1 if missing_industries else 0),
            'projects_with_awards': projects_with_awards,
            'entries_with_scores': entries_with_scores,
            'entries_with_awards': entries_with_awards
        }
    
    def _validate_temporal_consistency(self, data_dict: Dict[str, Any]) -> None:
        """Validate temporal consistency of data."""
        print("Validating temporal consistency...")
        
        project_info = data_dict.get('project_info', {})
        entry_info = data_dict.get('entry_info', {})
        
        temporal_violations = 0
        
        for project_id, project_data in project_info.items():
            project_start = project_data.get('start_date_dt')
            project_deadline = project_data.get('deadline_dt')
            
            if not project_start or not project_deadline:
                continue
            
            # Check entries are within project timeframe
            if project_id in entry_info:
                for entry_data in entry_info[project_id].values():
                    entry_date = entry_data.get('entry_created_at_dt')
                    
                    if entry_date:
                        if entry_date < project_start or entry_date > project_deadline:
                            temporal_violations += 1
        
        if temporal_violations > 0:
            self.warnings.append(f"Found {temporal_violations} entries outside project timeframes")
        
        self.validation_results['temporal_consistency'] = {
            'temporal_violations': temporal_violations
        }
    
    def _generate_validation_report(self) -> None:
        """Generate comprehensive validation report."""
        print("\\n" + "="*60)
        print("DATA VALIDATION REPORT")
        print("="*60)
        
        # Summary
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        print()
        
        # Errors
        if self.errors:
            print("CRITICAL ERRORS:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
            print()
        
        # Warnings
        if self.warnings:
            print("WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
            print()
        
        # Statistics
        print("DATA STATISTICS:")
        for component, stats in self.validation_results.items():
            print(f"  {component.upper()}:")
            for key, value in stats.items():
                print(f"    {key}: {value}")
            print()
        
        print("="*60)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get validation summary for programmatic use.
        
        Returns:
            Dictionary with validation summary
        """
        return {
            'errors': self.errors,
            'warnings': self.warnings,
            'results': self.validation_results,
            'is_valid': len(self.errors) == 0,
            'has_warnings': len(self.warnings) > 0
        }


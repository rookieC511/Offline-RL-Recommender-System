"""
Tests for data processing pipeline
Basic validation tests for the refactored data components.
"""

import unittest
import tempfile
import os
import json
import csv
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DataLoader
from src.data.feature_processor import FeatureProcessor
from src.data.data_validator import DataValidator, DataValidationError
from config.base_config import DATA_PATHS


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'DATA_PATHS': {
                'worker_quality': 'test_worker_quality.csv',
                'project_list': 'test_project_list.csv',
                'project_dir': 'test_project/',
                'entry_dir': 'test_entry/',
            },
            'reward_scale_reference': 20.0
        }
        
        self.temp_dir = tempfile.mkdtemp()
        self.data_loader = DataLoader(self.config, use_cache=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temp files if needed
        pass
    
    def test_init(self):
        """Test DataLoader initialization."""
        self.assertIsInstance(self.data_loader, DataLoader)
        self.assertEqual(self.data_loader.config, self.config)
        self.assertFalse(self.data_loader.use_cache)
    
    def test_get_embedding_sizes(self):
        """Test embedding size calculation."""
        # Mock some data
        self.data_loader.project_info = {
            1: {'category': 5, 'sub_category': 10, 'industry_id': 2},
            2: {'category': 3, 'sub_category': 8, 'industry_id': 1}
        }
        self.data_loader.entry_info = {
            1: {1: {'worker_id': 100}, 2: {'worker_id': 200}},
            2: {1: {'worker_id': 150}}
        }
        
        sizes = self.data_loader.get_embedding_sizes()
        
        self.assertEqual(sizes['num_categories'], 6)  # max(5,3) + 1
        self.assertEqual(sizes['num_sub_categories'], 11)  # max(10,8) + 1
        self.assertEqual(sizes['num_industries'], 3)  # max(2,1) + 1
        self.assertEqual(sizes['num_workers'], 201)  # max(100,200,150) + 1


class TestFeatureProcessor(unittest.TestCase):
    """Test cases for FeatureProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {}
        self.feature_processor = FeatureProcessor(self.config)
    
    def test_init(self):
        """Test FeatureProcessor initialization."""
        self.assertIsInstance(self.feature_processor, FeatureProcessor)
        self.assertFalse(self.feature_processor.is_fitted)
        self.assertEqual(self.feature_processor.total_numeric_features, 14)  # 5+4+4+1
    
    def test_min_max_scale_not_fitted(self):
        """Test scaling without fitting raises error."""
        with self.assertRaises(ValueError):
            self.feature_processor.min_max_scale(0.5, "test_feature")
    
    def test_min_max_scale_fitted(self):
        """Test scaling after fitting."""
        # Mock fitted state
        self.feature_processor.is_fitted = True
        self.feature_processor.feature_min_max = {
            "test_feature": (0.0, 10.0)
        }
        
        # Test scaling
        result = self.feature_processor.min_max_scale(5.0, "test_feature")
        self.assertEqual(result, 0.5)
        
        # Test clipping
        result = self.feature_processor.min_max_scale(-1.0, "test_feature")
        self.assertEqual(result, 0.0)
        
        result = self.feature_processor.min_max_scale(15.0, "test_feature")
        self.assertEqual(result, 1.0)
    
    def test_get_feature_info(self):
        """Test feature info retrieval."""
        info = self.feature_processor.get_feature_info()
        
        self.assertIn('num_worker_features', info)
        self.assertIn('total_numeric_features', info)
        self.assertIn('is_fitted', info)
        self.assertEqual(info['num_worker_features'], 5)
        self.assertEqual(info['total_numeric_features'], 14)


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {}
        self.validator = DataValidator(self.config)
    
    def test_init(self):
        """Test DataValidator initialization."""
        self.assertIsInstance(self.validator, DataValidator)
        self.assertEqual(len(self.validator.errors), 0)
        self.assertEqual(len(self.validator.warnings), 0)
    
    def test_validate_empty_data(self):
        """Test validation with empty data raises error."""
        empty_data = {
            'worker_quality_map': {},
            'project_list_data': {},
            'project_info': {},
            'entry_info': {},
            'industry_map': {}
        }
        
        with self.assertRaises(DataValidationError):
            self.validator.validate_all_data(empty_data)
    
    def test_validate_valid_data(self):
        """Test validation with valid data."""
        valid_data = {
            'worker_quality_map': {1: 0.8, 2: 0.6},
            'project_list_data': {1: 5, 2: 3},
            'project_info': {
                1: {
                    'id': 1,
                    'category': 1,
                    'sub_category': 2,
                    'start_date_dt': datetime(2020, 1, 1),
                    'deadline_dt': datetime(2020, 2, 1),
                    'total_awards': 100.0,
                    'industry_id': 1
                }
            },
            'entry_info': {
                1: {
                    1: {
                        'worker_id': 1,
                        'score': 4,
                        'entry_created_at_dt': datetime(2020, 1, 15)
                    }
                }
            },
            'industry_map': {'tech': 1}
        }
        
        # This should not raise an exception
        results = self.validator.validate_all_data(valid_data)
        self.assertIsInstance(results, dict)
    
    def test_get_validation_summary(self):
        """Test validation summary generation."""
        summary = self.validator.get_validation_summary()
        
        self.assertIn('errors', summary)
        self.assertIn('warnings', summary)
        self.assertIn('is_valid', summary)
        self.assertIn('has_warnings', summary)


class TestIntegration(unittest.TestCase):
    """Integration tests for data pipeline components."""
    
    def test_data_pipeline_integration(self):
        """Test integration between DataLoader, FeatureProcessor, and DataValidator."""
        # Mock configuration
        config = {
            'DATA_PATHS': {
                'worker_quality': 'nonexistent.csv',
                'project_list': 'nonexistent.csv',
                'project_dir': 'nonexistent/',
                'entry_dir': 'nonexistent/',
            }
        }
        
        # Test that components can be instantiated together
        data_loader = DataLoader(config, use_cache=False)
        feature_processor = FeatureProcessor(config)
        validator = DataValidator(config)
        
        self.assertIsInstance(data_loader, DataLoader)
        self.assertIsInstance(feature_processor, FeatureProcessor)
        self.assertIsInstance(validator, DataValidator)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)


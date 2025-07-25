"""
File: test_experiment_runner.py
Directory: memorization_reproduction/tests/

Comprehensive tests for experiment_runner.py module.
Validates high-level experimental interfaces and device-appropriate scaling.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from experiment_runner import (
    ExperimentConfig,
    ExperimentSuite,
    detect_device,
    create_device_appropriate_config,
    create_scaled_model_configs,
    create_scaled_training_config,
    run_capacity_suite,
    compare_to_morris_findings,
    calculate_reproduction_score,
    run_single_model_demo,
    print_experiment_summary,
    save_experiment_results,
    load_experiment_results,
    run_morris_reproduction_suite,
    quick_demo
)
from model_trainer import ModelConfig, TrainingConfig


class TestExperimentDataStructures:
    """Test experiment data structures."""
    
    def test_experiment_config_creation(self):
        """Test ExperimentConfig dataclass."""
        config = ExperimentConfig(
            device="cuda",
            use_cpu_optimizations=False,
            max_model_size=1000000,
            max_dataset_size=50000,
            n_seeds=3,
            save_results=True,
            results_dir="test_results"
        )
        
        assert config.device == "cuda"
        assert config.use_cpu_optimizations is False
        assert config.max_model_size == 1000000
        assert config.max_dataset_size == 50000
        assert config.n_seeds == 3
        assert config.save_results is True
        assert config.results_dir == "test_results"
    
    def test_experiment_suite_creation(self):
        """Test ExperimentSuite dataclass."""
        experiment_config = ExperimentConfig(
            device="cpu", use_cpu_optimizations=True, max_model_size=100000,
            max_dataset_size=1000, n_seeds=1, save_results=False, results_dir="test"
        )
        
        suite = ExperimentSuite(
            suite_name="test_suite",
            experiment_config=experiment_config,
            capacity_results={"test": "data"},
            validation_results={"test_valid": True},
            execution_time=120.5,
            timestamp="2024-01-01T00:00:00",
            morris_comparison={"score": 85.0}
        )
        
        assert suite.suite_name == "test_suite"
        assert suite.experiment_config == experiment_config
        assert suite.capacity_results == {"test": "data"}
        assert suite.validation_results == {"test_valid": True}
        assert suite.execution_time == 120.5
        assert suite.timestamp == "2024-01-01T00:00:00"
        assert suite.morris_comparison == {"score": 85.0}


class TestDeviceDetection:
    """Test device detection and configuration creation."""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.get_device_name')
    def test_detect_device_cuda(self, mock_get_name, mock_get_props, mock_cuda_available):
        """Test device detection with CUDA available."""
        mock_cuda_available.return_value = True
        mock_get_name.return_value = "Tesla V100"
        mock_props = Mock()
        mock_props.total_memory = 16 * 1024**3  # 16GB
        mock_get_props.return_value = mock_props
        
        device, use_cpu_optimizations = detect_device()
        
        assert device == "cuda"
        assert use_cpu_optimizations is False
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_detect_device_mps(self, mock_mps_available, mock_cuda_available):
        """Test device detection with Apple Silicon MPS."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True
        
        with patch('torch.backends', create=True) as mock_backends:
            mock_backends.mps.is_available.return_value = True
            device, use_cpu_optimizations = detect_device()
        
        assert device == "mps"
        assert use_cpu_optimizations is True
    
    @patch('torch.cuda.is_available')
    def test_detect_device_cpu(self, mock_cuda_available):
        """Test device detection with CPU only."""
        mock_cuda_available.return_value = False
        
        # Mock torch.backends to not have mps
        with patch('torch.backends', create=True) as mock_backends:
            mock_backends.mps = None
            device, use_cpu_optimizations = detect_device()
        
        assert device == "cpu"
        assert use_cpu_optimizations is True
    
    def test_create_device_appropriate_config_cpu(self):
        """Test configuration creation for CPU."""
        config = create_device_appropriate_config("cpu", use_cpu_optimizations=True)
        
        assert config.device == "cpu"
        assert config.use_cpu_optimizations is True
        assert config.max_model_size == 500000
        assert config.max_dataset_size == 10000
        assert config.n_seeds == 2
        assert config.save_results is True
        assert "cpu_experiments" in config.results_dir
    
    def test_create_device_appropriate_config_gpu(self):
        """Test configuration creation for GPU."""
        config = create_device_appropriate_config("cuda", use_cpu_optimizations=False)
        
        assert config.device == "cuda"
        assert config.use_cpu_optimizations is False
        assert config.max_model_size == 5000000
        assert config.max_dataset_size == 100000
        assert config.n_seeds == 3
        assert config.save_results is True
        assert "gpu_experiments" in config.results_dir


class TestConfigurationScaling:
    """Test configuration scaling based on device capabilities."""
    
    def test_create_scaled_model_configs_cpu(self):
        """Test model config creation for CPU."""
        experiment_config = ExperimentConfig(
            device="cpu", use_cpu_optimizations=True, max_model_size=200000,
            max_dataset_size=5000, n_seeds=1, save_results=False, results_dir="test"
        )
        
        configs = create_scaled_model_configs(experiment_config)
        
        assert len(configs) > 0
        
        # Check that all configs respect CPU constraints
        for config in configs:
            assert config.vocab_size == 1024  # Smaller for CPU
            assert config.max_seq_length == 32  # Shorter for CPU
            assert config.n_layers > 0
            assert config.d_model > 0
            assert config.n_heads > 0
    
    def test_create_scaled_model_configs_gpu(self):
        """Test model config creation for GPU."""
        experiment_config = ExperimentConfig(
            device="cuda", use_cpu_optimizations=False, max_model_size=2000000,
            max_dataset_size=50000, n_seeds=3, save_results=True, results_dir="test"
        )
        
        configs = create_scaled_model_configs(experiment_config)
        
        assert len(configs) > 0
        
        # Check that configs use GPU-appropriate settings
        for config in configs:
            assert config.vocab_size == 2048  # Morris et al. standard
            assert config.max_seq_length == 64  # Morris et al. standard
    
    def test_create_scaled_training_config_cpu(self):
        """Test training config creation for CPU."""
        experiment_config = ExperimentConfig(
            device="cpu", use_cpu_optimizations=True, max_model_size=100000,
            max_dataset_size=1000, n_seeds=1, save_results=False, results_dir="test"
        )
        
        training_config = create_scaled_training_config(experiment_config)
        
        assert training_config.batch_size == 8  # Smaller for CPU
        assert training_config.max_steps == 500  # Fewer steps for CPU
        assert training_config.warmup_steps == 50
        assert training_config.learning_rate == 1e-3
    
    def test_create_scaled_training_config_gpu(self):
        """Test training config creation for GPU."""
        experiment_config = ExperimentConfig(
            device="cuda", use_cpu_optimizations=False, max_model_size=1000000,
            max_dataset_size=10000, n_seeds=3, save_results=True, results_dir="test"
        )
        
        training_config = create_scaled_training_config(experiment_config)
        
        assert training_config.batch_size == 32  # Larger for GPU
        assert training_config.max_steps == 2000  # More steps for GPU
        assert training_config.warmup_steps == 200
        assert training_config.learning_rate == 6e-4


class TestMorrisComparison:
    """Test comparison to Morris et al. findings."""
    
    def test_compare_to_morris_findings_good_match(self):
        """Test comparison with results matching Morris et al."""
        capacity_results = {
            'summary_statistics': {
                'mean_bits_per_parameter': 3.7,  # Close to 3.6
                'std_bits_per_parameter': 0.2,   # Low variance
                'n_models': 4
            },
            'scaling_law': {
                'r_squared': 0.92  # Good scaling law fit
            }
        }
        
        experiment_config = ExperimentConfig(
            device="cuda", use_cpu_optimizations=False, max_model_size=1000000,
            max_dataset_size=10000, n_seeds=3, save_results=True, results_dir="test"
        )
        
        comparison = compare_to_morris_findings(capacity_results, experiment_config)
        
        assert comparison['morris_target_bpp'] == 3.6
        assert comparison['observed_bpp'] == 3.7
        assert abs(comparison['bpp_deviation'] - 0.1) < 1e-10  # Use tolerance for floating point
        assert comparison['bpp_relative_error'] < 0.05  # Small relative error
        assert comparison['scaling_meets_threshold'] is True
        assert comparison['consistent_across_models'] is True
        assert comparison['experiment_scale'] == 'full_scale'
        assert comparison['morris_reproduction_score'] > 80  # High score
    
    def test_compare_to_morris_findings_poor_match(self):
        """Test comparison with results not matching Morris et al."""
        capacity_results = {
            'summary_statistics': {
                'mean_bits_per_parameter': 1.5,  # Far from 3.6
                'std_bits_per_parameter': 1.5,   # High variance
                'n_models': 2
            },
            'scaling_law': {
                'r_squared': 0.3  # Poor scaling law fit
            }
        }
        
        experiment_config = ExperimentConfig(
            device="cpu", use_cpu_optimizations=True, max_model_size=100000,
            max_dataset_size=1000, n_seeds=1, save_results=False, results_dir="test"
        )
        
        comparison = compare_to_morris_findings(capacity_results, experiment_config)
        
        assert comparison['bpp_relative_error'] > 0.4  # Large relative error
        assert comparison['scaling_meets_threshold'] is False
        assert comparison['consistent_across_models'] is False
        assert comparison['experiment_scale'] == 'cpu_optimized'
        assert comparison['morris_reproduction_score'] < 50  # Low score
    
    def test_calculate_reproduction_score(self):
        """Test reproduction score calculation."""
        # Perfect match
        summary = {'mean_bits_per_parameter': 3.6, 'std_bits_per_parameter': 0.1}
        scaling = {'r_squared': 1.0}
        score = calculate_reproduction_score(summary, scaling, target_bpp=3.6)
        assert score > 95
        
        # Poor match
        summary = {'mean_bits_per_parameter': 1.0, 'std_bits_per_parameter': 2.0}
        scaling = {'r_squared': 0.2}
        score = calculate_reproduction_score(summary, scaling, target_bpp=3.6)
        assert score < 30
        
        # Medium match
        summary = {'mean_bits_per_parameter': 3.2, 'std_bits_per_parameter': 0.5}
        scaling = {'r_squared': 0.8}
        score = calculate_reproduction_score(summary, scaling, target_bpp=3.6)
        assert 60 < score < 90


class TestExperimentExecution:
    """Test experiment execution (heavily mocked for speed)."""
    
    def setup_method(self):
        """Set up test environment."""
        self.experiment_config = ExperimentConfig(
            device="cpu", use_cpu_optimizations=True, max_model_size=100000,
            max_dataset_size=1000, n_seeds=1, save_results=False, results_dir="test"
        )
    
    @patch('experiment_runner.run_capacity_experiments')
    @patch('experiment_runner.validate_capacity_experiment')
    @patch('experiment_runner.compare_to_morris_findings')
    def test_run_capacity_suite_mocked(
        self, mock_compare, mock_validate, mock_run_experiments
    ):
        """Test capacity suite execution with mocked expensive operations."""
        # Mock experiment results
        mock_capacity_results = {
            'summary_statistics': {
                'mean_bits_per_parameter': 3.5,
                'std_bits_per_parameter': 0.3,
                'n_models': 3
            },
            'scaling_law': {'r_squared': 0.85}
        }
        mock_run_experiments.return_value = mock_capacity_results
        
        # Mock validation
        mock_validation = {
            'bits_per_param_in_range': True,
            'scaling_law_significant': True,
            'experiment_valid': True
        }
        mock_validate.return_value = mock_validation
        
        # Mock comparison
        mock_comparison = {
            'morris_reproduction_score': 88.5,
            'bpp_deviation': 0.1,
            'morris_target_bpp': 3.6,
            'observed_bpp': 3.5,
            'bpp_relative_error': 0.03,
            'scaling_meets_threshold': True,
            'consistent_across_models': True,
            'experiment_scale': 'cpu_optimized',
            'n_models_tested': 3
        }
        mock_compare.return_value = mock_comparison
        
        # Run suite
        suite = run_capacity_suite(self.experiment_config)
        
        # Verify structure
        assert isinstance(suite, ExperimentSuite)
        assert suite.suite_name == "morris_capacity_reproduction"
        assert suite.experiment_config == self.experiment_config
        assert suite.capacity_results == mock_capacity_results
        assert suite.validation_results == mock_validation
        assert suite.morris_comparison == mock_comparison
        assert suite.execution_time > 0
        assert suite.timestamp is not None
    
    @patch('experiment_runner.generate_uniform_bitstrings')
    @patch('model_trainer.train_model')
    @patch('experiment_runner.calculate_total_memorization')
    @patch('experiment_runner.create_gpt_model')
    @patch('experiment_runner.count_parameters')
    def test_run_single_model_demo_mocked(
        self, mock_count_params, mock_create_model, mock_calc_memorization,
        mock_train, mock_generate_data
    ):
        """Test single model demo with mocked operations."""
        # Setup mocks
        mock_model_config = ModelConfig(
            n_layers=1, d_model=32, n_heads=2, vocab_size=100, max_seq_length=32
        )
        
        mock_count_params.return_value = 10000
        mock_create_model.return_value = Mock()
        mock_generate_data.return_value = [torch.tensor([1, 2, 3, 4])]
        mock_train.return_value = {"train_loss": [2.0, 1.5]}
        mock_calc_memorization.side_effect = [100, 200, 300, 350, 360]  # Plateau pattern
        
        # Run demo
        results = run_single_model_demo(
            mock_model_config, self.experiment_config, dataset_sizes=[100, 200, 300, 400, 500]
        )
        
        # Verify results structure
        assert 'model_config' in results
        assert 'param_count' in results
        assert 'dataset_sizes' in results
        assert 'memorization_values' in results
        assert 'estimated_capacity' in results
        assert 'bits_per_parameter' in results
        assert 'execution_time' in results
        assert 'morris_target_bpp' in results
        assert 'bpp_deviation' in results
        
        assert results['param_count'] == 10000
        assert results['estimated_capacity'] == 360  # Max memorization
        assert results['bits_per_parameter'] == 360 / 10000
        assert results['morris_target_bpp'] == 3.6


class TestFileOperations:
    """Test file saving and loading operations."""
    
    def test_save_and_load_experiment_results(self):
        """Test saving and loading experiment results."""
        # Create test suite
        experiment_config = ExperimentConfig(
            device="cpu", use_cpu_optimizations=True, max_model_size=100000,
            max_dataset_size=1000, n_seeds=1, save_results=True, results_dir="test_results"
        )
        
        suite = ExperimentSuite(
            suite_name="test_suite",
            experiment_config=experiment_config,
            capacity_results={"mean_bpp": 3.5, "data": [1, 2, 3]},
            validation_results={"valid": True},
            execution_time=123.45,
            timestamp="2024-01-01T00:00:00",
            morris_comparison={"score": 85.0}
        )
        
        # Test saving
        with tempfile.TemporaryDirectory() as temp_dir:
            suite.experiment_config.results_dir = temp_dir
            
            filepath = save_experiment_results(suite)
            
            # Verify file exists
            assert os.path.exists(filepath)
            assert filepath.endswith('.json')
            
            # Test loading
            loaded_data = load_experiment_results(filepath)
            
            # Verify loaded data matches (basic structure check)
            assert loaded_data['suite_name'] == suite.suite_name
            assert loaded_data['capacity_results']['mean_bpp'] == 3.5
            assert loaded_data['validation_results']['valid'] is True
            assert loaded_data['execution_time'] == 123.45
    
    def test_save_experiment_results_with_numpy_arrays(self):
        """Test saving results containing numpy arrays."""
        experiment_config = ExperimentConfig(
            device="cpu", use_cpu_optimizations=True, max_model_size=100000,
            max_dataset_size=1000, n_seeds=1, save_results=True, results_dir="test_results"
        )
        
        # Include numpy arrays in results
        suite = ExperimentSuite(
            suite_name="numpy_test",
            experiment_config=experiment_config,
            capacity_results={
                "memorization_values": np.array([100.0, 200.0, 300.0]),
                "dataset_sizes": np.array([100, 200, 300]),
                "float_value": np.float64(3.6),
                "int_value": np.int32(1000)
            },
            validation_results={"valid": True},
            execution_time=60.0,
            timestamp="2024-01-01T00:00:00",
            morris_comparison={"score": 90.0}
        )
        
        # Test saving with numpy arrays
        with tempfile.TemporaryDirectory() as temp_dir:
            suite.experiment_config.results_dir = temp_dir
            
            filepath = save_experiment_results(suite)
            
            # Verify file can be loaded as valid JSON
            with open(filepath, 'r') as f:
                loaded_data = json.load(f)
            
            # Verify numpy arrays were converted to lists
            assert isinstance(loaded_data['capacity_results']['memorization_values'], list)
            assert isinstance(loaded_data['capacity_results']['dataset_sizes'], list)
            assert isinstance(loaded_data['capacity_results']['float_value'], float)
            assert isinstance(loaded_data['capacity_results']['int_value'], int)


class TestHighLevelInterfaces:
    """Test high-level interface functions."""
    
    @patch('experiment_runner.detect_device')
    @patch('experiment_runner.run_capacity_suite')
    def test_run_morris_reproduction_suite(self, mock_run_suite, mock_detect_device):
        """Test main reproduction suite interface."""
        # Mock device detection
        mock_detect_device.return_value = ("cpu", True)
        
        # Mock suite execution
        mock_suite = Mock(spec=ExperimentSuite)
        mock_run_suite.return_value = mock_suite
        
        # Test quick mode
        result = run_morris_reproduction_suite(quick_mode=True, save_results=False)
        
        assert result == mock_suite
        mock_detect_device.assert_called_once()
        mock_run_suite.assert_called_once()
        
        # Verify config was created with quick mode settings
        args, kwargs = mock_run_suite.call_args
        config = args[0]
        assert config.use_cpu_optimizations is True
        assert config.save_results is False
    
    @patch('experiment_runner.detect_device')
    @patch('experiment_runner.run_single_model_demo')
    @patch('experiment_runner.create_gpt_model')
    @patch('experiment_runner.count_parameters')
    def test_quick_demo(self, mock_count_params, mock_create_model, mock_run_demo, mock_detect_device):
        """Test quick demo interface."""
        # Mock device detection
        mock_detect_device.return_value = ("cpu", True)
        
        # Mock model creation
        mock_create_model.return_value = Mock()
        mock_count_params.return_value = 50000
        
        # Mock demo execution
        mock_results = {"bits_per_parameter": 3.4, "param_count": 50000}
        mock_run_demo.return_value = mock_results
        
        # Test different model sizes
        for size in ["tiny", "small", "medium", "large"]:
            result = quick_demo(size)
            assert result == mock_results
        
        # Test invalid size (should default to small)
        result = quick_demo("invalid_size")
        assert result == mock_results


class TestPrintingAndSummary:
    """Test printing and summary functions."""
    
    def test_print_experiment_summary(self, capsys):
        """Test experiment summary printing."""
        experiment_config = ExperimentConfig(
            device="cuda", use_cpu_optimizations=False, max_model_size=1000000,
            max_dataset_size=10000, n_seeds=3, save_results=True, results_dir="test"
        )
        
        suite = ExperimentSuite(
            suite_name="test_print",
            experiment_config=experiment_config,
            capacity_results={
                'summary_statistics': {
                    'mean_bits_per_parameter': 3.65,
                    'std_bits_per_parameter': 0.15,
                    'n_models': 4
                },
                'scaling_law': {'r_squared': 0.92}
            },
            validation_results={
                'bits_per_param_in_range': True,
                'scaling_law_significant': True,
                'experiment_valid': True
            },
            execution_time=450.0,
            timestamp="2024-01-01T12:00:00",
            morris_comparison={
                'morris_target_bpp': 3.6,
                'observed_bpp': 3.65,
                'bpp_relative_error': 0.014,
                'morris_reproduction_score': 88.5
            }
        )
        
        # Call print function
        print_experiment_summary(suite)
        
        # Capture output
        captured = capsys.readouterr()
        
        # Verify key information is present
        assert "EXPERIMENT SUITE SUMMARY" in captured.out
        assert "Device: cuda" in captured.out
        assert "Mean Bits/Parameter: 3.65" in captured.out
        assert "Scaling Law R²: 0.920" in captured.out
        assert "Reproduction Score: 88.5/100" in captured.out
        assert "✓ PASSED" in captured.out or "PASSED" in captured.out


class TestIntegration:
    """Integration tests with minimal mocking."""
    
    def test_configuration_pipeline(self):
        """Test complete configuration creation pipeline."""
        # Test CPU path
        device, use_cpu_opt = "cpu", True
        config = create_device_appropriate_config(device, use_cpu_opt)
        
        model_configs = create_scaled_model_configs(config)
        training_config = create_scaled_training_config(config)
        
        # Verify all configs are valid
        assert len(model_configs) > 0
        for model_config in model_configs:
            assert isinstance(model_config, ModelConfig)
            assert model_config.vocab_size > 0
            assert model_config.max_seq_length > 0
            assert model_config.n_layers > 0
            assert model_config.d_model > 0
            assert model_config.n_heads > 0
            assert model_config.d_model % model_config.n_heads == 0
        
        assert isinstance(training_config, TrainingConfig)
        assert training_config.batch_size > 0
        assert training_config.learning_rate > 0
        assert training_config.max_steps > 0
        assert training_config.warmup_steps > 0
    
    def test_morris_comparison_realistic_data(self):
        """Test Morris comparison with realistic data."""
        # Simulate good results
        good_results = {
            'summary_statistics': {
                'mean_bits_per_parameter': 3.55,
                'std_bits_per_parameter': 0.25,
                'n_models': 5
            },
            'scaling_law': {'r_squared': 0.89}
        }
        
        config = ExperimentConfig(
            device="cuda", use_cpu_optimizations=False, max_model_size=2000000,
            max_dataset_size=50000, n_seeds=3, save_results=True, results_dir="test"
        )
        
        comparison = compare_to_morris_findings(good_results, config)
        reproduction_score = comparison['morris_reproduction_score']
        
        # Should be a good reproduction
        assert reproduction_score > 70
        assert comparison['scaling_meets_threshold'] is True
        assert comparison['bpp_relative_error'] < 0.1


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

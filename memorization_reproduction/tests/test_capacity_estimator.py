"""
File: test_capacity_estimator.py
Directory: memorization_reproduction/tests/

Comprehensive tests for capacity_estimator.py module.
Validates experimental methodology for reproducing Morris et al. capacity findings.
"""

import pytest
import torch
import numpy as np
import math
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from capacity_estimator import (
    CapacityExperimentResult,
    CapacityEstimate,
    estimate_model_capacity,
    detect_memorization_plateau,
    calculate_plateau_fit_quality,
    calculate_bits_per_parameter,
    fit_capacity_scaling_law,
    run_capacity_experiments,
    validate_capacity_experiment,
    create_morris_style_model_configs,
    analyze_memorization_vs_generalization
)
from model_trainer import ModelConfig, TrainingConfig, create_gpt_model, count_parameters


class TestCapacityDataStructures:
    """Test data structures for capacity experiments."""
    
    def test_capacity_experiment_result_creation(self):
        """Test CapacityExperimentResult dataclass."""
        model_config = ModelConfig(
            n_layers=1, d_model=32, n_heads=2, vocab_size=100, max_seq_length=64
        )
        training_config = TrainingConfig(
            batch_size=4, learning_rate=1e-3, max_steps=10, warmup_steps=2
        )
        
        result = CapacityExperimentResult(
            model_params=1000,
            dataset_size=500,
            total_memorization=1500.0,
            memorization_per_param=1.5,
            training_loss=2.3,
            experiment_id="test_exp",
            model_config=model_config,
            training_config=training_config
        )
        
        assert result.model_params == 1000
        assert result.dataset_size == 500
        assert result.total_memorization == 1500.0
        assert result.memorization_per_param == 1.5
        assert result.training_loss == 2.3
        assert result.experiment_id == "test_exp"
    
    def test_capacity_estimate_creation(self):
        """Test CapacityEstimate dataclass."""
        estimate = CapacityEstimate(
            estimated_capacity_bits=3600.0,
            bits_per_parameter=3.6,
            plateau_dataset_size=1000,
            memorization_values=[100, 200, 300, 300, 300],
            dataset_sizes=[100, 200, 500, 1000, 2000],
            r_squared=0.95,
            plateau_confidence=0.8
        )
        
        assert estimate.estimated_capacity_bits == 3600.0
        assert estimate.bits_per_parameter == 3.6
        assert estimate.plateau_dataset_size == 1000
        assert len(estimate.memorization_values) == 5
        assert estimate.r_squared == 0.95
        assert estimate.plateau_confidence == 0.8


class TestPlateauDetection:
    """Test memorization plateau detection algorithms."""
    
    def test_detect_plateau_perfect_case(self):
        """Test plateau detection with perfect plateau data."""
        # Create data that plateaus at size 500
        dataset_sizes = [100, 200, 300, 500, 1000, 2000]
        memorization_values = [50, 100, 150, 200, 200, 200]  # Plateaus at 200
        
        plateau_size, confidence = detect_memorization_plateau(
            dataset_sizes, memorization_values, tolerance=0.05
        )
        
        assert plateau_size >= 500  # Should detect plateau at or after size 500
        assert confidence > 0.5  # Should have reasonable confidence
    
    def test_detect_plateau_gradual_case(self):
        """Test plateau detection with gradual plateau."""
        dataset_sizes = [100, 200, 500, 1000, 2000, 5000]
        memorization_values = [100, 180, 220, 235, 240, 242]  # Gradual plateau
        
        plateau_size, confidence = detect_memorization_plateau(
            dataset_sizes, memorization_values, tolerance=0.1
        )
        
        assert plateau_size > 0
        assert 0 <= confidence <= 1
    
    def test_detect_plateau_no_plateau(self):
        """Test plateau detection with continuously increasing data."""
        dataset_sizes = [100, 200, 500, 1000, 2000]
        memorization_values = [50, 100, 250, 500, 1000]  # No plateau
        
        plateau_size, confidence = detect_memorization_plateau(
            dataset_sizes, memorization_values, tolerance=0.05
        )
        
        # Should still return a reasonable result (fallback method)
        assert plateau_size in dataset_sizes
        assert 0 <= confidence <= 1
    
    def test_detect_plateau_edge_cases(self):
        """Test plateau detection with edge cases."""
        # Empty data
        plateau_size, confidence = detect_memorization_plateau([], [], tolerance=0.05)
        assert plateau_size == 0
        assert confidence == 0.0
        
        # Single point
        plateau_size, confidence = detect_memorization_plateau([100], [50], tolerance=0.05)
        assert plateau_size == 100
        assert confidence == 0.0
        
        # Two points
        plateau_size, confidence = detect_memorization_plateau([100, 200], [50, 100], tolerance=0.05)
        assert plateau_size == 200
        assert confidence == 0.0
        
        # All zeros
        plateau_size, confidence = detect_memorization_plateau([100, 200, 300], [0, 0, 0], tolerance=0.05)
        assert plateau_size == 100
        assert confidence == 0.0
    
    def test_plateau_fit_quality(self):
        """Test plateau fit quality calculation."""
        dataset_sizes = [100, 200, 500, 1000, 2000]
        memorization_values = [50, 100, 200, 200, 200]  # Clear plateau at 200
        plateau_size = 500
        
        r_squared = calculate_plateau_fit_quality(dataset_sizes, memorization_values, plateau_size)
        
        assert 0 <= r_squared <= 1
        assert r_squared > 0.8  # Should be high for clear plateau
    
    def test_plateau_fit_quality_edge_cases(self):
        """Test plateau fit quality with edge cases."""
        # No plateau region
        r_squared = calculate_plateau_fit_quality([100, 200], [50, 100], plateau_size=500)
        assert r_squared == 0.0
        
        # Perfect plateau
        r_squared = calculate_plateau_fit_quality([100, 200, 300], [100, 100, 100], plateau_size=100)
        assert r_squared >= 0.9  # Should be very high


class TestBitsPerParameter:
    """Test bits-per-parameter calculations."""
    
    def test_bits_per_parameter_calculation(self):
        """Test basic bits-per-parameter calculation."""
        # Create small test model
        config = ModelConfig(
            n_layers=1, d_model=16, n_heads=2, vocab_size=50, max_seq_length=32
        )
        model = create_gpt_model(config)
        
        capacity_bits = 3600.0
        bits_per_param = calculate_bits_per_parameter(model, capacity_bits)
        
        expected_bpp = capacity_bits / count_parameters(model)
        assert abs(bits_per_param - expected_bpp) < 1e-6
    
    def test_bits_per_parameter_zero_params(self):
        """Test bits-per-parameter with zero parameters (edge case)."""
        # Mock model with zero parameters
        mock_model = Mock()
        with patch('capacity_estimator.count_parameters', return_value=0):
            bits_per_param = calculate_bits_per_parameter(mock_model, 1000.0)
            assert bits_per_param == 0.0
    
    def test_bits_per_parameter_zero_capacity(self):
        """Test bits-per-parameter with zero capacity."""
        config = ModelConfig(
            n_layers=1, d_model=16, n_heads=2, vocab_size=50, max_seq_length=32
        )
        model = create_gpt_model(config)
        
        bits_per_param = calculate_bits_per_parameter(model, 0.0)
        assert bits_per_param == 0.0


class TestScalingLawFitting:
    """Test scaling law fitting functionality."""
    
    def test_scaling_law_perfect_linear(self):
        """Test scaling law fitting with perfect linear data."""
        model_sizes = [1000, 2000, 3000, 4000, 5000]
        capacities = [3600, 7200, 10800, 14400, 18000]  # Perfect 3.6 bits/param
        
        slope, intercept, r_squared = fit_capacity_scaling_law(model_sizes, capacities)
        
        assert abs(slope - 3.6) < 0.1  # Should be close to 3.6
        assert abs(intercept) < 100  # Should be close to 0
        assert r_squared > 0.99  # Should be very high for perfect linear
    
    def test_scaling_law_noisy_data(self):
        """Test scaling law fitting with noisy data."""
        model_sizes = [1000, 2000, 3000, 4000]
        capacities = [3500, 7300, 10600, 14200]  # ~3.6 bits/param with noise
        
        slope, intercept, r_squared = fit_capacity_scaling_law(model_sizes, capacities)
        
        assert 3.0 < slope < 4.0  # Should be in reasonable range
        assert r_squared > 0.8  # Should still be reasonably high
    
    def test_scaling_law_edge_cases(self):
        """Test scaling law fitting with edge cases."""
        # Empty data
        slope, intercept, r_squared = fit_capacity_scaling_law([], [])
        assert slope == 0.0
        assert intercept == 0.0
        assert r_squared == 0.0
        
        # Single point
        slope, intercept, r_squared = fit_capacity_scaling_law([1000], [3600])
        assert slope == 0.0
        assert intercept == 0.0
        assert r_squared == 0.0
        
        # Two points (minimum for regression)
        slope, intercept, r_squared = fit_capacity_scaling_law([1000, 2000], [3600, 7200])
        assert abs(slope - 3.6) < 0.1
        assert r_squared >= 0.99  # Perfect fit with 2 points


class TestModelConfigGeneration:
    """Test Morris-style model configuration generation."""
    
    def test_create_morris_configs_default(self):
        """Test default Morris-style config generation."""
        configs = create_morris_style_model_configs()
        
        assert len(configs) > 0
        
        for config in configs:
            assert isinstance(config, ModelConfig)
            assert config.vocab_size == 2048  # Morris et al. standard
            assert config.max_seq_length == 64
            assert config.n_heads > 0
            assert config.d_model > 0
            assert config.n_layers > 0
            assert config.d_model % config.n_heads == 0  # Must be divisible
    
    def test_create_morris_configs_custom_targets(self):
        """Test Morris-style configs with custom target parameter counts."""
        target_params = [50000, 200000, 1000000]
        configs = create_morris_style_model_configs(target_params)
        
        assert len(configs) == len(target_params)
        
        # Check that actual parameter counts are reasonable
        for i, config in enumerate(configs):
            model = create_gpt_model(config)
            actual_params = count_parameters(model)
            target = target_params[i]
            
            # Should be within reasonable range of target (more lenient for small models)
            relative_error = abs(actual_params - target) / target
            assert relative_error < 0.7  # Within 70% of target (architecture constraints make exact matching hard)
    
    def test_config_architecture_progression(self):
        """Test that larger target params lead to larger architectures."""
        target_params = [100000, 500000, 2000000]
        configs = create_morris_style_model_configs(target_params)
        
        actual_params = []
        for config in configs:
            model = create_gpt_model(config)
            actual_params.append(count_parameters(model))
        
        # Should be roughly increasing
        for i in range(1, len(actual_params)):
            assert actual_params[i] > actual_params[i-1]


class TestCapacityEstimation:
    """Test full capacity estimation pipeline (mocked for speed)."""
    
    def setup_method(self):
        """Set up test environment with mocking."""
        self.device = "cpu"
        
        self.model_config = ModelConfig(
            n_layers=1, d_model=16, n_heads=2, vocab_size=50, max_seq_length=32
        )
        
        self.training_config = TrainingConfig(
            batch_size=2, learning_rate=1e-3, max_steps=5, warmup_steps=1
        )
    
    @patch('capacity_estimator.generate_uniform_bitstrings')
    @patch('capacity_estimator.train_model')
    @patch('capacity_estimator.calculate_total_memorization')
    def test_estimate_model_capacity_mocked(
        self, mock_calc_memorization, mock_train, mock_generate_data
    ):
        """Test capacity estimation with mocked expensive operations."""
        # Mock data generation
        mock_generate_data.return_value = [torch.tensor([1, 2, 3, 4])]
        
        # Mock training
        mock_train.return_value = {"train_loss": [2.5, 2.0, 1.5]}
        
        # Mock memorization calculation with plateau pattern
        dataset_sizes = [100, 200, 500, 1000]
        mock_calc_memorization.side_effect = [300, 550, 720, 720]  # Plateau at 720
        
        estimate = estimate_model_capacity(
            model_config=self.model_config,
            training_config=self.training_config,
            dataset_sizes=dataset_sizes,
            n_seeds=1,  # Single seed for testing
            device=self.device
        )
        
        assert isinstance(estimate, CapacityEstimate)
        assert estimate.estimated_capacity_bits > 0
        assert estimate.bits_per_parameter > 0
        assert estimate.plateau_dataset_size in dataset_sizes
        assert len(estimate.memorization_values) == len(dataset_sizes)
        assert 0 <= estimate.r_squared <= 1
        assert 0 <= estimate.plateau_confidence <= 1


class TestExperimentValidation:
    """Test experiment validation functionality."""
    
    def test_validate_capacity_experiment_good_results(self):
        """Test validation with good experimental results."""
        # Create mock results that should pass validation
        mock_results = {
            'summary_statistics': {
                'mean_bits_per_parameter': 3.7,  # Close to target 3.6
                'std_bits_per_parameter': 0.3,  # Low variance
            },
            'scaling_law': {
                'r_squared': 0.92  # Good fit
            },
            'estimated_capacities': [1000, 2000, 3000],  # All positive
            'capacity_estimates': [
                Mock(plateau_confidence=0.8),
                Mock(plateau_confidence=0.7),
                Mock(plateau_confidence=0.6)
            ]
        }
        
        validation = validate_capacity_experiment(mock_results)
        
        assert validation['bits_per_param_in_range'] is True
        assert validation['scaling_law_significant'] is True
        assert validation['positive_capacities'] is True
        assert validation['plateaus_detected'] is True
        assert validation['consistent_across_models'] is True
        assert validation['experiment_valid'] is True
    
    def test_validate_capacity_experiment_bad_results(self):
        """Test validation with poor experimental results."""
        mock_results = {
            'summary_statistics': {
                'mean_bits_per_parameter': 1.0,  # Far from target 3.6
                'std_bits_per_parameter': 2.0,  # High variance
            },
            'scaling_law': {
                'r_squared': 0.3  # Poor fit
            },
            'estimated_capacities': [1000, -500, 3000],  # Contains negative
            'capacity_estimates': [
                Mock(plateau_confidence=0.1),  # Low confidence
                Mock(plateau_confidence=0.2),
                Mock(plateau_confidence=0.1)
            ]
        }
        
        validation = validate_capacity_experiment(mock_results)
        
        assert validation['bits_per_param_in_range'] is False
        assert validation['scaling_law_significant'] is False
        assert validation['positive_capacities'] is False
        assert validation['plateaus_detected'] is False
        assert validation['consistent_across_models'] is False
        assert validation['experiment_valid'] is False
    
    def test_validate_capacity_experiment_custom_tolerance(self):
        """Test validation with custom tolerance."""
        mock_results = {
            'summary_statistics': {
                'mean_bits_per_parameter': 4.5,  # 0.9 away from 3.6
                'std_bits_per_parameter': 0.2,
            },
            'scaling_law': {
                'r_squared': 0.85
            },
            'estimated_capacities': [1000, 2000],
            'capacity_estimates': [Mock(plateau_confidence=0.5), Mock(plateau_confidence=0.6)]
        }
        
        # Should fail with default tolerance (0.5)
        validation_strict = validate_capacity_experiment(mock_results, tolerance=0.5)
        assert validation_strict['bits_per_param_in_range'] is False
        
        # Should pass with looser tolerance (1.0)
        validation_loose = validate_capacity_experiment(mock_results, tolerance=1.0)
        assert validation_loose['bits_per_param_in_range'] is True


class TestRunCapacityExperiments:
    """Test the full capacity experiment pipeline (heavily mocked)."""
    
    @patch('capacity_estimator.estimate_model_capacity')
    def test_run_capacity_experiments_basic(self, mock_estimate_capacity):
        """Test basic capacity experiments execution."""
        # Mock capacity estimation
        mock_estimate_capacity.return_value = CapacityEstimate(
            estimated_capacity_bits=3600.0,
            bits_per_parameter=3.6,
            plateau_dataset_size=1000,
            memorization_values=[100, 200, 300, 300],
            dataset_sizes=[100, 300, 500, 1000],
            r_squared=0.95,
            plateau_confidence=0.8
        )
        
        model_configs = [
            ModelConfig(n_layers=1, d_model=16, n_heads=2, vocab_size=50, max_seq_length=32),
            ModelConfig(n_layers=1, d_model=32, n_heads=4, vocab_size=50, max_seq_length=32)
        ]
        
        training_config = TrainingConfig(
            batch_size=2, learning_rate=1e-3, max_steps=5, warmup_steps=1
        )
        
        results = run_capacity_experiments(
            model_configs=model_configs,
            training_config=training_config,
            base_dataset_sizes=[100, 300, 500],
            n_seeds=1,
            device="cpu"
        )
        
        # Check result structure
        assert 'individual_results' in results
        assert 'capacity_estimates' in results
        assert 'model_sizes' in results
        assert 'estimated_capacities' in results
        assert 'scaling_law' in results
        assert 'summary_statistics' in results
        
        assert len(results['individual_results']) == len(model_configs)
        assert len(results['capacity_estimates']) == len(model_configs)
        assert len(results['model_sizes']) == len(model_configs)
        
        # Check scaling law structure
        scaling = results['scaling_law']
        assert 'bits_per_parameter' in scaling
        assert 'intercept' in scaling
        assert 'r_squared' in scaling
        
        # Check summary statistics
        summary = results['summary_statistics']
        assert 'mean_bits_per_parameter' in summary
        assert 'std_bits_per_parameter' in summary
        assert 'n_models' in summary
        assert summary['n_models'] == len(model_configs)


class TestMemorizationVsGeneralization:
    """Test memorization vs generalization analysis."""
    
    def test_analyze_memorization_vs_generalization_structure(self):
        """Test that analysis returns expected structure."""
        model_config = ModelConfig(
            n_layers=1, d_model=16, n_heads=2, vocab_size=50, max_seq_length=32
        )
        training_config = TrainingConfig(
            batch_size=2, learning_rate=1e-3, max_steps=5, warmup_steps=1
        )
        
        # Note: This is currently a placeholder implementation
        results = analyze_memorization_vs_generalization(
            model_config=model_config,
            training_config=training_config,
            text_data="sample text data",
            dataset_sizes=[100, 200, 500],
            device="cpu"
        )
        
        # Check expected structure
        assert 'dataset_sizes' in results
        assert 'memorization_values' in results
        assert 'generalization_metrics' in results
        assert 'transition_point' in results
        assert 'double_descent_detected' in results
        
        assert results['dataset_sizes'] == [100, 200, 500]


class TestIntegration:
    """Integration tests for capacity estimation pipeline."""
    
    def test_small_scale_capacity_experiment(self):
        """Test actual small-scale capacity experiment (no mocking)."""
        # Very small models and datasets for quick testing
        model_config = ModelConfig(
            n_layers=1, d_model=8, n_heads=2, vocab_size=16, max_seq_length=16  # Increased seq length
        )
        
        training_config = TrainingConfig(
            batch_size=2, learning_rate=1e-2, max_steps=3, warmup_steps=1
        )
        
        dataset_sizes = [5, 10, 15]  # Very small for speed
        
        # This will run actual training (but very quick due to small size)
        estimate = estimate_model_capacity(
            model_config=model_config,
            training_config=training_config,
            dataset_sizes=dataset_sizes,
            n_seeds=1,
            device="cpu",
            plateau_tolerance=0.2
        )
        
        # Verify structure and basic properties
        assert isinstance(estimate, CapacityEstimate)
        assert estimate.estimated_capacity_bits >= 0
        assert estimate.bits_per_parameter >= 0
        assert estimate.plateau_dataset_size in dataset_sizes
        assert len(estimate.memorization_values) == len(dataset_sizes)
        assert 0 <= estimate.r_squared <= 1
        assert 0 <= estimate.plateau_confidence <= 1
    
    def test_morris_config_to_actual_experiment(self):
        """Test that Morris-style configs work in actual experiments."""
        # Create small Morris-style config
        configs = create_morris_style_model_configs([50000])  # Small target
        
        assert len(configs) == 1
        config = configs[0]
        
        # Verify the config produces a working model
        model = create_gpt_model(config)
        param_count = count_parameters(model)
        
        assert param_count > 0
        assert param_count < 150000  # Should be reasonably small (more lenient)
        
        # Verify bits-per-parameter calculation works
        # Use actual parameter count for realistic calculation
        mock_capacity = param_count * 3.6  # Use realistic capacity based on actual params
        bpp = calculate_bits_per_parameter(model, mock_capacity)
        
        assert 3.5 < bpp < 3.7  # Should be close to 3.6
    
    def test_plateau_detection_with_real_data(self):
        """Test plateau detection with realistic memorization curves."""
        # Simulate realistic memorization curves that plateau
        dataset_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
        
        # Curve that saturates (mimicking capacity limits)
        def saturation_curve(x, capacity=1000, rate=0.001):
            return capacity * (1 - np.exp(-rate * x))
        
        memorization_values = [saturation_curve(size) for size in dataset_sizes]
        
        plateau_size, confidence = detect_memorization_plateau(
            dataset_sizes, memorization_values, tolerance=0.05
        )
        
        # Should detect plateau in the later dataset sizes
        assert plateau_size >= 1000
        assert confidence > 0.3
        
        # Test fit quality
        r_squared = calculate_plateau_fit_quality(
            dataset_sizes, memorization_values, plateau_size
        )
        assert r_squared > 0.5  # Should be reasonable for saturating curve


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

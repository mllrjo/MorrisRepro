"""
File: test_visualization.py
Directory: memorization_reproduction/tests/

Comprehensive tests for visualization.py module.
Validates plotting functions, figure generation, and publication-quality outputs.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
import matplotlib

# Use non-interactive backend for testing
matplotlib.use('Agg')

# Import the modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from visualization import (
    setup_figure_style,
    plot_memorization_vs_dataset_size,
    plot_capacity_vs_parameters,
    plot_scaling_law_validation,
    plot_training_dynamics,
    create_experimental_dashboard,
    compare_to_morris_results,
    save_all_figures,
    create_publication_figure_set,
    MORRIS_BITS_PER_PARAM,
    MORRIS_COLORS
)
from experiment_runner import ExperimentSuite, ExperimentConfig
from capacity_estimator import CapacityEstimate


class TestVisualizationSetup:
    """Test visualization setup and configuration."""
    
    def test_setup_figure_style(self):
        """Test figure style setup."""
        # Store original rcParams
        original_figsize = plt.rcParams.get('figure.figsize', (6.4, 4.8))
        original_dpi = plt.rcParams.get('figure.dpi', 100)
        
        # Apply style
        setup_figure_style()
        
        # Check that rcParams were modified - handle both tuple and list types
        figsize = plt.rcParams['figure.figsize']
        assert (tuple(figsize) == (10, 8) or list(figsize) == [10, 8] or 
                (len(figsize) == 2 and figsize[0] == 10 and figsize[1] == 8))
        
        assert plt.rcParams['figure.dpi'] == 300
        assert plt.rcParams['font.size'] == 12
        assert plt.rcParams['axes.titlesize'] == 16
        assert plt.rcParams['axes.labelsize'] == 14
        assert plt.rcParams['lines.linewidth'] == 2
        assert plt.rcParams['axes.spines.top'] is False
        assert plt.rcParams['axes.spines.right'] is False
    
    def test_morris_constants(self):
        """Test Morris et al. reference constants."""
        assert MORRIS_BITS_PER_PARAM == 3.6
        assert isinstance(MORRIS_COLORS, dict)
        assert 'target' in MORRIS_COLORS
        assert 'observed' in MORRIS_COLORS
        assert 'good_fit' in MORRIS_COLORS
        assert 'poor_fit' in MORRIS_COLORS


class TestMockDataCreation:
    """Helper class for creating mock experimental data."""
    
    @staticmethod
    def create_mock_capacity_results():
        """Create mock capacity results for testing."""
        return {
            'individual_results': [
                {
                    'model_params': 100000,
                    'capacity_estimate': CapacityEstimate(
                        estimated_capacity_bits=360000.0,
                        bits_per_parameter=3.6,
                        plateau_dataset_size=1000,
                        memorization_values=[50000, 150000, 250000, 350000, 360000],
                        dataset_sizes=[100, 300, 500, 1000, 2000],
                        r_squared=0.95,
                        plateau_confidence=0.8
                    )
                },
                {
                    'model_params': 200000,
                    'capacity_estimate': CapacityEstimate(
                        estimated_capacity_bits=720000.0,
                        bits_per_parameter=3.6,
                        plateau_dataset_size=2000,
                        memorization_values=[100000, 300000, 500000, 700000, 720000],
                        dataset_sizes=[200, 600, 1000, 2000, 4000],
                        r_squared=0.92,
                        plateau_confidence=0.75
                    )
                }
            ],
            'model_sizes': [100000, 200000],
            'estimated_capacities': [360000.0, 720000.0],
            'scaling_law': {
                'bits_per_parameter': 3.6,
                'intercept': 0.0,
                'r_squared': 0.93
            },
            'summary_statistics': {
                'mean_bits_per_parameter': 3.6,
                'std_bits_per_parameter': 0.1,
                'n_models': 2
            }
        }
    
    @staticmethod
    def create_mock_experiment_suite():
        """Create mock experiment suite for testing."""
        experiment_config = ExperimentConfig(
            device="cpu",
            use_cpu_optimizations=True,
            max_model_size=200000,
            max_dataset_size=5000,
            n_seeds=2,
            save_results=True,
            results_dir="test_results"
        )
        
        capacity_results = TestMockDataCreation.create_mock_capacity_results()
        
        return ExperimentSuite(
            suite_name="test_suite",
            experiment_config=experiment_config,
            capacity_results=capacity_results,
            validation_results={
                'bits_per_param_in_range': True,
                'scaling_law_significant': True,
                'positive_capacities': True,
                'plateaus_detected': True,
                'consistent_across_models': True,
                'experiment_valid': True
            },
            execution_time=120.5,
            timestamp="2024-01-01T00:00:00",
            morris_comparison={
                'morris_target_bpp': 3.6,
                'observed_bpp': 3.6,
                'bpp_deviation': 0.0,
                'bpp_relative_error': 0.0,
                'scaling_meets_threshold': True,
                'consistent_across_models': True,
                'experiment_scale': 'cpu_optimized',
                'n_models_tested': 2,
                'morris_reproduction_score': 95.0
            }
        )


class TestBasicPlotCreation:
    """Test basic plot creation functions."""
    
    def setup_method(self):
        """Set up mock data for testing."""
        self.capacity_results = TestMockDataCreation.create_mock_capacity_results()
    
    def test_plot_memorization_vs_dataset_size(self):
        """Test memorization vs dataset size plot creation."""
        fig = plot_memorization_vs_dataset_size(self.capacity_results)
        
        assert isinstance(fig, plt.Figure)
        
        # Check that the figure has the expected structure
        axes = fig.get_axes()
        assert len(axes) == 1
        
        ax = axes[0]
        assert ax.get_xlabel() == 'Training Set Size (number of datapoints)'
        assert ax.get_ylabel() == 'Unintended Memorization (bits)'
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'log'
        
        # Check that lines were plotted
        lines = ax.get_lines()
        assert len(lines) > 0
        
        plt.close(fig)
    
    def test_plot_capacity_vs_parameters(self):
        """Test capacity vs parameters plot creation."""
        fig = plot_capacity_vs_parameters(self.capacity_results)
        
        assert isinstance(fig, plt.Figure)
        
        axes = fig.get_axes()
        assert len(axes) == 1
        
        ax = axes[0]
        assert ax.get_xlabel() == 'Model Size (parameters)'
        assert ax.get_ylabel() == 'Total Memorization (bits)'
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'log'
        
        # Check for scatter plot and fitted line
        collections = ax.collections  # Scatter plots
        lines = ax.get_lines()       # Line plots
        
        assert len(collections) > 0 or len(lines) > 0
        
        plt.close(fig)
    
    def test_plot_scaling_law_validation(self):
        """Test scaling law validation plot creation."""
        fig = plot_scaling_law_validation(self.capacity_results)
        
        assert isinstance(fig, plt.Figure)
        
        # Should have multiple subplots
        axes = fig.get_axes()
        assert len(axes) >= 3  # Main plot, residuals, statistics panel
        
        plt.close(fig)
    
    def test_plot_training_dynamics(self):
        """Test training dynamics plot creation."""
        training_metrics = {
            'train_loss': [2.5, 2.0, 1.8, 1.5, 1.3],
            'step': [0, 100, 200, 300, 400]
        }
        memorization_values = [100, 200, 350, 360, 360]
        dataset_sizes = [100, 300, 500, 1000, 2000]
        
        fig = plot_training_dynamics(training_metrics, memorization_values, dataset_sizes)
        
        assert isinstance(fig, plt.Figure)
        
        axes = fig.get_axes()
        assert len(axes) == 2  # Loss curve and memorization curve
        
        plt.close(fig)
    
    def test_plot_training_dynamics_empty_metrics(self):
        """Test training dynamics with empty metrics."""
        training_metrics = {}
        memorization_values = [100, 200, 300]
        dataset_sizes = [100, 300, 500]
        
        fig = plot_training_dynamics(training_metrics, memorization_values, dataset_sizes)
        
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)


class TestDashboardCreation:
    """Test experimental dashboard creation."""
    
    def test_create_experimental_dashboard(self):
        """Test experimental dashboard creation."""
        suite = TestMockDataCreation.create_mock_experiment_suite()
        
        fig = create_experimental_dashboard(suite)
        
        assert isinstance(fig, plt.Figure)
        
        # Should have multiple subplots for comprehensive dashboard
        axes = fig.get_axes()
        assert len(axes) >= 5  # Multiple panels for different metrics
        
        plt.close(fig)
    
    def test_compare_to_morris_results(self):
        """Test Morris et al. comparison plot."""
        our_results = TestMockDataCreation.create_mock_capacity_results()
        
        fig = compare_to_morris_results(our_results)
        
        assert isinstance(fig, plt.Figure)
        
        axes = fig.get_axes()
        assert len(axes) == 4  # 2x2 subplot layout
        
        plt.close(fig)
    
    def test_compare_to_morris_results_custom_reference(self):
        """Test Morris comparison with custom reference values."""
        our_results = TestMockDataCreation.create_mock_capacity_results()
        custom_reference = {
            'bits_per_parameter': 3.5,
            'scaling_r_squared': 0.90,
            'model_sizes': [50000, 100000, 200000],
            'capacities': [175000, 350000, 700000]
        }
        
        fig = compare_to_morris_results(our_results, morris_reference=custom_reference)
        
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)


class TestFileSaving:
    """Test figure saving functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.capacity_results = TestMockDataCreation.create_mock_capacity_results()
        self.suite = TestMockDataCreation.create_mock_experiment_suite()
    
    def test_plot_with_save_path(self):
        """Test saving individual plots."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_plot.png")
            
            fig = plot_memorization_vs_dataset_size(
                self.capacity_results, save_path=save_path
            )
            
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0  # File is not empty
            
            plt.close(fig)
    
    def test_save_all_figures(self):
        """Test saving complete figure set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_paths = save_all_figures(
                self.suite, output_dir=temp_dir, prefix="test"
            )
            
            assert len(saved_paths) > 0
            
            # Check that all files exist and are non-empty
            for path in saved_paths:
                assert os.path.exists(path)
                assert os.path.getsize(path) > 0
                assert path.endswith('.png')
    
    def test_create_publication_figure_set(self):
        """Test publication figure generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            figures = create_publication_figure_set(self.suite, output_dir=temp_dir)
            
            assert len(figures) > 0
            
            # Check that all files exist and are PDFs
            for name, path in figures.items():
                assert os.path.exists(path)
                assert os.path.getsize(path) > 0
                assert path.endswith('.pdf')
                assert isinstance(name, str)


class TestDataHandling:
    """Test data handling and edge cases."""
    
    def test_empty_capacity_results(self):
        """Test handling of empty capacity results."""
        empty_results = {
            'individual_results': [],
            'model_sizes': [],
            'estimated_capacities': [],
            'scaling_law': {
                'bits_per_parameter': 0.0,
                'intercept': 0.0,
                'r_squared': 0.0
            },
            'summary_statistics': {
                'mean_bits_per_parameter': 0.0,
                'std_bits_per_parameter': 0.0,
                'n_models': 0
            }
        }
        
        # Should not crash with empty data
        fig = plot_memorization_vs_dataset_size(empty_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        fig = plot_capacity_vs_parameters(empty_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_single_model_results(self):
        """Test handling of single model results."""
        single_model_results = {
            'individual_results': [
                {
                    'model_params': 100000,
                    'capacity_estimate': CapacityEstimate(
                        estimated_capacity_bits=360000.0,
                        bits_per_parameter=3.6,
                        plateau_dataset_size=1000,
                        memorization_values=[100000, 300000, 360000],
                        dataset_sizes=[100, 500, 1000],
                        r_squared=0.95,
                        plateau_confidence=0.8
                    )
                }
            ],
            'model_sizes': [100000],
            'estimated_capacities': [360000.0],
            'scaling_law': {
                'bits_per_parameter': 3.6,
                'intercept': 0.0,
                'r_squared': 1.0  # Perfect fit with one point
            },
            'summary_statistics': {
                'mean_bits_per_parameter': 3.6,
                'std_bits_per_parameter': 0.0,
                'n_models': 1
            }
        }
        
        fig = plot_memorization_vs_dataset_size(single_model_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        fig = plot_capacity_vs_parameters(single_model_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_invalid_data_types(self):
        """Test handling of invalid data types."""
        # Test with None values
        invalid_results = {
            'individual_results': None,
            'model_sizes': None,
            'estimated_capacities': None,
            'scaling_law': None,
            'summary_statistics': None
        }
        
        # Should handle gracefully or raise appropriate errors
        with pytest.raises((AttributeError, TypeError, KeyError)):
            plot_memorization_vs_dataset_size(invalid_results)


class TestPlotCustomization:
    """Test plot customization and styling."""
    
    def setup_method(self):
        """Set up test data."""
        self.capacity_results = TestMockDataCreation.create_mock_capacity_results()
    
    def test_custom_titles(self):
        """Test custom plot titles."""
        custom_title = "Custom Test Title"
        
        fig = plot_memorization_vs_dataset_size(
            self.capacity_results, title=custom_title
        )
        
        axes = fig.get_axes()
        assert len(axes) > 0
        assert axes[0].get_title() == custom_title
        
        plt.close(fig)
    
    def test_color_scheme_consistency(self):
        """Test that plots use consistent color scheme."""
        fig = plot_capacity_vs_parameters(self.capacity_results)
        
        # This is a basic test - in practice you'd check specific color usage
        # Here we just verify the plot was created successfully
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_axis_scaling(self):
        """Test that axes use appropriate scaling."""
        fig = plot_memorization_vs_dataset_size(self.capacity_results)
        
        axes = fig.get_axes()
        ax = axes[0]
        
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'log'
        
        plt.close(fig)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_required_keys(self):
        """Test handling of missing required keys in data."""
        incomplete_results = {
            'model_sizes': [100000, 200000],
            # Missing other required keys
        }
        
        with pytest.raises(KeyError):
            plot_capacity_vs_parameters(incomplete_results)
    
    def test_mismatched_array_lengths(self):
        """Test handling of mismatched array lengths."""
        mismatched_results = {
            'individual_results': [],
            'model_sizes': [100000, 200000],
            'estimated_capacities': [360000.0],  # Different length
            'scaling_law': {
                'bits_per_parameter': 3.6,
                'intercept': 0.0,
                'r_squared': 0.95
            },
            'summary_statistics': {
                'mean_bits_per_parameter': 3.6,
                'std_bits_per_parameter': 0.1,
                'n_models': 2
            }
        }
        
        # Should handle gracefully or raise appropriate error
        try:
            fig = plot_capacity_vs_parameters(mismatched_results)
            plt.close(fig)
        except (ValueError, IndexError):
            pass  # Expected behavior for mismatched data
    
    def test_invalid_save_path(self):
        """Test handling of invalid save paths."""
        capacity_results = TestMockDataCreation.create_mock_capacity_results()
        
        # Test with invalid directory
        invalid_path = "/nonexistent/directory/test.png"
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((OSError, IOError)):
            plot_memorization_vs_dataset_size(capacity_results, save_path=invalid_path)


class TestNumericalStability:
    """Test numerical stability and extreme values."""
    
    def test_very_large_numbers(self):
        """Test handling of very large numerical values."""
        large_results = {
            'individual_results': [
                {
                    'model_params': 1e9,  # 1 billion parameters
                    'capacity_estimate': CapacityEstimate(
                        estimated_capacity_bits=3.6e9,
                        bits_per_parameter=3.6,
                        plateau_dataset_size=1000000,
                        memorization_values=[1e8, 2e8, 3e8, 3.6e9],
                        dataset_sizes=[100000, 500000, 1000000, 2000000],
                        r_squared=0.95,
                        plateau_confidence=0.8
                    )
                }
            ],
            'model_sizes': [1e9],
            'estimated_capacities': [3.6e9],
            'scaling_law': {
                'bits_per_parameter': 3.6,
                'intercept': 0.0,
                'r_squared': 0.95
            },
            'summary_statistics': {
                'mean_bits_per_parameter': 3.6,
                'std_bits_per_parameter': 0.1,
                'n_models': 1
            }
        }
        
        fig = plot_memorization_vs_dataset_size(large_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        fig = plot_capacity_vs_parameters(large_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_very_small_numbers(self):
        """Test handling of very small numerical values."""
        small_results = {
            'individual_results': [
                {
                    'model_params': 1000,  # Small model
                    'capacity_estimate': CapacityEstimate(
                        estimated_capacity_bits=3600.0,
                        bits_per_parameter=3.6,
                        plateau_dataset_size=10,
                        memorization_values=[100, 1000, 2000, 3600],
                        dataset_sizes=[1, 5, 10, 20],
                        r_squared=0.95,
                        plateau_confidence=0.8
                    )
                }
            ],
            'model_sizes': [1000],
            'estimated_capacities': [3600.0],
            'scaling_law': {
                'bits_per_parameter': 3.6,
                'intercept': 0.0,
                'r_squared': 0.95
            },
            'summary_statistics': {
                'mean_bits_per_parameter': 3.6,
                'std_bits_per_parameter': 0.1,
                'n_models': 1
            }
        }
        
        fig = plot_memorization_vs_dataset_size(small_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestIntegration:
    """Integration tests for complete visualization workflow."""
    
    def test_full_visualization_pipeline(self):
        """Test complete visualization pipeline."""
        suite = TestMockDataCreation.create_mock_experiment_suite()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test individual plots
            fig1 = plot_memorization_vs_dataset_size(suite.capacity_results)
            fig2 = plot_capacity_vs_parameters(suite.capacity_results)
            fig3 = create_experimental_dashboard(suite)
            
            assert all(isinstance(fig, plt.Figure) for fig in [fig1, fig2, fig3])
            
            # Test batch saving
            saved_paths = save_all_figures(suite, output_dir=temp_dir)
            assert len(saved_paths) > 0
            
            # Test publication figures
            pub_figures = create_publication_figure_set(suite, output_dir=temp_dir)
            assert len(pub_figures) > 0
            
            # Clean up
            for fig in [fig1, fig2, fig3]:
                plt.close(fig)
    
    def test_visualization_with_realistic_data(self):
        """Test visualization with realistic experimental data ranges."""
        # Create more realistic data that mimics actual experimental results
        realistic_suite = TestMockDataCreation.create_mock_experiment_suite()
        
        # Modify to have more realistic variations
        realistic_suite.capacity_results['summary_statistics']['mean_bits_per_parameter'] = 3.45
        realistic_suite.capacity_results['summary_statistics']['std_bits_per_parameter'] = 0.25
        realistic_suite.morris_comparison['observed_bpp'] = 3.45
        realistic_suite.morris_comparison['morris_reproduction_score'] = 87.5
        
        # Should handle realistic variations gracefully
        fig = create_experimental_dashboard(realistic_suite)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

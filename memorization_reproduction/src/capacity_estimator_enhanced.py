"""
File: capacity_estimator_enhanced.py
Directory: memorization_reproduction/src/

ENHANCED Capacity estimation module integrated with enhanced_model_trainer.
Fixes MAX_STEPS termination issues and provides reliable capacity measurement.
"""

from typing import List, Dict, Tuple, Optional, Callable, Any
import torch
import numpy as np
import math
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
import time

# Import existing modules
from model_trainer import (
    ModelConfig, TrainingConfig, create_gpt_model, count_parameters
)
from memorization_calculator import (
    calculate_total_memorization, create_synthetic_reference_model,
    validate_memorization_calculation
)
from data_generator import generate_uniform_bitstrings, create_dataset_size_series

# Import enhanced trainer
from enhanced_model_trainer import (
    enhanced_train_model_wrapper, 
    validate_memorization_achievement,
    create_enhanced_config_from_original
)


@dataclass
class EnhancedCapacityExperimentResult:
    """Enhanced results from capacity estimation experiment."""
    model_params: int
    dataset_size: int
    total_memorization: float
    memorization_per_param: float
    training_loss: float
    experiment_id: str
    model_config: ModelConfig
    training_config: TrainingConfig
    # Enhanced metrics
    convergence_achieved: bool
    convergence_reason: str
    final_status: str
    memorization_rate: float
    training_time: float


@dataclass
class EnhancedCapacityEstimate:
    """Enhanced model capacity estimation results."""
    estimated_capacity_bits: float
    bits_per_parameter: float
    plateau_dataset_size: int
    memorization_values: List[float]
    dataset_sizes: List[int]
    r_squared: float
    plateau_confidence: float
    # Enhanced metrics
    convergence_success_rate: float
    avg_training_time: float
    max_steps_elimination: bool  # True if no MAX_STEPS terminations
    morris_target_progress: float  # Progress toward 3.6 bits/param


def estimate_model_capacity_enhanced(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    dataset_sizes: List[int],
    n_seeds: int = 3,
    device: str = "cuda",
    plateau_tolerance: float = 0.05,
    use_enhanced_training: bool = True,
    enhanced_params: Optional[Dict] = None
) -> EnhancedCapacityEstimate:
    """
    ENHANCED capacity estimation using enhanced_model_trainer.
    
    Eliminates MAX_STEPS termination issues and provides reliable memorization measurement.
    
    Args:
        model_config: Model architecture configuration
        training_config: Training parameters
        dataset_sizes: List of dataset sizes to test (should span capacity)
        n_seeds: Number of random seeds for statistical robustness
        device: Device for computation
        plateau_tolerance: Relative tolerance for plateau detection
        use_enhanced_training: Whether to use enhanced trainer (recommended: True)
        enhanced_params: Additional parameters for enhanced training
        
    Returns:
        EnhancedCapacityEstimate with improved reliability and Morris progress tracking
    """
    
    # Default enhanced training parameters
    default_enhanced_params = {
        'memorization_threshold': 0.3,  # Reasonable threshold for capacity measurement
        'memorization_check_interval': 200,
        'min_memorization_steps': 500,
        'max_plateau_patience': 3000,
        'adaptive_lr': False  # Keep stable for capacity measurement
    }
    
    if enhanced_params:
        default_enhanced_params.update(enhanced_params)
    
    all_memorization_values = []
    all_results = []
    convergence_successes = 0
    total_training_time = 0
    max_steps_terminations = 0
    
    # Display progressive testing
    model_params = count_parameters(create_gpt_model(model_config))
    print(f"🔬 ENHANCED Capacity Estimation")
    print(f"Testing {len(dataset_sizes)} dataset sizes for model with {model_params:,} parameters:")
    print(f"Enhanced training: {'ENABLED' if use_enhanced_training else 'DISABLED'}")
    
    # Run experiments across seeds for statistical robustness
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        memorization_values = []
        
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")
        
        for i, dataset_size in enumerate(dataset_sizes):
            print(f"  Dataset {dataset_size:,} samples (#{i+1}/{len(dataset_sizes)})... ", end="", flush=True)
            
            # Generate uniform random data (no generalization possible)
            seq_length = min(64, model_config.max_seq_length)
            
            data = generate_uniform_bitstrings(
                n_samples=dataset_size,
                seq_length=seq_length,
                vocab_size=model_config.vocab_size,
                seed=seed + dataset_size
            )
            
            # Create and train model with ENHANCED training
            model = create_gpt_model(model_config)
            
            start_time = time.time()
            
            if use_enhanced_training:
                # Use enhanced trainer with capacity-optimized settings
                training_metrics = enhanced_train_model_wrapper(
                    model=model,
                    train_data=data,
                    original_config=training_config,
                    device=device,
                    enable_enhanced_training=True,
                    **default_enhanced_params
                )
                
                # Track enhanced metrics
                convergence_achieved = training_metrics.get('convergence_achieved', False)
                final_status = training_metrics.get('final_status', 'unknown')
                convergence_reason = training_metrics.get('convergence_reason', 'unknown')
                memorization_rate = training_metrics.get('final_memorization_rate', 0.0)
                
                if convergence_achieved:
                    convergence_successes += 1
                if final_status == "MAX_STEPS":
                    max_steps_terminations += 1
                    
            else:
                # Fallback to original training
                try:
                    from model_trainer import train_model
                    training_metrics = train_model(model, data, training_config, device)
                    # Add missing keys for compatibility
                    training_metrics.update({
                        'convergence_achieved': False,
                        'final_status': 'original_training',
                        'convergence_reason': 'original_method',
                        'final_memorization_rate': 0.0
                    })
                except ImportError:
                    # Use enhanced trainer as fallback
                    training_metrics = enhanced_train_model_wrapper(
                        model, data, training_config, device, False
                    )
            
            training_time = time.time() - start_time
            total_training_time += training_time
            
            # Create reference model (uniform for synthetic data)
            reference_model = create_synthetic_reference_model(
                model, model_config.vocab_size
            )
            
            # Calculate total memorization
            total_memorization = calculate_total_memorization(
                model, reference_model, data, device
            )
            
            memorization_values.append(total_memorization)
            
            # Display enhanced results
            bits_per_param = total_memorization / model_params
            status_indicator = "✓" if training_metrics.get('convergence_achieved', False) else "→"
            
            print(f"{status_indicator} {total_memorization:,.0f} bits ({bits_per_param:.3f} bits/param) "
                  f"[{training_metrics.get('final_status', 'unknown')}]")
            
            # Store detailed enhanced results
            result = EnhancedCapacityExperimentResult(
                model_params=count_parameters(model),
                dataset_size=dataset_size,
                total_memorization=total_memorization,
                memorization_per_param=total_memorization / count_parameters(model),
                training_loss=training_metrics.get('final_loss', float('nan')),
                experiment_id=f"enhanced_seed_{seed}_size_{dataset_size}",
                model_config=model_config,
                training_config=training_config,
                convergence_achieved=training_metrics.get('convergence_achieved', False),
                convergence_reason=training_metrics.get('convergence_reason', 'unknown'),
                final_status=training_metrics.get('final_status', 'unknown'),
                memorization_rate=training_metrics.get('final_memorization_rate', 0.0),
                training_time=training_time
            )
            all_results.append(result)
        
        all_memorization_values.append(memorization_values)
    
    # Average across seeds
    avg_memorization_values = np.mean(all_memorization_values, axis=0)
    
    # Enhanced plateau detection
    plateau_size, plateau_confidence = detect_memorization_plateau_enhanced(
        dataset_sizes, avg_memorization_values, plateau_tolerance, all_results
    )
    
    # Calculate enhanced metrics
    total_experiments = len(dataset_sizes) * n_seeds
    convergence_success_rate = convergence_successes / total_experiments
    avg_training_time = total_training_time / total_experiments
    max_steps_elimination = max_steps_terminations == 0
    
    # Estimate capacity as plateau value
    estimated_capacity = np.max(avg_memorization_values)
    
    # Calculate bits per parameter and Morris progress
    bits_per_parameter = estimated_capacity / model_params
    morris_target = 3.6
    morris_progress = min(bits_per_parameter / morris_target, 1.0)
    
    # Calculate R² for plateau fit quality
    r_squared = calculate_plateau_fit_quality(
        dataset_sizes, avg_memorization_values, plateau_size
    )
    
    # Enhanced progress reporting
    print(f"\n🎯 ENHANCED CAPACITY RESULTS:")
    print(f"  Estimated capacity: {estimated_capacity:,.0f} bits ({bits_per_parameter:.3f} bits/param)")
    print(f"  Convergence success rate: {convergence_success_rate:.1%}")
    print(f"  MAX_STEPS eliminations: {'✓ YES' if max_steps_elimination else '✗ NO'}")
    print(f"  Morris progress: {morris_progress:.1%} toward 3.6 bits/param target")
    print(f"  Average training time: {avg_training_time:.1f} seconds per experiment")
    
    return EnhancedCapacityEstimate(
        estimated_capacity_bits=estimated_capacity,
        bits_per_parameter=bits_per_parameter,
        plateau_dataset_size=plateau_size,
        memorization_values=avg_memorization_values.tolist(),
        dataset_sizes=dataset_sizes,
        r_squared=r_squared,
        plateau_confidence=plateau_confidence,
        convergence_success_rate=convergence_success_rate,
        avg_training_time=avg_training_time,
        max_steps_elimination=max_steps_elimination,
        morris_target_progress=morris_progress
    )


def detect_memorization_plateau_enhanced(
    dataset_sizes: List[int],
    memorization_values: List[float],
    tolerance: float = 0.05,
    experiment_results: List[EnhancedCapacityExperimentResult] = None
) -> Tuple[int, float]:
    """
    Enhanced plateau detection using convergence information from enhanced training.
    
    Uses both memorization values and convergence status to improve plateau detection.
    """
    
    if len(dataset_sizes) == 0:
        return 0, 0.0
    
    if len(dataset_sizes) < 3:
        return dataset_sizes[-1], 0.0
    
    # Original plateau detection
    sizes = np.array(dataset_sizes)
    values = np.array(memorization_values)
    
    max_memorization = np.max(values)
    if max_memorization == 0:
        return dataset_sizes[0], 0.0
    
    # Enhanced: Use convergence information if available
    plateau_candidates = []
    
    # Method 1: Convergence-informed plateau detection
    if experiment_results:
        # Group results by dataset size
        size_to_results = {}
        for result in experiment_results:
            if result.dataset_size not in size_to_results:
                size_to_results[result.dataset_size] = []
            size_to_results[result.dataset_size].append(result)
        
        # Look for dataset sizes where convergence rate drops (indicating capacity reached)
        for i, size in enumerate(dataset_sizes[:-1]):
            if size in size_to_results:
                current_convergence = np.mean([r.convergence_achieved for r in size_to_results[size]])
                
                # If convergence rate drops significantly, may indicate capacity reached
                if current_convergence < 0.7 and i > 0:  # Less than 70% convergence
                    plateau_candidates.append((size, 1.0 - current_convergence))
    
    # Method 2: Traditional relative change threshold
    for i in range(1, len(values)):
        if i >= len(values) - 2:
            continue
            
        current_value = values[i]
        remaining_values = values[i+1:]
        
        if len(remaining_values) > 0:
            max_remaining = np.max(remaining_values)
            relative_increase = (max_remaining - current_value) / current_value if current_value > 0 else 0
            
            if relative_increase < tolerance:
                plateau_candidates.append((sizes[i], relative_increase))
    
    # Method 3: Derivative-based detection (from original)
    if len(values) >= 4:
        derivatives = np.diff(values) / np.diff(sizes)
        
        if len(derivatives) >= 3:
            smoothed_derivatives = np.convolve(derivatives, [0.25, 0.5, 0.25], mode='valid')
            max_derivative = np.max(np.abs(smoothed_derivatives)) if len(smoothed_derivatives) > 0 else 0
            
            if max_derivative > 0:
                derivative_threshold = max_derivative * tolerance
                
                for i, deriv in enumerate(smoothed_derivatives):
                    if abs(deriv) < derivative_threshold:
                        plateau_idx = i + 1
                        if plateau_idx < len(sizes):
                            plateau_candidates.append((sizes[plateau_idx], abs(deriv) / max_derivative))
    
    # Choose best plateau candidate
    if plateau_candidates:
        # Sort by position (prefer earlier plateaus)
        plateau_candidates.sort(key=lambda x: x[0])
        plateau_size = plateau_candidates[0][0]
        confidence = 1.0 - plateau_candidates[0][1]
    else:
        # Fallback: use point where we reach 95% of maximum
        threshold_value = 0.95 * max_memorization
        plateau_idx = np.argmax(values >= threshold_value)
        plateau_size = sizes[plateau_idx]
        confidence = 0.5
    
    return int(plateau_size), confidence


def run_enhanced_capacity_experiments(
    model_configs: List[ModelConfig],
    training_config: TrainingConfig,
    base_dataset_sizes: List[int] = None,
    n_seeds: int = 3,
    device: str = "cuda",
    use_enhanced_training: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive ENHANCED capacity experiments across multiple model sizes.
    
    Reproduces Morris et al. methodology with enhanced training for reliable results.
    """
    
    if base_dataset_sizes is None:
        # Use Morris et al. scale but adjusted for enhanced training efficiency
        base_dataset_sizes = [500, 2000, 8000, 25000, 100000, 400000, 1000000, 2500000]
    
    all_results = []
    capacity_estimates = []
    model_sizes = []
    estimated_capacities = []
    
    print(f"🚀 ENHANCED CAPACITY EXPERIMENTS")
    print(f"Enhanced training: {'ENABLED' if use_enhanced_training else 'DISABLED'}")
    print(f"Testing {len(model_configs)} model configurations")
    
    for i, model_config in enumerate(model_configs):
        print(f"\n{'='*60}")
        print(f"Model {i+1}/{len(model_configs)}: {model_config.n_layers} layers, "
              f"d_model={model_config.d_model}")
        
        # Scale dataset sizes based on expected model capacity
        model_param_count = count_parameters(create_gpt_model(model_config))
        
        # More intelligent scaling based on enhanced training efficiency
        scale_factor = max(1.0, (model_param_count / 100000) ** 1.2)  # Less aggressive than original
        scaled_dataset_sizes = [int(size * scale_factor) for size in base_dataset_sizes]
        
        print(f"Model parameters: {model_param_count:,}")
        print(f"Dataset sizes: {scaled_dataset_sizes[0]:,} to {scaled_dataset_sizes[-1]:,} samples")
        
        # Enhanced parameters for different model sizes
        enhanced_params = {
            'memorization_threshold': max(0.2, 0.5 - model_param_count / 1000000),  # Stricter for larger models
            'memorization_check_interval': min(500, max(100, model_param_count // 1000)),
            'max_plateau_patience': min(5000, max(2000, model_param_count // 100))
        }
        
        # Estimate capacity for this model with enhanced training
        capacity_estimate = estimate_model_capacity_enhanced(
            model_config=model_config,
            training_config=training_config,
            dataset_sizes=scaled_dataset_sizes,
            n_seeds=n_seeds,
            device=device,
            use_enhanced_training=use_enhanced_training,
            enhanced_params=enhanced_params
        )
        
        capacity_estimates.append(capacity_estimate)
        model_sizes.append(model_param_count)
        estimated_capacities.append(capacity_estimate.estimated_capacity_bits)
        
        # Enhanced progress reporting
        print(f"📊 Results: {capacity_estimate.bits_per_parameter:.3f} bits/param "
              f"(Morris progress: {capacity_estimate.morris_target_progress:.1%})")
        print(f"   Convergence rate: {capacity_estimate.convergence_success_rate:.1%}")
        print(f"   MAX_STEPS eliminated: {'✓' if capacity_estimate.max_steps_elimination else '✗'}")
        
        # Store results
        result = {
            'model_config': model_config,
            'model_params': model_param_count,
            'capacity_estimate': capacity_estimate,
            'dataset_sizes': scaled_dataset_sizes,
            'enhanced_metrics': {
                'convergence_success_rate': capacity_estimate.convergence_success_rate,
                'max_steps_elimination': capacity_estimate.max_steps_elimination,
                'morris_progress': capacity_estimate.morris_target_progress
            }
        }
        all_results.append(result)
    
    # Enhanced scaling law analysis
    bits_per_param, intercept, r_squared = fit_capacity_scaling_law(
        model_sizes, estimated_capacities
    )
    
    # Calculate enhanced statistics
    bits_per_param_values = [cap / size for cap, size in zip(estimated_capacities, model_sizes)]
    mean_bits_per_param = np.mean(bits_per_param_values)
    std_bits_per_param = np.std(bits_per_param_values)
    
    # Enhanced convergence metrics
    overall_convergence_rate = np.mean([est.convergence_success_rate for est in capacity_estimates])
    max_steps_elimination_rate = np.mean([est.max_steps_elimination for est in capacity_estimates])
    morris_progress = np.mean([est.morris_target_progress for est in capacity_estimates])
    
    print(f"\n🎯 ENHANCED SCALING LAW RESULTS:")
    print(f"Mean bits-per-parameter: {mean_bits_per_param:.3f} (Morris target: 3.6)")
    print(f"Morris progress: {morris_progress:.1%}")
    print(f"Overall convergence success: {overall_convergence_rate:.1%}")
    print(f"MAX_STEPS elimination rate: {max_steps_elimination_rate:.1%}")
    print(f"Linear fit R²: {r_squared:.3f}")
    
    return {
        'individual_results': all_results,
        'capacity_estimates': capacity_estimates,
        'model_sizes': model_sizes,
        'estimated_capacities': estimated_capacities,
        'scaling_law': {
            'bits_per_parameter': bits_per_param,
            'intercept': intercept,
            'r_squared': r_squared
        },
        'enhanced_summary_statistics': {
            'mean_bits_per_parameter': mean_bits_per_param,
            'std_bits_per_parameter': std_bits_per_param,
            'n_models': len(model_configs),
            'morris_target': 3.6,
            'morris_progress': morris_progress,
            'deviation_from_target': abs(mean_bits_per_param - 3.6),
            'overall_convergence_rate': overall_convergence_rate,
            'max_steps_elimination_rate': max_steps_elimination_rate,
            'enhanced_training_used': use_enhanced_training
        }
    }


def validate_enhanced_capacity_experiment(
    experiment_results: Dict[str, Any],
    expected_bits_per_param: float = 3.6,
    tolerance: float = 0.5
) -> Dict[str, bool]:
    """
    Validate enhanced capacity experiment results against Morris et al. findings.
    
    Includes enhanced validation criteria for training reliability.
    """
    
    summary = experiment_results['enhanced_summary_statistics']
    scaling = experiment_results['scaling_law']
    
    validations = {
        'bits_per_param_in_range': abs(summary['mean_bits_per_parameter'] - expected_bits_per_param) <= tolerance,
        'scaling_law_significant': scaling['r_squared'] >= 0.8,
        'positive_capacities': all(cap > 0 for cap in experiment_results['estimated_capacities']),
        'plateaus_detected': all(
            est.plateau_confidence > 0.3 for est in experiment_results['capacity_estimates']
        ),
        'consistent_across_models': summary['std_bits_per_parameter'] <= 1.0,
        # Enhanced validation criteria
        'high_convergence_rate': summary['overall_convergence_rate'] >= 0.7,
        'max_steps_eliminated': summary['max_steps_elimination_rate'] >= 0.8,
        'morris_progress_significant': summary['morris_progress'] >= 0.3
    }
    
    # Overall validation
    core_validations = ['bits_per_param_in_range', 'scaling_law_significant', 'positive_capacities']
    enhanced_validations = ['high_convergence_rate', 'max_steps_eliminated']
    
    validations['core_experiment_valid'] = all(validations[key] for key in core_validations)
    validations['enhanced_training_effective'] = all(validations[key] for key in enhanced_validations)
    validations['experiment_fully_valid'] = validations['core_experiment_valid'] and validations['enhanced_training_effective']
    
    return validations


# Import remaining functions from original capacity_estimator for compatibility
from capacity_estimator import (
    calculate_bits_per_parameter,
    fit_capacity_scaling_law,
    calculate_plateau_fit_quality,
    create_morris_style_model_configs,
    analyze_memorization_vs_generalization
)


def create_enhanced_morris_configs(
    target_param_counts: List[int] = None,
    enhanced_training: bool = True
) -> List[ModelConfig]:
    """
    Create model configurations optimized for enhanced training and Morris reproduction.
    
    Args:
        target_param_counts: Target parameter counts (auto-generate if None)
        enhanced_training: Whether configs will be used with enhanced training
        
    Returns:
        List of ModelConfig objects optimized for enhanced training
    """
    
    if target_param_counts is None:
        if enhanced_training:
            # More aggressive scaling possible with enhanced training
            target_param_counts = [50000, 200000, 800000, 2000000, 5000000]
        else:
            # Conservative scaling for original training
            target_param_counts = [100000, 500000, 1000000, 5000000]
    
    configs = create_morris_style_model_configs(target_param_counts)
    
    if enhanced_training:
        # Optimize configs for enhanced training
        for config in configs:
            # Slightly lower dropout for better memorization
            config.dropout = max(0.05, config.dropout * 0.8)
    
    return configs


# Alias functions for backward compatibility
estimate_model_capacity = estimate_model_capacity_enhanced
run_capacity_experiments = run_enhanced_capacity_experiments
validate_capacity_experiment = validate_enhanced_capacity_experiment

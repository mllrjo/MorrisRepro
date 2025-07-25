"""
File: capacity_estimator.py
Directory: memorization_reproduction/src/

Capacity estimation module for reproducing Morris et al. memorization capacity findings.
Implements experiments to measure model capacity and validate 3.6 bits-per-parameter result.
"""

from typing import List, Dict, Tuple, Optional, Callable, Any
import torch
import numpy as np
import math
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings

# Import our modules
from model_trainer import (
    ModelConfig, TrainingConfig, create_gpt_model, train_model, count_parameters
)
from memorization_calculator import (
    calculate_total_memorization, create_synthetic_reference_model,
    validate_memorization_calculation
)
from data_generator import generate_uniform_bitstrings, create_dataset_size_series


@dataclass
class CapacityExperimentResult:
    """Results from a capacity estimation experiment."""
    model_params: int
    dataset_size: int
    total_memorization: float
    memorization_per_param: float
    training_loss: float
    experiment_id: str
    model_config: ModelConfig
    training_config: TrainingConfig


@dataclass
class CapacityEstimate:
    """Model capacity estimation results."""
    estimated_capacity_bits: float
    bits_per_parameter: float
    plateau_dataset_size: int
    memorization_values: List[float]
    dataset_sizes: List[int]
    r_squared: float
    plateau_confidence: float


def estimate_model_capacity(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    dataset_sizes: List[int],
    n_seeds: int = 3,
    device: str = "cuda",
    plateau_tolerance: float = 0.05
) -> CapacityEstimate:
    """
    Estimate model capacity by finding memorization plateau across dataset sizes.
    
    Core experiment from Morris et al.: train models on increasing dataset sizes
    until memorization plateaus, indicating capacity limit reached.
    
    Args:
        model_config: Model architecture configuration
        training_config: Training parameters
        dataset_sizes: List of dataset sizes to test (should span capacity)
        n_seeds: Number of random seeds for statistical robustness
        device: Device for computation
        plateau_tolerance: Relative tolerance for plateau detection
        
    Returns:
        CapacityEstimate with plateau detection and bits-per-parameter
    """
    all_memorization_values = []
    all_results = []
    
    # Display progressive testing
    model_params = count_parameters(create_gpt_model(model_config))
    print(f"Testing {len(dataset_sizes)} dataset sizes for model with {model_params:,} parameters:")
    
    # Run experiments across seeds for statistical robustness
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        memorization_values = []
        
        for i, dataset_size in enumerate(dataset_sizes):
            print(f"  Dataset {dataset_size:,} samples (#{i+1}/{len(dataset_sizes)})... ", end="", flush=True)
            
            # Generate uniform random data (no generalization possible)
            # Use model's max_seq_length to avoid embedding errors
            seq_length = min(64, model_config.max_seq_length)  # Morris et al. uses 64, but respect model limits
            
            data = generate_uniform_bitstrings(
                n_samples=dataset_size,
                seq_length=seq_length,
                vocab_size=model_config.vocab_size,
                seed=seed + dataset_size  # Ensure different data per size
            )
            
            # Create and train model
            model = create_gpt_model(model_config)
            training_metrics = train_model(model, data, training_config, device)

            # DEBUG: Check if training worked
            if training_metrics and "train_loss" in training_metrics:
                initial_loss = training_metrics["train_loss"][0] if training_metrics["train_loss"] else "unknown"
                final_loss = training_metrics["train_loss"][-1] if training_metrics["train_loss"] else "unknown"
                print(f"    Training: {initial_loss:.2f} → {final_loss:.2f} loss")
            else:
                print(f"    Training: no loss metrics returned")
            
            # Create reference model (uniform for synthetic data)
            reference_model = create_synthetic_reference_model(
                model, model_config.vocab_size
            )
            
            # Calculate total memorization
            total_memorization = calculate_total_memorization(
                model, reference_model, data, device
            )
            
            memorization_values.append(total_memorization)
            
            # Display result
            bits_per_param = total_memorization / model_params
            print(f"Memorization: {total_memorization:,.0f} bits ({bits_per_param:.3f} bits/param)")
            
            # Store detailed results
            result = CapacityExperimentResult(
                model_params=count_parameters(model),
                dataset_size=dataset_size,
                total_memorization=total_memorization,
                memorization_per_param=total_memorization / count_parameters(model),
                training_loss=training_metrics["train_loss"][-1] if training_metrics["train_loss"] else float('nan'),
                experiment_id=f"seed_{seed}_size_{dataset_size}",
                model_config=model_config,
                training_config=training_config
            )
            all_results.append(result)
        
        all_memorization_values.append(memorization_values)
    
    # Average across seeds
    avg_memorization_values = np.mean(all_memorization_values, axis=0)
    
    # Detect plateau
    plateau_size, plateau_confidence = detect_memorization_plateau(
        dataset_sizes, avg_memorization_values, plateau_tolerance
    )
    
    # Display plateau detection results
    print(f"  Plateau detected at dataset size: {plateau_size:,} (confidence: {plateau_confidence:.2f})")
    
    # Estimate capacity as plateau value
    estimated_capacity = np.max(avg_memorization_values)
    
    # Calculate bits per parameter
    model_params = count_parameters(create_gpt_model(model_config))
    bits_per_parameter = estimated_capacity / model_params
    
    # Calculate R² for plateau fit quality
    r_squared = calculate_plateau_fit_quality(
        dataset_sizes, avg_memorization_values, plateau_size
    )
    
    print(f"  Final capacity estimate: {estimated_capacity:,.0f} bits ({bits_per_parameter:.3f} bits/param)")
    
    return CapacityEstimate(
        estimated_capacity_bits=estimated_capacity,
        bits_per_parameter=bits_per_parameter,
        plateau_dataset_size=plateau_size,
        memorization_values=avg_memorization_values.tolist(),
        dataset_sizes=dataset_sizes,
        r_squared=r_squared,
        plateau_confidence=plateau_confidence
    )


def detect_memorization_plateau(
    dataset_sizes: List[int],
    memorization_values: List[float],
    tolerance: float = 0.05
) -> Tuple[int, float]:
    """
    Detect where memorization plateaus (capacity reached).
    
    Uses statistical methods to identify when memorization stops
    increasing significantly with dataset size.
    
    Args:
        dataset_sizes: List of dataset sizes
        memorization_values: Corresponding memorization measurements
        tolerance: Relative tolerance for plateau detection
        
    Returns:
        Tuple of (plateau_dataset_size, confidence_score)
    """
    if len(dataset_sizes) == 0:
        return 0, 0.0
    
    if len(dataset_sizes) < 3:
        return dataset_sizes[-1], 0.0
    
    # Convert to numpy arrays
    sizes = np.array(dataset_sizes)
    values = np.array(memorization_values)
    
    # Find maximum memorization value
    max_memorization = np.max(values)
    
    if max_memorization == 0:
        return dataset_sizes[0], 0.0
    
    # Method 1: Relative change threshold
    # Look for where relative increase becomes small
    plateau_candidates = []
    
    for i in range(1, len(values)):
        if i >= len(values) - 2:  # Need at least 2 more points
            continue
            
        # Check if subsequent points don't increase much
        current_value = values[i]
        remaining_values = values[i+1:]
        
        if len(remaining_values) > 0:
            max_remaining = np.max(remaining_values)
            relative_increase = (max_remaining - current_value) / current_value if current_value > 0 else 0
            
            if relative_increase < tolerance:
                plateau_candidates.append((sizes[i], relative_increase))
    
    # Method 2: Derivative-based detection
    # Look for where slope becomes small
    if len(values) >= 4:
        # Calculate discrete derivatives
        derivatives = np.diff(values) / np.diff(sizes)
        
        # Smooth derivatives
        if len(derivatives) >= 3:
            smoothed_derivatives = np.convolve(derivatives, [0.25, 0.5, 0.25], mode='valid')
            
            # Find where derivative becomes small
            max_derivative = np.max(np.abs(smoothed_derivatives)) if len(smoothed_derivatives) > 0 else 0
            
            if max_derivative > 0:
                derivative_threshold = max_derivative * tolerance
                
                for i, deriv in enumerate(smoothed_derivatives):
                    if abs(deriv) < derivative_threshold:
                        plateau_idx = i + 1  # Account for convolution offset
                        if plateau_idx < len(sizes):
                            plateau_candidates.append((sizes[plateau_idx], abs(deriv) / max_derivative))
    
    # Choose best plateau candidate
    if plateau_candidates:
        # Sort by position (prefer earlier plateaus)
        plateau_candidates.sort(key=lambda x: x[0])
        plateau_size = plateau_candidates[0][0]
        confidence = 1.0 - plateau_candidates[0][1]  # Higher confidence = lower relative change
    else:
        # Fallback: use point where we reach 95% of maximum
        threshold_value = 0.95 * max_memorization
        plateau_idx = np.argmax(values >= threshold_value)
        plateau_size = sizes[plateau_idx]
        confidence = 0.5  # Medium confidence for fallback method
    
    return int(plateau_size), confidence


def calculate_plateau_fit_quality(
    dataset_sizes: List[int],
    memorization_values: List[float],
    plateau_size: int
) -> float:
    """
    Calculate R² for plateau fit quality assessment.
    
    Args:
        dataset_sizes: Dataset sizes
        memorization_values: Memorization values
        plateau_size: Detected plateau size
        
    Returns:
        R² value for plateau fit quality
    """
    sizes = np.array(dataset_sizes)
    values = np.array(memorization_values)
    
    # Find plateau region (points at or beyond plateau size)
    plateau_mask = sizes >= plateau_size
    
    if np.sum(plateau_mask) < 2:
        return 0.0
    
    plateau_values = values[plateau_mask]
    plateau_mean = np.mean(plateau_values)
    
    # Calculate R² for plateau region
    ss_res = np.sum((plateau_values - plateau_mean) ** 2)
    ss_tot = np.sum((plateau_values - np.mean(values)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    r_squared = 1 - (ss_res / ss_tot)
    return max(0.0, r_squared)


def calculate_bits_per_parameter(
    model: torch.nn.Module,
    capacity_bits: float
) -> float:
    """
    Calculate bits-per-parameter ratio.
    
    Core metric from Morris et al.: should be ~3.6 for GPT models.
    
    Args:
        model: Model to analyze
        capacity_bits: Estimated capacity in bits
        
    Returns:
        Bits per parameter
    """
    param_count = count_parameters(model)
    if param_count == 0:
        return 0.0
    
    return capacity_bits / param_count


def fit_capacity_scaling_law(
    model_sizes: List[int],
    capacities: List[float]
) -> Tuple[float, float, float]:
    """
    Fit linear relationship between model parameters and capacity.
    
    Morris et al. finding: capacity scales linearly with parameters
    at approximately 3.6 bits-per-parameter.
    
    Args:
        model_sizes: List of model parameter counts
        capacities: List of corresponding capacities
        
    Returns:
        Tuple of (bits_per_parameter_slope, intercept, r_squared)
    """
    if len(model_sizes) < 2:
        return 0.0, 0.0, 0.0
    
    # Convert to numpy arrays
    X = np.array(model_sizes).reshape(-1, 1)
    y = np.array(capacities)
    
    # Fit linear regression
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    
    # Calculate R²
    y_pred = reg.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return reg.coef_[0], reg.intercept_, r_squared


def run_capacity_experiments(
    model_configs: List[ModelConfig],
    training_config: TrainingConfig,
    base_dataset_sizes: List[int] = None,
    n_seeds: int = 3,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Run comprehensive capacity experiments across multiple model sizes.
    
    Reproduces Morris et al. Figure 1 and capacity scaling analysis.
    
    Args:
        model_configs: List of model configurations to test
        training_config: Training configuration
        base_dataset_sizes: Base dataset sizes (scaled per model)
        n_seeds: Number of random seeds
        device: Device for computation
        
    Returns:
        Dictionary with experiment results and scaling law analysis
    """
    if base_dataset_sizes is None:
        # Use Morris et al. scale: up to 8M samples (Figure 1)
        base_dataset_sizes = [1000, 5000, 25000, 100000, 500000, 1000000, 2000000, 4000000, 8000000]
    
    all_results = []
    capacity_estimates = []
    model_sizes = []
    estimated_capacities = []
    
    for i, model_config in enumerate(model_configs):
        print(f"Running capacity experiment {i+1}/{len(model_configs)} for model with "
              f"{model_config.n_layers} layers, d_model={model_config.d_model}")
        
        # Scale dataset sizes based on expected model capacity
        # Larger models need larger datasets to reach plateau
        model_param_count = count_parameters(create_gpt_model(model_config))
        
        # Aggressive scaling: larger models need exponentially more data
        scale_factor = max(1.0, (model_param_count / 100000) ** 1.5)
        
        scaled_dataset_sizes = [int(size * scale_factor) for size in base_dataset_sizes]
        
        print(f"Dataset sizes for {model_param_count:,} parameter model: "
              f"{scaled_dataset_sizes[0]:,} to {scaled_dataset_sizes[-1]:,} samples")
        
        # Estimate capacity for this model
        capacity_estimate = estimate_model_capacity(
            model_config=model_config,
            training_config=training_config,
            dataset_sizes=scaled_dataset_sizes,
            n_seeds=n_seeds,
            device=device
        )
        
        capacity_estimates.append(capacity_estimate)
        model_sizes.append(model_param_count)
        estimated_capacities.append(capacity_estimate.estimated_capacity_bits)
        
        # Store results
        result = {
            'model_config': model_config,
            'model_params': model_param_count,
            'capacity_estimate': capacity_estimate,
            'dataset_sizes': scaled_dataset_sizes
        }
        all_results.append(result)
    
    # Fit scaling law across all models
    bits_per_param, intercept, r_squared = fit_capacity_scaling_law(
        model_sizes, estimated_capacities
    )
    
    # Calculate statistics
    bits_per_param_values = [cap / size for cap, size in zip(estimated_capacities, model_sizes)]
    mean_bits_per_param = np.mean(bits_per_param_values)
    std_bits_per_param = np.std(bits_per_param_values)
    
    print(f"\nScaling Law Results:")
    print(f"Mean bits-per-parameter: {mean_bits_per_param:.3f} (target: 3.6)")
    print(f"Standard deviation: {std_bits_per_param:.3f}")
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
        'summary_statistics': {
            'mean_bits_per_parameter': mean_bits_per_param,
            'std_bits_per_parameter': std_bits_per_param,
            'n_models': len(model_configs),
            'target_bits_per_param': 3.6,  # Morris et al. finding
            'deviation_from_target': abs(mean_bits_per_param - 3.6)
        }
    }


def validate_capacity_experiment(
    experiment_results: Dict[str, Any],
    expected_bits_per_param: float = 3.6,
    tolerance: float = 0.5
) -> Dict[str, bool]:
    """
    Validate capacity experiment results against Morris et al. findings.
    
    Args:
        experiment_results: Results from run_capacity_experiments
        expected_bits_per_param: Expected bits-per-parameter (3.6 from paper)
        tolerance: Tolerance for validation
        
    Returns:
        Dictionary of validation results
    """
    summary = experiment_results['summary_statistics']
    scaling = experiment_results['scaling_law']
    
    validations = {
        'bits_per_param_in_range': abs(summary['mean_bits_per_parameter'] - expected_bits_per_param) <= tolerance,
        'scaling_law_significant': scaling['r_squared'] >= 0.8,
        'positive_capacities': all(cap > 0 for cap in experiment_results['estimated_capacities']),
        'plateaus_detected': all(
            est.plateau_confidence > 0.3 for est in experiment_results['capacity_estimates']
        ),
        'consistent_across_models': summary['std_bits_per_parameter'] <= 1.0
    }
    
    # Overall validation
    validations['experiment_valid'] = all(validations.values())
    
    return validations


def create_morris_style_model_configs(
    target_param_counts: List[int] = None
) -> List[ModelConfig]:
    """
    Create model configurations matching Morris et al. experimental setup.
    
    Args:
        target_param_counts: Target parameter counts (auto-generate if None)
        
    Returns:
        List of ModelConfig objects
    """
    if target_param_counts is None:
        target_param_counts = [100000, 500000, 1000000, 5000000]
    
    configs = []
    vocab_size = 2048  # Following Morris et al.
    max_seq_length = 64
    
    for target_params in target_param_counts:
        # Heuristic to find reasonable architecture for target parameter count
        # Start with small model and scale up
        
        if target_params <= 200000:
            n_layers = 1
            d_model = 32
            n_heads = 2
        elif target_params <= 1000000:
            n_layers = 2
            d_model = 64
            n_heads = 4
        elif target_params <= 5000000:
            n_layers = 4
            d_model = 128
            n_heads = 8
        else:
            n_layers = 6
            d_model = 256
            n_heads = 16
        
        # Fine-tune d_model to hit target parameter count more precisely
        base_config = ModelConfig(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            vocab_size=vocab_size,
            max_seq_length=max_seq_length
        )
        
        # Create model to check actual parameter count
        actual_params = count_parameters(create_gpt_model(base_config))
        
        # Adjust d_model if needed (simple linear scaling approximation)
        if actual_params > 0:
            scale_factor = math.sqrt(target_params / actual_params)
            adjusted_d_model = max(16, int(d_model * scale_factor))
            
            # Ensure d_model is divisible by n_heads
            adjusted_d_model = (adjusted_d_model // n_heads) * n_heads
            
            final_config = ModelConfig(
                n_layers=n_layers,
                d_model=adjusted_d_model,
                n_heads=n_heads,
                vocab_size=vocab_size,
                max_seq_length=max_seq_length
            )
        else:
            final_config = base_config
        
        configs.append(final_config)
    
    return configs


def analyze_memorization_vs_generalization(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    text_data: str,
    dataset_sizes: List[int],
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Analyze memorization vs generalization on real text data.
    
    Reproduces Morris et al. text experiments showing transition from
    memorization to generalization as dataset size increases.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        text_data: Raw text for experiments
        dataset_sizes: Dataset sizes to test
        device: Device for computation
        
    Returns:
        Analysis results showing memorization/generalization trade-off
    """
    # This would require text preprocessing and tokenization
    # For now, return placeholder structure
    
    results = {
        'dataset_sizes': dataset_sizes,
        'memorization_values': [],
        'generalization_metrics': [],
        'transition_point': None,
        'double_descent_detected': False
    }
    
    # TODO: Implement full text experiment pipeline
    # Would need: tokenizer, text preprocessing, reference model training
    
    return results

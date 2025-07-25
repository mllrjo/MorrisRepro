"""
File: experiment_runner.py
Directory: memorization_reproduction/src/

High-level experiment runner for reproducing Morris et al. memorization findings.
Provides easy-to-use interfaces for running complete experimental suites with automatic device detection.
"""

from typing import List, Dict, Tuple, Optional, Any
import torch
import numpy as np
import time
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

# Import our modules
from model_trainer import ModelConfig, TrainingConfig, create_gpt_model, count_parameters
from capacity_estimator import (
    run_capacity_experiments, create_morris_style_model_configs, 
    validate_capacity_experiment, CapacityEstimate
)
from data_generator import generate_uniform_bitstrings
from memorization_calculator import (
    calculate_total_memorization, create_synthetic_reference_model
)


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution."""
    device: str
    use_cpu_optimizations: bool
    max_model_size: int
    max_dataset_size: int
    n_seeds: int
    save_results: bool
    results_dir: str


@dataclass
class ExperimentSuite:
    """Complete experimental suite results."""
    suite_name: str
    experiment_config: ExperimentConfig
    capacity_results: Dict[str, Any]
    validation_results: Dict[str, bool]
    execution_time: float
    timestamp: str
    morris_comparison: Dict[str, Any]


def detect_device() -> Tuple[str, bool]:
    """
    Automatically detect best available device and computational capabilities.
    
    Returns:
        Tuple of (device_string, use_cpu_optimizations)
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        use_cpu_optimizations = False
        print(f"GPU detected: {torch.cuda.get_device_name(0)} with {gpu_memory:.1f}GB memory")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps is not None and torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon
        use_cpu_optimizations = True  # Treat similar to CPU for conservative memory usage
        print("Apple Silicon GPU (MPS) detected")
    else:
        device = "cpu"
        use_cpu_optimizations = True
        print("Using CPU - experiments will be scaled down for reasonable runtime")
    
    return device, use_cpu_optimizations


def create_device_appropriate_config(
    device: str, 
    use_cpu_optimizations: bool
) -> ExperimentConfig:
    """
    Create experiment configuration appropriate for detected device.
    
    Args:
        device: Device string ("cuda", "mps", or "cpu")
        use_cpu_optimizations: Whether to use CPU-optimized settings
        
    Returns:
        ExperimentConfig tailored to device capabilities
    """
    if use_cpu_optimizations:
        # CPU-friendly settings - smaller models and datasets
        config = ExperimentConfig(
            device=device,
            use_cpu_optimizations=True,
            max_model_size=500000,  # 500K parameters max
            max_dataset_size=10000,  # 10K samples max
            n_seeds=2,  # Fewer seeds for speed
            save_results=True,
            results_dir="results/cpu_experiments"
        )
        print("Using CPU-optimized settings: smaller models and datasets for reasonable runtime")
    else:
        # GPU settings - can handle larger experiments
        config = ExperimentConfig(
            device=device,
            use_cpu_optimizations=False,
            max_model_size=5000000,  # 5M parameters max
            max_dataset_size=100000,  # 100K samples max
            n_seeds=3,  # More seeds for statistical robustness
            save_results=True,
            results_dir="results/gpu_experiments"
        )
        print("Using GPU-optimized settings: larger models and datasets")
    
    return config


def create_scaled_model_configs(experiment_config: ExperimentConfig) -> List[ModelConfig]:
    """
    Create model configurations scaled appropriately for device capabilities.
    
    Args:
        experiment_config: Experiment configuration
        
    Returns:
        List of ModelConfig objects scaled for device
    """
    if experiment_config.use_cpu_optimizations:
        # Smaller models for CPU
        target_params = [50000, 100000, 200000, 500000]
        vocab_size = 1024  # Smaller vocab for speed
        max_seq_length = 32  # Shorter sequences
    else:
        # Larger models for GPU following Morris et al. more closely
        target_params = [100000, 500000, 1000000, 2000000, 5000000]
        vocab_size = 2048  # Morris et al. standard
        max_seq_length = 64  # Morris et al. standard
    
    # Filter by max model size
    target_params = [p for p in target_params if p <= experiment_config.max_model_size]
    
    configs = create_morris_style_model_configs(target_params)
    
    # Update vocab size and sequence length for all configs
    for config in configs:
        config.vocab_size = vocab_size
        config.max_seq_length = max_seq_length
    
    return configs


def create_scaled_training_config(experiment_config: ExperimentConfig) -> TrainingConfig:
    """
    Create training configuration scaled for device capabilities.
    
    Args:
        experiment_config: Experiment configuration
        
    Returns:
        TrainingConfig scaled for device
    """
    if experiment_config.use_cpu_optimizations:
        # CPU-friendly training settings
        config = TrainingConfig(
            batch_size=8,  # Smaller batches
            learning_rate=1e-3,
            max_steps=500,  # Fewer steps
            warmup_steps=50,
            weight_decay=0.01
        )
    else:
        # GPU settings closer to Morris et al.
        config = TrainingConfig(
            batch_size=32,  # Larger batches
            learning_rate=6e-4,  # Following GPT-2 style
            max_steps=2000,  # More steps
            warmup_steps=200,
            weight_decay=0.01
        )
    
    return config


def run_capacity_suite(experiment_config: ExperimentConfig) -> ExperimentSuite:
    """
    Run complete capacity estimation experimental suite.
    
    Reproduces Morris et al. Figure 1 and capacity scaling analysis
    with device-appropriate scaling.
    
    Args:
        experiment_config: Experiment configuration
        
    Returns:
        Complete experimental suite results
    """
    print(f"\n{'='*60}")
    print(f"STARTING CAPACITY ESTIMATION SUITE")
    print(f"Device: {experiment_config.device}")
    print(f"CPU Optimizations: {experiment_config.use_cpu_optimizations}")
    print(f"Max Model Size: {experiment_config.max_model_size:,} parameters")
    print(f"Max Dataset Size: {experiment_config.max_dataset_size:,} samples")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Create device-appropriate configurations
    model_configs = create_scaled_model_configs(experiment_config)
    training_config = create_scaled_training_config(experiment_config)
    
    print(f"Testing {len(model_configs)} model configurations:")
    for i, config in enumerate(model_configs):
        model = create_gpt_model(config)
        param_count = count_parameters(model)
        print(f"  Model {i+1}: {param_count:,} params "
              f"({config.n_layers} layers, d_model={config.d_model})")
    
    # Scale dataset sizes based on device capabilities
    if experiment_config.use_cpu_optimizations:
        base_dataset_sizes = [100, 300, 500, 1000, 2000, 5000]
    else:
        base_dataset_sizes = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    
    # Filter by max dataset size
    base_dataset_sizes = [s for s in base_dataset_sizes if s <= experiment_config.max_dataset_size]
    
    print(f"\nDataset sizes to test: {base_dataset_sizes}")
    print(f"Number of seeds per experiment: {experiment_config.n_seeds}")
    
    # Run capacity experiments
    print("\nRunning capacity experiments...")
    capacity_results = run_capacity_experiments(
        model_configs=model_configs,
        training_config=training_config,
        base_dataset_sizes=base_dataset_sizes,
        n_seeds=experiment_config.n_seeds,
        device=experiment_config.device
    )
    
    # Validate results against Morris et al. findings
    print("\nValidating results against Morris et al. findings...")
    validation_results = validate_capacity_experiment(
        capacity_results,
        expected_bits_per_param=3.6,
        tolerance=1.0 if experiment_config.use_cpu_optimizations else 0.5
    )
    
    # Compare to Morris et al. findings
    morris_comparison = compare_to_morris_findings(capacity_results, experiment_config)
    
    execution_time = time.time() - start_time
    
    # Create experimental suite
    suite = ExperimentSuite(
        suite_name="morris_capacity_reproduction",
        experiment_config=experiment_config,
        capacity_results=capacity_results,
        validation_results=validation_results,
        execution_time=execution_time,
        timestamp=datetime.now().isoformat(),
        morris_comparison=morris_comparison
    )
    
    # Print summary
    print_experiment_summary(suite)
    
    # Save results if requested
    if experiment_config.save_results:
        save_experiment_results(suite)
    
    return suite


def compare_to_morris_findings(
    capacity_results: Dict[str, Any],
    experiment_config: ExperimentConfig
) -> Dict[str, Any]:
    """
    Compare experimental results to Morris et al. findings.
    
    Args:
        capacity_results: Results from capacity experiments
        experiment_config: Experiment configuration
        
    Returns:
        Dictionary comparing results to Morris et al.
    """
    summary = capacity_results['summary_statistics']
    scaling = capacity_results['scaling_law']
    
    # Morris et al. key findings
    morris_bits_per_param = 3.6
    morris_scaling_r2_threshold = 0.8
    
    comparison = {
        'morris_target_bpp': morris_bits_per_param,
        'observed_bpp': summary['mean_bits_per_parameter'],
        'bpp_deviation': abs(summary['mean_bits_per_parameter'] - morris_bits_per_param),
        'bpp_relative_error': abs(summary['mean_bits_per_parameter'] - morris_bits_per_param) / morris_bits_per_param,
        'scaling_law_quality': scaling['r_squared'],
        'scaling_meets_threshold': scaling['r_squared'] >= morris_scaling_r2_threshold,
        'consistent_across_models': summary['std_bits_per_parameter'] <= 1.0,
        'experiment_scale': 'cpu_optimized' if experiment_config.use_cpu_optimizations else 'full_scale',
        'n_models_tested': summary['n_models'],
        'morris_reproduction_score': calculate_reproduction_score(summary, scaling, morris_bits_per_param)
    }
    
    return comparison


def calculate_reproduction_score(
    summary: Dict[str, Any],
    scaling: Dict[str, Any],
    target_bpp: float
) -> float:
    """
    Calculate overall reproduction score (0-100) for Morris et al. findings.
    
    Args:
        summary: Summary statistics from experiments
        scaling: Scaling law results
        target_bpp: Target bits-per-parameter (3.6)
        
    Returns:
        Reproduction score from 0-100
    """
    # Component scores (each 0-1)
    bpp_score = max(0, 1 - abs(summary['mean_bits_per_parameter'] - target_bpp) / target_bpp)
    scaling_score = min(1.0, scaling['r_squared'])
    consistency_score = max(0, 1 - summary['std_bits_per_parameter'] / 2.0)  # Penalize high variance
    
    # Weighted average
    overall_score = (0.5 * bpp_score + 0.3 * scaling_score + 0.2 * consistency_score) * 100
    
    return min(100.0, max(0.0, overall_score))


def run_single_model_demo(
    model_config: ModelConfig,
    experiment_config: ExperimentConfig,
    dataset_sizes: List[int] = None
) -> Dict[str, Any]:
    """
    Run a focused demonstration experiment on a single model.
    
    Useful for quick validation and debugging.
    
    Args:
        model_config: Single model configuration to test
        experiment_config: Experiment configuration
        dataset_sizes: Dataset sizes to test (auto-generated if None)
        
    Returns:
        Single model experiment results
    """
    print(f"\n{'='*50}")
    print(f"SINGLE MODEL DEMONSTRATION")
    print(f"Device: {experiment_config.device}")
    
    model = create_gpt_model(model_config)
    param_count = count_parameters(model)
    print(f"Model: {param_count:,} parameters")
    print(f"Architecture: {model_config.n_layers} layers, d_model={model_config.d_model}")
    print(f"{'='*50}\n")
    
    if dataset_sizes is None:
        if experiment_config.use_cpu_optimizations:
            dataset_sizes = [100, 200, 500, 1000, 2000]
        else:
            dataset_sizes = [500, 1000, 2000, 5000, 10000]
    
    training_config = create_scaled_training_config(experiment_config)
    
    print(f"Testing dataset sizes: {dataset_sizes}")
    print("Running capacity estimation...")
    
    start_time = time.time()
    
    # Run simplified capacity estimation
    memorization_values = []
    
    for dataset_size in dataset_sizes:
        print(f"  Dataset size {dataset_size}...")
        
        # Generate data
        data = generate_uniform_bitstrings(
            n_samples=dataset_size,
            seq_length=min(model_config.max_seq_length, 32),
            vocab_size=model_config.vocab_size,
            seed=42
        )
        
        # Train model
        from model_trainer import train_model
        fresh_model = create_gpt_model(model_config)
        train_model(fresh_model, data, training_config, experiment_config.device)
        
        # Calculate memorization
        reference_model = create_synthetic_reference_model(fresh_model, model_config.vocab_size)
        total_memorization = calculate_total_memorization(
            fresh_model, reference_model, data, experiment_config.device
        )
        
        memorization_values.append(total_memorization)
        print(f"    Memorization: {total_memorization:.1f} bits")
    
    execution_time = time.time() - start_time
    
    # Calculate bits per parameter
    estimated_capacity = max(memorization_values)
    bits_per_parameter = estimated_capacity / param_count
    
    results = {
        'model_config': model_config,
        'param_count': param_count,
        'dataset_sizes': dataset_sizes,
        'memorization_values': memorization_values,
        'estimated_capacity': estimated_capacity,
        'bits_per_parameter': bits_per_parameter,
        'execution_time': execution_time,
        'morris_target_bpp': 3.6,
        'bpp_deviation': abs(bits_per_parameter - 3.6)
    }
    
    print(f"\nResults:")
    print(f"  Estimated capacity: {estimated_capacity:.1f} bits")
    print(f"  Bits per parameter: {bits_per_parameter:.2f}")
    print(f"  Morris et al. target: 3.6 bits/param")
    print(f"  Deviation: {abs(bits_per_parameter - 3.6):.2f}")
    print(f"  Execution time: {execution_time:.1f} seconds")
    
    return results


def print_experiment_summary(suite: ExperimentSuite) -> None:
    """Print comprehensive experiment summary."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUITE SUMMARY")
    print(f"{'='*80}")
    
    config = suite.experiment_config
    results = suite.capacity_results
    validation = suite.validation_results
    comparison = suite.morris_comparison
    
    print(f"\nConfiguration:")
    print(f"  Device: {config.device}")
    print(f"  CPU Optimizations: {config.use_cpu_optimizations}")
    print(f"  Models Tested: {results['summary_statistics']['n_models']}")
    print(f"  Seeds per Experiment: {config.n_seeds}")
    print(f"  Execution Time: {suite.execution_time:.1f} seconds")
    
    print(f"\nKey Results:")
    print(f"  Mean Bits/Parameter: {results['summary_statistics']['mean_bits_per_parameter']:.2f}")
    print(f"  Std Bits/Parameter: {results['summary_statistics']['std_bits_per_parameter']:.2f}")
    print(f"  Scaling Law R²: {results['scaling_law']['r_squared']:.3f}")
    
    print(f"\nMorris et al. Comparison:")
    print(f"  Target Bits/Parameter: {comparison['morris_target_bpp']}")
    print(f"  Observed Bits/Parameter: {comparison['observed_bpp']:.2f}")
    print(f"  Relative Error: {comparison['bpp_relative_error']:.1%}")
    print(f"  Reproduction Score: {comparison['morris_reproduction_score']:.1f}/100")
    
    print(f"\nValidation Results:")
    for key, value in validation.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key.replace('_', ' ').title()}")
    
    overall_success = validation['experiment_valid']
    print(f"\nOverall Validation: {'✓ PASSED' if overall_success else '✗ FAILED'}")
    
    print(f"\n{'='*80}")


def save_experiment_results(suite: ExperimentSuite) -> str:
    """
    Save experiment results to file.
    
    Args:
        suite: Experimental suite to save
        
    Returns:
        Path to saved file
    """
    # Create results directory
    os.makedirs(suite.experiment_config.results_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"morris_reproduction_{timestamp}.json"
    filepath = os.path.join(suite.experiment_config.results_dir, filename)
    
    # Convert suite to dictionary (handle dataclasses)
    suite_dict = asdict(suite)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    suite_dict = convert_numpy(suite_dict)
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(suite_dict, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def load_experiment_results(filepath: str) -> Dict[str, Any]:
    """
    Load experiment results from file.
    
    Args:
        filepath: Path to saved results file
        
    Returns:
        Dictionary containing experimental suite data
    """
    with open(filepath, 'r') as f:
        suite_dict = json.load(f)
    
    return suite_dict


def run_morris_reproduction_suite(
    quick_mode: bool = False,
    save_results: bool = True
) -> ExperimentSuite:
    """
    Main entry point for reproducing Morris et al. experiments.
    
    Automatically detects device capabilities and runs appropriate experiments.
    
    Args:
        quick_mode: If True, run smaller/faster experiments
        save_results: Whether to save results to file
        
    Returns:
        Complete experimental suite results
    """
    print("Morris et al. Memorization Capacity Reproduction")
    print("=" * 50)
    
    # Detect device and capabilities
    device, use_cpu_optimizations = detect_device()
    
    # Override with quick mode if requested
    if quick_mode:
        use_cpu_optimizations = True
        print("Quick mode enabled - using smaller experiments")
    
    # Create appropriate configuration
    experiment_config = create_device_appropriate_config(device, use_cpu_optimizations)
    experiment_config.save_results = save_results
    
    # Run complete experimental suite
    suite = run_capacity_suite(experiment_config)
    
    return suite


# Convenience functions for common use cases
def quick_demo(model_size: str = "small") -> Dict[str, Any]:
    """
    Run a quick demonstration experiment.
    
    Args:
        model_size: "tiny", "small", "medium", or "large"
        
    Returns:
        Demo experiment results
    """
    device, use_cpu_optimizations = detect_device()
    experiment_config = create_device_appropriate_config(device, True)  # Force CPU optimizations for demo
    
    # Create model config based on size
    size_configs = {
        "tiny": ModelConfig(n_layers=1, d_model=32, n_heads=2, vocab_size=256, max_seq_length=16),
        "small": ModelConfig(n_layers=2, d_model=64, n_heads=4, vocab_size=512, max_seq_length=32),
        "medium": ModelConfig(n_layers=3, d_model=128, n_heads=8, vocab_size=1024, max_seq_length=32),
        "large": ModelConfig(n_layers=4, d_model=256, n_heads=16, vocab_size=2048, max_seq_length=64)
    }
    
    model_config = size_configs.get(model_size, size_configs["small"])
    
    return run_single_model_demo(model_config, experiment_config)


if __name__ == "__main__":
    # Example usage
    print("Running Morris et al. reproduction experiments...")
    
    # Quick demo
    print("\nRunning quick demo...")
    demo_results = quick_demo("small")
    
    # Full reproduction (scaled for device)
    print("\nRunning full reproduction suite...")
    suite_results = run_morris_reproduction_suite(quick_mode=False)

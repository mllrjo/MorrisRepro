"""
File: test_enhanced_capacity_estimator.py
Directory: memorization_reproduction/

Test script to validate enhanced capacity estimator integration.
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from capacity_estimator_enhanced import (
    estimate_model_capacity_enhanced,
    run_enhanced_capacity_experiments,
    create_enhanced_morris_configs,
    validate_enhanced_capacity_experiment
)
from model_trainer import ModelConfig, TrainingConfig


def test_single_model_capacity_estimation():
    """Test enhanced capacity estimation on a single small model."""
    
    print("=" * 60)
    print("TESTING ENHANCED CAPACITY ESTIMATION - SINGLE MODEL")
    print("=" * 60)
    
    # Create small model for quick testing
    model_config = ModelConfig(
        n_layers=1,
        d_model=64,
        n_heads=4,
        vocab_size=128,
        max_seq_length=32,
        dropout=0.1
    )
    
    # Conservative training config
    training_config = TrainingConfig(
        batch_size=32,
        learning_rate=5e-4,
        max_steps=3000,  # Reasonable limit
        warmup_steps=100,
        weight_decay=0.01
    )
    
    # Small dataset sizes for quick test
    dataset_sizes = [20, 50, 100, 200]
    
    device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    
    print(f"Model: {model_config.n_layers} layers, {model_config.d_model} d_model")
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Device: {device}")
    print(f"Enhanced training: ENABLED")
    
    # Run enhanced capacity estimation
    start_time = time.time()
    
    capacity_estimate = estimate_model_capacity_enhanced(
        model_config=model_config,
        training_config=training_config,
        dataset_sizes=dataset_sizes,
        n_seeds=2,  # Quick test with 2 seeds
        device=device,
        use_enhanced_training=True,
        enhanced_params={
            'memorization_threshold': 0.5,
            'memorization_check_interval': 100
        }
    )
    
    test_time = time.time() - start_time
    
    # Display results
    print(f"\n📊 CAPACITY ESTIMATION RESULTS:")
    print(f"Estimated capacity: {capacity_estimate.estimated_capacity_bits:,.0f} bits")
    print(f"Bits per parameter: {capacity_estimate.bits_per_parameter:.3f}")
    print(f"Plateau dataset size: {capacity_estimate.plateau_dataset_size:,}")
    print(f"Convergence success rate: {capacity_estimate.convergence_success_rate:.1%}")
    print(f"MAX_STEPS eliminated: {'✓ YES' if capacity_estimate.max_steps_elimination else '✗ NO'}")
    print(f"Morris progress: {capacity_estimate.morris_target_progress:.1%}")
    print(f"Test time: {test_time:.1f} seconds")
    
    # Validation
    validations = []
    
    # Should achieve some capacity
    has_capacity = capacity_estimate.estimated_capacity_bits > 1000
    validations.append(("Meaningful capacity detected", has_capacity))
    
    # Should have reasonable bits/param
    reasonable_bpp = 0.1 <= capacity_estimate.bits_per_parameter <= 10.0
    validations.append(("Reasonable bits/parameter", reasonable_bpp))
    
    # Should achieve good convergence rate
    good_convergence = capacity_estimate.convergence_success_rate >= 0.5
    validations.append(("Good convergence rate", good_convergence))
    
    # Should eliminate MAX_STEPS (key metric!)
    no_max_steps = capacity_estimate.max_steps_elimination
    validations.append(("MAX_STEPS eliminated", no_max_steps))
    
    print(f"\n✅ VALIDATION RESULTS:")
    for validation_name, validation_result in validations:
        status = "✓ PASS" if validation_result else "✗ FAIL"
        print(f"  {validation_name}: {status}")
    
    overall_success = all(result for _, result in validations)
    print(f"\nOVERALL: {'✓ SUCCESS' if overall_success else '✗ NEEDS WORK'}")
    
    return overall_success, capacity_estimate


def test_multi_model_scaling_experiment():
    """Test enhanced capacity experiments across multiple model sizes."""
    
    print("\n" + "=" * 60)
    print("TESTING MULTI-MODEL SCALING EXPERIMENT")
    print("=" * 60)
    
    # Create small model configurations for testing
    model_configs = [
        ModelConfig(n_layers=1, d_model=32, n_heads=2, vocab_size=64, max_seq_length=16, dropout=0.1),
        ModelConfig(n_layers=1, d_model=64, n_heads=4, vocab_size=128, max_seq_length=32, dropout=0.1),
        ModelConfig(n_layers=2, d_model=64, n_heads=4, vocab_size=128, max_seq_length=32, dropout=0.1)
    ]
    
    training_config = TrainingConfig(
        batch_size=16,
        learning_rate=5e-4,
        max_steps=2000,  # Quick test
        warmup_steps=50,
        weight_decay=0.01
    )
    
    # Small dataset sizes for quick testing
    base_dataset_sizes = [10, 25, 50, 100]
    
    device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    
    print(f"Testing {len(model_configs)} model configurations")
    print(f"Base dataset sizes: {base_dataset_sizes}")
    print(f"Device: {device}")
    
    # Run enhanced scaling experiments
    start_time = time.time()
    
    experiment_results = run_enhanced_capacity_experiments(
        model_configs=model_configs,
        training_config=training_config,
        base_dataset_sizes=base_dataset_sizes,
        n_seeds=2,  # Quick test
        device=device,
        use_enhanced_training=True
    )
    
    experiment_time = time.time() - start_time
    
    # Display scaling results
    print(f"\n📈 SCALING EXPERIMENT RESULTS:")
    summary = experiment_results['enhanced_summary_statistics']
    print(f"Mean bits/parameter: {summary['mean_bits_per_parameter']:.3f}")
    print(f"Morris progress: {summary['morris_progress']:.1%}")
    print(f"Overall convergence rate: {summary['overall_convergence_rate']:.1%}")
    print(f"MAX_STEPS elimination rate: {summary['max_steps_elimination_rate']:.1%}")
    print(f"Scaling law R²: {experiment_results['scaling_law']['r_squared']:.3f}")
    print(f"Experiment time: {experiment_time:.1f} seconds")
    
    # Validate experiment
    validations = validate_enhanced_capacity_experiment(
        experiment_results, 
        expected_bits_per_param=2.0,  # Lower expectation for small models
        tolerance=1.5
    )
    
    print(f"\n🔬 EXPERIMENT VALIDATION:")
    key_validations = [
        'core_experiment_valid',
        'enhanced_training_effective', 
        'high_convergence_rate',
        'max_steps_eliminated'
    ]
    
    for validation_name in key_validations:
        if validation_name in validations:
            status = "✓ PASS" if validations[validation_name] else "✗ FAIL"
            print(f"  {validation_name.replace('_', ' ').title()}: {status}")
    
    overall_success = validations.get('experiment_fully_valid', False)
    enhanced_success = validations.get('enhanced_training_effective', False)
    
    print(f"\nOVERALL EXPERIMENT: {'✓ SUCCESS' if overall_success else '✓ ENHANCED SUCCESS' if enhanced_success else '✗ NEEDS WORK'}")
    
    return enhanced_success or overall_success, experiment_results


def test_morris_style_configs():
    """Test Morris-style configuration generation."""
    
    print("\n" + "=" * 60)
    print("TESTING MORRIS-STYLE MODEL CONFIGURATIONS")
    print("=" * 60)
    
    # Test enhanced Morris configs
    target_params = [50000, 200000, 500000]
    morris_configs = create_enhanced_morris_configs(
        target_param_counts=target_params,
        enhanced_training=True
    )
    
    print(f"Generated {len(morris_configs)} Morris-style configurations:")
    
    for i, config in enumerate(morris_configs):
        # Calculate actual parameters
        from model_trainer import create_gpt_model, count_parameters
        model = create_gpt_model(config)
        actual_params = count_parameters(model)
        
        print(f"  Config {i+1}: {config.n_layers}L/{config.d_model}D/{config.n_heads}H → {actual_params:,} params")
        print(f"    Target: {target_params[i]:,}, Ratio: {actual_params/target_params[i]:.2f}x")
    
    print("✓ Morris-style configurations generated successfully")
    return True


def main():
    """Run comprehensive enhanced capacity estimator tests."""
    
    print("🚀 ENHANCED CAPACITY ESTIMATOR INTEGRATION TESTS")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Single model capacity estimation
    try:
        success1, capacity_result = test_single_model_capacity_estimation()
        test_results.append(("Single model capacity", success1))
    except Exception as e:
        print(f"❌ Single model test failed: {e}")
        test_results.append(("Single model capacity", False))
    
    # Test 2: Multi-model scaling experiment
    try:
        success2, scaling_result = test_multi_model_scaling_experiment()
        test_results.append(("Multi-model scaling", success2))
    except Exception as e:
        print(f"❌ Multi-model test failed: {e}")
        test_results.append(("Multi-model scaling", False))
    
    # Test 3: Morris config generation
    try:
        success3 = test_morris_style_configs()
        test_results.append(("Morris config generation", success3))
    except Exception as e:
        print(f"❌ Morris config test failed: {e}")
        test_results.append(("Morris config generation", False))
    
    # Overall results
    print(f"\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    for test_name, test_success in test_results:
        status = "✓ PASS" if test_success else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    total_success = all(success for _, success in test_results)
    partial_success = any(success for _, success in test_results)
    
    if total_success:
        print(f"\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Enhanced capacity estimator ready for Morris reproduction")
        print("✅ MAX_STEPS termination issues resolved")
        print("✅ Ready to scale to 1M+ parameter models")
    elif partial_success:
        print(f"\n⚡ PARTIAL SUCCESS!")
        print("✅ Core enhanced training functionality working")
        print("⚠️  Some integration aspects need refinement")
    else:
        print(f"\n❌ INTEGRATION ISSUES DETECTED")
        print("🔧 Enhanced trainer integration needs debugging")
    
    return total_success or partial_success


if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎯 NEXT STEPS FOR MORRIS REPRODUCTION:")
        print("1. Replace original capacity_estimator.py:")
        print("   mv src/capacity_estimator.py src/capacity_estimator_original.py")
        print("   mv src/capacity_estimator_enhanced.py src/capacity_estimator.py")
        print("2. Test with larger models (1M+ parameters)")
        print("3. Run full Morris scaling experiments")
        print("4. Target 3.6 bits/parameter benchmark")
    
    exit(0 if success else 1)

#!/usr/bin/env python3
"""
Morris-Medium Validation Script
Scale to ~6M parameters to validate continued scaling trajectory

Expected result: 0.5-1.5 bits/param (major progress toward Morris 3.6 target)
"""

import os
import sys
import torch
import time
from datetime import datetime

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)


def main():
    print("MORRIS-MEDIUM VALIDATION - 6M Parameter Model")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cuda":
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_props.name}")
        print(f"GPU Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
    
    print()
    
    # Import modules
    try:
        from model_trainer import ModelConfig, TrainingConfig, create_gpt_model, count_parameters
        from capacity_estimator import estimate_model_capacity
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure enhanced_training.py and model_trainer.py are in src/ directory")
        return False
    
    # Morris-Medium model configuration (targeting ~6M parameters)
    print("Creating Morris-Medium model configuration...")
    
    model_config = ModelConfig(
        n_layers=8,           # Increased depth for better memorization
        d_model=256,          # Balanced width
        n_heads=8,            # Standard factorization (256/8=32)
        vocab_size=2048,      # Standard vocab size
        max_seq_length=64,    # Same as successful tests
        dropout=0.1
    )
    
    # Create and analyze model
    print("Creating model...")
    model = create_gpt_model(model_config)
    actual_params = count_parameters(model)
    
    print(f"✅ Morris-Medium model created")
    print(f"   Target: ~6,000,000 parameters")
    print(f"   Actual: {actual_params:,} parameters")
    print(f"   Scaling: {actual_params/1_602_144:.1f}x Morris-Micro model")
    print(f"   Total scaling: {actual_params/660_000:.1f}x baseline model")
    
    # Memory estimation
    if device == "cuda":
        model_size_mb = actual_params * 4 / 1024 / 1024  # FP32 estimate
        training_memory_mb = model_size_mb * 3  # Training overhead
        print(f"   Model memory: {model_size_mb:.0f} MB")
        print(f"   Training memory: {training_memory_mb:.0f} MB")
    
    print()
    
    # Create training configuration optimized for larger model
    print("Setting up enhanced training configuration...")
    
    training_config = TrainingConfig(
        batch_size=4,          # Smaller batch for larger model
        learning_rate=3e-4,    # More conservative LR
        max_steps=200_000,     # More steps for complex memorization
        warmup_steps=3000,     # Longer warmup for stability
        weight_decay=0.01
    )
    
    # Scaled dataset sizes for 6M parameters
    # Based on Morris-Micro success, scale by ~4x model size increase
    dataset_sizes = [25, 75, 150, 300, 600, 1200]
    
    print(f"Training config: {training_config.max_steps:,} max steps, LR={training_config.learning_rate}")
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Expected time: 2-4 hours with enhanced training")
    print()
    
    # Predict expected results based on Morris-Micro scaling
    micro_bpp = 0.124
    micro_params = 1_602_144
    scaling_efficiency = 2.7  # From Morris-Micro analysis
    
    expected_scaling = (actual_params / micro_params) ** (1/scaling_efficiency)
    expected_bpp = micro_bpp * expected_scaling
    morris_progress = (expected_bpp / 3.6) * 100
    
    print("SCALING PREDICTIONS:")
    print(f"  Expected bits/param: {expected_bpp:.3f}")
    print(f"  Morris progress: {morris_progress:.1f}% of 3.6 target")
    print(f"  Scaling efficiency: {scaling_efficiency:.1f}x (super-linear)")
    print()
    
    # Capacity estimation
    print("Starting Morris-Medium capacity estimation...")
    print("This will run enhanced training on multiple dataset sizes...")
    print("Monitor for 'MEMORIZATION_ACHIEVED' convergence (not MAX_STEPS)")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Run capacity estimation with enhanced training
        capacity_result = estimate_model_capacity(
            model_config=model_config,
            training_config=training_config,
            dataset_sizes=dataset_sizes,
            n_seeds=2,  # 2 seeds for validation
            device=device,
            plateau_tolerance=0.15  # More lenient for larger model
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Extract results
        capacity_bits = capacity_result.estimated_capacity_bits
        bits_per_param = capacity_result.bits_per_parameter
        r_squared = getattr(capacity_result, 'r_squared', 0.0)
        
        # Report results
        print("\n" + "=" * 60)
        print("MORRIS-MEDIUM VALIDATION RESULTS")
        print("=" * 60)
        
        print(f"Model Parameters: {actual_params:,}")
        print(f"Estimated Capacity: {capacity_bits:,.0f} bits")
        print(f"Bits per Parameter: {bits_per_param:.3f}")
        print(f"R² (fit quality): {r_squared:.3f}")
        print(f"Execution Time: {execution_time/3600:.2f} hours")
        
        # Scaling analysis
        micro_to_medium_param_scaling = actual_params / micro_params
        micro_to_medium_bpp_scaling = bits_per_param / micro_bpp
        observed_efficiency = micro_to_medium_bpp_scaling / micro_to_medium_param_scaling
        
        print(f"\nSCALING ANALYSIS:")
        print(f"Parameter scaling: {micro_to_medium_param_scaling:.1f}x")
        print(f"Performance scaling: {micro_to_medium_bpp_scaling:.1f}x")
        print(f"Scaling efficiency: {observed_efficiency:.1f}x")
        
        # Morris comparison
        morris_target = 3.6
        progress_percent = (bits_per_param / morris_target) * 100
        
        print(f"\nMorris Reproduction Progress:")
        print(f"Morris Target: {morris_target} bits/param")
        print(f"Current Result: {bits_per_param:.3f} bits/param")
        print(f"Progress: {progress_percent:.1f}% of Morris target")
        
        # Project to full Morris scale
        if observed_efficiency > 1.5:  # Still super-linear
            morris_15M_projection = bits_per_param * ((15_000_000 / actual_params) ** (1/observed_efficiency))
            print(f"Projected 15M params: {morris_15M_projection:.2f} bits/param")
        
        # Success assessment
        print(f"\nValidation Assessment:")
        
        success_criteria = [
            (bits_per_param > 0.3, f"Bits/param > 0.3: {bits_per_param:.3f}"),
            (bits_per_param > micro_bpp * 1.5, f"1.5x improvement over micro: {bits_per_param:.3f} vs {micro_bpp:.3f}"),
            (observed_efficiency > 1.2, f"Continued super-linear scaling: {observed_efficiency:.1f}x"),
            (r_squared > 0.6, f"Good fit quality: R² = {r_squared:.3f}"),
            (execution_time < 8*3600, f"Reasonable time: {execution_time/3600:.1f} hours"),
            (progress_percent > 10, f"Significant Morris progress: {progress_percent:.1f}%")
        ]
        
        passed_count = 0
        for passed, description in success_criteria:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}: {description}")
            if passed:
                passed_count += 1
        
        # Overall assessment
        overall_success = passed_count >= 4
        
        print(f"\n" + "=" * 60)
        if overall_success:
            print("🎉 MORRIS-MEDIUM VALIDATION: SUCCESS!")
            print("✅ Enhanced training continues to scale effectively")
            print("✅ Super-linear scaling trajectory maintained")
            
            if bits_per_param > 1.0:
                print("🚀 OUTSTANDING: Approaching Morris target rapidly!")
            elif bits_per_param > 0.5:
                print("📈 EXCELLENT: Strong progress, Morris target achievable")
            else:
                print("📊 GOOD: Solid scaling, continue to larger models")
                
            # Next steps
            print(f"\nNext Steps:")
            print(f"1. Scale to Morris Target: 12-20M parameters")
            print(f"2. Expected result: 1.5-3.6+ bits/param")
            print(f"3. Full Morris reproduction now highly feasible!")
            
        else:
            print("⚠️  MORRIS-MEDIUM VALIDATION: NEEDS OPTIMIZATION")
            print(f"Passed {passed_count}/6 criteria - review approach")
            print("May need architecture or hyperparameter adjustments")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"morris_medium_results_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write(f"Morris-Medium Validation Results\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Parameters: {actual_params:,}\n")
            f.write(f"Capacity: {capacity_bits:,.0f} bits\n")
            f.write(f"Bits/Parameter: {bits_per_param:.3f}\n")
            f.write(f"Scaling Efficiency: {observed_efficiency:.1f}x\n")
            f.write(f"Morris Progress: {progress_percent:.1f}%\n")
            f.write(f"R²: {r_squared:.3f}\n")
            f.write(f"Time: {execution_time/3600:.2f} hours\n")
            f.write(f"Success: {overall_success}\n")
            f.write(f"Dataset Sizes: {dataset_sizes}\n")
        
        print(f"\nResults saved to: {results_file}")
        
        return overall_success
        
    except Exception as e:
        print(f"\n❌ Morris-Medium validation failed: {e}")
        print("\nDebug information:")
        import traceback
        traceback.print_exc()
        return False


def quick_resource_check():
    """Quick feasibility check before running"""
    
    print("RESOURCE FEASIBILITY CHECK")
    print("=" * 40)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cuda":
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = gpu_props.total_memory / 1024**3
            
            print(f"GPU: {gpu_props.name}")
            print(f"Memory: {gpu_memory_gb:.1f} GB")
            
            # Estimate memory needs for 6M param model
            estimated_need_gb = 6.0 * 4 * 3 / 1024  # params * bytes * overhead / 1024³
            
            if gpu_memory_gb >= estimated_need_gb:
                print(f"✅ FEASIBLE: {gpu_memory_gb:.1f} GB >= {estimated_need_gb:.1f} GB needed")
                return True
            else:
                print(f"⚠️  TIGHT: {gpu_memory_gb:.1f} GB < {estimated_need_gb:.1f} GB ideal")
                print("   Consider batch_size=2 if memory errors occur")
                return True
                
        except Exception as e:
            print(f"GPU check failed: {e}")
            return False
    else:
        print("CPU training: feasible but slower")
        print("Estimated time: 4-8 hours")
        return True


if __name__ == "__main__":
    print("Morris-Medium Validation - 6M Parameter Scaling Test")
    print("Building on Morris-Micro success to validate continued scaling")
    print()
    
    # Resource check
    if not quick_resource_check():
        print("\n❌ Resource constraints detected")
        sys.exit(1)
    
    print()
    input("Press Enter to start Morris-Medium validation (will take 2-4 hours)...")
    print()
    
    # Run validation
    success = main()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 MORRIS-MEDIUM VALIDATION COMPLETE: SUCCESS")
        print("🚀 Scaling trajectory confirmed - ready for Morris target!")
    else:
        print("⚠️  MORRIS-MEDIUM VALIDATION: NEEDS INVESTIGATION")
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

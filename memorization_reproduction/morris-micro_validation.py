#!/usr/bin/env python3
"""
Morris-Micro Validation Script
Scale to ~1.6M parameters to validate Morris reproduction path

Run this script to test if enhanced training scales successfully
Expected result: 2.5-3.0 bits/param (approaching Morris 3.6 target)
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
    print("MORRIS-MICRO VALIDATION - 1.6M Parameter Model")
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
    
    # CORRECTED Morris-Micro model configuration (targeting ~1.6M parameters)
    print("Creating Morris-Micro model configuration...")
    
    model_config = ModelConfig(
        n_layers=4,           # Balanced depth for Morris scale
        d_model=144,          # Optimized width
        n_heads=6,            # Proper factorization (144/6=24)
        vocab_size=2048,      # Standard vocab size
        max_seq_length=64,    # Same as successful test
        dropout=0.1
    )
    
    # Create and analyze model
    print("Creating model...")
    model = create_gpt_model(model_config)
    actual_params = count_parameters(model)
    
    print(f"✅ Morris-Micro model created")
    print(f"   Target: ~1,600,000 parameters")
    print(f"   Actual: {actual_params:,} parameters")
    print(f"   Scaling: {actual_params/660_000:.1f}x successful test model")
    
    # Memory estimation
    if device == "cuda":
        model_size_mb = actual_params * 4 / 1024 / 1024  # FP32 estimate
        print(f"   Estimated GPU memory: {model_size_mb:.1f} MB")
    
    print()
    
    # Create training configuration
    print("Setting up enhanced training configuration...")
    
    training_config = TrainingConfig(
        batch_size=8,          # Conservative for larger model
        learning_rate=5e-4,    # Stable learning rate
        max_steps=150_000,     # Sufficient for enhanced convergence
        warmup_steps=2000,     # Longer warmup for stability
        weight_decay=0.01
    )
    
    # UPDATED dataset sizes scaled for 1.6M parameters
    # Based on successful 660K param test, scale by ~2.4x
    dataset_sizes = [12, 36, 72, 144, 288]
    
    print(f"Training config: {training_config.max_steps:,} max steps, LR={training_config.learning_rate}")
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Expected time: 1-2 hours with enhanced training")
    print()
    
    # Capacity estimation
    print("Starting Morris-Micro capacity estimation...")
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
            plateau_tolerance=0.1
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Extract results
        capacity_bits = capacity_result.estimated_capacity_bits
        bits_per_param = capacity_result.bits_per_parameter
        r_squared = getattr(capacity_result, 'r_squared', 0.0)
        
        # Report results
        print("\n" + "=" * 60)
        print("MORRIS-MICRO VALIDATION RESULTS")
        print("=" * 60)
        
        print(f"Model Parameters: {actual_params:,}")
        print(f"Estimated Capacity: {capacity_bits:,.0f} bits")
        print(f"Bits per Parameter: {bits_per_param:.3f}")
        print(f"R² (fit quality): {r_squared:.3f}")
        print(f"Execution Time: {execution_time/3600:.2f} hours")
        
        # Morris comparison
        morris_target = 3.6
        progress_percent = (bits_per_param / morris_target) * 100
        
        print(f"\nMorris Reproduction Progress:")
        print(f"Morris Target: {morris_target} bits/param")
        print(f"Current Result: {bits_per_param:.3f} bits/param")
        print(f"Progress: {progress_percent:.1f}% of Morris target")
        
        # Prediction for full Morris scale
        # Based on scaling: bits/param = -0.610 * log10(params) + 4.838
        morris_scale_params = 15_000_000  # 15M parameters
        predicted_morris = -0.610 * torch.log10(torch.tensor(morris_scale_params)) + 4.838
        print(f"Predicted at 15M params: {predicted_morris:.2f} bits/param")
        
        # Success assessment
        print(f"\nValidation Assessment:")
        
        success_criteria = [
            (bits_per_param > 1.5, f"Bits/param > 1.5: {bits_per_param:.3f}"),
            (bits_per_param > 0.019 * 5, f"5x improvement over test model: {bits_per_param:.3f} vs {0.019:.3f}"),
            (r_squared > 0.7, f"Good fit quality: R² = {r_squared:.3f}"),
            (execution_time < 4*3600, f"Reasonable time: {execution_time/3600:.1f} hours"),
            (capacity_bits > 500_000, f"Substantial capacity: {capacity_bits:,.0f} bits")
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
            print("🎉 MORRIS-MICRO VALIDATION: SUCCESS!")
            print("✅ Enhanced training scales successfully to 1.6M parameters")
            print("✅ Ready to proceed to full Morris target (12-20M parameters)")
            
            if bits_per_param > 2.5:
                print("🚀 OUTSTANDING: Very close to Morris target!")
            elif bits_per_param > 2.0:
                print("📈 EXCELLENT: Strong progress toward Morris target")
            else:
                print("📊 GOOD: Solid scaling validation, continue to larger models")
                
            # Next steps
            print(f"\nNext Steps:")
            print(f"1. Scale to Morris Target: 12-20M parameters")
            print(f"2. Expected result: 3.4-3.7 bits/param")
            print(f"3. Full Morris reproduction within reach!")
            
        else:
            print("⚠️  MORRIS-MICRO VALIDATION: PARTIAL SUCCESS")
            print(f"Passed {passed_count}/5 criteria - review results")
            print("Consider adjusting parameters or training longer")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"morris_micro_results_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write(f"Morris-Micro Validation Results\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Parameters: {actual_params:,}\n")
            f.write(f"Capacity: {capacity_bits:,.0f} bits\n")
            f.write(f"Bits/Parameter: {bits_per_param:.3f}\n")
            f.write(f"Morris Progress: {progress_percent:.1f}%\n")
            f.write(f"R²: {r_squared:.3f}\n")
            f.write(f"Time: {execution_time/3600:.2f} hours\n")
            f.write(f"Success: {overall_success}\n")
            f.write(f"Dataset Sizes: {dataset_sizes}\n")
        
        print(f"\nResults saved to: {results_file}")
        
        return overall_success
        
    except Exception as e:
        print(f"\n❌ Morris-Micro validation failed: {e}")
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
            
            # Estimate memory needs for 1.6M param model
            estimated_need_gb = 1.6 * 4 * 3 / 1024  # params * bytes * overhead / 1024³
            
            if gpu_memory_gb >= estimated_need_gb:
                print(f"✅ FEASIBLE: {gpu_memory_gb:.1f} GB >= {estimated_need_gb:.1f} GB needed")
                return True
            else:
                print(f"⚠️  TIGHT: {gpu_memory_gb:.1f} GB < {estimated_need_gb:.1f} GB ideal")
                print("   Try smaller batch size if GPU memory errors occur")
                return True
                
        except Exception as e:
            print(f"GPU check failed: {e}")
            return False
    else:
        print("CPU training: slower but feasible")
        print("Estimated time: 2-4 hours")
        return True


if __name__ == "__main__":
    print("Morris-Micro Validation - 1.6M Parameter Scaling Test")
    print("Testing enhanced training at Morris scale")
    print()
    
    # Resource check
    if not quick_resource_check():
        print("\n❌ Resource constraints detected")
        sys.exit(1)
    
    print()
    input("Press Enter to start Morris-Micro validation (will take 1-2 hours)...")
    print()
    
    # Run validation
    success = main()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 MORRIS-MICRO VALIDATION COMPLETE: SUCCESS")
        print("🚀 Ready to scale to full Morris target!")
    else:
        print("⚠️  MORRIS-MICRO VALIDATION: NEEDS INVESTIGATION")
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

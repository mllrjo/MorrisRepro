#!/usr/bin/env python3
"""
Dataset Scale Test - Morris-Micro with Large Datasets
Test if larger datasets unlock the true memorization capacity

HYPOTHESIS: Morris achieved 3.6 bits/param by testing much larger datasets
BASELINE: Morris-Micro with 288 samples = 0.124 bits/param
TEST: Morris-Micro with 2,880+ samples = ? bits/param
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
    print("DATASET SCALE TEST - MORRIS-MICRO LARGE DATASETS")
    print("=" * 60)
    print("Testing PROVEN Morris-Micro architecture with LARGE datasets")
    print("Hypothesis: Larger datasets unlock true memorization capacity")
    print("Baseline: Morris-Micro 288 samples = 0.124 bits/param")
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
        return False
    
    # IDENTICAL Morris-Micro configuration (proven best architecture)
    print("Using PROVEN Morris-Micro architecture...")
    
    model_config = ModelConfig(
        n_layers=4,           # PROVEN best architecture
        d_model=144,          # PROVEN best architecture
        n_heads=6,            # PROVEN best architecture
        vocab_size=2048,      # IDENTICAL to all tests
        max_seq_length=64,    # IDENTICAL to all tests
        dropout=0.1
    )
    
    # Create and analyze model
    print("Creating model...")
    model = create_gpt_model(model_config)
    actual_params = count_parameters(model)
    
    print(f"✅ Morris-Micro model created")
    print(f"   Parameters: {actual_params:,}")
    print(f"   Architecture: {model_config.n_layers}L×{model_config.d_model}D×{model_config.n_heads}H")
    print(f"   Proven performance: 0.124 bits/param with small datasets")
    
    print()
    
    # MODIFIED training configuration for larger datasets
    print("Setting up training configuration for LARGE datasets...")
    
    training_config = TrainingConfig(
        batch_size=8,          # Keep same for consistency
        learning_rate=5e-4,    # Keep same for consistency  
        max_steps=200_000,     # More steps for larger datasets
        warmup_steps=3000,     # Longer warmup for larger datasets
        weight_decay=0.01
    )
    
    # SCALED UP dataset sizes - 10x larger than Morris-Micro baseline
    original_sizes = [12, 36, 72, 144, 288]
    dataset_sizes = [120, 360, 720, 1440, 2880]  # 10x scale up
    
    print(f"Training config: {training_config.max_steps:,} max steps, LR={training_config.learning_rate}")
    print(f"Original dataset sizes: {original_sizes}")
    print(f"SCALED dataset sizes:   {dataset_sizes}")
    print(f"Scale factor: 10x larger datasets")
    print(f"Expected time: 2-4 hours (larger datasets take longer)")
    print()
    
    print("🔬 DATASET SCALE HYPOTHESIS:")
    print("   BASELINE: Morris-Micro 288 samples → 0.124 bits/param")
    print("   TEST: Morris-Micro 2,880 samples → ? bits/param")
    print("   MORRIS TARGET: ~8,000+ samples → 3.6 bits/param")
    print("   EXPECTATION: Higher bits/param with larger datasets")
    print()
    
    print("🎯 SUCCESS CRITERIA:")
    print("   - Significantly higher than 0.124 bits/param baseline")
    print("   - Good scaling curve fit (R² > 0.7)")
    print("   - Clear capacity plateau at large dataset sizes")
    print("   - Progress toward Morris 3.6 bits/param target")
    print()
    
    # Capacity estimation with large datasets
    print("Starting LARGE DATASET capacity estimation...")
    print("🔍 HYPOTHESIS: Larger datasets reveal true model capacity")
    print("Monitor for 'MEMORIZATION_ACHIEVED' convergence on large datasets")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Run capacity estimation with scaled datasets
        capacity_result = estimate_model_capacity(
            model_config=model_config,
            training_config=training_config,
            dataset_sizes=dataset_sizes,
            n_seeds=2,  # Keep same for consistency
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
        print("DATASET SCALE TEST RESULTS")
        print("=" * 60)
        
        print(f"Model Parameters: {actual_params:,}")
        print(f"Estimated Capacity: {capacity_bits:,.0f} bits")
        print(f"Bits per Parameter: {bits_per_param:.6f}")
        print(f"R² (fit quality): {r_squared:.3f}")
        print(f"Execution Time: {execution_time/3600:.2f} hours")
        
        # CRITICAL COMPARISON with baseline
        baseline_bpp = 0.124  # Morris-Micro baseline with small datasets
        improvement_factor = bits_per_param / baseline_bpp
        
        print(f"\n🔍 DATASET SCALE COMPARISON:")
        print(f"Small datasets (288): {baseline_bpp:.6f} bits/param")
        print(f"Large datasets (2880): {bits_per_param:.6f} bits/param")
        print(f"Improvement factor: {improvement_factor:.1f}x")
        
        # Assess the dataset scale hypothesis
        print(f"\n📊 DATASET SCALE HYPOTHESIS VALIDATION:")
        if improvement_factor > 5.0:
            print("🎉 MAJOR VALIDATION: Large datasets much better!")
            print("✅ Dataset scale was the PRIMARY limiting factor")
            print("🚀 Continue scaling to Morris target (8K+ samples)")
        elif improvement_factor > 2.0:
            print("📈 MODERATE VALIDATION: Large datasets significantly better")
            print("✅ Dataset scale is A limiting factor")
            print("📊 Test even larger datasets + other factors")
        elif improvement_factor > 1.3:
            print("📊 WEAK VALIDATION: Large datasets somewhat better")
            print("⚠️  Dataset scale has modest impact")
            print("🔍 Other factors still important")
        else:
            print("❌ HYPOTHESIS REJECTED: Large datasets not better")
            print("⚠️  Dataset scale is NOT the limiting factor")
            print("🔍 Focus on measurement methodology or other factors")
        
        # Scaling curve quality assessment
        print(f"\nScaling Curve Quality:")
        if r_squared > 0.8:
            print("✅ EXCELLENT fit: Clear scaling law identified")
        elif r_squared > 0.6:
            print("📈 GOOD fit: Reasonable scaling behavior")
        elif r_squared > 0.3:
            print("📊 MODERATE fit: Some scaling signal")
        else:
            print("⚠️  POOR fit: No clear scaling relationship")
        
        # Morris target projection
        morris_target = 3.6
        progress_percent = (bits_per_param / morris_target) * 100
        
        print(f"\n🎯 MORRIS TARGET PROGRESS:")
        print(f"Current result: {bits_per_param:.6f} bits/param")
        print(f"Morris target: {morris_target} bits/param")
        print(f"Progress: {progress_percent:.1f}% of Morris target")
        
        # Estimate what dataset size might reach Morris target
        if improvement_factor > 1.5 and r_squared > 0.5:
            remaining_factor = morris_target / bits_per_param
            estimated_dataset_size = 2880 * remaining_factor
            print(f"Estimated dataset size for Morris target: ~{estimated_dataset_size:,.0f} samples")
            
            if estimated_dataset_size < 50_000:
                print("✅ FEASIBLE: Morris target within reasonable dataset scale")
            else:
                print("⚠️  CHALLENGING: Would require very large datasets")
        
        # Success criteria assessment
        success_criteria = [
            (improvement_factor > 1.5, f"Significant improvement: {improvement_factor:.1f}x"),
            (bits_per_param > 0.3, f"Substantial capacity: {bits_per_param:.6f} bits/param"),
            (r_squared > 0.5, f"Good scaling fit: R² = {r_squared:.3f}"),
            (progress_percent > 10, f"Morris progress: {progress_percent:.1f}%"),
            (execution_time < 6*3600, f"Reasonable time: {execution_time/3600:.1f} hours")
        ]
        
        passed_count = sum(1 for passed, _ in success_criteria if passed)
        
        print(f"\nValidation Assessment:")
        for passed, description in success_criteria:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}: {description}")
        
        overall_success = passed_count >= 3
        
        print(f"\n" + "=" * 60)
        if overall_success:
            print("🎉 DATASET SCALE TEST: SUCCESS!")
            
            if improvement_factor > 3.0:
                print("✅ Dataset scale was a major factor in limitations")
                print("🚀 Next: Scale to Morris-level datasets (8K+ samples)")
                print("🎯 Morris 3.6 bits/param target now achievable")
            else:
                print("✅ Dataset scale provides meaningful improvement")
                print("📊 Continue investigating remaining factors")
                
        else:
            print("⚠️  DATASET SCALE TEST: MIXED RESULTS")
            print("🔍 Dataset scale may not be the primary limitation")
            print("📊 Investigate measurement methodology or other factors")
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"dataset_scale_test_results_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write(f"Dataset Scale Test Results\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Model: Morris-Micro {actual_params:,} params\n")
            f.write(f"Large dataset bits/param: {bits_per_param:.6f}\n")
            f.write(f"Small dataset baseline: {baseline_bpp:.6f}\n")
            f.write(f"Improvement factor: {improvement_factor:.1f}x\n")
            f.write(f"Morris progress: {progress_percent:.1f}%\n")
            f.write(f"R²: {r_squared:.3f}\n")
            f.write(f"Execution time: {execution_time/3600:.2f} hours\n")
            f.write(f"Dataset sizes: {dataset_sizes}\n")
        
        print(f"\nResults saved to: {results_file}")
        
        return overall_success
        
    except Exception as e:
        print(f"\n❌ Dataset scale test failed: {e}")
        print("\nDebug information:")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Dataset Scale Test - Morris-Micro Large Datasets")
    print("Testing if larger datasets unlock true memorization capacity")
    print()
    
    print("🔍 HYPOTHESIS:")
    print("Morris achieved 3.6 bits/param by testing much larger datasets")
    print("Our 288 samples may be too small to find true model capacity")
    print()
    
    input("Press Enter to start dataset scale test (2-4 hours)...")
    print()
    
    # Run test
    success = main()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 DATASET SCALE TEST COMPLETE")
        print("Key insight obtained about dataset size impact on capacity measurement")
    else:
        print("⚠️  DATASET SCALE TEST: INVESTIGATION NEEDED")
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

#!/usr/bin/env python3
"""
GPT-2 Standard Architecture Test
Test standard GPT-2 proportions vs our custom Morris-Micro ratios

HYPOTHESIS: Standard GPT-2 ratios will achieve higher bits/param than our custom 4L×144D×6H
BASELINE: Morris-Micro custom = 0.124 bits/param
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
    print("GPT-2 STANDARD ARCHITECTURE TEST")
    print("=" * 60)
    print("Testing STANDARD GPT-2 proportions vs custom Morris-Micro ratios")
    print("Baseline: Custom 4L×144D×6H = 0.124 bits/param")
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
    
    # STANDARD GPT-2 architecture configuration 
    print("Creating STANDARD GPT-2 architecture configuration...")
    
    model_config = ModelConfig(
        n_layers=4,           # Same depth as Morris-Micro for fair comparison
        d_model=256,          # Standard power-of-2 width (vs custom 144)
        n_heads=4,            # Standard head_dim=64 (256/4=64 vs 144/6=24)
        vocab_size=2048,      # IDENTICAL to Morris-Micro
        max_seq_length=64,    # IDENTICAL to Morris-Micro
        dropout=0.1
    )
    
    # Create and analyze model
    print("Creating model...")
    model = create_gpt_model(model_config)
    actual_params = count_parameters(model)
    
    print(f"✅ Standard GPT-2 model created")
    print(f"   Parameters: {actual_params:,}")
    print(f"   Architecture: {model_config.n_layers}L×{model_config.d_model}D×{model_config.n_heads}H")
    print(f"   Head dimension: {model_config.d_model // model_config.n_heads} (standard=64)")
    
    # Compare with Morris-Micro
    morris_micro_params = 1_602_144
    param_ratio = actual_params / morris_micro_params
    print(f"   vs Morris-Micro: {param_ratio:.1f}x parameters ({morris_micro_params:,})")
    
    print()
    
    # IDENTICAL training configuration (proven successful)
    print("Setting up IDENTICAL training configuration...")
    
    training_config = TrainingConfig(
        batch_size=8,          # IDENTICAL to Morris-Micro
        learning_rate=5e-4,    # IDENTICAL to Morris-Micro
        max_steps=150_000,     # IDENTICAL to Morris-Micro
        warmup_steps=2000,     # IDENTICAL to Morris-Micro
        weight_decay=0.01
    )
    
    # IDENTICAL dataset sizes (proven methodology)
    dataset_sizes = [12, 36, 72, 144, 288]  # IDENTICAL to Morris-Micro
    
    print(f"Training config: {training_config.max_steps:,} max steps, LR={training_config.learning_rate}")
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Expected time: 1-2 hours (same as Morris-Micro)")
    print()
    
    print("🔬 ARCHITECTURE COMPARISON:")
    print("   BASELINE (Morris-Micro): 4L×144D×6H → 0.124 bits/param")
    print("   TEST (Standard GPT-2):   4L×256D×4H → ? bits/param")
    print("   KEY DIFFERENCES:")
    print("     - Width: 144 → 256 (more capacity per layer)")
    print("     - Heads: 6 → 4 (standard head_dim 64 vs custom 24)")
    print("     - Standard power-of-2 dimensions")
    print()
    
    # Capacity estimation
    print("Starting GPT-2 standard architecture capacity estimation...")
    print("🎯 HYPOTHESIS: Should achieve HIGHER bits/param than 0.124")
    print("Monitor for 'MEMORIZATION_ACHIEVED' convergence")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Run capacity estimation with standard architecture
        capacity_result = estimate_model_capacity(
            model_config=model_config,
            training_config=training_config,
            dataset_sizes=dataset_sizes,
            n_seeds=2,  # Same as Morris-Micro for fair comparison
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
        print("GPT-2 STANDARD ARCHITECTURE RESULTS")
        print("=" * 60)
        
        print(f"Model Parameters: {actual_params:,}")
        print(f"Estimated Capacity: {capacity_bits:,.0f} bits")
        print(f"Bits per Parameter: {bits_per_param:.6f}")
        print(f"R² (fit quality): {r_squared:.3f}")
        print(f"Execution Time: {execution_time/3600:.2f} hours")
        
        # CRITICAL COMPARISON with Morris-Micro baseline
        morris_micro_bpp = 0.124  # Established baseline
        improvement_factor = bits_per_param / morris_micro_bpp
        
        print(f"\n🔍 ARCHITECTURE COMPARISON:")
        print(f"Morris-Micro (custom): {morris_micro_bpp:.6f} bits/param")
        print(f"Standard GPT-2:       {bits_per_param:.6f} bits/param")
        print(f"Improvement factor:    {improvement_factor:.1f}x")
        
        # Assess the architecture hypothesis
        print(f"\n📊 ARCHITECTURE HYPOTHESIS VALIDATION:")
        if improvement_factor > 3.0:
            print("🎉 MAJOR VALIDATION: Standard architecture much better!")
            print("✅ Architecture was the PRIMARY limiting factor")
            print("🚀 Scale standard architecture to Morris target (15M params)")
        elif improvement_factor > 1.5:
            print("📈 MODERATE VALIDATION: Standard architecture significantly better")
            print("✅ Architecture is A limiting factor")
            print("📊 Test larger standard models + dataset scale")
        elif improvement_factor > 1.1:
            print("📊 WEAK VALIDATION: Standard architecture slightly better")
            print("⚠️  Architecture has minor impact")
            print("🔍 Dataset scale or other factors likely more important")
        else:
            print("❌ HYPOTHESIS REJECTED: Standard architecture not better")
            print("⚠️  Architecture is NOT the limiting factor")
            print("🔍 Focus on dataset scale, measurement, or training factors")
        
        # Parameter efficiency comparison
        efficiency_improvement = improvement_factor / param_ratio
        print(f"\nParameter Efficiency: {efficiency_improvement:.1f}x")
        if efficiency_improvement > 1.0:
            print("✅ Standard architecture more efficient per parameter")
        else:
            print("⚠️  Standard architecture less efficient per parameter")
        
        # Success criteria assessment
        success_criteria = [
            (improvement_factor > 1.2, f"Better than baseline: {improvement_factor:.1f}x improvement"),
            (bits_per_param > 0.2, f"Substantial capacity: {bits_per_param:.6f} bits/param"),
            (execution_time < 3*3600, f"Reasonable time: {execution_time/3600:.1f} hours"),
            (r_squared > 0.5, f"Good fit quality: R² = {r_squared:.3f}")
        ]
        
        passed_count = sum(1 for passed, _ in success_criteria if passed)
        
        print(f"\nValidation Assessment:")
        for passed, description in success_criteria:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}: {description}")
        
        overall_success = passed_count >= 3
        
        print(f"\n" + "=" * 60)
        if overall_success:
            print("🎉 STANDARD ARCHITECTURE TEST: SUCCESS!")
            
            if improvement_factor > 2.0:
                print("✅ Architecture was a major factor in limitations")
                print("🚀 Next: Scale standard GPT-2 to Morris target (12L×768D×12H)")
                print("🎯 Expected: Much closer to Morris 3.6 bits/param")
            else:
                print("✅ Standard architecture provides improvement")
                print("📊 Also test dataset scale and other factors")
                
        else:
            print("⚠️  STANDARD ARCHITECTURE TEST: MIXED RESULTS")
            print("🔍 Architecture may not be the primary limitation")
            print("📊 Focus on dataset scale or measurement methodology")
        
        # Projection to Morris scale
        if improvement_factor > 1.2:
            morris_15M_params = 15_000_000
            scaling_factor = morris_15M_params / actual_params
            # Assume same efficiency at larger scale
            projected_morris_bpp = bits_per_param * (scaling_factor ** 0.5)  # Conservative scaling
            morris_target = 3.6
            
            print(f"\n🎯 MORRIS TARGET PROJECTION:")
            print(f"Standard arch at 15M params: ~{projected_morris_bpp:.2f} bits/param")
            print(f"Morris target: {morris_target} bits/param")
            print(f"Gap factor: {morris_target / projected_morris_bpp:.1f}x")
            
            if projected_morris_bpp > 1.0:
                print("🚀 PROMISING: Within striking distance of Morris target")
            else:
                print("📊 PROGRESS: Significant improvement, need additional factors")
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"gpt2_standard_architecture_results_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write(f"GPT-2 Standard Architecture Test Results\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Standard GPT-2 params: {actual_params:,}\n")
            f.write(f"Standard bits/param: {bits_per_param:.6f}\n")
            f.write(f"Morris-Micro baseline: {morris_micro_bpp:.6f}\n")
            f.write(f"Improvement factor: {improvement_factor:.1f}x\n")
            f.write(f"Parameter efficiency: {efficiency_improvement:.1f}x\n")
            f.write(f"Execution time: {execution_time/3600:.2f} hours\n")
            f.write(f"Architecture: {model_config.n_layers}L×{model_config.d_model}D×{model_config.n_heads}H\n")
        
        print(f"\nResults saved to: {results_file}")
        
        return overall_success
        
    except Exception as e:
        print(f"\n❌ Architecture test failed: {e}")
        print("\nDebug information:")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("GPT-2 Standard Architecture Validation")
    print("Testing if standard proportions improve memorization capacity")
    print()
    
    input("Press Enter to start architecture test (1-2 hours)...")
    print()
    
    # Run test
    success = main()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ARCHITECTURE TEST COMPLETE")
        print("Key insight obtained about GPT-2 standard vs custom proportions")
    else:
        print("⚠️  ARCHITECTURE TEST: INVESTIGATION NEEDED")
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

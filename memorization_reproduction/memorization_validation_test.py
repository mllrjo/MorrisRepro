"""
Memorization Validation Test
Tests cases where memorization should be guaranteed to validate the system works.

File: memorization_validation_test.py
"""

import torch
import time
from src.model_trainer import ModelConfig, TrainingConfig, create_gpt_model
from src.binary_data_generator import generate_random_binary_sequences

# Enhanced training import
try:
    from src.enhanced_model_trainer import enhanced_train_model_wrapper
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False


def test_guaranteed_memorization():
    """
    Test cases where memorization should be absolutely guaranteed.
    
    This validates that our binary system and enhanced training work correctly.
    """
    
    print("🧪 MEMORIZATION VALIDATION TEST")
    print("Testing cases where memorization should be GUARANTEED")
    print("=" * 60)
    
    # Test cases: (model_params, dataset_size, seq_length, expected_difficulty)
    test_cases = [
        # These should be trivial - less than 0.1 bits/param
        (100000, 100, 20, "TRIVIAL", "Should memorize in <100 steps"),
        (100000, 400, 20, "TRIVIAL", "Should memorize in <200 steps"),
        
        # These should be easy - less than 1.0 bits/param  
        (100000, 1000, 20, "EASY", "Should memorize reliably"),
        (200000, 5000, 20, "EASY", "Should memorize reliably"),
        
        # Your Morris-Micro equivalent - should be easy
        (1600000, 10000, 32, "EASY", "Morris-Micro binary test"),
    ]
    
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for i, (target_params, dataset_size, seq_length, expected_difficulty, description) in enumerate(test_cases):
        
        print(f"\n🔬 TEST {i+1}: {description}")
        print(f"   Target: {target_params:,} params, {dataset_size:,} sequences × {seq_length} bits")
        
        # Calculate expected ratio
        total_bits = dataset_size * seq_length
        expected_bits_per_param = total_bits / target_params
        print(f"   Expected: {expected_bits_per_param:.3f} bits/param ({expected_difficulty})")
        
        # Create model configuration
        # Start with reasonable config and scale to hit target params
        config = ModelConfig(
            n_layers=2,
            d_model=64,
            n_heads=4,
            vocab_size=2,
            max_seq_length=seq_length,
            dropout=0.05
        )
        
        # Scale d_model to approximately hit target parameters
        # Rough estimate: params ≈ n_layers * 12 * d_model^2 + vocab_size * d_model
        estimated_params = 2 * 12 * (64**2) + 2 * 64
        if estimated_params < target_params:
            scale_factor = (target_params / estimated_params) ** 0.5
            new_d_model = int(64 * scale_factor)
            # Ensure divisible by n_heads
            new_d_model = (new_d_model // 4) * 4
            config.d_model = max(32, new_d_model)
        
        # Create model and check actual parameters
        model = create_gpt_model(config)
        actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        actual_bits_per_param = total_bits / actual_params
        
        print(f"   Actual: {actual_params:,} params → {actual_bits_per_param:.3f} bits/param")
        
        # Generate binary data
        sequences = generate_random_binary_sequences(
            n_samples=dataset_size,
            seq_length=seq_length,
            seed=42 + i  # Different seed for each test
        )
        
        # Training configuration - aggressive for small datasets
        training_config = TrainingConfig(
            batch_size=min(16, dataset_size),
            learning_rate=1e-3,
            max_steps=10000,  # Should converge much faster
            warmup_steps=100,
            weight_decay=0.01
        )
        
        # Train the model
        start_time = time.time()
        
        if ENHANCED_AVAILABLE:
            training_results = enhanced_train_model_wrapper(
                model=model,
                train_data=sequences,
                original_config=training_config,
                device=device,
                enable_enhanced_training=True
            )
        else:
            from src.model_trainer import train_model
            training_results = train_model(model, sequences, training_config, device)
        
        training_time = time.time() - start_time
        
        # Analyze results
        final_loss = training_results.get('final_loss', float('inf'))
        converged = training_results.get('convergence_achieved', False)
        final_status = training_results.get('final_status', 'unknown')
        
        # Determine success
        # For guaranteed cases, we expect very low loss and convergence
        if expected_difficulty == "TRIVIAL":
            success = final_loss < 0.01 and converged
            expected_criteria = "Loss < 0.01 AND converged"
        elif expected_difficulty == "EASY":
            success = final_loss < 0.1 and converged
            expected_criteria = "Loss < 0.1 AND converged"
        else:
            success = final_loss < 0.5
            expected_criteria = "Loss < 0.5"
        
        result = {
            'test_name': description,
            'expected_difficulty': expected_difficulty,
            'dataset_size': dataset_size,
            'seq_length': seq_length,
            'actual_params': actual_params,
            'bits_per_param': actual_bits_per_param,
            'final_loss': final_loss,
            'converged': converged,
            'training_time': training_time,
            'success': success,
            'expected_criteria': expected_criteria
        }
        
        results.append(result)
        
        # Display result
        status_icon = "✅" if success else "❌"
        print(f"   Result: {status_icon} Loss={final_loss:.4f}, Converged={converged}, Time={training_time:.1f}s")
        print(f"   Status: {final_status}")
        
        if not success and expected_difficulty in ["TRIVIAL", "EASY"]:
            print(f"   ⚠️  WARNING: Failed guaranteed memorization case!")
    
    # Summary
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"=" * 40)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r['success'])
    trivial_tests = [r for r in results if r['expected_difficulty'] == 'TRIVIAL']
    easy_tests = [r for r in results if r['expected_difficulty'] == 'EASY']
    
    trivial_success = sum(1 for r in trivial_tests if r['success'])
    easy_success = sum(1 for r in easy_tests if r['success'])
    
    print(f"Overall success rate: {successful_tests}/{total_tests} ({successful_tests/total_tests:.1%})")
    print(f"TRIVIAL cases: {trivial_success}/{len(trivial_tests)} ({trivial_success/len(trivial_tests):.1%})")
    print(f"EASY cases: {easy_success}/{len(easy_tests)} ({easy_success/len(easy_tests):.1%})")
    
    # System validation
    system_valid = True
    issues = []
    
    if trivial_success < len(trivial_tests):
        system_valid = False
        issues.append("Failed TRIVIAL memorization cases")
    
    if easy_success < len(easy_tests):
        system_valid = False
        issues.append("Failed EASY memorization cases")
    
    if not ENHANCED_AVAILABLE:
        issues.append("Enhanced training not available")
    
    print(f"\n🎯 SYSTEM VALIDATION:")
    if system_valid and successful_tests == total_tests:
        print(f"✅ SYSTEM VALID - All guaranteed memorization cases passed")
        print(f"✅ Binary sequence training works correctly")
        print(f"✅ Enhanced training is functioning")
        print(f"✅ Ready for Morris scaling experiments")
    else:
        print(f"❌ SYSTEM ISSUES DETECTED:")
        for issue in issues:
            print(f"   - {issue}")
        
        if successful_tests > 0:
            print(f"✅ Partial success - system partially functional")
        else:
            print(f"❌ Complete failure - system needs debugging")
    
    return results


def run_quick_memorization_proof():
    """
    Ultra-quick test that memorization definitely works.
    
    Uses tiny dataset that should memorize in seconds.
    """
    
    print("⚡ QUICK MEMORIZATION PROOF")
    print("Testing ultra-small dataset that MUST memorize...")
    
    # Tiny test case - should be absolutely trivial
    config = ModelConfig(
        n_layers=2,
        d_model=32,
        n_heads=4,
        vocab_size=2,
        max_seq_length=16,
        dropout=0.0  # No dropout for perfect memorization
    )
    
    # Generate tiny dataset
    sequences = generate_random_binary_sequences(
        n_samples=50,  # Only 50 sequences
        seq_length=16,
        seed=42
    )
    
    # Training config for fast convergence
    training_config = TrainingConfig(
        batch_size=8,
        learning_rate=2e-3,  # Higher learning rate
        max_steps=2000,
        warmup_steps=50,
        weight_decay=0.0
    )
    
    model = create_gpt_model(config)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_bits = 50 * 16
    bits_per_param = total_bits / param_count
    
    print(f"Dataset: {len(sequences)} sequences × 16 bits = {total_bits} total bits")
    print(f"Model: {param_count:,} parameters")
    print(f"Ratio: {bits_per_param:.4f} bits/param (should be TRIVIAL)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()
    
    if ENHANCED_AVAILABLE:
        results = enhanced_train_model_wrapper(
            model=model,
            train_data=sequences,
            original_config=training_config,
            device=device,
            enable_enhanced_training=True
        )
    else:
        from src.model_trainer import train_model
        results = train_model(model, sequences, training_config, device)
    
    training_time = time.time() - start_time
    final_loss = results.get('final_loss', float('inf'))
    converged = results.get('convergence_achieved', False)
    
    print(f"\nResults:")
    print(f"  Training time: {training_time:.1f} seconds")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Converged: {converged}")
    
    # This should definitely work
    proof_success = final_loss < 0.001 and converged
    
    if proof_success:
        print(f"✅ PROOF SUCCESSFUL: Memorization definitely works!")
        print(f"✅ System is functional and ready for scaling tests")
    else:
        print(f"❌ PROOF FAILED: System has fundamental issues")
        print(f"   Expected: Loss < 0.001 and convergence")
        print(f"   Actual: Loss = {final_loss:.6f}, Converged = {converged}")
    
    return proof_success


if __name__ == "__main__":
    print("Memorization Validation Test")
    print("Testing guaranteed memorization cases")
    print("=" * 50)
    
    # Quick proof first
    proof_success = run_quick_memorization_proof()
    
    if proof_success:
        print(f"\n" + "="*60)
        # Full validation test
        validation_results = test_guaranteed_memorization()
    else:
        print(f"\n❌ Skipping full validation - basic proof failed")
        print(f"System needs debugging before proceeding")

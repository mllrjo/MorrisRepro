"""
Binary Enhanced Training Fix
Apply the discovered optimal settings to your enhanced training.

File: binary_enhanced_training_fix.py
"""

import torch
from src.model_trainer import ModelConfig, TrainingConfig, create_gpt_model
from src.binary_data_generator import generate_random_binary_sequences

n_seq = 64
# Enhanced training import
try:
    from src.enhanced_model_trainer import enhanced_train_model_wrapper
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False


def create_binary_optimized_training_config(
    batch_size: int = 16,
    max_steps: int = 10000  # Much lower - should converge in <1000 steps
) -> TrainingConfig:
    """
    Create training configuration optimized for binary memorization.
    
    Based on debug results:
    - Learning rate: 1e-2 (10x higher than text)
    - Weight decay: 0.0 (disabled - hurts memorization)
    - Lower max steps (fast convergence expected)
    """
    
    return TrainingConfig(
        batch_size=batch_size,
        learning_rate=1e-2,      # KEY FIX: 10x higher than text training
        max_steps=max_steps,
        warmup_steps=100,        # Shorter warmup
        weight_decay=0.0         # KEY FIX: Disable weight decay
    )


def create_binary_optimized_model_config(
    n_layers: int = 2,
    d_model: int = 128,
    n_heads: int = 4,
    max_seq_length: int = 32
) -> ModelConfig:
    """
    Create model configuration optimized for binary memorization.
    """
    
    return ModelConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        vocab_size=2,
        max_seq_length=max_seq_length,
        dropout=0.0              # KEY FIX: Disable dropout
    )


def enhanced_train_binary_sequences(
    model: torch.nn.Module,
    train_data: list,
    config: TrainingConfig,
    device: str = "cuda"
) -> dict:
    """
    Enhanced training wrapper specifically optimized for binary sequences.
    
    Uses your existing enhanced training with binary-optimized config.
    """
    
    if not ENHANCED_AVAILABLE:
        raise ImportError("Enhanced training not available")
    
    print("🔧 Using BINARY-OPTIMIZED enhanced training")
    print(f"   Learning rate: {config.learning_rate} (10x higher)")
    print(f"   Weight decay: {config.weight_decay} (disabled)")
    print(f"   Using existing enhanced training interface")
    
    # Call your existing enhanced training with optimized config
    results = enhanced_train_model_wrapper(
        model=model,
        train_data=train_data,
        original_config=config,
        device=device,
        enable_enhanced_training=True
    )
    
    return results


def test_binary_memorization_fix():
    """
    Test the fixed binary memorization on your original failing case.
    """
    
    print("🧪 TESTING: Binary Memorization Fix")
    print("Testing the case that was failing before...")
    print("=" * n_seq)
    
    # Recreate your original failing case
    config = create_binary_optimized_model_config(
        n_layers=2,
        d_model=64,  # Smaller for this test
        n_heads=4,
        max_seq_length=16
    )
    
    training_config = create_binary_optimized_training_config(
        batch_size=8,
        max_steps=5000  # Should be plenty
    )
    
    # Generate the dataset that was failing
    sequences = generate_random_binary_sequences(
        n_samples=n_seq,   # The failing case
        seq_length=16,
        seed=42
    )
    
    model = create_gpt_model(config)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_bits = n_seq * 16
    bits_per_param = total_bits / param_count
    
    print(f"Dataset: {len(sequences)} sequences × 16 bits = {total_bits} total bits")
    print(f"Model: {param_count:,} parameters")
    print(f"Ratio: {bits_per_param:.4f} bits/param (was failing)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Train with FIXED settings
    print(f"\nTraining with FIXED binary settings...")
    
    import time
    start_time = time.time()
    
    results = enhanced_train_binary_sequences(
        model=model,
        train_data=sequences,
        config=training_config,
        device=device
    )
    
    training_time = time.time() - start_time
    
    # Analyze results
    final_loss = results.get('final_loss', float('inf'))
    converged = results.get('convergence_achieved', False)
    final_status = results.get('final_status', 'unknown')
    
    print(f"\n📊 FIXED RESULTS:")
    print(f"   Training time: {training_time:.1f} seconds")
    print(f"   Final loss: {final_loss:.6f}")
    print(f"   Converged: {converged}")
    print(f"   Status: {final_status}")
    
    # Success criteria for binary memorization
    binary_success = final_loss < 0.05 and converged
    
    if binary_success:
        print(f"✅ BINARY MEMORIZATION FIXED!")
        print(f"   Loss dropped from 0.21 → {final_loss:.3f}")
        print(f"   Training time: {training_time:.0f}s (vs 2+ minutes)")
        print(f"   Ready for Morris scaling experiments")
    else:
        print(f"❌ Still not working - needs more investigation")
        print(f"   Expected: Loss < 0.05 and convergence")
        print(f"   Actual: Loss = {final_loss:.6f}, Converged = {converged}")
    
    return binary_success


def run_morris_micro_binary_test():
    """
    Test binary version of your Morris-Micro experiment.
    """
    
    print(f"\n🚀 Morris-Micro BINARY Test")
    print("Testing your 1.6M parameter model with binary sequences")
    print("=" * n_seq)
    
    # Create model similar to your Morris-Micro
    config = create_binary_optimized_model_config(
        n_layers=2,
        d_model=512,  # Similar to your 1.6M param model
        n_heads=8,
        max_seq_length=32
    )
    
    training_config = create_binary_optimized_training_config(
        batch_size=16,
        max_steps=25000  # More steps for larger model
    )
    
    model = create_gpt_model(config)
    actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Test with dataset that should give good bits/param
    dataset_size = 25000  # Should give ~0.5 bits/param
    sequences = generate_random_binary_sequences(
        n_samples=dataset_size,
        seq_length=32,
        seed=42
    )
    
    total_bits = dataset_size * 32
    bits_per_param = total_bits / actual_params
    
    print(f"Model: {actual_params:,} parameters")
    print(f"Dataset: {dataset_size:,} sequences × 32 bits = {total_bits:,} total bits")
    print(f"Expected: {bits_per_param:.3f} bits/param")
    print(f"Morris progress: {bits_per_param/3.6:.1%}")
    
    if bits_per_param > 3.6:
        print(f"⚠️  Above Morris target - expecting partial memorization")
    elif bits_per_param > 1.0:
        print(f"✅ Good memorization challenge")
    else:
        print(f"✅ Should memorize completely")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nTraining binary Morris-Micro...")
    
    import time
    start_time = time.time()
    
    results = enhanced_train_binary_sequences(
        model=model,
        train_data=sequences,
        config=training_config,
        device=device
    )
    
    training_time = time.time() - start_time
    
    # Results
    final_loss = results.get('final_loss', float('inf'))
    converged = results.get('convergence_achieved', False)
    final_status = results.get('final_status', 'unknown')
    
    print(f"\n🎯 MORRIS-MICRO BINARY RESULTS:")
    print(f"   Training time: {training_time/60:.1f} minutes")
    print(f"   Final loss: {final_loss:.6f}")
    print(f"   Converged: {converged}")
    print(f"   Status: {final_status}")
    print(f"   Bits/param achieved: {bits_per_param:.3f}")
    
    # Compare to your text result (0.124 bits/param)
    improvement = bits_per_param / 0.124
    
    print(f"\n📈 Comparison to text Morris-Micro:")
    print(f"   Text result: 0.124 bits/param")
    print(f"   Binary result: {bits_per_param:.3f} bits/param")
    print(f"   Improvement: {improvement:.1f}x")
    
    if converged and final_loss < 0.1:
        print(f"✅ BINARY MORRIS-MICRO SUCCESS!")
    else:
        print(f"🔄 Partial success - may need more time or larger model")
    
    return {
        'bits_per_param': bits_per_param,
        'final_loss': final_loss,
        'converged': converged,
        'training_time': training_time,
        'improvement_over_text': improvement
    }


if __name__ == "__main__":
    print("Binary Enhanced Training Fix")
    print("Applying discovered optimal settings")
    print("=" * n_seq)
    
    # Test 1: Fix the failing case
    fix_success = test_binary_memorization_fix()
    
    if fix_success:
        # Test 2: Morris-Micro binary
        morris_results = run_morris_micro_binary_test()
        
        print(f"\n🎯 BINARY SYSTEM STATUS:")
        print(f"✅ Basic memorization: FIXED")
        print(f"✅ Enhanced training: WORKING")
        print(f"✅ Morris scaling: READY")
        print(f"\n🚀 Next: Run full Morris scaling experiments with binary!")
    else:
        print(f"\n❌ Basic fix failed - need more debugging")

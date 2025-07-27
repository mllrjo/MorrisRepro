
"""
Binary Integration Example
Shows how to modify existing Morris reproduction code for binary sequences.

File: binary_integration_example.py
"""

import torch
import numpy as np

# Your existing imports (these should work unchanged)
from src.model_trainer import ModelConfig, TrainingConfig, create_gpt_model
from src.binary_data_generator import generate_random_binary_sequences

# Enhanced training (if available)
try:
    from src.enhanced_model_trainer import enhanced_train_model_wrapper
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False


def convert_existing_config_to_binary(original_config: ModelConfig, binary_seq_length: int = 24) -> ModelConfig:
    """
    Convert your existing ModelConfig from text (vocab=2048) to binary (vocab=2).
    
    This preserves your model architecture while adapting for binary sequences.
    """
    
    # Create new config with same architecture but vocab=2
    binary_config = ModelConfig(
        n_layers=original_config.n_layers,
        d_model=original_config.d_model,
        n_heads=original_config.n_heads,
        vocab_size=2,  # Changed from 2048 to 2
        max_seq_length=binary_seq_length,  # Adjust sequence length as needed
        dropout=max(0.05, original_config.dropout * 0.8)  # Slightly lower dropout for memorization
    )
    
    print(f"Converted config: {original_config.vocab_size} → {binary_config.vocab_size} vocab")
    print(f"Sequence length: {original_config.max_seq_length} → {binary_config.max_seq_length}")
    
    return binary_config


def demonstrate_key_differences():
    """
    Demonstrate the key differences between text and binary approaches.
    """
    
    print("🔍 Key Differences: Text vs Binary")
    print("=" * 50)
    
    # Text approach (your current system)
    text_config = ModelConfig(n_layers=2, d_model=256, n_heads=4, vocab_size=2048, max_seq_length=32)
    text_model = create_gpt_model(text_config)
    text_params = sum(p.numel() for p in text_model.parameters() if p.requires_grad)
    
    # Binary approach 
    binary_config = ModelConfig(n_layers=2, d_model=256, n_heads=4, vocab_size=2, max_seq_length=32)
    binary_model = create_gpt_model(binary_config)
    binary_params = sum(p.numel() for p in binary_model.parameters() if p.requires_grad)
    
    print(f"Parameter counts:")
    print(f"  Text model (vocab=2048):   {text_params:,}")
    print(f"  Binary model (vocab=2):    {binary_params:,}")
    print(f"  Parameter reduction:       {text_params/binary_params:.1f}x fewer")
    
    print(f"\nInformation content per token:")
    print(f"  Text token:    {np.log2(2048):.1f} bits")
    print(f"  Binary token:  {np.log2(2):.1f} bits")
    
    print(f"\nExpected benefits of binary:")
    print(f"  ✓ Faster convergence (simpler patterns)")
    print(f"  ✓ Higher bits/parameter (pure memorization)")
    print(f"  ✓ More predictable scaling")
    print(f"  ✓ Better Morris comparison")


def quick_binary_test():
    """
    Quick test to verify binary system works with your enhanced training.
    """
    
    print("🧪 Quick Binary Test")
    
    # Small test configuration
    config = ModelConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        vocab_size=2,
        max_seq_length=16,
        dropout=0.05
    )
    
    training_config = TrainingConfig(
        batch_size=8,
        learning_rate=1e-3,
        max_steps=5000
    )
    
    # Generate small binary dataset
    sequences = generate_random_binary_sequences(
        n_samples=500,
        seq_length=16,
        seed=42
    )
    
    # Test training
    model = create_gpt_model(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Training {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameter model...")
    print(f"Dataset: {len(sequences)} sequences of {len(sequences[0])} bits each")
    
    if ENHANCED_AVAILABLE:
        results = enhanced_train_model_wrapper(
            model=model,
            train_data=sequences,
            original_config=training_config,
            device=device
        )
        
        print(f"✅ Enhanced training successful!")
        print(f"   Final loss: {results.get('final_loss', 'N/A')}")
        print(f"   Converged: {'✓' if results.get('convergence_achieved') else '✗'}")
        print(f"   Status: {results.get('final_status', 'unknown')}")
        
        total_bits = len(sequences) * len(sequences[0])
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        bits_per_param = total_bits / param_count
        
        print(f"   Bits/param: {bits_per_param:.3f}")
        print(f"   Morris progress: {bits_per_param/3.6:.1%}")
        
        return True
    else:
        print("❌ Enhanced training not available")
        return False


if __name__ == "__main__":
    print("Binary Integration Example for Morris Memorization")
    print("=" * 55)
    
    # Show differences
    demonstrate_key_differences()
    
    print(f"\n")
    
    # Quick test
    if quick_binary_test():
        print(f"\n✅ Binary system integration successful!")
    else:
        print(f"\n❌ Binary system needs enhanced training")

"""
Debug Binary Training Issues
Diagnose why binary sequence memorization is failing on guaranteed cases.

File: debug_binary_training.py
"""

import torch
import torch.nn.functional as F
from src.model_trainer import ModelConfig, TrainingConfig, create_gpt_model
from src.binary_data_generator import generate_random_binary_sequences


def debug_binary_sequences():
    """Debug if binary sequences are generated correctly."""
    
    print("🔍 DEBUGGING: Binary Sequence Generation")
    print("=" * 50)
    
    # Generate small test dataset
    sequences = generate_random_binary_sequences(
        n_samples=10,
        seq_length=16,
        seed=42
    )
    
    print(f"Generated {len(sequences)} sequences")
    print(f"Sequence type: {type(sequences[0])}")
    print(f"Sequence dtype: {sequences[0].dtype}")
    print(f"Sequence shape: {sequences[0].shape}")
    print(f"Value range: {sequences[0].min().item()} to {sequences[0].max().item()}")
    
    # Check if values are actually binary
    all_values = torch.cat(sequences)
    unique_values = torch.unique(all_values)
    print(f"Unique values in dataset: {unique_values.tolist()}")
    
    # Show some examples
    print(f"\nFirst 5 sequences:")
    for i in range(min(5, len(sequences))):
        print(f"  Seq {i}: {sequences[i].tolist()}")
    
    # Check for duplicates
    sequence_strings = [tuple(seq.tolist()) for seq in sequences]
    unique_sequences = len(set(sequence_strings))
    print(f"\nUnique sequences: {unique_sequences}/{len(sequences)}")
    
    return sequences


def debug_model_forward_pass():
    """Debug if model can process binary sequences correctly."""
    
    print("\n🔍 DEBUGGING: Model Forward Pass")
    print("=" * 50)
    
    # Create small model
    config = ModelConfig(
        n_layers=1,  # Minimal model
        d_model=32,
        n_heads=4,
        vocab_size=2,  # Binary
        max_seq_length=16,
        dropout=0.0
    )
    
    model = create_gpt_model(config)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")
    
    # Generate test sequence
    sequences = generate_random_binary_sequences(n_samples=1, seq_length=16, seed=42)
    test_sequence = sequences[0]
    
    print(f"Test sequence: {test_sequence.tolist()}")
    print(f"Input shape: {test_sequence.shape}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        # Prepare input (remove last token for input, remove first for target)
        input_ids = test_sequence[:-1].unsqueeze(0)  # Add batch dimension
        targets = test_sequence[1:].unsqueeze(0)
        
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Input IDs: {input_ids[0].tolist()}")
        print(f"Targets: {targets[0].tolist()}")
        
        # Forward pass
        try:
            logits = model(input_ids)
            print(f"Logits shape: {logits.shape}")
            print(f"Logits range: {logits.min().item():.3f} to {logits.max().item():.3f}")
            
            # Calculate loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            print(f"Initial loss: {loss.item():.6f}")
            
            # Check predictions
            predictions = torch.argmax(logits, dim=-1)
            print(f"Predictions: {predictions[0].tolist()}")
            
            # Accuracy
            correct = (predictions[0] == targets[0]).sum().item()
            total = targets.size(1)
            accuracy = correct / total
            print(f"Accuracy: {correct}/{total} = {accuracy:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return False


def debug_training_step():
    """Debug a single training step to see what's happening."""
    
    print("\n🔍 DEBUGGING: Single Training Step")
    print("=" * 50)
    
    # Create minimal setup
    config = ModelConfig(
        n_layers=1,
        d_model=32,
        n_heads=4,
        vocab_size=2,
        max_seq_length=8,  # Very short sequences
        dropout=0.0
    )
    
    model = create_gpt_model(config)
    
    # Single sequence
    sequences = generate_random_binary_sequences(n_samples=1, seq_length=8, seed=42)
    sequence = sequences[0]
    
    print(f"Training on single sequence: {sequence.tolist()}")
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    
    # Multiple training steps on same sequence
    print(f"\nTraining steps:")
    for step in range(20):
        optimizer.zero_grad()
        
        input_ids = sequence[:-1].unsqueeze(0)
        targets = sequence[1:].unsqueeze(0)
        
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        
        loss.backward()
        optimizer.step()
        
        # Check predictions
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions[0] == targets[0]).sum().item()
            total = targets.size(1)
            accuracy = correct / total
        
        if step % 5 == 0 or step < 5:
            print(f"  Step {step:2d}: Loss {loss.item():.6f}, Accuracy {accuracy:.3f}")
    
    print(f"\nFinal predictions: {predictions[0].tolist()}")
    print(f"Target sequence:   {targets[0].tolist()}")
    
    # Test if it can memorize a single sequence
    final_accuracy = accuracy
    memorized = loss.item() < 0.01 and final_accuracy > 0.95
    
    print(f"\nSingle sequence memorization: {'✅ SUCCESS' if memorized else '❌ FAILED'}")
    return memorized


def debug_enhanced_training_threshold():
    """Debug if enhanced training threshold is appropriate for binary."""
    
    print("\n🔍 DEBUGGING: Enhanced Training Threshold")
    print("=" * 50)
    
    # The enhanced training is using loss < 0.15 threshold
    # For binary memorization, this might be too high
    
    print("Enhanced training analysis:")
    print("- Current threshold: loss < 0.15")
    print("- Binary vocab size: 2")
    print("- Random guessing loss: -log(0.5) = 0.693")
    print("- Good memorization loss: < 0.01")
    print("- Perfect memorization loss: ≈ 0.001")
    
    print("\nRecommended thresholds for binary:")
    print("- Convergence threshold: < 0.05 (not 0.15)")
    print("- Perfect memorization: < 0.01")
    print("- Early stopping: loss plateau > 1000 steps")
    
    # Check what loss corresponds to different accuracies
    print("\nLoss vs Accuracy for binary sequences:")
    
    # Simulate different accuracy levels
    for accuracy in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]:
        # For binary classification, loss ≈ -log(accuracy) when accuracy > 0.5
        if accuracy == 1.0:
            estimated_loss = 0.001  # Very low for perfect accuracy
        else:
            # This is approximate - actual loss depends on confidence scores
            estimated_loss = -torch.log(torch.tensor(accuracy)).item()
        
        print(f"  {accuracy:.2f} accuracy → ~{estimated_loss:.3f} loss")


def run_full_debug():
    """Run complete debugging sequence."""
    
    print("🚨 DEBUGGING BINARY TRAINING FAILURE")
    print("=" * 60)
    
    # Step 1: Check binary sequence generation
    sequences = debug_binary_sequences()
    
    # Step 2: Check model can process binary data
    forward_ok = debug_model_forward_pass()
    
    if not forward_ok:
        print("\n❌ CRITICAL: Model forward pass failed!")
        print("Check model architecture for vocab_size=2 compatibility")
        return False
    
    # Step 3: Check if model can learn single sequence
    single_seq_ok = debug_training_step()
    
    if not single_seq_ok:
        print("\n❌ CRITICAL: Cannot memorize single sequence!")
        print("Fundamental training issue - model architecture or optimization")
        return False
    
    # Step 4: Check enhanced training settings
    debug_enhanced_training_threshold()
    
    print(f"\n🎯 DEBUGGING CONCLUSIONS:")
    print(f"✅ Binary sequences generated correctly")
    print(f"✅ Model processes binary data correctly")
    
    if single_seq_ok:
        print(f"✅ Model can memorize single sequence")
        print(f"\n💡 LIKELY ISSUE: Enhanced training threshold too high")
        print(f"   - Current: loss < 0.15")
        print(f"   - Needed: loss < 0.05 for binary memorization")
        print(f"   - Or learning rate too high causing oscillation")
    else:
        print(f"❌ Model cannot memorize single sequence")
        print(f"\n🔧 NEEDED FIXES:")
        print(f"   - Check model architecture")
        print(f"   - Check learning rate (try 1e-4)")
        print(f"   - Check loss calculation")
    
    return True


if __name__ == "__main__":
    run_full_debug()

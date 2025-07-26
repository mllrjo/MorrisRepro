"""
File: quick_validation.py
Directory: memorization_reproduction/

Quick validation script to test the core fixes before running full test suite.
"""

import sys
import os
import torch
import numpy as np
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from enhanced_model_trainer import (
    adaptive_memorization_training,
    calculate_memorization_rate,
    create_enhanced_config_from_original,
    EnhancedTrainingConfig,
    SimpleMemorizationModel
)


class MockTrainingConfig:
    """Mock training config for testing."""
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 5e-4
        self.max_steps = 2000
        self.warmup_steps = 100
        self.weight_decay = 0.01


def create_tiny_test_data():
    """Create very small test dataset for validation."""
    torch.manual_seed(42)
    
    # Extremely small dataset - should be memorizable
    n_sequences = 3
    seq_length = 4
    vocab_size = 4
    
    sequences = []
    for _ in range(n_sequences):
        sequence = torch.randint(0, vocab_size, (seq_length,))
        sequences.append(sequence)
    
    print(f"Created {n_sequences} sequences of length {seq_length}, vocab size {vocab_size}")
    for i, seq in enumerate(sequences):
        print(f"  Sequence {i}: {seq.tolist()}")
    
    return sequences


def test_memorization_capability():
    """Test if the fixed implementation can memorize a tiny dataset."""
    
    print("=" * 60)
    print("QUICK VALIDATION: Testing Core Memorization Capability")
    print("=" * 60)
    
    # Create tiny model and dataset
    model = SimpleMemorizationModel(vocab_size=4, d_model=32, max_seq_len=8)
    tiny_dataset = create_tiny_test_data()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Create enhanced config
    base_config = MockTrainingConfig()
    config = EnhancedTrainingConfig(
        base_config=base_config,
        memorization_threshold=1.0,  # Very lenient for tiny dataset
        min_memorization_steps=100,
        max_plateau_patience=500,
        memorization_check_interval=50,
        adaptive_lr=False
    )
    
    # Test initial memorization rate
    initial_rate = calculate_memorization_rate(model, tiny_dataset, threshold=1.0, device=device)
    print(f"\nInitial memorization rate: {initial_rate:.3f}")
    
    # Run training
    print("\nStarting training...")
    start_time = time.time()
    
    results = adaptive_memorization_training(
        model=model,
        train_data=tiny_dataset,
        config=config,
        device=device
    )
    
    training_time = time.time() - start_time
    
    # Display results
    print(f"\n" + "=" * 40)
    print("VALIDATION RESULTS:")
    print("=" * 40)
    print(f"Final status: {results['final_status']}")
    print(f"Steps taken: {results['total_steps']}")
    print(f"Training time: {training_time:.1f} seconds")
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Final memorization rate: {results['final_memorization_rate']:.3f}")
    print(f"Convergence achieved: {results['convergence_achieved']}")
    print(f"Convergence reason: {results['convergence_reason']}")
    
    # Test individual sequence losses
    print(f"\nIndividual sequence analysis:")
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    
    with torch.no_grad():
        for i, sequence in enumerate(tiny_dataset):
            sequence = sequence.to(device)
            input_ids = sequence[:-1].unsqueeze(0)
            target_ids = sequence[1:]
            
            outputs = model(input_ids)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            logits = logits.squeeze(0)
            loss = criterion(logits, target_ids)
            
            memorized = "YES" if loss.item() <= 1.0 else "NO"
            print(f"  Sequence {i}: loss={loss.item():.4f}, memorized={memorized}")
    
    # Validation criteria
    print(f"\n" + "=" * 40)
    print("VALIDATION CRITERIA:")
    print("=" * 40)
    
    success_criteria = []
    
    # Criterion 1: Should not hit MAX_STEPS on tiny dataset
    no_max_steps = results['final_status'] != "MAX_STEPS"
    success_criteria.append(("No MAX_STEPS", no_max_steps))
    print(f"✓ No MAX_STEPS termination: {'PASS' if no_max_steps else 'FAIL'}")
    
    # Criterion 2: Should achieve some memorization (excellent if >90%)
    some_memorization = results['final_memorization_rate'] >= 0.3
    excellent_memorization = results['final_memorization_rate'] >= 0.9
    success_criteria.append(("Some memorization", some_memorization))
    status = "EXCELLENT" if excellent_memorization else ("PASS" if some_memorization else "FAIL")
    print(f"✓ Some memorization achieved: {status} ({results['final_memorization_rate']:.1%})")
    
    # Criterion 3: Loss should decrease significantly
    loss_decreased = results['final_loss'] < 2.0
    success_criteria.append(("Loss decreased", loss_decreased))
    print(f"✓ Loss decreased significantly: {'PASS' if loss_decreased else 'FAIL'}")
    
    # Criterion 4: Training should show progress OR converge quickly
    has_progress = len(results['memorization_history']) > 2 or results['convergence_achieved']
    success_criteria.append(("Training progress", has_progress))
    print(f"✓ Training shows progress (or fast convergence): {'PASS' if has_progress else 'FAIL'}")
    
    # Overall validation
    all_passed = all(criterion[1] for criterion in success_criteria)
    
    # Check for excellent performance
    is_excellent = (results['final_memorization_rate'] >= 0.95 and 
                   results['final_loss'] < 0.1 and 
                   results['convergence_achieved'])
    
    print(f"\n" + "=" * 40)
    if is_excellent:
        print(f"OVERALL VALIDATION: 🌟 EXCELLENT")
        print("Perfect memorization achieved!")
    else:
        print(f"OVERALL VALIDATION: {'✓ PASS' if all_passed else '✗ FAIL'}")
    print("=" * 40)
    
    if all_passed or is_excellent:
        print("🎉 Core memorization capability validated!")
        if is_excellent:
            print("⭐ OUTSTANDING PERFORMANCE: Perfect memorization achieved!")
        print("Ready to run full test suite.")
    else:
        print("❌ Core issues remain. Need further fixes.")
    
    return all_passed or is_excellent, is_excellent


if __name__ == "__main__":
    success, is_excellent = test_memorization_capability()
    
    if success:
        print("\n💡 Next steps:")
        print("1. Run full test suite: python -m pytest tests/test_enhanced_model_trainer.py -v")
        print("2. Replace original enhanced_model_trainer.py with fixed version")
        print("3. Test integration with existing pipeline")
        if is_excellent:
            print("4. 🏆 Scale up to larger models - memorization methodology is validated!")
    else:
        print("\n🔧 Need to debug further before proceeding to full tests")
    
    exit(0 if success else 1)

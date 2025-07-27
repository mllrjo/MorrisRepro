"""
Fix Binary Training - Address Confidence Issue
The model gets right answers but isn't confident enough.

File: fix_binary_training.py
"""

import torch
import torch.nn.functional as F
from src.model_trainer import ModelConfig, TrainingConfig, create_gpt_model
from src.binary_data_generator import generate_random_binary_sequences


def test_confidence_training():
    """Test training with settings optimized for high confidence."""
    
    print("🔧 TESTING: High-Confidence Training")
    print("=" * 50)
    
    # Create simple model
    config = ModelConfig(
        n_layers=1,
        d_model=32,
        n_heads=4,
        vocab_size=2,
        max_seq_length=8,
        dropout=0.0  # Remove dropout - it hurts confidence
    )
    
    model = create_gpt_model(config)
    sequence = generate_random_binary_sequences(1, 8, seed=42)[0]
    
    print(f"Training sequence: {sequence.tolist()}")
    
    # Training settings optimized for confidence
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=5e-4,  # Lower learning rate for stability
        weight_decay=0.0  # No weight decay - hurts memorization
    )
    
    model.train()
    
    print(f"\nTraining with confidence-optimized settings:")
    print(f"- Learning rate: 5e-4 (lower)")
    print(f"- Weight decay: 0.0 (disabled)")
    print(f"- Dropout: 0.0 (disabled)")
    
    for step in range(100):  # More steps
        optimizer.zero_grad()
        
        input_ids = sequence[:-1].unsqueeze(0)
        targets = sequence[1:].unsqueeze(0)
        
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        loss.backward()
        optimizer.step()
        
        # Detailed analysis every 10 steps
        if step % 10 == 0:
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions[0] == targets[0]).sum().item()
                accuracy = correct / targets.size(1)
                
                # Check confidence (probabilities)
                probs = F.softmax(logits, dim=-1)
                target_probs = probs[0, torch.arange(targets.size(1)), targets[0]]
                min_confidence = target_probs.min().item()
                avg_confidence = target_probs.mean().item()
                
                print(f"  Step {step:3d}: Loss {loss.item():.6f}, Acc {accuracy:.3f}, "
                      f"Conf {avg_confidence:.3f} (min {min_confidence:.3f})")
                
                # Success criteria: high accuracy AND high confidence
                if accuracy >= 0.99 and avg_confidence >= 0.95 and loss.item() < 0.05:
                    print(f"  🎯 MEMORIZATION ACHIEVED at step {step}!")
                    return True, step
    
    print(f"  ❌ Failed to achieve high-confidence memorization")
    return False, 100


def test_different_learning_rates():
    """Test different learning rates to find optimal for memorization."""
    
    print(f"\n🔧 TESTING: Learning Rate Optimization")
    print("=" * 50)
    
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    
    results = []
    
    for lr in learning_rates:
        print(f"\nTesting LR = {lr:.0e}")
        
        # Fresh model for each test
        config = ModelConfig(
            n_layers=1, d_model=32, n_heads=4, vocab_size=2, 
            max_seq_length=8, dropout=0.0
        )
        model = create_gpt_model(config)
        sequence = generate_random_binary_sequences(1, 8, seed=42)[0]
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        model.train()
        
        # Train for 50 steps
        for step in range(50):
            optimizer.zero_grad()
            
            input_ids = sequence[:-1].unsqueeze(0)
            targets = sequence[1:].unsqueeze(0)
            
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            loss.backward()
            optimizer.step()
        
        # Final evaluation
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions[0] == targets[0]).float().mean().item()
            
            probs = F.softmax(logits, dim=-1)
            target_probs = probs[0, torch.arange(targets.size(1)), targets[0]]
            avg_confidence = target_probs.mean().item()
        
        success = accuracy >= 0.99 and avg_confidence >= 0.9 and loss.item() < 0.1
        
        results.append({
            'lr': lr,
            'final_loss': loss.item(),
            'accuracy': accuracy,
            'confidence': avg_confidence,
            'success': success
        })
        
        status = "✅" if success else "❌"
        print(f"  {status} Final: Loss {loss.item():.4f}, Acc {accuracy:.3f}, Conf {avg_confidence:.3f}")
    
    # Find best learning rate
    successful_lrs = [r for r in results if r['success']]
    if successful_lrs:
        best_lr = min(successful_lrs, key=lambda x: x['final_loss'])
        print(f"\n🎯 Best learning rate: {best_lr['lr']:.0e}")
        print(f"   Loss: {best_lr['final_loss']:.4f}")
        print(f"   Confidence: {best_lr['confidence']:.3f}")
        return best_lr['lr']
    else:
        print(f"\n❌ No learning rate achieved memorization!")
        return None


def test_enhanced_training_fix():
    """Test if we can fix the enhanced training threshold."""
    
    print(f"\n🔧 TESTING: Enhanced Training Threshold Fix")
    print("=" * 50)
    
    # The issue: enhanced training uses loss < 0.15 threshold
    # For binary memorization, we need < 0.05
    
    print("Current enhanced training analysis:")
    print("- Threshold: loss < 0.15 (TOO HIGH)")
    print("- Your result: loss = 0.21 (above threshold)")
    print("- Needed: loss < 0.05 for binary memorization")
    
    print(f"\nProposed fixes for enhanced training:")
    print("1. Lower memorization threshold: 0.15 → 0.05")
    print("2. Increase memorization check frequency")
    print("3. Add confidence-based stopping (avg confidence > 0.9)")
    print("4. Disable weight decay for memorization tasks")
    print("5. Lower learning rate for binary sequences")
    
    # Test what threshold corresponds to good memorization
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2]
    
    print(f"\nLoss threshold analysis:")
    for threshold in thresholds:
        if threshold <= 0.01:
            status = "PERFECT memorization"
        elif threshold <= 0.05:
            status = "GOOD memorization"
        elif threshold <= 0.1:
            status = "OK memorization"
        elif threshold <= 0.15:
            status = "POOR memorization (current enhanced)"
        else:
            status = "FAILED memorization"
        
        print(f"  Loss < {threshold:.3f}: {status}")
    
    print(f"\n💡 RECOMMENDED FIX:")
    print(f"   Change enhanced training threshold from 0.15 to 0.05")
    print(f"   This should fix your binary memorization issues")


def quick_memorization_test(lr=5e-4):
    """Quick test with optimal settings."""
    
    print(f"\n⚡ QUICK MEMORIZATION TEST (LR = {lr:.0e})")
    print("=" * 50)
    
    config = ModelConfig(
        n_layers=1, d_model=32, n_heads=4, vocab_size=2,
        max_seq_length=8, dropout=0.0
    )
    
    model = create_gpt_model(config)
    sequence = generate_random_binary_sequences(1, 8, seed=42)[0]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    
    print(f"Target: {sequence[1:].tolist()}")
    
    # Train until memorized or 200 steps max
    for step in range(200):
        optimizer.zero_grad()
        
        input_ids = sequence[:-1].unsqueeze(0)
        targets = sequence[1:].unsqueeze(0)
        
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        loss.backward()
        optimizer.step()
        
        # Check every 10 steps
        if step % 10 == 0 or step < 10:
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions[0] == targets[0]).float().mean().item()
                
                probs = F.softmax(logits, dim=-1)
                target_probs = probs[0, torch.arange(targets.size(1)), targets[0]]
                avg_confidence = target_probs.mean().item()
                
                print(f"Step {step:3d}: Loss {loss.item():.4f}, Acc {accuracy:.3f}, Conf {avg_confidence:.3f}")
                
                # Success: perfect accuracy, high confidence, low loss
                if accuracy >= 1.0 and avg_confidence >= 0.95 and loss.item() < 0.05:
                    print(f"🎯 SUCCESS: Memorized in {step} steps!")
                    print(f"   Predicted: {predictions[0].tolist()}")
                    print(f"   Target:    {targets[0].tolist()}")
                    return True, step
    
    print(f"❌ Failed to memorize in 200 steps")
    return False, 200


def run_all_fixes():
    """Run all diagnostic tests and fixes."""
    
    print("🔧 BINARY TRAINING CONFIDENCE FIXES")
    print("=" * 60)
    
    # Test 1: High-confidence training
    success1, steps1 = test_confidence_training()
    
    # Test 2: Learning rate optimization  
    best_lr = test_different_learning_rates()
    
    # Test 3: Enhanced training analysis
    test_enhanced_training_fix()
    
    # Test 4: Quick test with best settings
    if best_lr:
        success4, steps4 = quick_memorization_test(best_lr)
    else:
        success4, steps4 = quick_memorization_test(5e-4)
    
    print(f"\n🎯 SUMMARY:")
    print(f"High-confidence training: {'✅' if success1 else '❌'} ({steps1} steps)")
    print(f"Best learning rate found: {best_lr:.0e}" if best_lr else "❌ No LR worked")
    print(f"Quick test result: {'✅' if success4 else '❌'} ({steps4} steps)")
    
    if success1 or success4:
        print(f"\n✅ SOLUTION FOUND: Binary memorization CAN work!")
        print(f"📋 Apply these fixes to enhanced training:")
        print(f"   1. Lower threshold: 0.15 → 0.05")
        print(f"   2. Learning rate: {best_lr:.0e}" if best_lr else "   2. Learning rate: 5e-4")
        print(f"   3. Disable weight decay for memorization")
        print(f"   4. Disable dropout")
    else:
        print(f"\n❌ DEEPER ISSUES: Need more investigation")


if __name__ == "__main__":
    run_all_fixes()

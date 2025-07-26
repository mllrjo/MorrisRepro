#!/usr/bin/env python3
"""
Quick test to verify enhanced training fixes MAX_STEPS convergence issue
Run this to validate the fix before scaling up
"""

import os
import sys
import torch

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

def test_enhanced_training_fix():
    """Test enhanced training on small model to verify MAX_STEPS fix"""
    
    print("Testing Enhanced Training Fix for MAX_STEPS Issue")
    print("=" * 60)
    
    # Import modules
    try:
        from data_generator import generate_uniform_bitstrings
        from model_trainer import create_gpt_model
        from enhanced_training import adaptive_memorization_training, EnhancedTrainingConfig
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure enhanced_training.py is in src/ directory")
        return False
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")
    
    # Create test model (small for quick testing)
    class TestModelConfig:
        def __init__(self):
            self.n_layers = 2
            self.d_model = 128
            self.n_heads = 4
            self.vocab_size = 1000
            self.max_seq_length = 32
            self.dropout = 0.1
    
    # Generate test data (small dataset)
    print("Generating test data...")
    train_data = generate_uniform_bitstrings(
        n_samples=25,  # Small for quick test
        seq_length=16,  # Short sequences
        vocab_size=1000,
        seed=42
    )
    print(f"✅ Generated {len(train_data)} test sequences")
    
    # Create model
    print("Creating test model...")
    model_config = TestModelConfig()
    model = create_gpt_model(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Created model with {n_params:,} parameters")
    
    # Enhanced training configuration
    enhanced_config = EnhancedTrainingConfig(
        batch_size=8,
        learning_rate=1e-3,
        max_steps=15_000,  # Reasonable for quick test
        memorization_threshold=0.15,
        patience=3000,
        memorization_check_interval=200
    )
    
    print("\nEnhanced Training Configuration:")
    print(f"  Max steps: {enhanced_config.max_steps:,}")
    print(f"  Memorization threshold: {enhanced_config.memorization_threshold}")
    print(f"  Patience: {enhanced_config.patience:,}")
    print(f"  Adaptive LR: {enhanced_config.use_adaptive_lr}")
    
    # Run enhanced training
    print("\nStarting enhanced training test...")
    print("This should converge WITHOUT hitting MAX_STEPS")
    print("-" * 40)
    
    try:
        metrics = adaptive_memorization_training(
            model=model,
            train_data=train_data,
            config=enhanced_config,
            device=device
        )
        
        # Analyze results
        print("\n" + "=" * 60)
        print("ENHANCED TRAINING TEST RESULTS")
        print("=" * 60)
        
        convergence_info = metrics['convergence_info']
        
        if convergence_info['converged']:
            print("✅ SUCCESS: Training converged!")
            print(f"   Reason: {convergence_info['reason']}")
            print(f"   Final step: {convergence_info['final_step']:,}")
            print(f"   Final loss: {convergence_info['final_loss']:.4f}")
            
            if convergence_info['reason'] == 'MEMORIZATION_ACHIEVED':
                print("🎉 PERFECT: Achieved memorization threshold!")
            elif convergence_info['reason'] == 'HIGH_MEMORIZATION_RATE':
                print("🎉 EXCELLENT: High memorization rate achieved!")
            elif convergence_info['reason'] == 'LOSS_PLATEAU':
                print("✅ GOOD: Converged via loss plateau (better than MAX_STEPS)")
                
        else:
            print("❌ ISSUE: Still hitting MAX_STEPS")
            print(f"   Final step: {convergence_info['final_step']:,}")
            print(f"   Final loss: {convergence_info['final_loss']:.4f}")
            print("   Consider increasing max_steps or adjusting parameters")
        
        # Show training progress
        if len(metrics['loss']) > 0:
            initial_loss = metrics['loss'][0]
            final_loss = metrics['loss'][-1]
            improvement = initial_loss - final_loss
            print(f"\nTraining Progress:")
            print(f"   Initial loss: {initial_loss:.4f}")
            print(f"   Final loss: {final_loss:.4f}")
            print(f"   Improvement: {improvement:.4f}")
            
        if len(metrics['memorization_rate']) > 0:
            final_mem_rate = metrics['memorization_rate'][-1]
            print(f"   Final memorization rate: {final_mem_rate:.3f}")
        
        return convergence_info['converged']
        
    except Exception as e:
        print(f"❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return False


def integration_instructions():
    """Print instructions for integrating into main pipeline"""
    
    print("\n" + "=" * 60)
    print("INTEGRATION INSTRUCTIONS")
    print("=" * 60)
    
    print("""
If the test above succeeded, integrate enhanced training into your pipeline:

1. MODIFY model_trainer.py:
   Add this at the top:
   ```python
   from enhanced_training import enhanced_train_model_wrapper
   ```
   
   Replace train_model calls with:
   ```python
   metrics = enhanced_train_model_wrapper(model, train_data, config, device)
   ```

2. OR modify capacity_estimator.py directly:
   ```python
   from enhanced_training import adaptive_memorization_training, EnhancedTrainingConfig
   
   enhanced_config = EnhancedTrainingConfig(
       batch_size=config.batch_size,
       learning_rate=config.learning_rate,
       max_steps=100_000,  # Increased!
       memorization_threshold=0.15
   )
   
   metrics = adaptive_memorization_training(model, train_data, enhanced_config, device)
   ```

3. RUN full pipeline test:
   ```bash
   python test_complete_pipeline.py
   ```
   
   You should see models converge with reasons like:
   - "MEMORIZATION_ACHIEVED" 
   - "HIGH_MEMORIZATION_RATE"
   Instead of "MAX_STEPS"

4. SCALE UP with confidence:
   - Models should now converge reliably
   - Ready to test 2M+ parameter models
   - Path to Morris target is clear
""")


if __name__ == "__main__":
    # Run the test
    success = test_enhanced_training_fix()
    
    # Print integration instructions
    integration_instructions()
    
    # Final summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if success:
        print("🎉 ENHANCED TRAINING FIX: SUCCESS")
        print("✅ Ready to integrate into main pipeline")
        print("✅ Ready to scale to larger models")
        print("✅ MAX_STEPS convergence issue resolved")
    else:
        print("❌ ENHANCED TRAINING FIX: NEEDS WORK")
        print("   Check error messages above")
        print("   May need parameter tuning")
        
    print("\nNext steps:")
    print("1. If successful: Integrate into main pipeline")
    print("2. Run full test_complete_pipeline.py")
    print("3. Scale to Morris target models (2-20M parameters)")

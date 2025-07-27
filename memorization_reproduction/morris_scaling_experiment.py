"""
File: morris_scaling_experiment.py
Example of how to use enhanced training for Morris reproduction scaling
"""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_trainer import TrainingConfig, create_gpt_model, ModelConfig
from data_generator import generate_uniform_bitstrings
from enhanced_model_trainer import enhanced_train_model_wrapper

def run_morris_scaling_experiment():
    """Example of progressive scaling for Morris reproduction."""
    
    # Define model configurations for scaling
    model_configs = [
        # Small scale validation
        ModelConfig(n_layers=2, d_model=128, n_heads=4, vocab_size=256, max_seq_length=32),
        
        # Morris-Micro scale (2M params)
        ModelConfig(n_layers=8, d_model=384, n_heads=6, vocab_size=2048, max_seq_length=64),
        
        # Morris-Target scale (15M params)  
        ModelConfig(n_layers=12, d_model=768, n_heads=12, vocab_size=2048, max_seq_length=64)
    ]
    
    # Training configuration
    training_config = TrainingConfig(
        batch_size=16,
        learning_rate=1e-3,
        max_steps=50000,
        warmup_steps=1000,
        weight_decay=0.01
    )
    
    # Generate dataset
    dataset = generate_uniform_bitstrings(
        n_samples=100,
        seq_length=32,
        vocab_size=256,
        seed=42
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Progressive scaling experiment
    for i, model_config in enumerate(model_configs):
        print(f"\n=== Model {i+1}: {model_config.n_layers} layers, {model_config.d_model} d_model ===")
        
        model = create_gpt_model(model_config)
        model = model.to(device)
        
        # Enhanced training with memorization tracking
        results = enhanced_train_model_wrapper(
            model=model,
            train_data=dataset,
            original_config=training_config,
            device=device,
            memorization_threshold=0.5,           # Explicit
            memorization_check_interval=250,      # Explicit  
            enable_enhanced_training=True         # Explicit
        )
        
        # Now you get memorization metrics:
        memorization_rate = results['final_memorization_rate']
        convergence_achieved = results['convergence_achieved']
        final_loss = results['final_loss']
        total_steps = results['total_steps']
        
        print(f"Results:")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Memorization rate: {memorization_rate:.3f}")
        print(f"  Converged: {convergence_achieved}")
        print(f"  Steps taken: {total_steps}")

if __name__ == "__main__":
    run_morris_scaling_experiment()
